import os
from typing import Any, Callable, List, Optional, Tuple, Union

import gym
import numpy as np

from env.worker import (
    DummyEnvWorker,
    EnvWorker,
    RayEnvWorker,
    SubprocEnvWorker,
)


class BaseVectorEnv(gym.Env):

    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        worker_fn: Callable[[Callable[[], gym.Env]], EnvWorker],
        wait_num: Optional[int] = None,
        timeout: Optional[float] = None,
        norm_obs: bool = False,
        update_obs_rms: bool = True,
    ) -> None:
        self._env_fns = env_fns
        # A VectorEnv contains a pool of EnvWorkers, which corresponds to
        # interact with the given envs (one worker <-> one env).
        self.workers = [worker_fn(fn) for fn in env_fns]
        self.worker_class = type(self.workers[0])
        assert issubclass(self.worker_class, EnvWorker)
        assert all([isinstance(w, self.worker_class) for w in self.workers])

        self.env_num = len(env_fns)
        self.wait_num = wait_num or len(env_fns)
        assert 1 <= self.wait_num <= len(env_fns), \
            f"wait_num should be in [1, {len(env_fns)}], but got {wait_num}"
        self.timeout = timeout
        assert self.timeout is None or self.timeout > 0, \
            f"timeout is {timeout}, it should be positive if provided!"
        self.is_async = self.wait_num != len(env_fns) or timeout is not None
        self.waiting_conn: List[EnvWorker] = []
        # environments in self.ready_id is actually ready
        # but environments in self.waiting_id are just waiting when checked,
        # and they may be ready now, but this is not known until we check it
        # in the step() function
        self.waiting_id: List[int] = []
        # all environments are ready in the beginning
        self.ready_id = list(range(self.env_num))
        self.is_closed = False

        # initialize observation running mean/std
        self.norm_obs = norm_obs
        self.update_obs_rms = update_obs_rms
        self.__eps = np.finfo(np.float32).eps.item()

    def _assert_is_not_closed(self) -> None:
        assert not self.is_closed, \
            f"Methods of {self.__class__.__name__} cannot be called after close."

    def __len__(self) -> int:
        """Return len(self), which is the number of environments."""
        return self.env_num

    def __getattribute__(self, key: str) -> Any:
        """Switch the attribute getter depending on the key.

        Any class who inherits ``gym.Env`` will inherit some attributes, like
        ``action_space``. However, we would like the attribute lookup to go straight
        into the worker (in fact, this vector env's action_space is always None).
        """
        if key in [
            'metadata', 'reward_range', 'spec', 'action_space', 'observation_space'
        ]:  # reserved keys in gym.Env
            return self.get_env_attr(key)
        else:
            return super().__getattribute__(key)

    def get_env_attr(
        self,
        key: str,
        id: Optional[Union[int, List[int], np.ndarray]] = None
    ) -> List[Any]:
        """Get an attribute from the underlying environments.

        If id is an int, retrieve the attribute denoted by key from the environment
        underlying the worker at index id. The result is returned as a list with one
        element. Otherwise, retrieve the attribute for all workers at indices id and
        return a list that is ordered correspondingly to id.

        :param str key: The key of the desired attribute.
        :param id: Indice(s) of the desired worker(s). Default to None for all env_id.

        :return list: The list of environment attributes.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        return [self.workers[j].get_env_attr(key) for j in id]

    def set_env_attr(
        self,
        key: str,
        value: Any,
        id: Optional[Union[int, List[int], np.ndarray]] = None
    ) -> None:
        """Set an attribute in the underlying environments.

        If id is an int, set the attribute denoted by key from the environment
        underlying the worker at index id to value.
        Otherwise, set the attribute for all workers at indices id.

        :param str key: The key of the desired attribute.
        :param Any value: The new value of the attribute.
        :param id: Indice(s) of the desired worker(s). Default to None for all env_id.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)
        for j in id:
            self.workers[j].set_env_attr(key, value)

    def _wrap_id(
        self,
        id: Optional[Union[int, List[int], np.ndarray]] = None
    ) -> Union[List[int], np.ndarray]:
        if id is None:
            return list(range(self.env_num))
        return [id] if np.isscalar(id) else id  # type: ignore

    def _assert_id(self, id: Union[List[int], np.ndarray]) -> None:
        for i in id:
            assert i not in self.waiting_id, \
                f"Cannot interact with environment {i} which is stepping now."
            assert i in self.ready_id, \
                f"Can only interact with ready environments {self.ready_id}."

    def reset(
        self, id: Optional[Union[int, List[int], np.ndarray]] = None
    ) -> np.ndarray:
        """Reset the state of some envs and return initial observations.

        If id is None, reset the state of all the environments and return
        initial observations, otherwise reset the specific environments with
        the given id, either an int or a list.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)
        # send(None) == reset() in worker
        for i in id:
            self.workers[i].send(None)
        obs_list = [self.workers[i].recv() for i in id]
        try:
            obs = np.stack(obs_list)
        except ValueError:  # different len(obs)
            obs = np.array(obs_list, dtype=object)
        return self.normalize_obs(obs)

    def step(
        self,
        action: np.ndarray,
        id: Optional[Union[int, List[int], np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run one timestep of some environments' dynamics.

        If id is None, run one timestep of all the environments’ dynamics;
        otherwise run one timestep for some environments with given id,  either
        an int or a list. When the end of episode is reached, you are
        responsible for calling reset(id) to reset this environment’s state.

        Accept a batch of action and return a tuple (batch_obs, batch_rew,
        batch_done, batch_info) in numpy format.

        :param numpy.ndarray action: a batch of action provided by the agent.

        :return: A tuple including four items:

            * ``obs`` a numpy.ndarray, the agent's observation of current environments
            * ``rew`` a numpy.ndarray, the amount of rewards returned after \
                previous actions
            * ``done`` a numpy.ndarray, whether these episodes have ended, in \
                which case further step() calls will return undefined results
            * ``info`` a numpy.ndarray, contains auxiliary diagnostic \
                information (helpful for debugging, and sometimes learning)

        For the async simulation:

        Provide the given action to the environments. The action sequence
        should correspond to the ``id`` argument, and the ``id`` argument
        should be a subset of the ``env_id`` in the last returned ``info``
        (initially they are env_ids of all the environments). If action is
        None, fetch unfinished step() calls instead.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if not self.is_async:
            assert len(action) == len(id)
            for i, j in enumerate(id):
                self.workers[j].send(action[i])
            result = []
            for j in id:
                obs, rew, done, info = self.workers[j].recv()
                info["env_id"] = j
                result.append((obs, rew, done, info))
        else:
            if action is not None:
                self._assert_id(id)
                assert len(action) == len(id)
                for act, env_id in zip(action, id):
                    self.workers[env_id].send(act)
                    self.waiting_conn.append(self.workers[env_id])
                    self.waiting_id.append(env_id)
                self.ready_id = [x for x in self.ready_id if x not in id]
            ready_conns: List[EnvWorker] = []
            while not ready_conns:
                ready_conns = self.worker_class.wait(
                    self.waiting_conn, self.wait_num, self.timeout
                )
            result = []
            for conn in ready_conns:
                waiting_index = self.waiting_conn.index(conn)
                self.waiting_conn.pop(waiting_index)
                env_id = self.waiting_id.pop(waiting_index)
                obs, rew, done, info = conn.recv()
                info["env_id"] = env_id
                result.append((obs, rew, done, info))
                self.ready_id.append(env_id)
        obs_list, rew_list, done_list, info_list = zip(*result)
        try:
            obs_stack = np.stack(obs_list)
        except ValueError:  # different len(obs)
            obs_stack = np.array(obs_list, dtype=object)
        rew_stack, done_stack, info_stack = map(
            np.stack, [rew_list, done_list, info_list]
        )
        return self.normalize_obs(obs_stack), rew_stack, done_stack, info_stack

    def action_interpret(
        self,
        logits,
        action: np.ndarray,
        dones, 
        id: Optional[Union[int, List[int], np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        assert len(action) == len(id)
        for i, j in enumerate(id):
            self.workers[j].send2(logits[i], action[i], dones[i])
        logp_list = []
        entro_list = []
        for j in id:
            logp, entro = self.workers[j].recv2()
            # info["env_id"] = j
            logp_list.append(logp)
            entro_list += entro

        # logp_list, entro_list = zip(*result)
        return logp_list, entro_list
        # logp_stack, entro_stack = map(
        #     np.stack, [logp_list, entro_list]
        # )
        # return logp_stack, entro_stack



    def seed(
        self,
        seed: Optional[Union[int, List[int]]] = None
    ) -> List[Optional[List[int]]]:
        """Set the seed for all environments.

        Accept ``None``, an int (which will extend ``i`` to
        ``[i, i + 1, i + 2, ...]``) or a list.

        :return: The list of seeds used in this env's random number generators.
            The first value in the list should be the "main" seed, or the value
            which a reproducer pass to "seed".
        """
        self._assert_is_not_closed()
        seed_list: Union[List[None], List[int]]
        if seed is None:
            seed_list = [seed] * self.env_num
        elif isinstance(seed, int):
            seed_list = [seed + i for i in range(self.env_num)]
        else:
            seed_list = seed
        return [w.seed(s) for w, s in zip(self.workers, seed_list)]

    def render(self, **kwargs: Any) -> List[Any]:
        """Render all of the environments."""
        self._assert_is_not_closed()
        if self.is_async and len(self.waiting_id) > 0:
            raise RuntimeError(
                f"Environments {self.waiting_id} are still stepping, cannot "
                "render them now."
            )
        return [w.render(**kwargs) for w in self.workers]

    def close(self) -> None:
        """Close all of the environments.

        This function will be called only once (if not, it will be called during
        garbage collected). This way, ``close`` of all workers can be assured.
        """
        self._assert_is_not_closed()
        for w in self.workers:
            w.close()
        self.is_closed = True

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observations by statistics in obs_rms."""
        # if self.obs_rms and self.norm_obs:
        #     clip_max = 10.0  # this magic number is from openai baselines
        #     # see baselines/common/vec_env/vec_normalize.py#L10
        #     obs = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.__eps)
        #     obs = np.clip(obs, -clip_max, clip_max)
        return obs


class DummyVectorEnv(BaseVectorEnv):
    """Dummy vectorized environment wrapper, implemented in for-loop.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for other APIs' usage.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], **kwargs: Any) -> None:
        super().__init__(env_fns, DummyEnvWorker, **kwargs)


class SubprocVectorEnv(BaseVectorEnv):
    """Vectorized environment wrapper based on subprocess.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for other APIs' usage.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], **kwargs: Any) -> None:

        def worker_fn(fn: Callable[[], gym.Env]) -> SubprocEnvWorker:
            return SubprocEnvWorker(fn, share_memory=False)

        super().__init__(env_fns, worker_fn, **kwargs)


class ShmemVectorEnv(BaseVectorEnv):
    """Optimized SubprocVectorEnv with shared buffers to exchange observations.

    ShmemVectorEnv has exactly the same API as SubprocVectorEnv.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for other APIs' usage.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], **kwargs: Any) -> None:

        def worker_fn(fn: Callable[[], gym.Env]) -> SubprocEnvWorker:
            return SubprocEnvWorker(fn, share_memory=True)

        super().__init__(env_fns, worker_fn, **kwargs)


class RayVectorEnv(BaseVectorEnv):
    """Vectorized environment wrapper based on ray.

    This is a choice to run distributed environments in a cluster.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for other APIs' usage.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], **kwargs: Any) -> None:
        try:
            import ray
        except ImportError as exception:
            raise ImportError(
                "Please install ray to support RayVectorEnv: pip install ray"
            ) from exception
        if not ray.is_initialized():
            ray.init()
        super().__init__(env_fns, RayEnvWorker, **kwargs)
