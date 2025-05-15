import os
from typing import Any

import cloudpickle
import numpy as np


class Info:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get(self):
        return self.kwargs

    def add(self, key, value):
        self.kwargs[key] = value


# random walk sampler

class CloudpickleWrapper(object):
    """A cloudpickle wrapper used in SubprocVectorEnv."""

    def __init__(self, data: Any) -> None:
        self.data = data

    def __getstate__(self) -> str:
        return cloudpickle.dumps(self.data)

    def __setstate__(self, data: str) -> None:
        self.data = cloudpickle.loads(data)
