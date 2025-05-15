import os
import numpy as np
import time, copy
import sys, inspect
from tqdm import tqdm

# ============================== Tool functions ============================== #
def rotatefunc(x, Mr):
    return np.matmul(Mr, x.transpose()).transpose()


def sr_func(x, Os, Mr, sh=1):  # shift and rotate
    y = (x - Os) * sh
    return np.matmul(Mr, y.transpose()).transpose()


def rotate_gen(dim):  # Generate a rotate matrix
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        mat = np.eye(dim)
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
    # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H


def osc_transform(x):
    """
    Implementing the oscillating transformation on objective values or/and decision values.

    :param x: If x represents objective values, x is a 1-D array in shape [NP] if problem is single objective,
              or a 2-D array in shape [NP, number_of_objectives] if multi-objective.
              If x represents decision values, x is a 2-D array in shape [NP, dim].
    :return: The array after transformation in the shape of x.
    """
    y = x.copy()
    idx = (x > 0.)
    y[idx] = np.log(x[idx]) / 0.1
    y[idx] = np.exp(y[idx] + 0.49 * (np.sin(y[idx]) + np.sin(0.79 * y[idx]))) ** 0.1
    idx = (x < 0.)
    y[idx] = np.log(-x[idx]) / 0.1
    y[idx] = -np.exp(y[idx] + 0.49 * (np.sin(0.55 * y[idx]) + np.sin(0.31 * y[idx]))) ** 0.1
    return y


def asy_transform(x, beta):
    """
    Implementing the asymmetric transformation on decision values.

    :param x: Decision values in shape [NP, dim].
    :param beta: beta factor.
    :return: The array after transformation in the shape of x.
    """
    NP, dim = x.shape
    idx = (x > 0.)
    y = x.copy()
    y[idx] = y[idx] ** (1. + beta * np.linspace(0, 1, dim).reshape(1, -1).repeat(repeats=NP, axis=0)[idx] * np.sqrt(y[idx]))
    return y


def pen_func(x, ub):
    """
    Implementing the penalty function on decision values.

    :param x: Decision values in shape [NP, dim].
    :param ub: the upper-bound as a scalar.
    :return: Penalty values in shape [NP].
    """
    return np.sum(np.maximum(0., np.abs(x) - ub) ** 2, axis=-1)


def get_split(n_components, dim):
    splits = np.ones(n_components, dtype=int) * 2
    remain = dim - np.sum(splits)
    pos = 0
    while remain > 0 :
        if pos == (n_components - 1):
            splits[pos] += remain
            break
        split = int(remain * np.random.rand())
        remain -= split
        splits[pos] += split
        pos+=1
    return splits


def get_problem_classes():
    pc = []
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj) and issubclass(obj, Basic_Problem) and obj.__name__ != 'Basic_Problem' and obj.__name__ != 'Composition' and obj.__name__ != 'Hybrid':
            pc.append(obj)
    return pc

# ============================== Basic problems ============================== #
# Basic problem class
class Basic_Problem:
    """
    Abstract super class for problems and applications.
    """
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        self.dim = dim
        self.shift = shift
        self.rotate = rotate
        self.bias = bias
        if self.shift is None:
            self.shift = np.random.rand(dim) * (ub - lb) * 0.8 + lb + (ub - lb) * 0.1
        if self.rotate is None:
            self.rotate = rotate_gen(dim)
        if self.bias is None:
            self.bias = np.random.randint(1, 10) * 100
        self.lb = lb
        self.ub = ub
        self.FEs = 0
        self.opt = self.shift
        self.MaxFEs = int(maxfes)
        self.optimum = 0  # self.eval(self.get_optimal())
        # self.optimum = self.func(self.get_optimal().reshape(1, -1))[0]
        self.name = self.__class__.__name__

    def get_optimal(self):
        return self.opt
    
    # def optimum(self):
    #     return self.func(self.get_optimal().reshape(1, -1))[0]
    def __str__(self) -> str:
        return self.name
    
    def reset(self):
        self.T1=0
        self.FEs = 0

    def eval(self, x):
        """
        A general version of func() with adaptation to evaluate both individual and population.
        """
        start=time.perf_counter()

        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 1:  # x is a single individual
            y=self.func(x.reshape(1, -1))[0]
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y
        elif x.ndim == 2:  # x is a whole population
            y=self.func(x)
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y
        else:
            y=self.func(x.reshape(-1, x.shape[-1]))
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y

    def func(self, x):
        raise NotImplementedError


class Sphere(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        self.shrink = 1
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)

    def func(self, x):
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        return np.sum(z ** 2, -1)


class Schwefel12(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        self.shrink = 1
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)

    def func(self, x):
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        res = np.zeros(x.shape[0])
        for i in range(self.dim):
            tmp = np.power(np.sum(z[:,:i+1], -1), 2)
            res += tmp
        return res


class Ellipsoidal(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        self.shrink = 1
        self.condition = 1e4
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)

    def func(self, x):
        
        nx = self.dim
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        i = np.arange(nx)
        return np.sum(np.power(self.condition, i / (nx - 1)) * (z ** 2), -1)


class Ellipsoidal_high_cond(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        self.shrink = 1
        self.condition = 1e6
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)

    def func(self, x):
        
        nx = self.dim
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        i = np.arange(nx)
        return np.sum(np.power(self.condition, i / (nx - 1)) * (z ** 2), -1)


class Bent_cigar(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        self.shrink = 1
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)

    def func(self, x):
        
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        return z[:, 0] ** 2 + np.sum(np.power(10, 6) * (z[:, 1:] ** 2), -1)


class Discus(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        self.shrink = 1
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)

    def func(self, x):
        
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        return np.power(10, 6) * (z[:, 0] ** 2) + np.sum(z[:, 1:] ** 2, -1)


class Dif_powers(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        self.shrink = 1
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)

    def func(self, x):
        
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        i = np.arange(self.dim)
        return np.power(np.sum(np.power(np.fabs(z), 2 + 4 * i / max(1, self.dim - 1)), -1), 0.5)


class Rosenbrock(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        self.shrink = 2.048 / ub
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)

    def func(self, x):
        
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        z += 1
        z_ = z[:, 1:]
        z = z[:, :-1]
        tmp1 = z ** 2 - z_
        return np.sum(100 * tmp1 * tmp1 + (z - 1) ** 2, -1)


class Ackley(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        self.shrink = 1
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)

    def func(self, x):
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        sum1 = -0.2 * np.sqrt(np.sum(z ** 2, -1) / self.dim)
        sum2 = np.sum(np.cos(2 * np.pi * z), -1) / self.dim
        return np.round(np.e + 20 - 20 * np.exp(sum1) - np.exp(sum2), 15)


class Weierstrass(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        self.shrink = 0.5 / ub
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)

    def func(self, x):
        
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        a, b, k_max = 0.5, 3.0, 20
        sum1, sum2 = 0, 0
        for k in range(k_max + 1):
            sum1 += np.sum(np.power(a, k) * np.cos(2 * np.pi * np.power(b, k) * (z + 0.5)), -1)
            sum2 += np.power(a, k) * np.cos(2 * np.pi * np.power(b, k) * 0.5)
        return sum1 - self.dim * sum2


class Griewank(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        self.shrink = 6
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)

    def func(self, x):
        
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        s = np.sum(z ** 2, -1)
        p = np.ones(x.shape[0])
        for i in range(self.dim):
            p *= np.cos(z[:, i] / np.sqrt(1 + i))
        return 1 + s / 4000 - p


class Rastrigin(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        self.shrink = 5.12 / ub
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)

    def func(self, x):
        
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        return np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z) + 10, -1)


class Buche_Rastrigin(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        self.shrink = ((10. ** 0.5) ** np.linspace(0, 1, dim))
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)
        self.shift[::2] = np.abs(self.shift[::2])

    def func(self, x):
        
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        z = osc_transform(z)
        even = z[:, ::2]
        even[even > 0.] *= 10.
        z *= self.shrink
        return 10 * (self.dim - np.sum(np.cos(2 * np.pi * z), axis=-1)) + np.sum(z ** 2, axis=-1) + 100 * pen_func(x, self.ub)


class Mod_Schwefel(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        self.shrink = 10
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)

    def func(self, x):
        
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        a = 4.209687462275036e+002
        b = 4.189828872724338e+002
        z += a
        g1 = z * np.sin(np.sqrt(np.abs(z)))
        g2 = (500 - z % 500) * np.sin(np.sqrt(np.fabs(500 - z % 500))) - (z - 500) ** 2 / (10000 * self.dim)
        g3 = (-z % 500 - 500) * np.sin(np.sqrt(np.fabs(500 - -z % 500))) - (z + 500) ** 2 / (10000 * self.dim)
        g = np.where(np.fabs(z) <= 500, g1, 0) + np.where(z > 500, g2, 0) + np.where(z < -500, g3, 0)
        return b * self.dim - g.sum(-1)


class Katsuura(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        self.shrink = 5 / ub
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)

    def func(self, x):
        
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        tmp3 = np.power(self.dim, 1.2)
        tmp1 = np.repeat(np.power(np.ones((1, 32)) * 2, np.arange(1, 33)), x.shape[0], 0)
        res = np.ones(x.shape[0])
        for i in range(self.dim):
            tmp2 = tmp1 * np.repeat(z[:, i, None], 32, 1)
            temp = np.sum(np.fabs(tmp2 - np.floor(tmp2 + 0.5)) / tmp1, -1)
            res *= np.power(1 + (i + 1) * temp, 10 / tmp3)
        tmp = 10 / self.dim / self.dim
        return res * tmp - tmp


class Grie_rosen(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        self.shrink = 5 / ub
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)

    def func(self, x):
        
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        z += 1
        z_ = np.concatenate((z[:, 1:], z[:, :1]), -1)
        _z = z
        tmp1 = _z ** 2 - z_
        temp = 100 * tmp1 * tmp1 + (_z - 1) ** 2
        res = np.sum(temp * temp / 4000 - np.cos(temp) + 1, -1)
        return res


class Escaffer6(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        self.shrink = 1
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)

    def func(self, x):
        
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        z_ = np.concatenate((z[:, 1:], z[:, :1]), -1)
        return np.sum(0.5 + (np.sin(np.sqrt(z ** 2 + z_ ** 2)) ** 2 - 0.5) / ((1 + 0.001 * (z ** 2 + z_ ** 2)) ** 2), -1)


class Happycat(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        self.shrink = 5 / ub
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)

    def get_optimal(self):
        return self.opt

    def func(self, x):
        
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        z -= 1
        sum_z = np.sum(z, -1)
        r2 = np.sum(z ** 2, -1)
        return np.power(np.fabs(r2 - self.dim), 1 / 4) + (0.5 * r2 + sum_z) / self.dim + 0.5


class Hgbat(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        self.shrink = 5 / ub
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)

    def func(self, x):
        
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        z -= 1
        sum_z = np.sum(z, -1)
        r2 = np.sum(z ** 2, -1)
        return np.sqrt(np.fabs(np.power(r2, 2) - np.power(sum_z, 2))) + (0.5 * r2 + sum_z) / self.dim + 0.5


class Lunacek_bi_Rastrigin(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)

    def func(self, x):
        
        d = 1
        s = 1 - 1 / (2 * np.sqrt(self.dim + 20) - 8.2)
        u0 = 2.5
        u1 = -np.sqrt((u0 * u0 - d) / s)
        y = 10 * (x - self.shift) / 100
        tmpx = 2 * y
        tmpx[:, self.shift < 0] *= -1
        z = rotatefunc(tmpx, self.rotate)
        tmpx += u0
        tmp1 = 10 * (self.dim - np.sum(np.cos(2 * np.pi * z), -1))
        tmp2 = np.minimum(np.sum((tmpx - u0) ** 2, -1), d * self.dim + s * np.sum((tmpx - u1) ** 2, -1))
        return tmp1 + tmp2


class Zakharov(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        self.shrink = 1.0
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)

    def func(self, x):
        
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        return np.sum(np.power(z, 2), -1) + np.power(np.sum(0.5 * z, -1), 2) + np.power(np.sum(0.5 * z, -1), 4)


class Levy(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        self.shrink = 1.0
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)

    def func(self, x):
        
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        w = 1 + (z - 1) / 4
        _w_ = w[:, :-1]
        return np.power(np.sin(np.pi * w[:, 0]), 2) + \
               np.sum(np.power(_w_ - 1, 2) * (1 + 10 * np.power(np.sin(np.pi * _w_ - 1), 2)), -1) + \
               np.power(w[:, -1] - 1, 2) * (1 + np.power(np.sin(2 * np.pi * w[:, -1]), 2))


class Scaffer_F7(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        self.shrink = 1
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)

    def func(self, x):
        
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        s = np.sqrt(z[:, :-1] ** 2 + z[:, 1:] ** 2)
        return np.power(1 / (self.dim - 1) * np.sum(np.sqrt(s) * (np.sin(50 * np.power(s, 0.2)) + 1), -1), 2)


class Step_Rastrigin(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        self.shrink = 5.12 / ub
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)
        self.subproblem = Rastrigin(self.dim, np.zeros(self.dim), np.eye(self.dim), 0, lb, ub)

    def func(self, x):
        
        y = sr_func(x, self.shift, self.rotate, self.shrink)
        y[np.fabs(y) > 0.5] = np.round(2 * y[np.fabs(y) > 0.5]) / 2.
        return self.subproblem.func(y)


class Linear_Slope(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        self.shrink = (10. ** 0.5) ** np.linspace(0, 1, dim)
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)
        self.shift = np.sign(self.shift)
        self.shift[self.shift == 0.] = np.random.choice([-1., 1.], size=(self.shift == 0.).sum())
        self.shift = self.shift * ub

    def func(self, x):
        
        z = x.copy()
        exceed_bound = (x * self.shift) > (self.ub ** 2)
        z[exceed_bound] = np.sign(z[exceed_bound]) * self.ub  # clamp back into the domain
        s = np.sign(self.shift) * self.shrink
        return np.sum(self.ub * np.abs(s) - z * s, axis=-1)


class Attractive_Sector(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        self.shrink = (10. ** 0.5) ** np.linspace(0, 1, dim)
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)
        self.rotate = np.matmul(np.matmul(rotate_gen(dim), np.diag(self.shrink)), self.rotate)

    def func(self, x):
        
        z = sr_func(x, self.shift, self.rotate)
        idx = (z * self.shift) > 0.
        z[idx] *= 100.
        return osc_transform(np.sum(z ** 2, -1)) ** 0.9


class Step_Ellipsoidal(Basic_Problem):
    def __init__(self, dim, shift, rotate, bias, lb, ub, maxfes=50000, Q_rotate=None):
        self.shrink = (10. ** 0.5) ** np.linspace(0, 1, dim)
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)
        self.rotate = np.matmul(np.diag(self.shrink), self.rotate)
        self.Q_rotate = Q_rotate
        if self.Q_rotate is None:
            self.Q_rotate = rotate_gen(dim)

    def func(self, x):
        z_hat = sr_func(x, self.shift, self.rotate)
        z = np.matmul(np.where(np.abs(z_hat) > 0.5, np.floor(0.5 + z_hat), np.floor(0.5 + 10. * z_hat) / 10.), self.Q_rotate.T)
        return 0.1 * np.maximum(np.abs(z_hat[:, 0]) / 1e4, np.sum(100 ** np.linspace(0, 1, self.dim) * (z ** 2), axis=-1))


class Sharp_Ridge(Basic_Problem):
    def __init__(self, dim, shift, rotate, bias, lb, ub, maxfes=50000, Q_rotate=None):
        self.shrink = (10 ** 0.5) ** np.linspace(0, 1, dim)
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)
        self.Q_rotate = Q_rotate
        if self.Q_rotate is None:
            self.Q_rotate = rotate_gen(dim)
        self.rotate = np.matmul(np.matmul(self.Q_rotate, np.diag(self.shrink)), self.rotate)

    def func(self, x):
        
        z = sr_func(x, self.shift, self.rotate)
        return z[:, 0] ** 2. + 100. * np.sqrt(np.sum(z[:, 1:] ** 2., axis=-1))


class Rastrigin_F15(Basic_Problem):
    def __init__(self, dim, shift, rotate, bias, lb, ub, maxfes=50000, linearTF=None):
        self.shrink = (10 ** 0.5) ** np.linspace(0, 1, dim)
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)
        self.linearTF = linearTF
        if self.linearTF is None:
            self.linearTF = np.matmul(np.matmul(self.rotate, np.diag(self.shrink)), rotate_gen(dim))

    def func(self, x):
        
        z = sr_func(x, self.shift, self.rotate)
        z = asy_transform(osc_transform(z), beta=0.2)
        z = np.matmul(z, self.linearTF.T)
        return 10. * (self.dim - np.sum(np.cos(2. * np.pi * z), axis=-1)) + np.sum(z ** 2, axis=-1)


class Schwefel(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        lb=-5
        ub=5
        self.shrink = (10 ** 0.5) ** np.linspace(0, 1, dim)
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)
        self.shift = 0.5 * 4.2096874633 * np.sign(self.shift)

    def func(self, x):
        
        tmp = 2 * np.abs(self.shift)
        z = 2 * np.sign(self.shift) * x
        z[:, 1:] += 0.25 * (z[:, :-1] - tmp[:-1])
        z = 100. * (self.shrink * (z - tmp) + tmp)
        b = 4.189828872724339
        return b - 0.01 * np.mean(z * np.sin(np.sqrt(np.abs(z))), axis=-1) + 100 * pen_func(z / 100, self.ub)


class Gallagher101(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        lb = -5
        ub = 5
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)
        self.shrink = 1
        self.n_peaks = 101
        opt_shrink = 1.
        global_opt_alpha = 1e3
        self.y = opt_shrink * (np.random.rand(self.n_peaks, dim) * (ub - lb) + lb)  # [n_peaks, dim]
        self.y[0] = self.shift * opt_shrink  # the global optimum
        self.shift = self.y[0]

        # generate the matrix C[i]
        sqrt_alpha = 1000 ** np.random.permutation(np.linspace(0, 1, self.n_peaks - 1))
        sqrt_alpha = np.insert(sqrt_alpha, obj=0, values=np.sqrt(global_opt_alpha))
        self.C = [np.random.permutation(sqrt_alpha[i] ** np.linspace(-0.5, 0.5, dim)) for i in range(self.n_peaks)]
        self.C = np.vstack(self.C)  # [n_peaks, dim]

        # generate the weight w[i]
        self.w = np.insert(np.linspace(1.1, 9.1, self.n_peaks - 1), 0, 10.)  # [n_peaks]


    def func(self, x):
        
        z = np.matmul(np.expand_dims(x, axis=1).repeat(self.n_peaks, axis=1) - self.y, self.rotate.T)  # [NP, n_peaks, dim]
        z = np.max(self.w * np.exp((-0.5 / self.dim) * np.sum(self.C * (z ** 2), axis=-1)), axis=-1)  # [NP]
        return osc_transform(10 - z) ** 2


class Gallagher21(Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000):
        lb = -5
        ub = 5
        Basic_Problem.__init__(self, dim, shift, rotate, bias, lb, ub, maxfes)
        self.shrink = 1
        self.n_peaks = 21
        opt_shrink = 0.98
        global_opt_alpha = 1e6
        self.y = opt_shrink * (np.random.rand(self.n_peaks, dim) * (ub - lb) + lb)  # [n_peaks, dim]
        self.y[0] = self.shift * opt_shrink  # the global optimum
        self.shift = self.y[0]

        # generate the matrix C[i]
        sqrt_alpha = 1000 ** np.random.permutation(np.linspace(0, 1, self.n_peaks - 1))
        sqrt_alpha = np.insert(sqrt_alpha, obj=0, values=np.sqrt(global_opt_alpha))
        self.C = [np.random.permutation(sqrt_alpha[i] ** np.linspace(-0.5, 0.5, dim)) for i in range(self.n_peaks)]
        self.C = np.vstack(self.C)  # [n_peaks, dim]

        # generate the weight w[i]
        self.w = np.insert(np.linspace(1.1, 9.1, self.n_peaks - 1), 0, 10.)  # [n_peaks]


    def func(self, x):
        
        z = np.matmul(np.expand_dims(x, axis=1).repeat(self.n_peaks, axis=1) - self.y, self.rotate.T)  # [NP, n_peaks, dim]
        z = np.max(self.w * np.exp((-0.5 / self.dim) * np.sum(self.C * (z ** 2), axis=-1)), axis=-1)  # [NP]
        return osc_transform(10 - z) ** 2


# ============================== Composition & Hybrid problems ============================== #
class Composition(Basic_Problem):
    def __init__(self, dim, sub_problems=None, weights=None, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000, include_classes=None):
        super().__init__(dim, shift, rotate, bias, lb, ub, maxfes)
        self.sub_problems = sub_problems
        self.weights = weights
        if self.sub_problems is None:
            if include_classes is None:
                include_classes = get_problem_classes()
            self.n_sub_problem = np.random.randint(2,6)
            self.sub_problems = []
            for i in range(self.n_sub_problem):
                sp = np.random.choice(include_classes)(dim, np.zeros(dim), np.eye(dim), 0, lb, ub)
                self.sub_problems.append(sp)
        else:
            self.n_sub_problem = len(sub_problems)
            self.sub_problems = []
            for i in range(self.n_sub_problem):
                self.sub_problems.append(sub_problems[i].__class__(dim, np.zeros(dim), np.eye(dim), 0, lb, ub))
        if self.weights is None:
            self.weights = np.random.rand(self.n_sub_problem)

    def func(self, x):
        res = np.zeros(x.shape[0])
        z = sr_func(x, self.shift, self.rotate)
        for i in range(self.n_sub_problem):
            if 'Gallagher' in self.sub_problems[i].__class__.__name__ or isinstance(self.sub_problems[i], Schwefel): #
                res += self.weights[i] * self.sub_problems[i].func(z * 5 / self.ub)
            else:
                try:
                    res += self.weights[i] * self.sub_problems[i].func(z)
                except ValueError:
                    print(self.dim, self.sub_problems[i].dim)
                    print(x.shape, z.shape)
                    print(self.sub_problems[i])
        return res
    

class Hybrid(Basic_Problem):
    def __init__(self, dim, sub_problems=None, splits=None, permu=None, shift=None, rotate=None, bias=0, lb=-5, ub=5, maxfes=50000, include_classes=None):
        super().__init__(dim, shift, rotate, bias, lb, ub, maxfes)
        self.sub_problems = sub_problems
        self.splits = splits
        self.permu = permu
        if self.splits is None and self.sub_problems is None:
            self.n_sub_problem = np.random.randint(2, min(5, dim//2+1))
        elif self.splits is not None:
            self.n_sub_problem = len(self.splits)
        elif self.sub_problems is not None:
            self.n_sub_problem = len(self.sub_problems)

        if self.splits is None:
            self.splits = get_split(self.n_sub_problem, dim)
        if self.sub_problems is None:
            if include_classes is None:
                include_classes = get_problem_classes()
            self.sub_problems = []
            for i in range(self.n_sub_problem):
                sp = np.random.choice(include_classes)(self.splits[i], np.zeros(self.splits[i]), np.eye(self.splits[i]), 0, lb, ub)
                self.sub_problems.append(sp)
        else:
            self.n_sub_problem = len(sub_problems)
            self.sub_problems = []
            for i in range(self.n_sub_problem):
                self.sub_problems.append(sub_problems[i].__class__(self.splits[i], np.zeros(self.splits[i]), np.eye(self.splits[i]), 0, lb, ub))
        if self.permu is None:
            self.permu = np.random.permutation(dim)

    def func(self, x):
        res = np.zeros(x.shape[0])
        z = sr_func(x, self.shift, self.rotate)
        z = z[:,self.permu]
        index = 0
        for i in range(self.n_sub_problem):
            if 'Gallagher' in self.sub_problems[i].__class__.__name__ or isinstance(self.sub_problems[i], Schwefel): #
                res += self.sub_problems[i].func(z[:,index:index+self.splits[i]] * 5 / self.ub)
            else:
                try:
                    res += self.sub_problems[i].func(z[:,index:index+self.splits[i]])
                except ValueError:
                    print(self.splits)
            index += self.splits[i]
        return res
    

# ============================== Problem Dataset ============================== #
from torch.utils.data import Dataset
class Synthetic_Dataset(Dataset):
    def __init__(self,
                 data,
                 batch_size=1,
                 infinite=False):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.N = len(self.data)
        self.infinite = infinite
        if self.infinite:
            self.ptr = 0
        else:
            self.ptr = [i for i in range(0, self.N, batch_size)]
        self.index = np.arange(self.N)

    @staticmethod
    def get_datasets(Dim=None,
                     shifted=True,
                     rotated=True,
                     biased=False,
                     lb=-5, ub=5,
                     split='random',  
                     # random: random train-test division; 
                     # average: both train & test contain all problem classes, train_size / num of all instances * n_expand instances of each class are assigned for training;
                     # instance: split train-test according to instance list in split_list, train_size is ignored;
                     split_list={'train': [], 'test': []},  # should be assigned when split == instance, names in train + test == basic + com + hyb
                     n_expand=100,  # expand each instance (class) to n_expand instances with random shift and rotation
                     include_class=None, # the list of included basic problem classes, default to all
                     composition_num=48,
                     hybrid_num=48,
                     # 32 basic + 48 com + 48 hyb = 128
                     train_size=9600,
                     train_batch_size=1,
                     test_batch_size=1,
                     instance_seed=3849,
                     infinite=False,):
        if instance_seed > 0:
            np.random.seed(instance_seed)
        MaxFEs_range = np.array([10000, 20000, 30000, 40000, 50000], dtype=np.int32)
        bound_range = np.array([[-5, 5], [-10, 10], [-20, 20], [-50, 50]], dtype=np.int32)
        overall = {}
        if include_class is None:
            include_class = get_problem_classes()
        # include basic problem instances
        pbar = tqdm(total=(len(include_class) + composition_num + hybrid_num) * n_expand, desc='Dataset generating')
        for i in range(len(include_class)):
            overall[include_class[i].__name__] = []
            for j in range(n_expand):
                if Dim is None:
                    dim = np.random.choice([5, 10, 20, 50])
                else:
                    dim = Dim
                shift = None
                H = None
                bias = 0
                bound = bound_range[np.random.choice(len(bound_range))]
                # lbi = lb
                # ubi = ub
                lbi, ubi = bound
                if not shifted:
                    shift = np.zeros(dim)
                if not rotated:
                    H = np.eye(dim)
                if biased:
                    bias = np.random.randint(1, 26) * 100
                if lb is None:
                    lbi = -np.random.randint(1, 20, dim) * 5
                if ub is None:
                    ubi = -lbi
                ins = include_class[i](dim, shift, H, bias, lbi, ubi, np.random.choice(MaxFEs_range))
                ins.name = include_class[i].__name__ + f'-{len(overall[include_class[i].__name__])}'
                overall[include_class[i].__name__].append(ins)
                pbar.update()

        # include composition problem instances
        for i in range(composition_num):
            name = f'Composition_{i}'
            overall[name] = []
            if Dim is None:
                dim = np.random.choice([5, 10, 20, 50])
            else:
                dim = Dim
            shift = None
            H = None
            bias = 0
            bound = bound_range[np.random.choice(len(bound_range))]
            # lbi = lb
            # ubi = ub
            lbi, ubi = bound
            if not shifted:
                shift = np.zeros(dim)
            if not rotated:
                H = np.eye(dim)
            if biased:
                bias = np.random.randint(1, 26) * 100
            if lb is None:
                lbi = -np.random.randint(1, 20, dim) * 5
            if ub is None:
                ubi = -lbi
            ins = Composition(dim=dim, shift=shift, rotate=H, bias=bias, lb=lbi, ub=ubi, maxfes=np.random.choice(MaxFEs_range), include_classes=include_class)
            ins.name = name + '-0'
            overall[name].append(copy.deepcopy(ins))
            pbar.update()
            
            sub_problems = ins.sub_problems
            weights = ins.weights
            dim = ins.dim
            for j in range(n_expand - 1):
                if Dim is None:
                    dim = np.random.choice([5, 10, 20, 50])
                else:
                    dim = Dim
                weights = None
                shift = None
                H = None
                bias = 0
                bound = bound_range[np.random.choice(len(bound_range))]
                # lbi = lb
                # ubi = ub
                lbi, ubi = bound
                if not shifted:
                    shift = np.zeros(dim)
                if not rotated:
                    H = np.eye(dim)
                if biased:
                    bias = np.random.randint(1, 26) * 100
                if lb is None:
                    lbi = -np.random.randint(1, 20, dim) * 5
                if ub is None:
                    ubi = -lbi
                inss = Composition(dim=dim, sub_problems=sub_problems, weights=weights, shift=shift, rotate=H, bias=bias, lb=lbi, ub=ubi, maxfes=np.random.choice(MaxFEs_range), include_classes=include_class)
                inss.name = name + f'-{j+1}'
                overall[name].append(copy.deepcopy(inss))
                pbar.update()

        # include hybrid problem instances
        for i in range(hybrid_num):
            name = f'Hybrid_{i}'
            overall[name] = []
            if Dim is None:
                dim = np.random.choice([5, 10, 20, 50])
            else:
                dim = Dim
            shift = None
            H = None
            bias = 0
            bound = bound_range[np.random.choice(len(bound_range))]
            # lbi = lb
            # ubi = ub
            lbi, ubi = bound
            if not shifted:
                shift = np.zeros(dim)
            if not rotated:
                H = np.eye(dim)
            if biased:
                bias = np.random.randint(1, 26) * 100
            if lb is None:
                lbi = -np.random.randint(1, 21, dim) * 5
            if ub is None:
                ubi = -lbi
            ins = Hybrid(dim=dim, shift=shift, rotate=H, bias=bias, lb=lbi, ub=ubi, maxfes=np.random.choice(MaxFEs_range), include_classes=include_class)
            ins.name = name + '-0'
            overall[name].append(copy.deepcopy(ins))
            pbar.update()
            
            sub_problems = ins.sub_problems
            splits = ins.splits
            permu = ins.permu
            dim = ins.dim
            for j in range(n_expand - 1):
                if Dim is None:
                    dim = np.random.choice([10, 20, 50])
                else:
                    dim = Dim
                splits = None
                permu = None
                shift = None
                H = None
                bias = 0
                bound = bound_range[np.random.choice(len(bound_range))]
                # lbi = lb
                # ubi = ub
                lbi, ubi = bound
                if not shifted:
                    shift = np.zeros(dim)
                if not rotated:
                    H = np.eye(dim)
                if biased:
                    bias = np.random.randint(1, 26) * 100
                if lb is None:
                    lbi = -np.random.randint(1, 21, dim) * 5
                if ub is None:
                    ubi = -lbi
                inss = Hybrid(dim=dim, sub_problems=sub_problems, splits=splits, permu=permu, shift=shift, rotate=H, bias=bias, lb=lbi, ub=ubi, maxfes=np.random.choice(MaxFEs_range), include_classes=include_class)
                inss.name = name + f'-{j+1}'
                overall[name].append(copy.deepcopy(inss))
                pbar.update()

        train_set = []
        test_set = []
        if split == 'random':
            all_ins = []
            for inss in list(overall.values()):
                all_ins = all_ins + inss 
            train_list = np.sort(np.random.choice(np.arange(len(all_ins)), size=train_size, replace=False))
            j = 0
            for i, ins in enumerate(all_ins):
                if j < train_size and i == train_list[j]:
                    train_set.append(copy.deepcopy(ins))
                    j += 1
                else:
                    test_set.append(copy.deepcopy(ins))
        elif split == 'average':
            nall = (len(include_class) + composition_num + hybrid_num) * n_expand
            ratio = int(train_size / nall * n_expand)
            for cla in overall.keys():
                train_set += overall[cla][:ratio]
                test_set += overall[cla][ratio:]
        elif split == 'instance':
            for name in split_list['train']:
                if isinstance(name, str):
                    train_set += overall[name]
                else:
                    raise ValueError
            for name in split_list['test']:
                if isinstance(name, str):
                    test_set += overall[name]
                else:
                    raise ValueError
        else:
            raise NotImplementedError
        pbar.close()
        return Synthetic_Dataset(train_set, train_batch_size, infinite), Synthetic_Dataset(test_set, test_batch_size, infinite)

    def __getitem__(self, item):
        if self.infinite:
            index = self.index[self.ptr: min(self.ptr + self.batch_size, self.N)]
            if self.ptr + self.batch_size > self.N:
                index = np.concatenate([index, self.index[:(self.ptr + self.batch_size - self.N)]])
            res = []
            for i in range(len(index)):
                res.append(self.data[index[i]])
            self.ptr = (self.ptr + self.batch_size) % self.N
        else:
            ptr = self.ptr[item]
            index = self.index[ptr: min(ptr + self.batch_size, self.N)]
            res = []
            for i in range(len(index)):
                res.append(self.data[index[i]])
        return res

    def __len__(self):
        return self.N

    def __add__(self, other: 'Synthetic_Dataset'):
        return Synthetic_Dataset(self.data + other.data, self.batch_size, self.infinite)

    def shuffle(self):
        self.index = np.random.permutation(self.N)


