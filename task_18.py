import numpy as np
from typing import Callable
import matplotlib.pyplot as plt

def f(x: np.ndarray, t: np.ndarray):
    return t**2 / (np.abs(x) + 3) + np.sin(4*t)

def r(x1: np.ndarray, x2: np.ndarray):
    return np.max(np.abs(x1 -x2))

def x0(t: np.ndarray):
    return t**2

def x(t: np.ndarray, x0: np.ndarray):
    return t**2 / (np.abs(x0) + 3) + np.sin(4*t) - x0

def draw(t, x):
    fig, ax = plt.subplots()

    ax.set_title('Task 18')
    ax.set_xlabel('t')
    ax.set_ylabel('x')

    ax.set_xlim(-2.5, 2.5)
    ax.plot(t, x)

    plt.savefig('task_18.png')

if __name__ == '__main__':
    e = 0.00001
    eps = 0.01
    t = np.arange(-2.5, 2.5 + e, e)

    xn = x0(t)

    sup = np.max(x(t, x0(t)))
    k = 25 / 36

    n0 = np.log(eps * (1-k) / sup) / np.log(k)

    n = int(n0 // 1 + (n0 % 1 != 0))

    for i in range(1, n):
        xn = f(xn, t)

    draw(t, xn)
