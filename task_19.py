import numpy as np
import matplotlib.pyplot as plt

def x0(t: np.ndarray):
    return np.zeros(t.shape)

def f(t: np.ndarray, x: np.ndarray):
    e = 0.0001
    l = 0.5

    s = np.arange(0, 1 + e, e)
    i = np.sum(np.sin(np.pi*(t[:, None] - s[None, :])) * x, axis=1) * e

    return l * i + 1

def exact(t: np.ndarray):
    c1 = 4 / (np.pi * 4.25)
    c2 = 1 / (np.pi * 4.25)

    return -c1 * np.cos(np.pi * t) - c2 * np.sin(np.pi * t) + np.ones(t.shape)

def r(x1, x2):
    return np.max(np.abs(x1 - x2))

def draw(t, x, ax):
    ax.set_xlabel('t')
    ax.set_ylabel('x')

    ax.set_xlim(0, 1)
    ax.plot(t, x)

    plt.savefig('task_19.png')

if __name__ == '__main__':
    eps = 0.0001

    fig, axes = plt.subplots(1,2, figsize=(10, 5))
    t = np.arange(0, 1 + eps, eps)
    ex = exact(t)
    axes[0].set_title('Точное решение')
    draw(t, ex, axes[0])

    xn = x0(t)

    sup = np.max(np.abs(f(t, xn) - xn))
    print(f'SUP = {sup}')

    k = 0.5
    n0 = np.log(eps * (1-k) / sup) / np.log(k)
    n = int(n0 // 1 + (n0 % 1 != 0))

    print(f'Число шагов: {n}')

    for i in range(1, n):
        xn = f(t, xn)

    print(f'Расстояние между функциями: {r(xn, exact(t))}')

    axes[1].set_title('Приближенное решение')
    draw(t, xn, axes[1])

    


    