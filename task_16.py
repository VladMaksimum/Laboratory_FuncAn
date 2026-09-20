import numpy as np

np.set_printoptions(suppress=True)

def qr_algorithm(A: np.matrix, num_iter, eps):
    T = A.copy()

    for i in range(num_iter):
        Q, R = np.linalg.qr(T)
        T = R @ Q

        if np.sum(np.abs(np.tril(T, k=-1))) < eps:
            break
    return T

def check(x):
    return x**4 - 60*x**3 + 904*x**2 - 3856*x + 3600

def f(x: np.matrix, c: np.matrix, d: np.matrix):
    return c @ x + d

def solve_apr(c: np.matrix, d: np.matrix, x0: np.matrix, n: int):
    for i in range(1, n+1):
        x0 = f(x0, c, d)

        #if i % 10 == 0:
            #print(i, x0)

    return x0

def solve_apostpr(c: np.matrix, d: np.matrix, x0: np.matrix, eps: float, k: float):
    x1 = f(x0, c, d)
    cnt = 1

    while True:
        cnt += 1
        x0 = x1
        x1 = f(x0, c, d)

        if k / (1-k) * np.linalg.norm(x0 - x1) < eps:
            break
    
    return x0, cnt

if __name__ == '__main__':
    A = np.matrix([[2,1,2,3],
                   [3,0,3,0],
                   [2,-1,0,3],
                   [1,2,-1,2]])

    b = np.matrix([[8],
                   [6],
                   [4],
                   [4]])
    At = A.T

    AtA = At @ A

    T = qr_algorithm(AtA, 1000, 10**(-6))

    for i, l in enumerate(np.diag(T)):
        print(f'l{i} = {l}, погрешность: {check(l)}')

    x0 = np.matrix([[0],
                   [0],
                   [0],
                   [0]])

    lma = max(np.diag(T))
    lmi = min(np.diag(T))
    c = np.eye(A.shape[0]) - AtA / lma
    d = At @ b / lma
    r = np.linalg.norm(x0 - f(x0, c, d))
    k = 1 - lmi / lma

    eps = [10**(-2), 10**(-4)]
    for e in eps:
        n = (np.log(e * (1-k) / r) / np.log(k))
        n = int((n // 1) + (n % 1 != 0))

        res, cnt = solve_apostpr(c, d, x0, e, k)

        print(f'Погрешность: {e}')
        print(f'Приближенное решение (априорная оценка n={n}):\n{solve_apr(c, d, x0, n)}')
        print(f'Приближенное решение (апостериорная оценка n={cnt}):\n{res}')
    

