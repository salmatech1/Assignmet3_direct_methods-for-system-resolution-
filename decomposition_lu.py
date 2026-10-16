import numpy as np

EPS = 1e-12


def descente(L, c):
    n = len(c)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (c[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y


def remontee(U, c):
    n = len(c)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (c[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x


def lu_decomposition(A):
    U = np.array(A, dtype=float)
    n = U.shape[0]
    L = np.eye(n)
    P = np.eye(n)

    for k in range(n - 1):
        p = k + np.argmax(np.abs(U[k:, k]))
        if abs(U[p, k]) < EPS:
            raise ValueError("Matrice singulière : pivot nul.")
        if p != k:
            U[[k, p]] = U[[p, k]]
            P[[k, p]] = P[[p, k]]
            L[[k, p], :k] = L[[p, k], :k]

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] = U[i, k:] - L[i, k] * U[k, k:]

    if abs(U[n - 1, n - 1]) < EPS:
        raise ValueError("Matrice singulière : dernier pivot nul.")

    return P, L, U


def resoudre_lu(A, b):
    b = np.array(b, dtype=float).reshape(-1)
    P, L, U = lu_decomposition(A)
    y = descente(L, P @ b)
    x = remontee(U, y)
    return x


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    A1 = np.array([[1, 2, 1], [2, 3, 1], [5, 6, 9]], dtype=float)
    b1 = [0, -1, 20]
    P, L, U = lu_decomposition(A1)
    print("P =\n", P)
    print("L =\n", L)
    print("U =\n", U)
    print("P A = L U ?", np.allclose(P @ A1, L @ U))
    x1 = resoudre_lu(A1, b1)
    print("Exemple 1 : x =", x1)
    print("Vérification :", np.allclose(x1, np.linalg.solve(A1, b1)), "\n")

    A2 = np.array([[0, 2, 1], [1, 1, 1], [2, 1, 3]], dtype=float)
    b2 = [3, 3, 6]
    P, L, U = lu_decomposition(A2)
    print("P =\n", P)
    print("L =\n", L)
    print("U =\n", U)
    print("P A = L U ?", np.allclose(P @ A2, L @ U))
    x2 = resoudre_lu(A2, b2)
    print("Exemple 2 : x =", x2)
    print("Vérification :", np.allclose(x2, np.linalg.solve(A2, b2)))
