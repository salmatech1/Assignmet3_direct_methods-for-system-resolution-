import numpy as np

EPS = 1e-12


def remontee(U, c):
    n = len(c)
    x = np.zeros(n)
    x[n - 1] = c[n - 1] / U[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (c[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x


def gauss_pivot_total(A, b, verbose=False):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)
    n = len(b)
    perm = np.arange(n)

    for k in range(n - 1):
        sous = np.abs(A[k:, k:])
        i_rel, j_rel = np.unravel_index(np.argmax(sous), sous.shape)
        p, q = k + i_rel, k + j_rel
        if abs(A[p, q]) < EPS:
            raise ValueError("Matrice singulière : pivot nul.")

        if p != k:
            A[[k, p]] = A[[p, k]]
            b[[k, p]] = b[[p, k]]
            if verbose:
                print(f"Étape {k + 1} : échange L{k + 1} <-> L{p + 1}")
        if q != k:
            A[:, [k, q]] = A[:, [q, k]]
            perm[[k, q]] = perm[[q, k]]
            if verbose:
                print(f"Étape {k + 1} : échange C{k + 1} <-> C{q + 1}")

        if verbose:
            ordre = ", ".join(f"x{j + 1}" for j in perm)
            print(f"Ordre des inconnues : ({ordre})")

        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] = A[i, k:] - m * A[k, k:]
            b[i] = b[i] - m * b[k]

        if verbose:
            print(f"Après l'étape {k + 1} :\n{np.column_stack((A, b))}\n")

    if abs(A[n - 1, n - 1]) < EPS:
        raise ValueError("Matrice singulière : dernier pivot nul.")

    y = remontee(A, b)
    x = np.zeros(n)
    x[perm] = y
    return x


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    A1 = [[1, 2, 1], [2, 3, 1], [5, 6, 9]]
    b1 = [0, -1, 20]
    x1 = gauss_pivot_total(A1, b1, verbose=True)
    print("Exemple 1 : x =", x1)
    print("Vérification :", np.allclose(x1, np.linalg.solve(A1, b1)), "\n")

    A2 = [[2, 1, -1, 3], [4, 5, 2, -1], [-2, 3, 8, 1], [1, -2, 1, 6]]
    b2 = [5, 10, 14, 9]
    x2 = gauss_pivot_total(A2, b2, verbose=True)
    print("Exemple 2 : x =", x2)
    print("Vérification :", np.allclose(x2, np.linalg.solve(A2, b2)))
