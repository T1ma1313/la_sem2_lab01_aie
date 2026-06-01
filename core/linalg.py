# core/linalg.py

"""
Базовая линейная алгебра для DenseTensor.

Этот модуль содержит reference-реализации операций:
    - eye
    - transpose
    - diag
    - matmul
    - qr
    - svd

Все функции работают поверх DenseTensor и не зависят от backend API.
Их можно использовать как из CPUBackend, так и напрямую.
Ожидается, что этот файл не будет редактироваться.
"""

from __future__ import annotations

import math

from core.dense_tensor import DenseTensor


# ────────────────────────────────────────────
# Базовые операции
# ────────────────────────────────────────────

def eye(n: int) -> DenseTensor:
    """
    Возвращает единичную матрицу размера (n, n).

    Args:
        n: размер матрицы
    """
    data: list[float] = [0.0] * (n * n)
    for i in range(n):
        data[i * n + i] = 1.0
    return DenseTensor((n, n), data=data)


def transpose(matrix: DenseTensor) -> DenseTensor:
    """
    Возвращает транспонированную 2D-матрицу.

    Args:
        matrix: DenseTensor с ndim == 2
    """
    if matrix.ndim != 2:
        raise ValueError(f"transpose ожидает 2D, получено ndim={matrix.ndim}")

    m: int
    n: int
    m, n = matrix.shape
    result: DenseTensor = DenseTensor.zeros((n, m))

    for i in range(m):
        for j in range(n):
            result[j, i] = matrix[i, j]

    return result


def diag(vector: DenseTensor) -> DenseTensor:
    """
    Возвращает диагональную матрицу, полученную из 1D-вектора.

    Args:
        vector: DenseTensor с ndim == 1
    """
    if vector.ndim != 1:
        raise ValueError(f"diag ожидает 1D, получено ndim={vector.ndim}")

    n: int = vector.shape[0]
    result: DenseTensor = DenseTensor.zeros((n, n))

    for i in range(n):
        result[i, i] = vector.data[i]

    return result


def matmul(a: DenseTensor, b: DenseTensor) -> DenseTensor:
    """
    Возвращает матричное произведение двух 2D-тензоров.

    Args:
        a: DenseTensor формы (m, k)
        b: DenseTensor формы (k, n)
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"matmul ожидает 2D тензоры, получено {a.ndim}D и {b.ndim}D"
        )

    m: int
    k1: int
    m, k1 = a.shape
    k2: int
    n: int
    k2, n = b.shape

    if k1 != k2:
        raise ValueError(
            f"Несовместимые размеры для matmul: ({m},{k1}) x ({k2},{n})"
        )

    k: int = k1
    result: DenseTensor = DenseTensor.zeros((m, n))

    a_data: list[float] = a.data
    b_data: list[float] = b.data
    r_data: list[float] = result.data

    for i in range(m):
        i_offset: int = i * k
        r_offset: int = i * n
        for j in range(n):
            s: float = 0.0
            for p in range(k):
                s += a_data[i_offset + p] * b_data[p * n + j]
            r_data[r_offset + j] = s

    return result


# ────────────────────────────────────────────
# QR-разложение
# ────────────────────────────────────────────

def qr(matrix: DenseTensor) -> tuple[DenseTensor, DenseTensor]:
    """
    Тонкое QR-разложение модифицированным методом Грама-Шмидта
    с одним шагом реортогонализации.

    Args:
        matrix: DenseTensor формы (m, n), где m >= n

    Returns:
        Q: DenseTensor формы (m, n)
        R: DenseTensor формы (n, n)
    """
    if matrix.ndim != 2:
        raise ValueError(f"qr ожидает 2D, получено ndim={matrix.ndim}")

    m: int
    n: int
    m, n = matrix.shape
    if m < n:
        raise ValueError(
            f"Для тонкого QR нужно m >= n, получено ({m}, {n})"
        )

    # Извлекаем столбцы матрицы как списки
    cols: list[list[float]] = []
    for j in range(n):
        col: list[float] = [matrix.data[i * n + j] for i in range(m)]
        cols.append(col)

    q_cols: list[list[float]] = []
    R: list[list[float]] = [[0.0] * n for _ in range(n)]

    for j in range(n):
        v: list[float] = cols[j][:]

        # Первый проход ортогонализации
        for i in range(len(q_cols)):
            dot: float = 0.0
            for row in range(m):
                dot += q_cols[i][row] * v[row]
            R[i][j] += dot
            for row in range(m):
                v[row] -= dot * q_cols[i][row]

        # Второй проход
        for i in range(len(q_cols)):
            dot = 0.0
            for row in range(m):
                dot += q_cols[i][row] * v[row]
            R[i][j] += dot
            for row in range(m):
                v[row] -= dot * q_cols[i][row]

        norm_v: float = math.sqrt(sum(x * x for x in v))
        R[j][j] = norm_v

        if norm_v > 1e-15:
            for row in range(m):
                v[row] /= norm_v
        else:
            # Столбец линейно зависим
            v = [0.0] * m

        q_cols.append(v)

    Q: DenseTensor = DenseTensor.zeros((m, n))
    for j in range(n):
        for i in range(m):
            Q[i, j] = q_cols[j][i]

    R_tensor: DenseTensor = DenseTensor.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            R_tensor[i, j] = R[i][j]

    return Q, R_tensor


# ────────────────────────────────────────────
# SVD-разложение
# ────────────────────────────────────────────

def svd(
    matrix: DenseTensor,
    full_matrices: bool = False
) -> tuple[DenseTensor, DenseTensor, DenseTensor]:
    """
    Тонкое SVD-разложение: A = U @ diag(S) @ Vt

    Args:
        matrix: DenseTensor формы (m, n)
        full_matrices: полная или сокращенная форма матрицы

    Returns:
        U:  DenseTensor формы (m, k)
        S:  DenseTensor формы (k,)
        Vt: DenseTensor формы (k, n)

    где k = min(m, n)
    """
    if matrix.ndim != 2:
        raise ValueError(f"svd ожидает 2D, получено ndim={matrix.ndim}")

    m: int
    n: int
    m, n = matrix.shape
    k: int = min(m, n)
    tol: float = 1e-12

    A: list[list[float]] = _to_list_of_rows(matrix, m, n)

    if m >= n:
        return _svd_tall(A, m, n, k, tol)
    return _svd_wide(A, m, n, k, tol)


def _svd_tall(
    A: list[list[float]],
    m: int,
    n: int,
    k: int,
    tol: float
) -> tuple[DenseTensor, DenseTensor, DenseTensor]:
    """
    SVD для m >= n через собственное разложение A^T A.
    """
    ata: list[list[float]] = _build_ata(A, m, n)

    eigenvalues: list[float]
    eigenvectors: list[list[float]]
    eigenvalues, eigenvectors = _jacobi_eigen_symmetric(ata, tol=tol)

    order: list[int] = sorted(range(n), key=lambda i: -eigenvalues[i])

    singular_values: list[float] = []
    v_cols: list[list[float]] = []

    for idx in order[:k]:
        lam: float = max(eigenvalues[idx], 0.0)
        s: float = math.sqrt(lam)
        singular_values.append(s)
        col: list[float] = [eigenvectors[row][idx] for row in range(n)]
        v_cols.append(col)

    # Ортонормируем V на случай вырождения
    v_cols = _orthonormalize_vectors(v_cols, n, tol)

    # U_i = A @ v_i / s_i
    u_cols: list[list[float] | None] = []
    for i in range(k):
        s = singular_values[i]
        if s > tol:
            v: list[float] = v_cols[i]
            u: list[float] = [0.0] * m
            for row in range(m):
                acc: float = 0.0
                for col_idx in range(n):
                    acc += A[row][col_idx] * v[col_idx]
                u[row] = acc / s
            u_cols.append(u)
        else:
            u_cols.append(None)

    # Дополняем/ортонормируем U
    u_cols_final: list[list[float]] = _orthonormalize_vectors(u_cols, m, tol)

    U: DenseTensor = _tensor_from_columns(u_cols_final, m)
    S: DenseTensor = DenseTensor((k,), data=singular_values)
    Vt: DenseTensor = _tensor_from_rows(v_cols, n)

    return U, S, Vt


def _svd_wide(
    A: list[list[float]],
    m: int,
    n: int,
    k: int,
    tol: float
) -> tuple[DenseTensor, DenseTensor, DenseTensor]:
    """
    SVD для m < n через собственное разложение A A^T.
    """
    aat: list[list[float]] = _build_aat(A, m, n)

    eigenvalues: list[float]
    eigenvectors: list[list[float]]
    eigenvalues, eigenvectors = _jacobi_eigen_symmetric(aat, tol=tol)

    order: list[int] = sorted(range(m), key=lambda i: -eigenvalues[i])

    singular_values: list[float] = []
    u_cols: list[list[float]] = []

    for idx in order[:k]:
        lam: float = max(eigenvalues[idx], 0.0)
        s: float = math.sqrt(lam)
        singular_values.append(s)
        col: list[float] = [eigenvectors[row][idx] for row in range(m)]
        u_cols.append(col)

    # Ортонормируем U
    u_cols = _orthonormalize_vectors(u_cols, m, tol)

    # v_i = A^T @ u_i / s_i
    v_rows: list[list[float] | None] = []
    for i in range(k):
        s = singular_values[i]
        if s > tol:
            u: list[float] = u_cols[i]
            v: list[float] = [0.0] * n
            for col_idx in range(n):
                acc: float = 0.0
                for row in range(m):
                    acc += A[row][col_idx] * u[row]
                v[col_idx] = acc / s
            v_rows.append(v)
        else:
            v_rows.append(None)

    # Дополняем/ортонормируем V
    v_rows_final: list[list[float]] = _orthonormalize_vectors(v_rows, n, tol)

    U: DenseTensor = _tensor_from_columns(u_cols, m)
    S: DenseTensor = DenseTensor((k,), data=singular_values)
    Vt: DenseTensor = _tensor_from_rows(v_rows_final, n)

    return U, S, Vt


# ────────────────────────────────────────────
# Вспомогательные функции для SVD
# ────────────────────────────────────────────

def _to_list_of_rows(
    matrix: DenseTensor,
    m: int,
    n: int
) -> list[list[float]]:
    """
    Конвертирует DenseTensor (m, n) в список строк.
    """
    rows: list[list[float]] = []
    for i in range(m):
        row: list[float] = []
        for j in range(n):
            row.append(matrix.data[i * n + j])
        rows.append(row)
    return rows


def _build_ata(
    A: list[list[float]],
    m: int,
    n: int
) -> list[list[float]]:
    """
    Строит симметричную матрицу A^T A размера (n, n).
    """
    result: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s: float = 0.0
            for r in range(m):
                s += A[r][i] * A[r][j]
            result[i][j] = s
            result[j][i] = s
    return result


def _build_aat(
    A: list[list[float]],
    m: int,
    n: int
) -> list[list[float]]:
    """
    Строит симметричную матрицу A A^T размера (m, m).
    """
    result: list[list[float]] = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i, m):
            s: float = 0.0
            for c in range(n):
                s += A[i][c] * A[j][c]
            result[i][j] = s
            result[j][i] = s
    return result


def _jacobi_eigen_symmetric(
    matrix: list[list[float]],
    tol: float = 1e-12,
    max_iter: int | None = None
) -> tuple[list[float], list[list[float]]]:
    """
    Находит собственные значения и собственные векторы симметричной
    матрицы методом вращений Якоби.

    Args:
        matrix: список списков n x n
        tol: порог сходимости
        max_iter: максимум итераций

    Returns:
        eigenvalues: список длины n
        eigenvectors: матрица n x n, где собственные векторы лежат в столбцах
    """
    n: int = len(matrix)

    if n == 0:
        return [], []
    if n == 1:
        return [matrix[0][0]], [[1.0]]

    A: list[list[float]] = [row[:] for row in matrix]
    V: list[list[float]] = [
        [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)
    ]

    if max_iter is None:
        max_iter = 100 * n * n

    for _ in range(max_iter):
        p: int = 0
        q: int = 1
        max_off: float = abs(A[0][1])

        for i in range(n):
            for j in range(i + 1, n):
                val: float = abs(A[i][j])
                if val > max_off:
                    max_off = val
                    p, q = i, j

        if max_off < tol:
            break

        app: float = A[p][p]
        aqq: float = A[q][q]
        apq: float = A[p][q]

        if abs(apq) < tol:
            continue

        theta: float = (aqq - app) / (2.0 * apq)

        t: float
        if theta >= 0.0:
            t = 1.0 / (theta + math.sqrt(1.0 + theta * theta))
        else:
            t = -1.0 / (-theta + math.sqrt(1.0 + theta * theta))

        c: float = 1.0 / math.sqrt(1.0 + t * t)
        s: float = t * c

        for k in range(n):
            if k != p and k != q:
                akp: float = A[k][p]
                akq: float = A[k][q]

                A[k][p] = c * akp - s * akq
                A[p][k] = A[k][p]

                A[k][q] = s * akp + c * akq
                A[q][k] = A[k][q]

        A[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        A[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        A[p][q] = 0.0
        A[q][p] = 0.0

        for k in range(n):
            vkp: float = V[k][p]
            vkq: float = V[k][q]
            V[k][p] = c * vkp - s * vkq
            V[k][q] = s * vkp + c * vkq

    eigenvalues: list[float] = [A[i][i] for i in range(n)]
    return eigenvalues, V


def _orthonormalize_vectors(
    vectors: list[list[float] | None],
    dim: int,
    tol: float = 1e-12
) -> list[list[float]]:
    """
    Ортонормирует список векторов длины dim.
    Элементы списка могут быть None.

    Returns:
        список ортонормированных векторов той же длины
    """
    result: list[list[float] | None] = []
    accepted: list[list[float]] = []

    for vec in vectors:
        if vec is None:
            result.append(None)
            continue

        w: list[float] = vec[:]

        # Первый проход
        for q_vec in accepted:
            dot: float = sum(w[i] * q_vec[i] for i in range(dim))
            for i in range(dim):
                w[i] -= dot * q_vec[i]

        # Реортогонализация
        for q_vec in accepted:
            dot = sum(w[i] * q_vec[i] for i in range(dim))
            for i in range(dim):
                w[i] -= dot * q_vec[i]

        norm_w: float = math.sqrt(sum(x * x for x in w))
        if norm_w < tol:
            result.append(None)
            continue

        w = [x / norm_w for x in w]
        result.append(w)
        accepted.append(w)

    basis_idx: int = 0
    for idx in range(len(result)):
        if result[idx] is not None:
            continue

        while basis_idx < dim:
            w = [0.0] * dim
            w[basis_idx] = 1.0
            basis_idx += 1

            for q_vec in accepted:
                dot = sum(w[i] * q_vec[i] for i in range(dim))
                for i in range(dim):
                    w[i] -= dot * q_vec[i]

            for q_vec in accepted:
                dot = sum(w[i] * q_vec[i] for i in range(dim))
                for i in range(dim):
                    w[i] -= dot * q_vec[i]

            norm_w = math.sqrt(sum(x * x for x in w))
            if norm_w >= tol:
                w = [x / norm_w for x in w]
                result[idx] = w
                accepted.append(w)
                break

        if result[idx] is None:
            raise RuntimeError("Не удалось дополнить ортонормированный базис")

    return result  # type: ignore[return-value]


def _tensor_from_columns(
    columns: list[list[float]],
    num_rows: int
) -> DenseTensor:
    """
    Собирает DenseTensor формы (num_rows, len(columns))
    из списка столбцов.
    """
    k: int = len(columns)
    data: list[float] = [0.0] * (num_rows * k)

    for j in range(k):
        col: list[float] = columns[j]
        for i in range(num_rows):
            data[i * k + j] = col[i]

    return DenseTensor((num_rows, k), data=data)


def _tensor_from_rows(
    rows: list[list[float]],
    num_cols: int
) -> DenseTensor:
    """
    Собирает DenseTensor формы (len(rows), num_cols)
    из списка строк.
    """
    k: int = len(rows)
    data: list[float] = [0.0] * (k * num_cols)

    for i in range(k):
        row: list[float] = rows[i]
        for j in range(num_cols):
            data[i * num_cols + j] = row[j]

    return DenseTensor((k, num_cols), data=data)
