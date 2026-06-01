# algorithms/canonical_form.py

"""
Приведение TT-тензора в канонические формы (полная правая и
левая ортогонализация ядер).
"""

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def left_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в лево-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    order = tt.order
    if order < 2:
        return tt.copy()

    canonical_cores = [backend.copy(core) for core in tt.cores]

    for idx in range(order - 1):
        current = canonical_cores[idx]
        left_rank, mode_size, right_rank = current.shape

        mat = backend.reshape(current, (left_rank * mode_size, right_rank))
        U, S, Vt = backend.svd(mat, full_matrices=False)
        new_rank = S.shape[0]

        canonical_cores[idx] = backend.reshape(U, (left_rank, mode_size, new_rank))
        transfer = _multiply_diag_matrix(S, Vt, new_rank, backend)

        nxt = canonical_cores[idx + 1]
        nxt_left, nxt_mode, nxt_right = nxt.shape
        if nxt_left != right_rank:
            raise ValueError("несогласованные размеры при левой каноникализации")

        nxt_matrix = backend.reshape(nxt, (right_rank, nxt_mode * nxt_right))
        contracted = backend.matmul(transfer, nxt_matrix)
        canonical_cores[idx + 1] = backend.reshape(
            contracted, (new_rank, nxt_mode, nxt_right)
        )

    return TTTensor(canonical_cores)


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в право-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    order = tt.order
    if order < 2:
        return tt.copy()

    canonical_cores = [backend.copy(core) for core in tt.cores]

    for idx in range(order - 1, 0, -1):
        current = canonical_cores[idx]
        left_rank, mode_size, right_rank = current.shape

        mat = backend.reshape(current, (left_rank, mode_size * right_rank))
        U, S, Vt = backend.svd(mat, full_matrices=False)
        new_rank = S.shape[0]

        canonical_cores[idx] = backend.reshape(Vt, (new_rank, mode_size, right_rank))
        transfer = _multiply_columns_by_diag(U, S, backend)

        prev = canonical_cores[idx - 1]
        prev_left, prev_mode, prev_right = prev.shape
        if prev_right != left_rank:
            raise ValueError("несогласованные размеры при правой каноникализации")

        prev_matrix = backend.reshape(prev, (prev_left * prev_mode, left_rank))
        contracted = backend.matmul(prev_matrix, transfer)
        canonical_cores[idx - 1] = backend.reshape(
            contracted, (prev_left, prev_mode, new_rank)
        )

    return TTTensor(canonical_cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _numerical_rank(
    S: DenseTensor,
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-12
) -> int:
    """
    Возвращает числовой ранг матрицы по вектору сингулярных значений.

    Сингулярное число \sigma_i считаем ненулевым, если:
        |\sigma_i| > max(abs_tol, rel_tol * max(\sigma_1, ..., \sigma_n))

    Args:
        S:       одномерный тензор формы (k,) — сингулярные значения
                 в порядке убывания
        rel_tol: относительный допуск (по умолчанию 1e-8)
        abs_tol: абсолютный допуск (по умолчанию 1e-12)
    """
    if S.ndim != 1:
        raise ValueError("S должен быть 1D вектором")
    if S.size == 0:
        return 0

    threshold = max(abs_tol, rel_tol * max(abs(x) for x in S.data))
    return sum(1 for sigma in S.data if abs(sigma) > threshold)


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.

    Используется после SVD для усечения матрицы левых сингулярных векторов:
        U in R^{m x n} -> U_trunc in R^{m x rank}

    Args:
        matrix:  двумерный тензор формы (m, n)
        rank:    число сохраняемых столбцов
        backend: интерфейс backend
    """
    if matrix.ndim != 2:
        raise ValueError("matrix должен быть 2D")

    m, n = matrix.shape
    if rank < 0 or rank > n:
        raise ValueError(f"rank должен быть в диапазоне [0, {n}]")

    out = backend.zeros((m, rank))
    for i in range(m):
        for j in range(rank):
            out[i, j] = matrix[i, j]
    return out


def _truncate_rows(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank строк исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (k, n)
        rank:    число сохраняемых строк
        backend: интерфейс backend
    """
    if matrix.ndim != 2:
        raise ValueError("matrix должен быть 2D")

    k, n = matrix.shape
    if rank < 0 or rank > k:
        raise ValueError(f"rank должен быть в диапазоне [0, {k}]")

    out = backend.zeros((rank, n))
    for i in range(rank):
        for j in range(n):
            out[i, j] = matrix[i, j]
    return out


def _truncate_vector(
    vector: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает вектор, состоящий из первых rank элементов исходного вектора.

    Args:
        vector:  одномерный тензор формы (k,)
        rank:    число сохраняемых элементов
        backend: интерфейс backend
    """
    if vector.ndim != 1:
        raise ValueError("vector должен быть 1D")

    k = vector.shape[0]
    if rank < 0 or rank > k:
        raise ValueError(f"rank должен быть в диапазоне [0, {k}]")

    out = backend.zeros((rank,))
    for i in range(rank):
        out[i] = vector[i]
    return out


def _multiply_diag_matrix(
    diag_vec: DenseTensor,
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает произведение диагональной матрицы на обычную матрицу:
        diag(diag_vec) @ matrix

    Args:
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        matrix:   двумерный тензор формы (rank, n)
        rank:     длина диагонального вектора
        backend:  интерфейс backend
    """
    if diag_vec.ndim != 1:
        raise ValueError("diag_vec должен быть 1D")
    if matrix.ndim != 2:
        raise ValueError("matrix должен быть 2D")

    if diag_vec.shape[0] < rank:
        raise ValueError("rank превышает длину diag_vec")
    if matrix.shape[0] < rank:
        raise ValueError("число строк matrix должно быть >= rank")

    _, n = matrix.shape
    out = backend.zeros((rank, n))
    for i in range(rank):
        scale = diag_vec[i]
        for j in range(n):
            out[i, j] = scale * matrix[i, j]
    return out


def _multiply_columns_by_diag(
    matrix: DenseTensor,
    diag_vec: DenseTensor,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает результат произведения обычной матрицы на диагональную:
        matrix @ diag(diag_vec)

    Args:
        matrix:   двумерный тензор формы (m, n)
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        backend:  интерфейс backend
    """
    if matrix.ndim != 2:
        raise ValueError("matrix должен быть 2D")
    if diag_vec.ndim != 1:
        raise ValueError("diag_vec должен быть 1D")

    m, n = matrix.shape
    if diag_vec.shape[0] != n:
        raise ValueError("длина diag_vec должна совпадать с числом столбцов matrix")

    out = backend.zeros((m, n))
    for i in range(m):
        for j in range(n):
            out[i, j] = matrix[i, j] * diag_vec[j]
    return out

