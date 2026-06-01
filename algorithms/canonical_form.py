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

    cores = [backend.copy(core) for core in tt.cores]

    for pos in range(order - 1):
        core = cores[pos]
        r_left, mode, r_right = core.shape

        mat = backend.reshape(core, (r_left * mode, r_right))
        if r_left * mode >= r_right:
            q_factor, r_factor = backend.qr(mat)
            next_rank = q_factor.shape[1]
            cores[pos] = backend.reshape(q_factor, (r_left, mode, next_rank))
        else:
            u_factor, s_vec, vt_factor = backend.svd(mat, full_matrices=False)
            next_rank = s_vec.shape[0]
            cores[pos] = backend.reshape(u_factor, (r_left, mode, next_rank))
            r_factor = _multiply_diag_matrix(s_vec, vt_factor, next_rank, backend)

        nxt = cores[pos + 1]
        nxt_left, nxt_mode, nxt_right = nxt.shape
        if nxt_left != r_right:
            raise ValueError("несогласованные размеры при левой каноникализации")

        nxt_mat = backend.reshape(nxt, (r_right, nxt_mode * nxt_right))
        pushed = backend.matmul(r_factor, nxt_mat)
        cores[pos + 1] = backend.reshape(pushed, (next_rank, nxt_mode, nxt_right))

    return TTTensor(cores)


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

    cores = [backend.copy(core) for core in tt.cores]

    for pos in range(order - 1, 0, -1):
        core = cores[pos]
        r_left, mode, r_right = core.shape

        mat = backend.reshape(core, (r_left, mode * r_right))
        use_triangular_push = False
        if mode * r_right >= r_left:
            mat_t = backend.transpose(mat)
            q_t, r_t = backend.qr(mat_t)
            q = backend.transpose(q_t)
            r = backend.transpose(r_t)
            next_rank = q.shape[0]
            cores[pos] = backend.reshape(q, (next_rank, mode, r_right))
            use_triangular_push = True
        else:
            u_factor, s_vec, vt_factor = backend.svd(mat, full_matrices=False)
            next_rank = s_vec.shape[0]
            cores[pos] = backend.reshape(vt_factor, (next_rank, mode, r_right))
            r = _multiply_columns_by_diag(u_factor, s_vec, backend)

        prev = cores[pos - 1]
        prev_left, prev_mode, prev_right = prev.shape
        if prev_right != r_left:
            raise ValueError("несогласованные размеры при правой каноникализации")

        prev_mat = backend.reshape(prev, (prev_left * prev_mode, r_left))
        if use_triangular_push:
            pushed = _matmul_with_lower_triangular(prev_mat, r, backend)
        else:
            pushed = backend.matmul(prev_mat, r)
        cores[pos - 1] = backend.reshape(pushed, (prev_left, prev_mode, next_rank))

    return TTTensor(cores)


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


def _matmul_with_lower_triangular(
    left: DenseTensor,
    lower_tri: DenseTensor,
    backend: BackendInterface
) -> DenseTensor:
    """
    Умножает матрицу на нижнетреугольную матрицу справа:
        left @ lower_tri

    Для уменьшения накопления ошибки использует суммирование Kahan.
    """
    if left.ndim != 2 or lower_tri.ndim != 2:
        raise ValueError("ожидаются 2D матрицы")

    m, k = left.shape
    k2, n = lower_tri.shape
    if k != k2:
        raise ValueError("несогласованные размеры матриц")
    if k2 != n:
        raise ValueError("правая матрица должна быть квадратной")

    out = backend.zeros((m, n))
    for i in range(m):
        for j in range(n):
            # Для нижнетреугольной матрицы элементы выше диагонали равны 0.
            start = j
            acc = 0.0
            comp = 0.0
            for t in range(start, k):
                y = left[i, t] * lower_tri[t, j] - comp
                tmp = acc + y
                comp = (tmp - acc) - y
                acc = tmp
            out[i, j] = acc
    return out
