# algorithms/tt_svd.py

"""
TT-SVD алгоритм: разложение плотного тензора в TT-формат.
"""

import math

from algorithms.canonical_form import (
    _multiply_diag_matrix as _cf_multiply_diag_matrix,
    _truncate_columns as _cf_truncate_columns,
    _truncate_rows as _cf_truncate_rows,
    _truncate_vector as _cf_truncate_vector,
)
from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def tt_svd(
    tensor: DenseTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — тензор в TT-формате.

    Args:
        tensor:   DenseTensor с shape (n_0, n_1, ..., n_{d-1})
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    if max_rank is not None and (
        isinstance(max_rank, bool) or not isinstance(max_rank, int) or max_rank <= 0
    ):
        raise ValueError("max_rank должен быть > 0 или None")

    order = tensor.ndim
    if order == 1:
        return TTTensor([backend.reshape(tensor, (1, tensor.shape[0], 1))])

    base_norm = backend.norm(tensor)
    delta = eps * base_norm / math.sqrt(order - 1) if base_norm > 1e-30 else 0.0

    cores: list[DenseTensor] = []
    residual = backend.copy(tensor)
    left_rank = 1

    for axis in range(order - 1):
        mode = tensor.shape[axis]
        rows = left_rank * mode
        cols = residual.size // rows

        matrix = backend.reshape(residual, (rows, cols))
        U, S, Vt = backend.svd(matrix, full_matrices=False)

        next_rank = _compute_truncated_rank(S, delta, max_rank)

        left_factor = _truncate_columns(U, next_rank, backend)
        cores.append(backend.reshape(left_factor, (left_rank, mode, next_rank)))

        sigma = _truncate_vector(S, next_rank, backend)
        right_factor = _truncate_rows(Vt, next_rank, backend)
        residual = _multiply_diag_matrix(sigma, right_factor, next_rank, backend)

        left_rank = next_rank

    n_last = tensor.shape[-1]
    last_core = backend.reshape(residual, (left_rank, n_last, 1))
    cores.append(last_core)

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_truncated_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает ранг усечения по сингулярным значениям.

    Args:
        S:        DenseTensor (k,) — сингулярные значения по убыванию
        delta:    порог усечения
        max_rank: максимальный ранг (None = без ограничения)
    """
    if S.ndim != 1:
        raise ValueError("S должен быть вектором (1D)")

    size = S.shape[0]
    if size == 0:
        return 1

    cutoff = max(1e-12, 1e-8 * abs(S[0]))
    numerical_rank = sum(1 for x in S.data if abs(x) > cutoff)
    numerical_rank = max(1, numerical_rank)

    keep_rank = numerical_rank
    if delta > 0.0:
        dropped_sq = 0.0
        while keep_rank > 1:
            next_sq = dropped_sq + S[keep_rank - 1] * S[keep_rank - 1]
            if next_sq > delta * delta:
                break
            dropped_sq = next_sq
            keep_rank -= 1

    if max_rank is not None:
        keep_rank = min(keep_rank, max_rank)
    return max(1, keep_rank)


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
    return _cf_truncate_columns(matrix, rank, backend)


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
    return _cf_truncate_rows(matrix, rank, backend)


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
    return _cf_truncate_vector(vector, rank, backend)


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
        rank:     число строк матрицы и длина диагонального вектора
        backend:  интерфейс backend
    """
    return _cf_multiply_diag_matrix(diag_vec, matrix, rank, backend)
