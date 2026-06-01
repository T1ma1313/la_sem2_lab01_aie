# algorithms/tt_round.py

"""
TT-округление.
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
from algorithms.canonical_form import right_canonicalize


def tt_round(
    tt: TTTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор с уменьшенными рангами

    Args:
        tt:       исходный тензор
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    if max_rank is not None and (
        isinstance(max_rank, bool) or not isinstance(max_rank, int) or max_rank <= 0
    ):
        raise ValueError("max_rank должен быть > 0 или None")

    order = tt.order
    if order < 2:
        return tt.copy()

    right_canon = right_canonicalize(tt, backend)
    cores = [backend.copy(core) for core in right_canon.cores]

    norm_ref = backend.norm(cores[0])
    delta = eps * norm_ref / math.sqrt(order - 1) if norm_ref > 1e-30 else 0.0

    for idx in range(order - 1):
        current = cores[idx]
        left_rank, mode_size, right_rank = current.shape

        matrix = backend.reshape(current, (left_rank * mode_size, right_rank))
        U, S, Vt = backend.svd(matrix, full_matrices=False)

        keep_rank = _compute_rank(S, delta, max_rank)

        left_factor = _truncate_columns(U, keep_rank, backend)
        cores[idx] = backend.reshape(left_factor, (left_rank, mode_size, keep_rank))

        sigma = _truncate_vector(S, keep_rank, backend)
        right_factor = _truncate_rows(Vt, keep_rank, backend)
        transfer = _multiply_diag_matrix(sigma, right_factor, keep_rank, backend)

        nxt = cores[idx + 1]
        nxt_left, nxt_mode, nxt_right = nxt.shape
        if nxt_left != right_rank:
            raise ValueError("несогласованные ранги при TT-round")

        nxt_matrix = backend.reshape(nxt, (right_rank, nxt_mode * nxt_right))
        contracted = backend.matmul(transfer, nxt_matrix)
        cores[idx + 1] = backend.reshape(contracted, (keep_rank, nxt_mode, nxt_right))

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает int ранг усечения по вектору сингулярных значений.

    Args:
        S:        одномерный тензор формы (k,) — сингулярные значения
                  в порядке убывания
        delta:    абсолютный порог усечения (0 — без усечения по delta)
        max_rank: максимально допустимый ранг (None = без ограничения)
    """
    size = S.shape[0]
    if size == 0:
        return 1

    keep_rank = size
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
