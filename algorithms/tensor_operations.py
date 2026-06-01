# algorithms/tensor_operations.py

"""
Базовые операции с TT-тензорами.

Все операции работают напрямую с TT-ядрами,
не восстанавливая полный тензор.

Содержит:
    - tt_add:         поэлементное сложение
    - tt_scalar_mul:  умножение на скаляр
    - tt_hadamard:    поэлементное произведение (Адамар)
    - tt_dot:         скалярное произведение <A, B>
    - tt_norm:        Фробениусова норма
    - tt_diff_norm:   ||A - B||_F без восстановления полных тензоров

Все операции через backend.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


Number = int | float


def _assert_same_shape(tt1: TTTensor, tt2: TTTensor) -> None:
    if tt1.shape != tt2.shape:
        raise ValueError(f"shape не совпадают: {tt1.shape} != {tt2.shape}")


def _copy_core_region(
    dst: DenseTensor,
    src: DenseTensor,
    left_shift: int,
    right_shift: int
) -> None:
    left_rank, mode_size, right_rank = src.shape
    for a in range(left_rank):
        for i in range(mode_size):
            for b in range(right_rank):
                dst[a + left_shift, i, b + right_shift] = src[a, i, b]


def _extract_mode_matrix(
    core: DenseTensor,
    mode_index: int,
    backend: BackendInterface
) -> DenseTensor:
    left_rank, _, right_rank = core.shape
    mat = backend.zeros((left_rank, right_rank))
    for a in range(left_rank):
        for b in range(right_rank):
            mat[a, b] = core[a, mode_index, b]
    return mat


def tt_add(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного сложения двух TT-тензоров.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    _assert_same_shape(tt1, tt2)

    out_cores: list[DenseTensor] = []
    last_axis = tt1.order - 1

    for k in range(tt1.order):
        a = tt1.cores[k]
        b = tt2.cores[k]

        ra0, n, ra1 = a.shape
        rb0, n2, rb1 = b.shape
        if n != n2:
            raise ValueError("несогласованные модовые размеры ядер")

        if k == 0:
            merged = backend.zeros((1, n, ra1 + rb1))
            _copy_core_region(merged, a, left_shift=0, right_shift=0)
            _copy_core_region(merged, b, left_shift=0, right_shift=ra1)
        elif k == last_axis:
            merged = backend.zeros((ra0 + rb0, n, 1))
            _copy_core_region(merged, a, left_shift=0, right_shift=0)
            _copy_core_region(merged, b, left_shift=ra0, right_shift=0)
        else:
            merged = backend.zeros((ra0 + rb0, n, ra1 + rb1))
            _copy_core_region(merged, a, left_shift=0, right_shift=0)
            _copy_core_region(merged, b, left_shift=ra0, right_shift=ra1)

        out_cores.append(merged)

    return TTTensor(out_cores)


def tt_scalar_mul(
    tt: TTTensor,
    alpha: Number,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат умножения TT-тензора на скаляр.
    Модифицируем только первое ядро.

    Args:
        tt:      TTTensor
        alpha:   число
        backend: интерфейс backend
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha должен быть числом")

    cores = [backend.copy(core) for core in tt.cores]
    cores[0] = backend.scale(cores[0], alpha)
    return TTTensor(cores)


def tt_hadamard(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного произведения (произведения Адамара).

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    _assert_same_shape(tt1, tt2)

    d = tt1.order
    out_cores: list[DenseTensor] = []

    for k in range(d):
        a = tt1.cores[k]
        b = tt2.cores[k]

        ra0, n, ra1 = a.shape
        rb0, n2, rb1 = b.shape
        if n != n2:
            raise ValueError("несогласованные модовые размеры ядер")

        hadamard_core = backend.zeros((ra0 * rb0, n, ra1 * rb1))
        for i in range(n):
            for pa in range(ra0):
                for pb in range(rb0):
                    left_idx = pa * rb0 + pb
                    for qa in range(ra1):
                        a_val = a[pa, i, qa]
                        right_offset = qa * rb1
                        for qb in range(rb1):
                            right_idx = right_offset + qb
                            hadamard_core[left_idx, i, right_idx] = a_val * b[pb, i, qb]

        out_cores.append(hadamard_core)

    return TTTensor(out_cores)


def tt_dot(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> Number:
    """
    Возвращает скалярное произведение двух TT-тензоров: <tt1, tt2>.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    _assert_same_shape(tt1, tt2)

    accumulator = backend.ones((1, 1))

    for k in range(tt1.order):
        a = tt1.cores[k]
        b = tt2.cores[k]

        ra0, n, ra1 = a.shape
        rb0, n2, rb1 = b.shape
        if n != n2:
            raise ValueError("несогласованные модовые размеры ядер")
        if accumulator.shape != (ra0, rb0):
            raise ValueError("несогласованные ранги в tt_dot")

        next_accumulator = backend.zeros((ra1, rb1))
        for i in range(n):
            a_slice = _extract_mode_matrix(a, i, backend)
            b_slice = _extract_mode_matrix(b, i, backend)
            left = backend.matmul(backend.transpose(a_slice), accumulator)
            next_accumulator = backend.add(next_accumulator, backend.matmul(left, b_slice))

        accumulator = next_accumulator

    return accumulator[0, 0]


def tt_norm(
    tt: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает Фробениусову норму TT-тензора.

    Args:
        tt:      TTTensor
        backend: интерфейс backend
    """
    val = tt_dot(tt, tt, backend)
    return math.sqrt(max(0.0, float(val)))


def tt_diff_norm(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает норму разности: ||tt1 - tt2||_F.
    Вычисляется без восстановления полных тензоров:

    Args:
        tt1, tt2: TTTensor
        backend:  интерфейс backend
    """
    _assert_same_shape(tt1, tt2)

    n1_sq = tt_dot(tt1, tt1, backend)
    n2_sq = tt_dot(tt2, tt2, backend)
    dot12 = tt_dot(tt1, tt2, backend)

    val = float(n1_sq) + float(n2_sq) - 2.0 * float(dot12)
    return math.sqrt(max(0.0, val))
