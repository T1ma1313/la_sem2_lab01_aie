# core/tt_tensor.py

"""
Тензор в TT-формате (Tensor Train).

TT-тензор порядка d с shape (n_0, n_1, ..., n_{d-1}) хранится как
список d ядер (cores), где k-е ядро — это 3D DenseTensor с shape:
    (r_k, n_k, r_{k+1})

Граничные условия: r_0 = r_d = 1.

TT-ранги: (r_0, r_1, ..., r_d) = (1, r_1, ..., r_{d-1}, 1).
"""

from __future__ import annotations

import random

from core.dense_tensor import DenseTensor
from core.utils import validate_shape, compute_size, flat_to_multi_index


class TTTensor:
    """
    Тензор в TT-формате.

    Атрибуты:
        cores:  список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        order:  порядок тензора d (число мод)
        shape:  кортеж (n_0, n_1, ..., n_{d-1})
        ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d), r_0 = r_d = 1
    """

    __slots__ = ('cores', 'order', 'shape', 'ranks')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(self, cores: list[DenseTensor]) -> None:
        """
        Создаёт TT-тензор из списка ядер.

        Args:
            cores: список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        """
        if not isinstance(cores, list) or len(cores) == 0:
            raise ValueError("cores должен быть непустым списком DenseTensor")

        copied_cores: list[DenseTensor] = []
        mode_sizes: list[int] = []
        ranks: list[int] = []

        prev_rank: int | None = None
        for idx, core in enumerate(cores):
            if not isinstance(core, DenseTensor):
                raise TypeError("каждое ядро должно быть DenseTensor")
            if core.ndim != 3:
                raise ValueError(
                    f"ядро {idx} должно быть 3D, получено ndim={core.ndim}"
                )

            left_rank, mode_size, right_rank = core.shape
            if prev_rank is None:
                if left_rank != 1:
                    raise ValueError("первый TT-ранг должен быть равен 1")
                ranks.append(1)
            elif left_rank != prev_rank:
                raise ValueError(
                    f"несогласованные ранги ядер на позиции {idx}: "
                    f"ожидался левый ранг {prev_rank}, получен {left_rank}"
                )

            mode_sizes.append(mode_size)
            ranks.append(right_rank)
            prev_rank = right_rank
            copied_cores.append(core.copy())

        if ranks[-1] != 1:
            raise ValueError("последний TT-ранг должен быть равен 1")

        self.cores = copied_cores
        self.order = len(copied_cores)
        self.shape = validate_shape(tuple(mode_sizes))
        self.ranks = tuple(ranks)

    @staticmethod
    def random(shape, ranks, seed=None):
        """
        Создаёт случайный TT-тензор с заданными рангами.

        Args:
            shape:  кортеж размеров мод (n_0, ..., n_{d-1})
            ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d)
                    или список внутренних рангов (r_1, ..., r_{d-1})
            seed:   seed для воспроизводимости

        NB: это отладочная функция, она не проверяется тестами
        """
        shp = validate_shape(shape)
        d = len(shp)

        if not isinstance(ranks, (tuple, list)):
            raise TypeError("ranks должен быть tuple или list")

        rank_list = list(ranks)
        if len(rank_list) == d - 1:
            rank_list = [1, *rank_list, 1]
        elif len(rank_list) != d + 1:
            raise ValueError(
                "ranks должен содержать либо d+1 рангов, либо d-1 внутренних рангов"
            )

        if rank_list[0] != 1 or rank_list[-1] != 1:
            raise ValueError("граничные TT-ранги должны быть равны 1")

        for rank in rank_list:
            if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
                raise ValueError("все TT-ранги должны быть положительными целыми")

        rng = random.Random(seed)
        cores = []
        for k in range(d):
            left_rank = rank_list[k]
            mode = shp[k]
            right_rank = rank_list[k + 1]
            core_data = [rng.uniform(-1.0, 1.0) for _ in range(left_rank * mode * right_rank)]
            cores.append(DenseTensor((left_rank, mode, right_rank), data=core_data))

        return TTTensor(cores)

    # ────────────────────────────────────────────
    # Доступ к элементам
    # ────────────────────────────────────────────

    def get_element(
        self,
        indices: tuple[int, ...] | list[int]
    ) -> float:
        """
        Возвращает элемент TT-тензора по его мультииндексу.

        Args:
            indices: кортеж/список длины d
        """
        if not isinstance(indices, (tuple, list)):
            raise TypeError("indices должен быть tuple или list")
        if len(indices) != self.order:
            raise ValueError(
                f"ожидается {self.order} индексов, получено {len(indices)}"
            )

        idx_tuple = tuple(indices)
        for axis, (idx, limit) in enumerate(zip(idx_tuple, self.shape)):
            if isinstance(idx, bool) or not isinstance(idx, int):
                raise TypeError("индексы должны быть целыми числами")
            if idx < 0 or idx >= limit:
                raise IndexError(
                    f"индекс {idx} вне диапазона [0, {limit}) "
                    f"для оси {axis}"
                )

        state = [1.0]
        for axis, core in enumerate(self.cores):
            left_rank, _, right_rank = core.shape
            mode_index = idx_tuple[axis]
            next_state = [0.0] * right_rank
            for left in range(left_rank):
                if state[left] == 0.0:
                    continue
                coeff = state[left]
                for right in range(right_rank):
                    next_state[right] += coeff * core[left, mode_index, right]
            state = next_state

        return float(state[0])

    # ────────────────────────────────────────────
    # Восстановление полного тензора
    # ────────────────────────────────────────────

    def full(self) -> DenseTensor:
        """Возвращает полный DenseTensor из его TT-формата."""
        total = compute_size(self.shape)
        data = [
            self.get_element(flat_to_multi_index(flat_idx, self.shape))
            for flat_idx in range(total)
        ]
        return DenseTensor(self.shape, data=data)

    # ────────────────────────────────────────────
    # Информация и отладка
    # ────────────────────────────────────────────

    def core_sizes(self) -> list[tuple[int, ...]]:
        """Возвращает размеры всех ядер."""
        return [core.shape for core in self.cores]

    def total_storage(self) -> int:
        """
        Возвращает общее число элементов во всех ядрах.
        Это то, сколько памяти реально занимает TT-тензор.
        """
        return sum(core.size for core in self.cores)

    def compression_ratio(self) -> float:
        """
        Возвращает отношение числа элементов полного тензора к числу
        элементов TT-тензора. Показывает, насколько TT-формат компактнее.
        """
        full_size = compute_size(self.shape)
        storage = self.total_storage()
        if storage == 0:
            return float("inf")
        return full_size / storage

    def copy(self) -> TTTensor:
        """Возвращает глубокую копию TT-тензора."""
        return TTTensor([core.copy() for core in self.cores])

    def __repr__(self) -> str:
        """
        Возвращает строковое представление TT-тензора для отладки.

        Формирует многострочную строку с основной служебной информацией
        об объекте:
            - порядок тензора (order),
            - исходная форма (shape),
            - TT-ранги (ranks),
            - размеры TT-ядер (cores),
            - суммарный объём хранения в элементах.

        NB: это отладочная функция, которая не покрывается тестами
        """
        return (
            "TTTensor(\n"
            f"  order={self.order},\n"
            f"  shape={self.shape},\n"
            f"  ranks={self.ranks},\n"
            f"  cores={self.core_sizes()},\n"
            f"  storage={self.total_storage()} elements\n"
            ")"
        )

    def __str__(self) -> str:
        """
        Возвращает строковое представление TT-тензора.

        Делегирует работу методу __repr__, обеспечивая единый формат
        отображения при вызове.
        """
        return self.__repr__()
