# core/dense_tensor.py

"""Функции для работы с тензорами в стандартной плотной форме."""


from __future__ import annotations

import random
import math

from core.utils import (
    validate_shape,
    compute_size,
    compute_strides,
    multi_index_to_flat,
    check_shapes_match,
)


class DenseTensor:
    """
    Плотный тензор произвольного порядка.

    Атрибуты:
        shape:   кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        ndim:    порядок тензора (число мод)
        size:    общее число элементов
        data:    плоский список значений (row-major / C-order)
        strides: шаги для перевода мультииндекса в плоский индекс
    """

    __slots__ = ('shape', 'ndim', 'size', 'data', 'strides')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(
        self,
        shape: tuple[int, ...] | list[int],
        data: list[float] | None = None,
        fill: float = 0.0
    ) -> None:
        """
        Создаёт тензор заданной формы.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
            data:  плоский список значений (если None — заполняется fill)
            fill:  значение для заполнения (по умолчанию 0.0)
        """
        self.shape = validate_shape(shape)
        self.ndim = len(self.shape)
        self.size = compute_size(self.shape)
        self.strides = compute_strides(self.shape)

        if data is None:
            self.data = [float(fill) for _ in range(self.size)]
        else:
            if not isinstance(data, list):
                raise TypeError("data должен быть списком")
            if len(data) != self.size:
                raise ValueError(
                    f"длина data ({len(data)}) не совпадает с размером тензора ({self.size})"
                )
            self.data = [float(value) for value in data]

    @staticmethod
    def zeros(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает тензор, заполненный нулями.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        """
        return DenseTensor(shape, fill=0.0)

    @staticmethod
    def ones(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает тензор, заполненный единицами.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        """
        return DenseTensor(shape, fill=1.0)

    @staticmethod
    def random(
        shape: tuple[int, ...] | list[int],
        low: int = -5,
        high: int = 5,
        integer: bool = True,
        seed: int | None = None
    ) -> DenseTensor:
        """
        Возвращает тензор со случайными значениями.

        Args:
            shape:   кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
            low:     нижняя граница значений тензора
            high:    верхняя граница значений тензора
            integer: True — целые числа, False — вещественные
            seed:    seed для воспроизводимости (None — без фиксации)

        NB: эта функция не тестируется, ее можно использовать для отладки
        """
        shp: tuple[int, ...] = validate_shape(shape)
        size: int = compute_size(shp)

        rng = random.Random(seed)
        if integer:
            data = [float(rng.randint(low, high)) for _ in range(size)]
        else:
            data = [rng.uniform(low, high) for _ in range(size)]

        return DenseTensor(shp, data=data)

    @staticmethod
    def from_nested_list(nested: list) -> DenseTensor:
        """
        Создаёт тензор из вложенного списка Python.
        Автоматически определяет shape.

        Args:
            nested: список
        """
        if not isinstance(nested, (list, tuple)):
            raise TypeError("nested должен быть списком или кортежем")

        shape: list[int] = []
        probe = nested
        while isinstance(probe, (list, tuple)):
            if len(probe) == 0:
                raise ValueError("пустые вложенные списки не поддерживаются")
            shape.append(len(probe))
            probe = probe[0]

        if len(shape) == 0:
            raise ValueError("ожидается хотя бы одномерный список")

        def _check_rectangular(node, depth: int) -> None:
            if depth == len(shape):
                if isinstance(node, bool) or not isinstance(node, (int, float)):
                    raise TypeError("элементы nested должны быть числами")
                return

            if not isinstance(node, (list, tuple)) or len(node) != shape[depth]:
                raise ValueError("вложенный список должен быть прямоугольным")
            for child in node:
                _check_rectangular(child, depth + 1)

        _check_rectangular(nested, 0)

        flat_data: list[float] = []
        stack = [nested]
        while stack:
            node = stack.pop()
            if isinstance(node, (list, tuple)):
                for child in reversed(node):
                    stack.append(child)
            else:
                flat_data.append(float(node))

        return DenseTensor(tuple(shape), data=flat_data)

    # ────────────────────────────────────────────
    # Индексация
    # ────────────────────────────────────────────

    def _validate_index(
        self,
        multi_index: tuple[int, ...] | int
    ) -> tuple[int, ...]:
        """
        Возвращает нормализованный мультииндекс в виде кортежа.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
        """
        if isinstance(multi_index, int) and not isinstance(multi_index, bool):
            if self.ndim != 1:
                raise IndexError(
                    f"скалярная индексация допустима только для 1D, ndim={self.ndim}"
                )
            raw_index = (multi_index,)
        elif isinstance(multi_index, tuple):
            raw_index = multi_index
        else:
            raise TypeError("индекс должен быть int или tuple[int, ...]")

        if len(raw_index) != self.ndim:
            raise IndexError(
                f"ожидается {self.ndim} индексов, получено {len(raw_index)}"
            )

        validated: list[int] = []
        for axis, (idx, upper) in enumerate(zip(raw_index, self.shape)):
            if isinstance(idx, bool) or not isinstance(idx, int):
                raise TypeError("каждый индекс должен быть целым числом")
            if idx < 0 or idx >= upper:
                raise IndexError(
                    f"индекс {idx} вне диапазона [0, {upper}) "
                    f"для оси {axis}"
                )
            validated.append(idx)

        return tuple(validated)

    def __getitem__(self, multi_index: tuple[int, ...] | int) -> float:
        """
        Возвращает значение элемента по заданному мультииндексу.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
        """
        mi = self._validate_index(multi_index)
        flat = multi_index_to_flat(mi, self.strides)
        return self.data[flat]

    def __setitem__(
        self,
        multi_index: tuple[int, ...] | int,
        value: float
    ) -> None:
        """
        Устанавливает новое значение элемента по заданному мультииндексу.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
            value:       новое значение (число)
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("value должен быть числом")

        mi = self._validate_index(multi_index)
        flat = multi_index_to_flat(mi, self.strides)
        self.data[flat] = float(value)

    # ────────────────────────────────────────────
    # Преобразования формы
    # ────────────────────────────────────────────

    def reshape(self, new_shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает новый объект тензора с новой формой и скопированными данными.

        Args:
            new_shape: кортеж новых размеров (n'_0, n'_1, ..., n'_{k-1})
        """
        shp = validate_shape(new_shape)
        if compute_size(shp) != self.size:
            raise ValueError(
                f"нельзя изменить форму {self.shape} на {shp}: "
                "разное число элементов"
            )
        return DenseTensor(shp, data=self.data.copy())

    def unfolding(self, mode: int) -> DenseTensor:
        """
        Возвращает матрицу — развертку тензора по моде n.

        Args:
            mode: номер моды (0 ≤ mode < ndim), которая становится индексом строк
        """
        if isinstance(mode, bool) or not isinstance(mode, int):
            raise TypeError("mode должен быть целым числом")
        if mode < 0 or mode >= self.ndim:
            raise ValueError(f"mode должен быть в диапазоне [0, {self.ndim})")

        rows = self.shape[mode]
        cols = self.size // rows
        out = DenseTensor.zeros((rows, cols))
        for row in range(rows):
            for col in range(cols):
                rem = col
                idx = [0] * self.ndim
                for axis in range(self.ndim - 1, -1, -1):
                    if axis == mode:
                        continue
                    dim = self.shape[axis]
                    idx[axis] = rem % dim
                    rem //= dim
                idx[mode] = row
                out[row, col] = self[tuple(idx)]

        return out

    def left_unfolding(self, k: int) -> DenseTensor:
        """
        Возвращает матрицу — "левую развертку" тензора для TT-SVD.

        Args:
            k: номер границы разбиения (0 ≤ k < ndim - 1)
        """
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k должен быть целым числом")
        if self.ndim < 2:
            raise ValueError("left_unfolding определена только для ndim >= 2")
        if k < 0 or k >= self.ndim - 1:
            raise ValueError(f"k должен быть в диапазоне [0, {self.ndim - 1})")

        left_shape = self.shape[:k + 1]
        right_shape = self.shape[k + 1:]
        rows = compute_size(left_shape)
        cols = compute_size(right_shape)

        return self.reshape((rows, cols))

    # ────────────────────────────────────────────
    # Копирование
    # ────────────────────────────────────────────

    def copy(self) -> DenseTensor:
        """Возвращает глубокую копию тензора."""
        return DenseTensor(self.shape, data=self.data.copy())

    # ────────────────────────────────────────────
    # Арифметика
    # ────────────────────────────────────────────

    def norm(self) -> float:
        """Возвращает Фробениусову норму тензора."""
        return math.sqrt(sum(x * x for x in self.data))

    def __add__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного сложения: t1 + t2.

        Args:
            other: t2
        """
        other_shape = getattr(other, "shape", None)
        other_data = getattr(other, "data", None)
        if other_shape is None or other_data is None:
            return NotImplemented
        check_shapes_match(self.shape, tuple(other_shape))
        if len(other_data) != self.size:
            raise ValueError("некорректная длина data у второго тензора")
        data = [0.0] * self.size
        for i in range(self.size):
            data[i] = self.data[i] + float(other_data[i])
        return DenseTensor(self.shape, data=data)

    def __sub__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного вычитания: t1 - t2.

        Args:
            other: t2
        """
        other_shape = getattr(other, "shape", None)
        other_data = getattr(other, "data", None)
        if other_shape is None or other_data is None:
            return NotImplemented
        check_shapes_match(self.shape, tuple(other_shape))
        if len(other_data) != self.size:
            raise ValueError("некорректная длина data у второго тензора")
        data = [0.0] * self.size
        for i in range(self.size):
            data[i] = self.data[i] - float(other_data[i])
        return DenseTensor(self.shape, data=data)

    def __mul__(self, scalar: float | int) -> DenseTensor:
        """
        Возвращает тензор — результат умножения тензора на скаляр: t1 * scalar.

        Args:
            scalar: число
        """
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
            return NotImplemented
        alpha = float(scalar)
        return DenseTensor(self.shape, data=[alpha * value for value in self.data])

    def __rmul__(self, scalar: float | int) -> DenseTensor:
        """
        Возвращает тензор — результат умножения тензора на скаляр: scalar * t1.

        Args:
            scalar: число, на которое умножаем
        """
        return self.__mul__(scalar)

    def __neg__(self) -> DenseTensor:
        """Возвращает тензор — результат умножения тензора на -1."""
        return DenseTensor(self.shape, data=[-value for value in self.data])

    # ────────────────────────────────────────────
    # Сравнение и отладка
    # ────────────────────────────────────────────

    def allclose(
        self,
        other: DenseTensor,
        atol: float = 1e-8,
        rtol: float = 1e-5
    ) -> bool:
        """
        Возвращает True, если тензоры равны с заданной точностью.

        Условие равенства: shape равны и для каждой пары элементов
        тензоров с равными индексами выполняется:
            |a - b| <= atol + rtol * max(|a|, |b|)


        Args:
            other: DenseTensor для сравнения
            atol:  абсолютная погрешность (по умолчанию 1e-8)
            rtol:  относительная погрешность (по умолчанию 1e-5)
        """
        other_shape = getattr(other, "shape", None)
        other_data = getattr(other, "data", None)
        if other_shape is None or other_data is None:
            return False
        if self.shape != tuple(other_shape):
            return False
        if len(other_data) != self.size:
            return False

        for i in range(self.size):
            a = self.data[i]
            b = float(other_data[i])
            if abs(a - b) > atol + rtol * max(abs(a), abs(b)):
                return False
        return True

    def to_nested_list(self) -> list:
        """Возвращает тензор в формате вложенного списка."""
        def _build(shape: tuple[int, ...], flat: list[float]) -> list:
            if len(shape) == 1:
                return flat[:shape[0]]

            chunk = compute_size(shape[1:])
            result: list = []
            for i in range(shape[0]):
                start = i * chunk
                result.append(_build(shape[1:], flat[start:start + chunk]))
            return result

        return _build(self.shape, self.data)

    def __repr__(self) -> str:
        """
        Возвращает строковое представление тензора для отладки.

        NB: эта функция не проверяется тестами, ее реализация может быть произвольной
        """
        preview_count = min(8, self.size)
        preview = ", ".join(f"{x:.6g}" for x in self.data[:preview_count])
        if self.size > preview_count:
            preview += ", ..."
        return (
            f"DenseTensor(shape={self.shape}, ndim={self.ndim}, size={self.size}, "
            f"data=[{preview}])"
        )

    def __str__(self) -> str:
        """Возвращает строковое представление тензора для отладки."""
        return self.__repr__()
