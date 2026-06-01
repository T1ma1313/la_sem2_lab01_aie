# core/utils.py

"""Вспомогательные функции для работы с тензорами."""


def validate_shape(
    shape: tuple[int, ...] | list[int]
) -> tuple[int, ...]:
    """
    Проверяет корректность формы тензора и приводит её к стандартному виду.

    Убеждается, что shape является последовательностью положительных целых
    чисел. Преобразует список в кортеж для единообразия.

    Args:
        shape: кортеж или список размеров тензора по каждой моде

    Returns:
        tuple: проверенный кортеж положительных целых чисел

    Raises:
        TypeError:  если shape не является tuple или list
        ValueError: если хотя бы один элемент shape не является
                    положительным целым числом
    """
    if not isinstance(shape, (tuple, list)):
        raise TypeError("shape должен быть tuple или list")

    normalized = tuple(shape)
    if len(normalized) == 0:
        raise ValueError("shape не должен быть пустым")

    checked: list[int] = []
    for dim in normalized:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise ValueError("размерности shape должны быть целыми числами")
        if dim <= 0:
            raise ValueError("размерности shape должны быть положительными")
        checked.append(dim)
    return tuple(checked)


def compute_size(shape: tuple[int, ...]) -> int:
    """
    Возвращает общее число элементов тензора заданной формы.

    Args:
        shape: кортеж размеров тензора (n_0, n_1, ..., n_{d-1})
    """
    size = 1
    for dim in shape:
        size *= dim
    return size


def compute_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """
    Возвращает кортеж strides, содержащий для каждой моды k свой strides[k].

    Stride по моде k — это число элементов в плоском списке, на которое
    нужно сдвинуться, чтобы перейти к следующему элементу вдоль моды k.

    Args:
        shape: кортеж размеров тензора (n_0, n_1, ..., n_{d-1})
    """
    step = 1
    reversed_strides: list[int] = []
    for dim in reversed(shape):
        reversed_strides.append(step)
        step *= dim

    return tuple(reversed(reversed_strides))


def multi_index_to_flat(
    multi_index: tuple[int, ...],
    strides: tuple[int, ...]
) -> int:
    """
    Возвращает позицию элемента в плоском списке данных по его
    многомерным координатам и заранее вычисленным strides.

    Args:
        multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1})
        strides:     кортеж шагов   (s_0, s_1, ..., s_{d-1})
    """
    if len(multi_index) != len(strides):
        raise ValueError("длины multi_index и strides должны совпадать")

    flat = 0
    for idx, stride in zip(multi_index, strides):
        flat += idx * stride
    return flat


def flat_to_multi_index(
    flat_index: int,
    shape: tuple[int, ...]
) -> tuple[int, ...]:
    """
    Возвращает мультииндекс на основе плоского индекса.

    Args:
        flat_index: плоский индекс в списке данных
        shape:      кортеж размеров тензора (n_0, n_1, ..., n_{d-1})
    """
    if isinstance(flat_index, bool) or not isinstance(flat_index, int):
        raise TypeError("flat_index должен быть целым числом")

    total = compute_size(shape)
    if flat_index < 0 or flat_index >= total:
        raise IndexError(
            f"flat_index={flat_index} вне диапазона [0, {total})"
        )

    strides = compute_strides(shape)
    out: list[int] = []
    rest = flat_index

    for stride in strides:
        value = rest // stride
        out.append(value)
        rest -= value * stride

    return tuple(out)


def check_shapes_match(
    shape1: tuple[int, ...],
    shape2: tuple[int, ...]
) -> None:
    """
    Проверяет совпадение форм двух тензоров.

    Используется перед поэлементными операциями (сложение, вычитание),
    чтобы гарантировать совместимость тензоров.

    Args:
        shape1: кортеж размеров первого тензора
        shape2: кортеж размеров второго тензора

    Raises:
        ValueError: если формы не совпадают
    """
    if shape1 != shape2:
        raise ValueError(f"формы тензоров не совпадают: {shape1} != {shape2}")
