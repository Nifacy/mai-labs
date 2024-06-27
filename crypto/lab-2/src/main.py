from typing import Iterable, Literal
from math import gcd
import os.path
import json


CURDIR = os.path.abspath(os.path.join(__file__, os.path.pardir))


TaskVariant = dict[Literal["n1", "n2"], int]


def load_task() -> dict[str, TaskVariant]:
    with open(os.path.join(CURDIR, "task.json")) as task_file:
        return json.load(task_file)


def get_numbers(task: dict[str, TaskVariant]) -> list[int]:
    return [variant["n1"] for variant in task.values()]


def find_first_delimeter(target_number: int, numbers: Iterable[int]) -> int:
    for number in numbers:
        if (d := gcd(number, target_number)) != 1:
            return d
    raise ValueError(f"Can't find delimeter for {target_number}")


def main():
    task = load_task()
    target_task = int(input("Номер варианта: "))
    target_number = task[str(target_task)]["n2"]
    numbers = [
        variant["n2"] for task_number, variant in task.items()
        if variant["n2"] != target_number
    ]

    # поиск делителей
    a = find_first_delimeter(target_number, numbers)
    b = target_number // a

    # вывод результата в консоль для пользователя
    print(f"n: {target_number}")
    print(f"a: {a}")
    print(f"b: {b}")

    # проверка полученного результата
    print("Check:")
    print(f"a * b = {a * b}")
    print("Result: {}".format(["FAILED", "SUCCESS"][a * b == target_number]))


if __name__ == "__main__":
    main()
