import argparse
from typing import Iterable, Iterator
from curve import Curve, Point, iter_points
from get_divisors import find_divisors


def find_dot_orders(
    curve: Curve,
    points: Iterable[Point],
    divisors: list[int],
) -> Iterator[tuple[int, Point]]:
    for point in points:
        for d in divisors:
            if curve.mult(d, point) == Point(0, 0):
                yield d, point
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="find_dot",
        description="Находит порядок точки на эллиптической кривой.",
    )

    parser.add_argument("a", type=int, help="Коэффициент a уравнения эллиптической кривой.")
    parser.add_argument("b", type=int, help="Коэффициент b уравнения эллиптической кривой.")
    parser.add_argument("p", type=int, help="Модуль p уравнения эллиптической кривой.")
    parser.add_argument("n", type=int, help="Порядок кривой.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    a, b, p = args.a, args.b, args.p
    n = args.n

    curve = Curve(a, b, p)
    d = find_divisors(n)
    points = iter_points(curve)

    for order, point in find_dot_orders(curve, points, d):
        print(f"point: {point}, order: {order}")
