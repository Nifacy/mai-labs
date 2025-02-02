import argparse
import time
from curve import Curve, Point, point_order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="find_order",
        description="Вычисляет порядок точки на эллиптической кривой.",
    )

    parser.add_argument("a", type=int, help="Коэффициент a уравнения эллиптической кривой.")
    parser.add_argument("b", type=int, help="Коэффициент b уравнения эллиптической кривой.")
    parser.add_argument("p", type=int, help="Модуль p уравнения эллиптической кривой.")
    parser.add_argument("x", type=int, help="Координата x точки на кривой.")
    parser.add_argument("y", type=int, help="Координата y точки на кривой.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    a, b, p = args.a, args.b, args.p
    x, y = args.x, args.y

    curve = Curve(a, b, p)
    point = Point(x, y)

    start = time.time()
    order = point_order(curve, point)
    end = time.time()

    print(f"point: {point}")
    print(f"order: {order}")
    print(f"time: {end - start}")
