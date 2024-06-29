import argparse
import time
from curve import Curve, Point, iter_points, point_order


def calc_order_time(curve: Curve, point: Point) -> tuple[int, float]:
    start = time.time()
    order = point_order(curve, point)
    end = time.time()
    return order, end - start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="calc_speed",
        description="Утилита для вычисления скорости вычисления порядка точки на эллиптической кривой."
    )
    parser.add_argument("a", type=int, help="Коэффициент a уравнения эллиптической кривой.")
    parser.add_argument("b", type=int, help="Коэффициент b уравнения эллиптической кривой.")
    parser.add_argument("p", type=int, help="Модуль p уравнения эллиптической кривой.")
    parser.add_argument("expected", type=float, help="Ожидаемое время выполнения (секунды).")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    a, b, p = args.a, args.b, args.p
    expected = args.expected

    curve = Curve(a, b, p)

    results = []
    max_time = 0.0

    for point in iter_points(curve):
        order, elapsed = calc_order_time(curve, point)

        if elapsed > max_time:
            max_time = elapsed
            print(f"point: {point}, order: {order}, time: {elapsed}")

        results.append((order, elapsed))

    coefs = [order / elapsed for order, elapsed in results if elapsed]
    max_coef = int(max(coefs))

    print("iterations per second:", max_coef)
    print("for expected execution time you need order:", int(expected * max_coef))
