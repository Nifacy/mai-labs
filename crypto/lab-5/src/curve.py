from dataclasses import dataclass
from typing import Iterator
import random


@dataclass(slots=True, frozen=True)
class Point:
    x: int
    y: int


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    stack = []

    while a:
        stack.append((a, b))
        a, b = b % a, a

    gcd, x, y = b, 0, 1
    while stack:
        a, b = stack.pop()
        x, y = y - (b // a) * x, x

    return gcd, x, y


def mod_inverse(a, p):
    gcd, x, _ = extended_gcd(a, p)

    if gcd != 1:
        raise Exception("Can't find inverse element")

    else:
        return x % p


class Curve:
    def __init__(self, a: int, b: int, p: int) -> None:
        self._a = a
        self._b = b
        self._p = p

    def _check_curve_coefs(self) -> None:
        a, b, p = self._a, self._b, self._p

        if (4 * a ** 3 + 27 * b ** 2) % p == 0:
            raise ValueError("Invalid curve coefficients")

    def contains(self, point: Point) -> bool:
        x, y = point.x, point.y
        a, b, p = self._a, self._b, self._p

        return (y * y) % p == (x * x * x + a * x + b) % p

    def add(self, p: Point, q: Point) -> Point:
        px, py, qx, qy = p.x, p.y, q.x, q.y

        if p == Point(0, 0):
            return q

        if q == Point(0, 0):
            return p

        if py == (-qy) % self._p:
            return Point(0, 0)

        if p == q:
            coef = mod_inverse(2 * py % self._p, self._p)
            m = ((3 * px ** 2 + self._a) * coef) % self._p
        else:
            coef = mod_inverse((px - qx) % self._p, self._p)
            m = ((py - qy) * coef) % self._p

        rx = (m ** 2 - px - qx) % self._p
        ry = (qy + m * (rx - qx)) % self._p

        return Point(rx, self._p - ry)

    def mult(self, n: int, p: Point) -> Point:
        result = Point(0, 0)

        while n:
            if n & 1:
                result = self.add(result, p)
            p = self.add(p, p)
            n >>= 1

        return result

    def get_order(self) -> int:
        N = 1

        for x in range(0, self._p):
            for y in range(0, self._p):
                left = (y * y) % self._p
                right = (x * x * x + self._a * x + self._b) % self._p
                N += (left == right)

        return N


def iter_points(curve: Curve) -> Iterator[Point]:
    p = curve._p

    for x in range(0, p):
        for y in range(0, p):
            point = Point(x, y)

            if curve.contains(point):
                yield point


def get_random_point(curve: Curve) -> Point:
    index = random.randint(128, 256)
    for number, point in enumerate(iter_points(curve), 1):
        if number == index:
            return point
    return point


def point_order(curve: Curve, point: Point) -> int:
    order = 1
    zero_point = Point(0, 0)
    curr = point

    while curr != zero_point:
        order += 1
        curr = curve.add(curr, point)

    return order
