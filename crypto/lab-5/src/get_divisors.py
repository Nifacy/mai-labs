import argparse


def find_divisors(n: int) -> list[int]: 
    divisors_left = [] 
    divisors_right = [] 
 
    for k in range(2, int(n ** 0.5) + 1): 
        if n % k == 0: 
            divisors_left.append(k) 
            divisors_right.insert(0, n // k) 
 
    return divisors_left + divisors_right 


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="get_divisors",
        description="Находит делители заданного числа.",
    )
    parser.add_argument(
        "number",
        type=int,
        help="Число, для которого нужно найти делители.",
    )
    return parser.parse_args()

 
if __name__ == "__main__":
    args = parse_args()
    n = args.number 
    d = find_divisors(n) 
    print("divisors:", ", ".join(map(str, d)))
