from random import randint

def random_bytes_string(n: int) -> bytes:
    return bytes([randint(0, 255) for _ in range(n)])

def invert_bit(a: bytes, i: int) -> bytes:
    b = list(a)
    b[i // 8] ^= 1 << (i % 8)
    return bytes(b)

def diff(a: bytes, b: bytes) -> int:
    assert len(a) == len(b)
    diff_bits = 0

    for x, y in zip(a, b):
        d = x ^ y
        while d:
            diff_bits += d & 1
            d >>= 1
    
    return diff_bits

def to_bits(a: bytes) -> str:
    return "".join(bin(x)[2:].zfill(8) for x in a)
