from pygost import gost34112012256


def get_last_bits(b: bytes) -> int:
    return b[-1] & 0b1111


name = input("ФИО: ")
hex_table = '0123456789ABCDEF'
encoded = gost34112012256.new(name.encode()).digest()
variant = get_last_bits(encoded)

print("Хеш:", encoded)
print("Ваш вариант:", hex_table[variant])
