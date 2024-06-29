from common import random_bytes_string, invert_bit, diff
import json
from skein import Threefish512, bytes2words, words2bytes

DATASET_SIZE = 100
ROUNDS = range(1, 73)

def encrypt(msg: bytes, rounds: int) -> bytes:
    th = Threefish512(rounds=rounds)
    th.prepare_key()
    th.prepare_tweak()
    return words2bytes(th.encrypt_block(bytes2words(msg)))
 

def main():
    data = []

    for rounds in ROUNDS:
        result = []

        for t in range(DATASET_SIZE):
            print(f'\rrounds: {rounds}, iteration: {t + 1} / {DATASET_SIZE}', end='')
            a = random_bytes_string(64)

            for i in range(0, len(a) * 8, 4):
                b = invert_bit(a, i)
                hash1 = encrypt(a, rounds)
                hash2 = encrypt(b, rounds)
                result.append(diff(hash1, hash2))

        mean = sum(result) / len(result)
        print(f" mean: {mean}")
        data.append({"rounds": rounds, "bits_changed": mean})

    with open("threefish.json", "w") as file:
        file.write(json.dumps(data, indent=4))

if __name__ == "__main__":
    main()
