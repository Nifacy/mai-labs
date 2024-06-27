from itertools import chain, combinations
import random
import string
from typing import List, TextIO

SKIP_CHARS = '.,:?`!;‘()/[]"“…”’\'>-'


def normalize_text(text: str) -> str:
    """
    Нормализует текст путем удаления специальных символов и
    перевода в нижний регистр
    """
    return ''.join(char for char in text if char not in SKIP_CHARS).lower()


def read_text(file: TextIO) -> str:
    """
    Считывает и нормализует текст из файла
    """
    return ' '.join(
        chain.from_iterable(
            normalize_text(line).split()
            for line in file
        )
    )


def variance(elements: List[float]) -> float:
    """
    Считает дисперсию на основе списка значений случайной величины
    """
    mean = sum(elements) / len(elements)
    return sum((x - mean) ** 2 for x in elements) / len(elements)


def random_char_text(length: int) -> str:
    """
    Генерирует строку заданной длины из случайных букв
    """
    letters = string.ascii_lowercase + ' ' + string.digits
    return ''.join(random.choice(letters) for _ in range(length))


def random_words_text(length: int, words: List[str]) -> str:
    """
    Генерирует строку заданной длины из случайных слов
    """
    text = ' '.join(random.choice(words) for _ in range(length))
    return text[:length]


def accuracy(a: str, b: str) -> float:
    """
    Подсчет процента совпадения букв для двух текстов
    """
    matches = sum(x == y for x, y in zip(a, b))
    return matches / len(a)


def print_accuracy_results(accuracies: List[float], title: str) -> None:
    """
    Вывод информации о проценте совпадения
    """
    acc_mean = sum(accuracies) / len(accuracies)
    acc_variance = variance(accuracies)

    print(f'{title}:')
    print(f'accuracies: {", ".join(map(str, map(lambda x: round(x, 5), accuracies)))}')
    print(f'mean: {acc_mean:.5f}')
    print(f'variance: {acc_variance:.5f}')
    print()


# Считываем осмысленные текста из файлов
text_paths = [
    'texts/literature/heart-of-a-dog.txt',
    'texts/literature/fight-club.txt',
    'texts/literature/master-and-margarita.txt',
    'texts/literature/new-life.txt',
    'texts/literature/the-castle.txt',
    'texts/podcasts/1.txt',
    'texts/podcasts/2.txt',
    'texts/podcasts/3.txt',
    'texts/podcasts/4.txt',
    'texts/podcasts/5.txt',
]
human_texts = [read_text(open(path, 'r', encoding='utf-8')) for path in text_paths]

# Считываем английские слова
with open('words.txt', 'r', encoding='utf-8') as words_file:
    random_words = [line.strip() for line in words_file]

# 2 осмысленных текста
N = min(map(len, human_texts))
accuracies = [accuracy(a[:N], b[:N]) for a, b in combinations(human_texts, 2)]
print_accuracy_results(accuracies, '2 human texts')

# осмысленный текст и текст из случайных букв
accuracies = [accuracy(text, random_char_text(len(text))) for text in human_texts]
print_accuracy_results(accuracies, 'human and random char texts')

# осмысленный текст и текст из случайных слов
accuracies = [accuracy(text, random_words_text(len(text), random_words)) for text in human_texts]
print_accuracy_results(accuracies, 'human and random word texts')

# два текста из случайных букв
N = 100_000
random_char_texts = [random_char_text(N) for _ in range(8)]
accuracies = [accuracy(a, b) for a, b in combinations(random_char_texts, 2)]
print_accuracy_results(accuracies, '2 random char texts')

# два текста из случайных слов
random_word_texts = [random_words_text(N, random_words) for _ in range(8)]
accuracies = [accuracy(a, b) for a, b in combinations(random_word_texts, 2)]
print_accuracy_results(accuracies, '2 random word texts')
