from itertools import permutations
import numpy as np


A = np.array([
    [3, 9, 4, -2],
    [7, -5, 1, 7],
    [1, 2, 2, -2],
    [4, 9, 8, 1],
    [9, -2, -5, -2]
])

p = np.array([0.29, 0.11, 0.33])

# упрощение матрицы

n = len(A)
strategies = list(range(n))

while True:
    for i, j in permutations(strategies, 2):
        a, b = A[i], A[j]
        if (a >= b).all():
            print(f'A[{i + 1}] >= A[{j + 1}]')
            strategies.remove(j)
            break
    else:
        break

print('Итоговая матрица')
for i in strategies:
    print(f'A[{i + 1}] = {A[i]}')
print()

A = A[strategies]

print('== Восстановление П4 ==')
p4 = 1.0 - p.sum()
p = np.append(p, p4)
print(f'П4 = {p4}, П = {p}')
print()

print('== Оптимальная стратегия (мат ожидание) ==')
means = np.array([
    np.matmul(a, p.T)
    for a in A
])
best_strategy = np.argmax(means)

for i, a, m in zip(strategies, A, means):
    print(f'A[{i + 1}] : {a} : {m}')
print(f'Лучшая стратегия: A[{strategies[best_strategy] + 1}]')
print()

print('== Оптимальная стратегия (оптимистичный критерий) ==')
maxs = np.max(A, axis=-1)
best_strategy = np.argmax(maxs)

for i, a, m in zip(strategies, A, maxs):
    print(f'A[{i + 1}] : {a} : {m}')
print(f'Лучшая стратегия: A[{strategies[best_strategy] + 1}]')
print()

print('== Оптимальная стратегия (пессимистичный критерий) ==')
mins = np.min(A, axis=-1)
best_strategy = np.argmax(mins)

for i, a, m in zip(strategies, A, mins):
    print(f'A[{i + 1}] : {a} : {m}')
print(f'Лучшая стратегия: A[{strategies[best_strategy] + 1}]')
print()

alpha = 0.3
print(f'== Оптимальная стратегия (критерий Гурвица {alpha}) ==')
k = alpha * maxs + (1.0 - alpha) * mins
best_strategy = np.argmax(k)

for i, a, mn, mx, k_el in zip(strategies, A, mins, maxs, k):
    print(f'A[{i + 1}] : {a} : {mn} : {mx} : {round(k_el, 4)}')
print(f'Лучшая стратегия: A[{strategies[best_strategy] + 1}]')
print()

print('== Оптимальная стратегия (пессимистичный критерий) ==')
mins = np.min(A, axis=-1)
best_strategy = np.argmax(mins)

for i, a, m in zip(strategies, A, mins):
    print(f'A[{i + 1}] : {a} : {m}')
print(f'Лучшая стратегия: A[{strategies[best_strategy] + 1}]')
print()

print(f'== Оптимальная стратегия (критерий Сэвиджа) ==')
max_cols = np.max(A, axis=0)
n, m = A.shape
R = np.array([
    [max_cols[j] - A[i][j] for j in range(m)]
    for i in range(n)
])
r_maxs = R.max(axis=-1)
best_strategy = np.argmin(r_maxs)

for i, r, m in zip(strategies, R, r_maxs):
    print(f'A[{i + 1}] : {r} : {m}')
print(f'Лучшая стратегия: A[{strategies[best_strategy] + 1}]')
print()
