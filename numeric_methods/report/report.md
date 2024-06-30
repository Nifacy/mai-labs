# Курсовая работа по дисциплине "Численные методы"

## Лабораторная работа №1

### О программе

Для выполнения данной лабораторной работы были написаны программы на языке программирования C++ для каждого из заданий. Для поддержки матричных операций был реализован класс `TMatrix`, объекты которого поддерживали операции сложения, матричного умножения матриц, чтение из файлов и форматированный вывод матрицы.

### Запуск программ

Для запуска программ необходимо скомпилировать исходный код путем ввода команды `g++ lab-1/task_{n}/main.cpp`, где `n` – номер задания. После чего запустить скомпилированный файл.

### Алгоритм LU-разложения матрицы

$LU$ разложение матрицы является методом решения задачи разложения матрицы $A$ в виде произведения нижней и верхней треугольных матриц $A = LU$, где $L$ - нижняя треугольная матрица, $U$ - верхняя треугольная матрица.

Одним из вариантов построения $LU$ разложения может быть использование метода Гаусса. Для понимания работы алгоритма рассмотрим $k$-й шаг метода Гаусса, на котором осуществляется обнуление поддиагональных элементов $k$-го столбца матрицы $A ^ {(k-1)}$. Элементы результирующей матрицы можно описать следующий образом:

$$
a_{ij} ^ {(k)} =
a_{ij} ^ {(k - 1)} - \mu_i ^ {(k)} \cdot a_{kj} ^ {(k - 1)}, \
\text{где} \
\mu_i ^ {(k)} = \frac{a_{ik} ^ {(k - 1)}}{a_{kk} ^ (k - 1)}; \
i = \overline{k + 1, n}; \
j = \overline{k, n}
$$

Если выразить данный процесс в виде матричных операций, то увидим, что данная запись эквивалентна записи $A ^ {(k)} = M_k \cdot A ^ {(k - 1)}$, где матрица $M$ определяется следующим образом:

$$
m_{ij} ^ k = \left\{ \begin{matrix}
    1, && i = j \\
    0, && i \ne j, j \ne k \\
    - \mu_{k + 1} ^ {(k)}, && i \ne j, j = k \\
\end{matrix} \right.
$$

Рассмотрим теперь, как можно выразить прямой ход метода Гаусса при помощи выше описанных матричных операций:

$$
A = A ^ {(0)} = M_{1} ^ {-1} \cdot A ^ {(1)} =
M_1 ^ {-1} \cdot M_2 ^ {-1} \cdot A ^ {(2)} =
M_1 ^ {-1} \cdot M_2 ^ {-1} \cdot \ldots \cdot M_{n - 1} ^ {-1} \cdot A ^ {(n - 1)}
$$

В результате прямого хода метода Гаусса получим $A ^ {(n-1)} = U$, где $U$ - верхняя треугольная матрица. Соответствующая матрице $U$ матрица $L$ может быть выражена следующий образом $L = M_1 ^ {-1} \cdot M_2 ^ {-1} \cdot \ldots \cdot M_{n - 1} ^ {-1}$. Если вычислить это выражение, то получим следующий вид матрицы

$$
L = \left( \begin{matrix}
    1 && 0 && 0 && 0 && 0 && 0 \\
    \mu_2 ^ {(1)} && 1 && 0 && 0 && 0 && 0 \\
    \mu_3 ^ {(1)} && \mu_3 ^ {(2)} && 1 && 0 && 0 && 0 \\
    \vdots && \ddots && \mu_{k + 1} ^ {(k)} && 1 && 0 && 0 \\
    \ldots && \ldots && \ldots && \ldots && \ldots && \ldots \\
    \mu_n ^ {(1)} && \mu_n ^ {(2)} && \mu_n ^ {(k + 1)} && \ldots && \mu_n ^ {(n - 1)} && 1\\ 
\end{matrix} \right)
$$

Полученное $LU$-разложение может быть эффективно использовано в задачах решения системы линейных алгебраических уравнений вида $A \cdot x = b$, в поиске детерминанта матрицы $\det ⁡A$, поиске обратной матрицы $A ^ {-1}$.

Рассмотрим более подробно процесс решения линейных алгебраических уравнений с помощью $LU$-разложения.

1. Первым шагом является разложение матрицы коэффициентов $A$ на произведение матриц $A = L \cdot U$. В итоге, исходное уравнение можно представить в виде $L \cdot U \cdot x = b$, или в виде системы

$$
\left\{ \begin{matrix}
    L \cdot z = b \\
    U \cdot x = z \\
\end{matrix} \right.
$$

2. На втором шаге решается уравнение $L \cdot z = b$. Поскольку матрица системы – нижняя треугольная, решение можно записать в явном виде: 

$$
y_1 = b_1, \
y_i = b_i - \sum_{j = 1} ^ {i - 1} l_{ij} \cdot y_j, \
i = \overline{2, n}
$$

3. И, в заключении, решается система $U \cdot x = z$ с верхней треугольной матрицей. Здесь, как и на предыдущем этапе, решение представляется в явном виде:

$$
x_1 = \frac{y_n}{u_{nn}}, \
x_i = \frac{1}{u_{ii}} \cdot \left( y_i - \sum_{j = i + 1} ^ {n} u_{ij} \cdot y_j \right), \
i = \overline{n - 1, 1}
$$

Также, задачу поиска определителя матрицы $A$ можно легко свести к задаче $LU$-разложения. Разложив матрицу в виде произведения $A = L \cdot U$ ее определитель можно найти по формуле:

$$
\det A = \prod_{i = 1} ^ {n} l_{ii} \cdot u_{ii}
$$

Задачу поиска обратной матрицы также можно свести к задаче $LU$-разложения, если воспользоваться свойством $A \cdot A ^ {-1} = L \cdot U \cdot A ^ {-1} = E$. Это уравнение также можно решить методом $LU$ разложения. 

### Входные данные

Ниже приведен пример входных данных для решения систему линейный алгебраических уравнений, соответствующий моему варианту

$$
\left\{ \begin{matrix}
    x_1 - 5 \cdot x_2 - 7 \cdot x_3 + x_4 = -75 \\
    x_1 - 3 \cdot x_2 - 9 \cdot x_3 - 4 \cdot x_4 = -41 \\
    -2 \cdot x_1 + 4 \cdot x_2 + 2 \cdot x_3 + x_4 = 18 \\
    -9 \cdot x_1 + 9 \cdot x_2 + 5 \cdot x_3 + 3 \cdot x_4 = 29 \\
\end{matrix} \right.
$$

### Реализация LU-разложения

Ниже предоставлен исходный код, содержащий реализацию алгоритма $LU$-разложения. Также, стоит отметить, что данная реализация имеет оптимизацию в виде введения дополнительной матрицы перестановок $P$, что применимо конкретно к выполнения алгоритма на ЭВМ.

```cpp
std::pair<int, int> ForwardStep(Matrix::TMatrix& m, int k, std::vector<float>& coef) {
    int n = m.GetSize().first;

    // find element with maximum square to avoid small dividers
    int swapIndex = k;

    for (int i = k + 1; i < n; ++i) {
        float a = m.Get(swapIndex, k);
        float b = m.Get(i, k);

        if (a * a > b * b) {
            swapIndex = i;
        }
    }

    m.SwapRaws(k, swapIndex);

    // try to find a string with a non-zero element
    // and swap it with the current
    if (m.Get(k, k) == 0.0) {
        return {k, k};
    }

    // convert raws below so that the elements are zero
    for (int i = k + 1; i < n; ++i) {
        float c = m.Get(i, k) / m.Get(k, k);

        for (int j = k; j < n; ++j) {
            m.Set(i, j, m.Get(i, j) - c * m.Get(k, j)); 
        }

        coef.push_back(c);
    }
    return {k, swapIndex};
}

void LUDecompose(
    const Matrix::TMatrix& a,
    Matrix::TMatrix& l,
    Matrix::TMatrix& u,
    Matrix::TMatrix& p
) {
    int n = a.GetSize().first;
    std::vector<float> coef;

    p = Matrix::TMatrix::Eye(n);
    l = Matrix::TMatrix::Eye(n);
    u = a;

    coef.reserve(n); // reserve max possible number of bytes to avoid allocations

    for (int k = 0; k < n - 1; ++k) {
        std::pair<int, int> swap = ForwardStep(u, k, coef);

        for (int i = 0; i < coef.size(); ++i) {
            l.Set(i + k + 1, k, coef[i]);
        }

        p.SwapRaws(swap.first, swap.second);

        coef.clear();
    }
}
```

### Реализация решения СЛАУ с помощью LU-разложения

Ниже представлен код, содеражщий реализацию алгоритма решения системы линейных алгебраических уравнений с помощью ранее описанного $LU$-разложения.

```cpp
void SolveWithL(const Matrix::TMatrix& l, const Matrix::TMatrix& b, Matrix::TMatrix& x) {
    int n = l.GetSize().first;

    for (int i = 0; i < n; ++i) {
        float c = b.Get(i, 0);
        for (int j = 0; j < i; ++j) {
            c -= x.Get(j, 0) * l.Get(i, j);
        }
        x.Set(i, 0, c);
    }
}

void SolveWithU(const Matrix::TMatrix& u, const Matrix::TMatrix& b, Matrix::TMatrix& x) {
    int n = u.GetSize().first;

    for (int i = n - 1; i >= 0; --i) {
        float c = b.Get(i, 0);

        for (int j = i + 1; j < n; ++j) {
            c -= x.Get(j, 0) * u.Get(i, j);
        }

        c /= u.Get(i, i);
        x.Set(i, 0, c);
    }
}

void SolveSystem(
    const Matrix::TMatrix& l,
    const Matrix::TMatrix& u,
    const Matrix::TMatrix& p,
    const Matrix::TMatrix& b,
    Matrix::TMatrix& x
) {
    Matrix::TMatrix z(b.GetSize().first, 1);
    SolveWithL(l, p * b, z);
    SolveWithU(u, z, x);
}
```

### Результат

Ниже предоставлен пример работы программы с входными данными, соответствующими моему варианту

![](images/1.png)
