# Курсовая работа по дисциплине "Численные методы"

## Лабораторная работа №1

### О программе

Для выполнения данной лабораторной работы были написаны программы на языке программирования C++ для каждого из заданий. Для поддержки матричных операций был реализован класс `TMatrix`, объекты которого поддерживали операции сложения, матричного умножения матриц, чтение из файлов и форматированный вывод матрицы.

### Запуск программ

Для запуска программ необходимо скомпилировать исходный код путем ввода команды `g++ lab-1/task_{n}/main.cpp`, где `n` - номер задания. После чего запустить скомпилированный файл.

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

2. На втором шаге решается уравнение $L \cdot z = b$. Поскольку матрица системы - нижняя треугольная, решение можно записать в явном виде: 

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

#### Входные данные

Ниже приведен пример входных данных для решения систему линейный алгебраических уравнений, соответствующий моему варианту

$$
\left\{ \begin{matrix}
    x_1 - 5 \cdot x_2 - 7 \cdot x_3 + x_4 = -75 \\
    x_1 - 3 \cdot x_2 - 9 \cdot x_3 - 4 \cdot x_4 = -41 \\
    -2 \cdot x_1 + 4 \cdot x_2 + 2 \cdot x_3 + x_4 = 18 \\
    -9 \cdot x_1 + 9 \cdot x_2 + 5 \cdot x_3 + 3 \cdot x_4 = 29 \\
\end{matrix} \right.
$$

#### Реализация LU-разложения

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

#### Реализация решения СЛАУ с помощью LU-разложения

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

#### Результат

Ниже предоставлен пример работы программы с входными данными, соответствующими моему варианту

![](images/1.png)

### Метод прогонки

Для случаев, когда в СЛАУ матрица коэффициентов имеет трехдиагональный вид:

$$
\left\{ \begin{matrix}
    b_1 \cdot x_1 + c_1 \cdot x_2 = d_1 \\
    a_2 \cdot x_1 + b_2 \cdot x_2 + c_2 \cdot x_3 = d_2 \\
    a_3 \cdot x_2 + b_3 \cdot x_3 + c_3 \cdot x_4 = d_3 \\
    \ldots \\
    a_{n - 1} \cdot x_{n - 2} + b_{n - 1} \cdot x_{n - 1} + c_{n - 1} \cdot x_n = d_{n - 1} \\
    a_n \cdot x_{n - 1} + b_n \cdot x_n = d_n \\
\end{matrix} \right.
$$

может быть применен более оптимизированный вид метода Гаусса под названием метод прогонки. При использовании метода прогонки, решение системы происходит итеративно следующим образом

$$
x_i = P_i \cdot x_{i + 1} + Q_i, \
i = \overline{1, n}
$$

где $P_i$, $Q_i$ - прогоночные коэффициенты, определяемые по формулам

$$
\begin{matrix}
P_i = \left\{ \begin{matrix}
    - \frac{c_1}{b_1}, && i = 1 \\
    \frac{-c_i}{b_i + a_i \cdot P_{i - 1}}, && \text{иначе} \\ 
\end{matrix} \right.

&&

Q_i = \left\{ \begin{matrix}
    \frac{d_1}{b_1}, && i = 1 \\
    \frac{d_i - a_i \cdot Q_{i - 1}}{b_i + a_i \cdot P_{i - 1}}, && \text{иначе} \\
\end{matrix} \right.
\end{matrix}
$$

После того как будут найдены все прогоночные коэффициенты, что соответствует этапу прямого хода метода прогонк, можно вычислить значения неизвестных путем обратной подстановки (обратный ход):

$$
\begin{matrix}
    x_n = P_n \cdot x_{n + 1} + Q_n = 0 \cdot x_{n + 1} + Q_n = Q_n \\
    x_{n - 1} = P_{n - 1} \cdot x_n + Q_{n - 1} \\
    x_{n - 2} = P_{n - 2} \cdot x_{n - 1} + Q_{n - 2} \\
    \ldots \\
    x_1 = P_1 \cdot x_2 + Q_1 \\
\end{matrix}
$$

Для того, чтобы удостовериться, что данные подходят для решения методом прогонки, можно воспользоваться достаточным условием корректности и устойчивости к погрешностям вычислений является условие преобладания диагональных коэффициентов:

$$
\left| b_i \right| \ge \left| a_i \right| + \left| c_i \right|
$$

#### Входные данные

Ниже приведен пример входных данных для решения систему линейный алгебраических уравнений, соответствующий моему варианту

$$
\left\{ \begin{matrix}
    15 \cdot x_1 + 8 \cdot x_2 = 92 \\
    2 \cdot x_1 - 15 \cdot x_2 + 4 \cdot x_3 = -84 \\
    4 \cdot x_2 + 11 \cdot x_3 + 5 \cdot x_4 = -77 \\
    -3 \cdot x_3 + 16 \cdot x_4 + (-7) \cdot x_5 = 15 \\
    3 \cdot x_4 + 8 \cdot x_5 = -11 \\
\end{matrix} \right.
$$

#### Реализация решения СЛАУ с помощью метода прогонки

Ниже предоставлен пример кода, содержащего реализацию метода прогонки и решения СЛАУ с трехдиагональной матрицей с помощью него

```cpp
void CalculateRunCoefficients(const Matrix::TMatrix& A, const Matrix::TMatrix& B, Matrix::TMatrix& result) {
    int n = A.GetSize().first;

    for (int i = 0; i < n; ++i) {
        float a = A.Get(i, 0), b = A.Get(i, 1), c = A.Get(i, 2);
        float d = B.Get(i, 0);

        if (i == 0) {
            if (B.Get(i, 0) == 0.0) {
                throw std::runtime_error("Can't find solution of system");
            }

            result.Set(i, 0, - c / b);
            result.Set(i, 1, d / b);
        }

        else {
            float PLast = result.Get(i - 1, 0), QLast = result.Get(i - 1, 1);
            float t = b + a * PLast;

            if (t == 0.0) {
                throw std::runtime_error("Can't find solution of system");
            }

            result.Set(i, 0, - c / t);
            result.Set(i, 1, (d - a * QLast) / t);
        }
    }
}

void SolveUsingRunCoefficients(const Matrix::TMatrix& runCoefs, Matrix::TMatrix& result) {
    int n = runCoefs.GetSize().first;

    result.Set(n - 1, 0, runCoefs.Get(n - 1, 1));
    for (int i = n - 2; i >= 0; --i) {
        result.Set(i, 0, runCoefs.Get(i, 0) * result.Get(i + 1, 0) + runCoefs.Get(i, 1));
    }
}
```

#### Результат

Ниже предоставлен пример работы программы с входными данными, соответствующими моему варианту

![](images/2.png)

### Итерационные методы решения СЛАУ

Итерационные методы позволяют решать СЛАУ поэтапно. На каждом этапе решение рассматриваемой системы становится все более точным. Для дальнейшего рассмотрения итерационных методов, рассмотрим СЛАУ

$$
\left\{ \begin{matrix}
    a_{11} \cdot x_1 + a_{12} \cdot x_2 + \ldots + a_{1n} \cdot x_n = b_1 \\
    a_{21} \cdot x_1 + a_{22} \cdot x_2 + \ldots + a_{2n} \cdot x_n = b_1 \\
    \ldots \\
    a_{2n} \cdot x_1 + a_{n2} \cdot x_2 + \ldots + a_{nn} \cdot x_n = b_n \\
\end{matrix} \right.
$$

где матрица коэффициентов является невырожденной.

Приведем СЛАУ к следующему виду путем эквивалентных преобразований

$$
\left\{ \begin{matrix}
    x_1 = \beta_1 + \alpha_{11} \cdot x_1 + \alpha_{12} \cdot x_2 + \ldots + \alpha_{1n} \cdot x_n \\
    x_2 = \beta_1 + \alpha_{21} \cdot x_1 + \alpha_{22} \cdot x_2 + \ldots + \alpha_{2n} \cdot x_n \\
    \ldots \\
    x_2 = \beta_1 + \alpha_{21} \cdot x_1 + \alpha_{22} \cdot x_2 + \ldots + \alpha_{2n} \cdot x_n \\
\end{matrix} \right.
$$

или в векторно-матричной форме $x = \beta + \alpha \cdot x$. Мы получили общий вид решения системы уравнений с помощью итерационных методов

$$
\left\{ \begin{matrix}
    x ^ {(0)} = \beta \\
    x ^ {(1)} = \beta + \alpha \cdot x ^ {(0)} \\
    x ^ {(2)} = \beta + \alpha \cdot x ^ {(1)} \\
    \ldots \\
    x ^ {(k)} = \beta + \alpha \cdot x ^ {(k - 1)} \\
\end{matrix} \right.
$$

Однако, при таком определении мы сталкиваемся с проблемой бесконечного поиска решения СЛАУ. В этом случае применяют критерий окончания итерационного процесса. Например, им может послужить неравенство

$$
\left| x ^ {(k)} - x ^ {(k + 1)} \right| \le \varepsilon
$$

В итоге, задача сводится к определению элементов $\beta$, $\alpha$. Одним из наиболее распространенных является следующий. Разрешим систему относительно неизвестных при ненулевых диагональных элементах $a_{ii} \ne 0, i = \overline{1, n}$ (если какой-либо коэффициент на главной диагонали равен нулю, достаточно соответствующее уравнение поменять местами с любым другим уравнением). Получим следующие выражения для компонентов вектора $\beta$ и матрицы $\alpha$ эквивалентной системы:

$$
\begin{matrix}
    \beta_i = \frac{b_i}{a_{ii}} \\
    \alpha_{ij} = - \frac{a_{ij}}{a_{ii}}, i \ne j; a_{ij} = 0, i = j \\
\end{matrix}
$$

В итоге мы пришли к определению одного из методов итераций - методу простых итераций. Его достаточным условием сходимости является диагональное преобладание матрицы $A$ по строкам или по столбцам:

$$
\left| a_{ii} \right| > \sum_{j = 1, i \ne j} ^ {n} \left| a_{ij} \right|
$$

Одним из недостатков метода простых итераций является его довольно медленная сходимость. Для решения этой проблемы используют **метод Зейделя**, заключающийся в том, что при вычислении компонента $x_i ^ {k + 1}$ вектора неизвестных на $(k + 1)$-й итерации используются $x_1 ^ {k + 1}, x_2 ^ {k + 1}, \ldots, x_{i - 1} ^ {k + 1}$, уже вычисленные на $(k + 1)$-й итерации. Тогда метод Зейделя для известного вектора на k-ой итерации имеет вид:

$$
\left\{ \begin{matrix}
    x_1 ^ {k + 1} = \beta_1 + \alpha_{11} \cdot x_1 ^ {k} + \alpha_{12} \cdot x_2 ^ {k} + \ldots + \alpha_{1n} \dot x_n ^ k \\
    x_2 ^ {k + 1} = \beta_2 + \alpha_{21} \cdot x_1 ^ {k} + \alpha_{22} \cdot x_2 ^ {k} + \ldots + \alpha_{2n} \dot x_n ^ k \\
    x_3 ^ {k + 1} = \beta_3 + \alpha_{31} \cdot x_1 ^ {k} + \alpha_{32} \cdot x_2 ^ {k} + \ldots + \alpha_{3} \dot x_n ^ k \\
    \ldots \\
    x_n ^ {k + 1} = \beta_n + \alpha_{n1} \cdot x_1 ^ {k} + \alpha_{n2} \cdot x_2 ^ {k} + \ldots + \alpha_{nn} \dot x_n ^ k \\
\end{matrix} \right.
$$

#### Входные данные

Ниже приведен пример входных данных для решения системы линейный алгебраических уравнений с использованием методов итераций, соответствующий моему варианту

$$
\left\{ \begin{matrix}
    29 \cdot x_1 + 8 \cdot x_2 + 9 \cdot x_3 - 9 \cdot x_4 = 197 \\
    -7 \cdot x_1 - 25 \cdot x_2 + 9 \cdot x_4 = -226 \\
    x_1 + 6 \cdot x_2 + 16 \cdot x_3 - 2 \cdot x_4 = -95 \\
    -7 \cdot x_1 + 4 \cdot x_2 - 2 \cdot x_3 + 17 \cdot x_4 = -58 \\
\end{matrix} \right.
$$

#### Реализация метода простых итераций

Ниже предоставлена реализация метода простых реализаций на языке программирования C++

```cpp
struct IterativeMethodResult {
    Matrix::TMatrix result;
    int iterations;
};

float Norm(const Matrix::TMatrix& m) {
    float matrixSum = 0.0;
    float element;
    int n = m.GetSize().first;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            element = m.Get(i, j);
            matrixSum += element * element;
        }
    }

    return std::sqrt(matrixSum);
}

void JakobiMethod(const Matrix::TMatrix& A, const Matrix::TMatrix& b, Matrix::TMatrix& alpha, Matrix::TMatrix& beta) {
    int n = A.GetSize().first;
    float diagAlpha;

    for (int i = 0; i < n; ++i) {
        diagAlpha = A.Get(i, i);
        beta.Set(i, 0, b.Get(i, 0) / diagAlpha);

        for (int j = 0; j < n; ++j) {
            alpha.Set(
                i, j,
                (i == j) ? 0.0 : - A.Get(i, j) / diagAlpha
            );
        }
    }
}

void inverseMatrix(const Matrix::TMatrix& m, Matrix::TMatrix& result) {
    int n = m.GetSize().first;
    Matrix::TMatrix l(n, n), u(n, n), p(n, n);

    LUDecompose(m, l, u, p);
    InverseMatrix(l, u, p, result);
}

void SeidelMethod(const Matrix::TMatrix& A, const Matrix::TMatrix& b, Matrix::TMatrix& alpha, Matrix::TMatrix& beta) {
    int n = A.GetSize().first;
    Matrix::TMatrix B(n, n), C(n, n), T(n, n);
    Matrix::TMatrix E = Matrix::TMatrix::Eye(n);

    JakobiMethod(A, b, alpha, beta);

    // split alpha matrix B, C: alpha = B + C
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (j >= i) C.Set(i, j, alpha.Get(i, j));
            else B.Set(i, j, alpha.Get(i, j));
        }
    }

    inverseMatrix(E + B * (-1.0), T);
    alpha = T * C;
    beta = T * beta;
}

float Epsilon(float alphaNorm, const Matrix::TMatrix& x1, const Matrix::TMatrix& x2) {
    float coef = alphaNorm / (1 - alphaNorm);
    Matrix::TMatrix diff = x1 + x2 * (-1.0);
    return coef * Norm(diff);
}

IterativeMethodResult IterativeMethod(const Matrix::TMatrix& alpha, const Matrix::TMatrix& beta, float eps) {
    int n = alpha.GetSize().first;
    float alphaNorm = Norm(alpha);
    int iterations = 0;

    Matrix::TMatrix x = beta;
    Matrix::TMatrix x2 = beta;

    while (true) {
        iterations++;
        x2 = alpha * x + beta;

        if (Epsilon(alphaNorm, x, x2) <= eps) {
            break;
        }

        x = x2;
    }

    return {
        .result=x2,
        .iterations=iterations
    };
}
```

#### Результат

Ниже предоставлен пример работы программы с входными данными, соответствующими моему варианту

![](images/3.png)

### Метод вращений

Среди численных методов для решения задачи поиска собственных значений матрицы существует метода вращений Якоби. Данный метод имеет ограничения на входящую матрицу в виде симметричности ($A ^ T = A$). Его задачей является решение полной проблемы собственных значений и собственных векторов матриц.

Данный метод основывается на идее итерационном применении преобразований подобия

$$
\Lambda = U ^ {-1} \cdot A \cdot U
$$

поскольку для симметрических матриц $A$ матрица преобразования подобия $U$ является ортогональной ($U ^ {-1} = U ^ T$), то $\Lambda = U ^ T \cdot A \cdot U$, где $\Lambda$ - диагональная матрица с собственными значениями на главной диагонали.

Сформулируем алгоритм работы методы вращений Якоби. Пусть дана симметрическая матрица $A$. Требуется вычислить все собственные значения и соответствующие им собственные векторы с заданной точностью $\varepsilon$.

Пусть известна матрица $A^ {(k)}$ на $k$-й итерации, при этом для $k = 0$: $A ^ {(0)} = A$.

1. На начальной итерации (при $k = 0$) имеем $A ^ {(0)} = A$;

2. На каждой итерации выбирается максимальный по модулю недиагональный элемент $a_{ij} ^ {(k)}$ матрицы $A ^ {(k)}$ ($\left| a_{ij} ^ {(k)} \right| = \max_{l < m} \left| a_{lm} ^ {(k)} \right|$);

3. Ставится задача найти такую ортогональную матрицу $U ^ {(k)}$, чтобы в результате преобразования подобия $A ^ {(k + 1)} = U ^ {(k) T} \cdot A ^ {(k)} \cdot U ^ {(k)}$ произошло обнуление элемента $a_{ij} ^ {(k+1)}$ матрицы $A ^ {(k + 1)}$. В качестве ортогональной матрицы выбирается матрица вращения, имеющая следующий вид:

   ![](images/4.png)

   Угол вращения $\varphi ^ {(k)}$ определяется из условия $a_{ij} ^ {(k + 1)} = 0$:

   $$
   \varphi ^ {(k)} = \frac{1}{2} \cdot \arctg{
       \frac{
            2 \cdot a_{ij} ^ {(k)}
        }{
            a_{ii} ^ {(k)} - a_{jj} ^ {(k)}
        }
   }
   $$

   причем если $a_{ii} ^ {(k)} = a_{jj} ^ {(k)}$, то $\varphi ^ {(k)} = \frac{\pi}{4}$.

4. Строится матрица $A ^ {(k + 1)}$

   $$
   A ^ {(k + 1)} = U ^ {(k) T} \cdot A ^ {(k)} \cdot U ^ {(k)}
   $$

   Так как метод вращений Якоби также является итерационным, то возникает необходимость в определении критерия окончания итерационного процесса. Он имеет следующий вид:

   $$
   t \left( A ^ {(k + 1)} \right) =
   \sqrt{\sum_{l, m; l < m} \left( a_{lm} ^ {(k + 1)} \right) ^ 2}
   $$

   Данное неравенство описывает условие малости суммы квадратов в недиагональных элементах.

#### Входные данные

Ниже приведен пример входных данных для решения задачи поиска собственных значений с использованием метода вращений Якоби, соответствующий моему варианту

$$
\left( \begin{matrix}
    -6 &&  6 && -8 \\
     6 && -4 &&  9 \\
    -8 &&  9 && -2 \\
\end{matrix} \right)
$$

#### Реализация метода вращений Якоби

Ниже предоставлена программная реализация метода вращения Якоби

```cpp
std::pair<int, int> FindMaxNotDiagonalElement(const Matrix::TMatrix& A) {
    int n = A.GetSize().first;
    int maxI = 0, maxJ = 1;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (fabsf(A.Get(i, j)) > fabsf(A.Get(maxI, maxJ))) {
                maxI = i;
                maxJ = j;
            }
        }
    }
    return {maxI, maxJ};
}

float GetRotationAngle(const Matrix::TMatrix& A, int i, int j) {
    float aii = A.Get(i, i), ajj = A.Get(j, j), aij = A.Get(i, j);
    if (aii == ajj) return M_PI / 4.0;
    return 0.5 * atan((2.0 * aij) / (aii - ajj));
}

Matrix::TMatrix GetRotationMatrix(int n, int i, int j, float phi) {
    Matrix::TMatrix U = Matrix::TMatrix::Eye(n);
    float c = cos(phi), s = sin(phi);
    U.Set(i, i, c);
    U.Set(j, j, c);
    U.Set(i, j, -s);
    U.Set(j, i, s);
    return U;
}

Matrix::TMatrix GetNextA(const Matrix::TMatrix& A, const Matrix::TMatrix& U) {
    return Matrix::Transpose(U) * A * U;
}

float t(const Matrix::TMatrix& A) {
    int n = A.GetSize().first;
    float s = 0.0, el;

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            el = A.Get(i, j);
            s += el * el;
        }
    }
    return sqrt(s);
}

EigenTaskResult SolveEigenTask(const Matrix::TMatrix& M, float eps) {
    Matrix::TMatrix A(M);
    int n = A.GetSize().first;
    Matrix::TMatrix U = Matrix::TMatrix::Eye(n);
    EigenTaskResult result;
    result.iterations = 0;

    while (t(A) > eps) {
        std::pair<int, int> pos = FindMaxNotDiagonalElement(A);
        float phi = GetRotationAngle(A, pos.first, pos.second);
        Matrix::TMatrix u = GetRotationMatrix(n, pos.first, pos.second, phi);
        A = GetNextA(A, u);
        U = U * u;
        result.iterations++;
    }

    for (int i = 0; i < n; ++i) {
        result.eigenValues.push_back(A.Get(i, i));
    }

    for (int i = 0; i < n; ++i) {
        Matrix::TMatrix v(n, 1);
        for (int j = 0; j < n; ++j) {
            v.Set(j, 0, U.Get(j, i));
        }
        result.eigenVectors.push_back(v);
    }

    return result;
}
```

#### Результат

Ниже предоставлен пример работы программы с входными данными, соответствующими моему варианту

![](images/5.png)

### QR-алгоритм нахождения собственных значений матриц

Также, для решения задачи поиска собственных значений и собственных векторов матрицы может быть применен метод QR-разложения. Его суть заключается в представлении исходной матрицы $A$ в виде следующего произведения

$$
A = Q \cdot R
$$

где $Q$ - ортогональная матрица, а $R$ - верхняя треугольная. Одним из возможных способов построения $QR$ разложения является использование преобразования Хаусхолдера, который позволяет обратить в нуль группу поддиагональных элементов столбца матрицы.

Преобразование Хаусхолдера осуществляется с использованием матрицы Хаусхолдера, имеющей следующий вид:

$$
H = E - \frac{2}{\nu ^ T \cdot \nu}\cdot \nu \cdot \nu ^ T
$$

где $v$ - произвольный ненулевой столбец.

Далее, для решения поставленной задачи, необходимо рассмотреть случай, когда необходимо обратить в нуль все элементы вектора кроме первого. Для этого необходимо построить матрицу Хаусхолдера такую, чтобы она удовлетворяла следующему равенству

$$
\widetilde{b} = H \cdot b
$$

где $\widetilde{b} = \left( \widetilde{b_1},\ 0, \ldots, 0 \right) ^ T $ - столбец $b$, в котором все элементы, кроме первого, стали нулевыми.

В этом случае вектор $\nu$ имеет следующий вид:

$$
\nu = b + sign \left( b_1 \right) \cdot \left| \left| b \right| \right|_2 \cdot e_1
$$

Применяя описанную процедуру для обнуления поддиагональных элементов каждого столбца исходной матрицы, можно за фиксированное количество шагов выполнить ее QR-разложение.

Процедура QR-разложения многократно применяется в QR-алгоритме для вычисления собственных значений. При этом строится следующий
итерационный процесс:

$$
\begin{matrix}
A ^ {(0)} = A \\
A ^ {(0)} = Q ^ {(0)} \cdot R^{(0)} \\
A ^ {(1)} = R ^ {(0)} \cdot Q ^ {(0)} \\
\ldots \\
A ^ {(k)} = Q ^ {(k)} \cdot R ^ {(k)} \\
A ^ {(k + 1)} = R ^ {(k)} \cdot Q ^ {(k)} \\
\end{matrix}
$$

Таким образом, каждая итерация выполняется в два этапа. На первом этапе матрица $A ^ {(k)}$ разлагается на произведение ортогональной матрицы $Q ^ {(k)}$ и верхней треугольной матрицы $R ^ {(k)}$. На втором этапе эти матрицы перемножаются в обратном порядке.

Если у матрицы нет кратных собственных значений, последовательность $A ^ {(k)}$ сходится к верхней треугольной матрице (если все собственные значения вещественные) или к верхней квазитреугольной матрице (если имеются комплексно-сопряженные пары собственных значений).

Таким образом, каждому вещественному собственному значению будет соответствовать столбец с поддиагональными элементами, стремящимися к нулю. В качестве критерия сходимости итерационного процесса для таких собственных значений можно использовать следующее неравенство:

$$
\sqrt{\sum_{l = m + 1} ^ n \left( a_{lm} ^ {(k)} \right)^{2}} \leq \varepsilon
$$

#### Входные данные

Ниже приведен пример входных данных для решения задачи поиска собственных значений с использованием метода $QR$-разложения,
соответствующий моему варианту

$$
\left( \begin{matrix} 
     9 &&  0 && 2 \\
    -6 &&  4 && 4 \\
    -2 && -7 && 5 \\
\end{matrix} \right)
$$

#### Реализация

Ниже предоставлена программная реализация алгоритма $QR$-разложения. Она также включает в себя вычисление матрицы Хаусхолдера.

```cpp
using ComplexPair = std::pair<std::complex<float>, std::complex<float>>;

using EigenValues = std::vector<std::complex<float>>;

using ChangeHistory = std::vector<float>;

const int HISTORY_SIZE = 5;

void GetHouseholderMatrix(const Matrix::TMatrix& A, int i, Matrix::TMatrix& H) {
    int n = A.GetSize().first;
    Matrix::TMatrix v(n, 1);

    for (int j = 0; j < n; ++j) {
        if (j < i) {
            v.Set(j, 0, 0);
        }

        else if (j > i) {
            v.Set(j, 0, A.Get(j, i));
        }

        else {
            float aDiag = A.Get(i, i);
            float signA = float((aDiag > 0) - (aDiag < 0));
            float sum = 0.0;

            for (int t = j; t < n; ++t) {
                sum += A.Get(t, i) * A.Get(t, i);
            }

            v.Set(
                j, 0,
                aDiag + signA * sqrt(sum)
            );
        }
    }

    float k = (Matrix::Transpose(v) * v).Get(0, 0);
    H = v * Matrix::Transpose(v) * (-2.0 / k) + Matrix::TMatrix::Eye(n);
}

void QRDecompose(const Matrix::TMatrix& A, Matrix::TMatrix& Q, Matrix::TMatrix& R) {
    int n = A.GetSize().first;
    Matrix::TMatrix H(n, n);
    Q = Matrix::TMatrix::Eye(n);
    R = A;

    for (int i = 0; i < n - 1; ++i) {
        GetHouseholderMatrix(R, i, H);
        Q = Q * H;
        R = H * R;
    }
}

ComplexPair FindComplexEigeValues(const Matrix::TMatrix& A, int i) {
    float a1 = A.Get(i, i), a2 = A.Get(i + 1, i + 1);
    float a3 = A.Get(i + 1, i), a4 = A.Get(i, i + 1);
    float b = - a1 - a2;
    float c = a1 * a2 - a3 * a4;
    float d = b * b - 4.0 * c;

    std::complex<float> dSqrt = std::sqrt(std::complex<float>(d, 0));
    std::complex<float> bComplex = std::complex<float>(b, 0.0);
    std::complex<float> k = 0.5;

    return { k * (-bComplex + dSqrt), k * (-bComplex - dSqrt) };
}

bool tReal(const Matrix::TMatrix& A, int i, int j, float eps) {
    int n = A.GetSize().first;
    float sum = 0.0;
    for (int t = j; t < n; ++t) {
        sum += A.Get(t, i) * A.Get(t, i);
    }

    return std::sqrt(sum) <= eps;
}

float tComplex(const Matrix::TMatrix& Ai, int i, float eps) {
    int n = Ai.GetSize().first;

    Matrix::TMatrix Q(n, n), R(n, n);
    QRDecompose(Ai, Q, R);
    Matrix::TMatrix ANext = R * Q;

    ComplexPair lambda1 = FindComplexEigeValues(Ai, i);
    ComplexPair lambda2 = FindComplexEigeValues(ANext, i);

    return (std::abs(lambda2.first - lambda1.first) <= eps) && (std::abs(lambda2.second - lambda1.second) <= eps);
}

bool IsEigenValueReal(const ChangeHistory& history) {
    int startIndex = std::max(1, int(history.size()) - HISTORY_SIZE);

    for (int i = startIndex; i < history.size(); ++i) {
        if (history[i] >= history[i - 1]) {
            return false;
        }
    }

    return true;

}

void UpdateChangeHistory(const Matrix::TMatrix& A, std::vector<ChangeHistory>& history) {
    int n = A.GetSize().first;

    for (int i = 0; i < n - 1; ++i) {
        history[i].push_back(std::abs(A.Get(i + 1, i)));
    }
}

EigenValues GetEigenValues(const Matrix::TMatrix& A, float eps) {
    int n = A.GetSize().first;

    Matrix::TMatrix Q(n, n), R(n, n);
    Matrix::TMatrix Ai = A;
    EigenValues values;

    std::vector<std::vector<float>> history(n);
    int i = 0;

    while (i < n) {
        QRDecompose(Ai, Q, R);
        Ai = R * Q;
        UpdateChangeHistory(Ai, history);

        if (tReal(Ai, i, i + 1, eps)) {
            values.push_back(Ai.Get(i, i));
            i++;
        } else if (tComplex(Ai, i, eps) && tReal(Ai, i, i + 2, eps)) {
            ComplexPair p = FindComplexEigeValues(Ai, i);
            values.push_back(p.first);
            values.push_back(p.second);
            i += 2;
        }
    }

    return values;
}
```

#### Результат

Ниже предоставлен пример работы программы с входными данными,
соответствующими моему варианту

![](images/6.png)
