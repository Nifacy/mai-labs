#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

void sort(float* array, size_t n) {
    bool swapped;
    float tmp;
    size_t i, j;

    for (i = 0; i < n - 1; ++i) {
        swapped = false;

        for (j = 0; j < n - i - 1; ++j) {
            if (array[j] > array[j + 1]) {
                tmp = array[j];
                array[j] = array[j + 1];
                array[j + 1] = tmp;
                swapped = true;
            }
        }

        if (!swapped) {
            break;
        }
    }
}

void printArray(const float* array, size_t n) {
    for (size_t index = 0; index < n; ++index) {
        printf("%.6e ", array[index]);
    }
    printf("\n");
}

int main() {
    size_t n;
    scanf("%lu", &n);

    float* array = (float*) malloc(sizeof(float) * n);

    for (size_t index = 0; index < n; ++index) {
        scanf("%f", &array[index]);
    }

    sort(array, n);
    printArray(array, n);

    return 0;
}
