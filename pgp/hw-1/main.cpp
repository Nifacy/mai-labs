#include <math.h>
#include <stdint.h>
#include <stdio.h>

enum RootsAmount { ANY, INCORRECT, IMAGINARY, ONE, TWO };

typedef struct Result {
  enum RootsAmount rootsAmount;
  float roots[2];
} Result;

Result solve(float a, float b, float c) {
  Result result;
  float D, sqrtD;

  if (a == 0.0 && b == 0.0) {
    if (c == 0.0) {
      result.rootsAmount = ANY;
    } else {
      result.rootsAmount = INCORRECT;
    }
    return result;
  }

  if (a == 0.0) {
    result.rootsAmount = ONE;
    result.roots[0] = -c / b;
    return result;
  }

  D = b * b - 4.0 * a * c;

  if (D > 0.0) {
    sqrtD = sqrt(D);
    result.rootsAmount = TWO;
    result.roots[0] = (-b + sqrtD) / (2.0 * a);
    result.roots[1] = (-b - sqrtD) / (2.0 * a);
  } else if (D == 0.0) {
    result.rootsAmount = ONE;
    result.roots[0] = -b / (2 * a);
  } else {
    result.rootsAmount = IMAGINARY;
  }

  return result;
}

int main() {
  float a, b, c;
  Result result;

  scanf("%f %f %f", &a, &b, &c);
  result = solve(a, b, c);

  switch (result.rootsAmount) {
    case ANY:
      printf("any\n");
      break;
    case INCORRECT:
      printf("incorrect\n");
      break;
    case IMAGINARY:
      printf("imaginary\n");
      break;
    case ONE:
      printf("%.6f\n", result.roots[0]);
      break;
    case TWO:
      printf("%.6f %.6f\n", result.roots[0], result.roots[1]);
      break;
  }

  return 0;
}
