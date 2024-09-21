#include <stdint.h>
#include <stdio.h>


const uint32_t DEVICE_BLOCKS = 512;
const uint32_t DEVICE_THREADS = 512;


#define EXIT_WITH_ERROR(message)                                   \
  {                                                                \
    fprintf(stderr, "ERROR: [line %d] %s\n", __LINE__, (message)); \
    exit(0);                                                       \
  }


__global__ void kernel(double* array, size_t n) {
  size_t offset = gridDim.x * blockDim.x;
  size_t index = blockDim.x * blockIdx.x + threadIdx.x;
  double tmp;

  while (index < n / 2) {
    tmp = array[index];
    array[index] = array[n - index - 1];
    array[n - index - 1] = tmp;
    index += offset;
  }
}


int main() {
  size_t n;
  size_t sizeInBytes;
  double* array;
  double* deviceArray;
  cudaError_t result;

  scanf("%lu", &n);

  sizeInBytes = sizeof(double) * n;
  array = (double*)malloc(sizeInBytes);
  if (array == NULL) {
    EXIT_WITH_ERROR("Host allocation error");
  }

  for (size_t index = 0; index < n; ++index) {
    scanf("%lf", &array[index]);
  }

  result = cudaMalloc(&deviceArray, sizeInBytes);
  if (result != cudaSuccess) {
    EXIT_WITH_ERROR(cudaGetErrorString(result));
  }

  result = cudaMemcpy(deviceArray, array, sizeInBytes, cudaMemcpyHostToDevice);
  if (result != cudaSuccess) {
    EXIT_WITH_ERROR(cudaGetErrorString(result));
  }

  kernel<<<DEVICE_BLOCKS, DEVICE_THREADS>>>(deviceArray, n);
  cudaDeviceSynchronize();

  result = cudaMemcpy(array, deviceArray, sizeInBytes, cudaMemcpyDeviceToHost);
  if (result != cudaSuccess) {
    EXIT_WITH_ERROR(cudaGetErrorString(result));
  }

  for (size_t index = 0; index < n; ++index) {
    printf("%.10e ", array[index]);
  }
  printf("\n");

  result = cudaFree(deviceArray);
  if (result != cudaSuccess) {
    EXIT_WITH_ERROR(cudaGetErrorString(result));
  }
  free(array);

  return 0;
}
