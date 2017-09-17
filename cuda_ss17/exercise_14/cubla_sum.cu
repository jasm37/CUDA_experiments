#include <cublas_v2.h>
#include <iostream>

int main()
{
    float num[100000];
    float *d_num;
    int n = 100000;
    size_t nbytes = n * sizeof(float);
    // fill array :
    for(int i = 0; i < n; i++) num[i] = i;

    cudaMalloc(&d_num, nbytes);
    cudaMemcpy(d_num, num, nbytes, cudaMemcpyHostToDevice);

    // Start handle to use cublas
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Variable where to store sum:
    float sum_num = 0;
    
    cublasSasum(handle, n, d_num, 1, &sum_num);
    std::cout << "Sum is "<< sum_num << std::endl;

    cublasDestroy(handle);
    cudaFree(d_num);

     
}
