// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2017, September 11 - October 9
// ###

#include <cuda_runtime.h>
#include <iostream>
using namespace std;



// cuda error checking
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(string file, int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        cout << endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << endl;
        exit(1);
    }
}


__global__
void sumFunc(float *a, float *b, float *c, int n)
{
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i<n) c[i] = a[i] + b[i];
}


int main(int argc, char **argv)
{
    // alloc and init input arrays on host (CPU)
    int n = 20;
    float *a = new float[n];
    float *b = new float[n];
    float *c = new float[n];
    for(int i=0; i<n; i++)
    {
        a[i] = i;
        b[i] = (i%5)+1;
        c[i] = 0;
    }

    // CPU computation
    for(int i=0; i<n; i++) c[i] = a[i] + b[i];

    // print result
    cout << "CPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << " + " << b[i] << " = " << c[i] << endl;
    cout << endl;
    // init c
    for(int i=0; i<n; i++) c[i] = 0;
    
    float *d_a, *d_b, *d_c = NULL;
    size_t nbytes = size_t(n)*sizeof(float);

    // Initialize stuff
    dim3 block = dim3(128,1,1);
    dim3 grid = dim3((n + block.x -1) / block.x, 1, 1);

    cudaMalloc(&d_a, nbytes);
    cudaMalloc(&d_b, nbytes);
    cudaMalloc(&d_c, nbytes);
    CUDA_CHECK;

    cudaMemcpy( d_a, a, nbytes, cudaMemcpyHostToDevice );
    cudaMemcpy( d_b, b, nbytes, cudaMemcpyHostToDevice );
    cudaMemcpy( d_c, c, nbytes, cudaMemcpyHostToDevice );
    CUDA_CHECK;

    //Call function
    sumFunc<<<grid, block>>>(d_a, d_b, d_c, n);

    cudaMemcpy( a, d_a, nbytes, cudaMemcpyDeviceToHost );
    cudaMemcpy( b, d_b, nbytes, cudaMemcpyDeviceToHost );
    cudaMemcpy( c, d_c, nbytes, cudaMemcpyDeviceToHost );
    CUDA_CHECK;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    CUDA_CHECK;



    // GPU computation
    // ###
    // ### TODO: Implement the array addition on the GPU, store the result in "c"
    // ###
    // ### Notes:
    // ### 1. Remember to free all GPU arrays after the computation
    // ### 2. Always use the macro CUDA_CHECK after each CUDA call, e.g. "cudaMalloc(...); CUDA_CHECK;"
    // ###    For convenience this macro is defined directly in this file, later we will only include "helper.h"
    


    // print result
    cout << "GPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << " + " << b[i] << " = " << c[i] << endl;
    cout << endl;

    // free CPU arrays
    delete[] a;
    delete[] b;
    delete[] c;
}



