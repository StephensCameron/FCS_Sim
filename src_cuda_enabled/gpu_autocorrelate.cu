#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <math.h>
#include "cublas_v2.h"


// nvcc --shared -o DLLs_cuda_enabled/gpu_autocorrelate.dll src_cuda_enabled/gpu_autocorrelate.cu
// nvcc --shared -o DLLs_cuda_enabled/gpu_autocorrelate.dll src_cuda_enabled/gpu_autocorrelate.cu -lcublas



__global__ void elementwiseMul(float *A, float *B, float *C, int N) {
    // Calculate global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check to prevent out-of-bounds memory access
    if (i < N) {
        C[i] = A[i] * B[i];
    }
}

extern "C" __declspec(dllexport) void call_elementwiseMul(float *A, float *B, float *C, int N) {
    size_t size = N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    elementwiseMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}



extern "C" __declspec(dllexport) int my_cublas_autocorrelate(float *I, float *corr, int N, int max_lag_indx) {
    size_t size_corr = max_lag_indx * sizeof(float);
    // check that max_lag_indx is not greater than N
    if (max_lag_indx > N) {
        return -1;
    }

    // create cublas handle
    cublasHandle_t handle;
    cublasStatus_t cublas_status = cublasCreate(&handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        return -2;
    }

    // cublasSdot writes into device memory below, so use device pointer mode.
    cublas_status = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(handle);
        return -3;
    }


    // allocate memory on device:
    float *d_I, *d_corr;
    cudaMalloc(&d_I, N * sizeof(float));
    cudaMalloc(&d_corr, size_corr);

    cudaMemcpy(d_I, I, N * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize correlation array on device to zero
    cudaMemset(d_corr, 0, size_corr);

    // Your autocorrelation computation would go here
    // initialize length of each lag segment in the correlation array
    int n = N;
    for (int lag = 0; lag < max_lag_indx; ++lag) {
        // this is actually a very fast operation since we do not have to move memory from device to host more than once
        // even though it's a loop, all the data is already on the device

        // psudocode: 
        // corr[i] = dot(I[i:],I[:-i])/n
        cublas_status = cublasSdot(handle, n, &d_I[lag], 1, &d_I[0], 1, &d_corr[lag]);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            cudaFree(d_I);
            cudaFree(d_corr);
            cublasDestroy(handle);
            return -4;
        }
        cudaDeviceSynchronize(); // Ensure the dot product is completed before moving to the next lag
        n--; // decrement the length of the segment for the next lag
    }

    // Copy the result back to host
    cudaMemcpy(corr, d_corr, size_corr, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_I);
    cudaFree(d_corr);

    cublasDestroy(handle);
    return 0;
}
    



// int main() {
//     int N = 1 << 20; // 1,048,576 elements
//     size_t size = N * sizeof(float);

//     // 1. Allocate Host Memory
//     float *h_A = (float*)malloc(size);
//     float *h_B = (float*)malloc(size);
//     float *h_C = (float*)malloc(size);

//     // Initialize Host Arrays
//     for(int i = 0; i < N; i++) {
//         h_A[i] = 2.0f;
//         h_B[i] = 3.0f;
//     }

//     // 2. Allocate Device Memory
//     float *d_A, *d_B, *d_C;
//     cudaMalloc(&d_A, size);
//     cudaMalloc(&d_B, size);
//     cudaMalloc(&d_C, size);

//     // 3. Copy Data from Host to Device
//     cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

//     // 4. Configure Execution Parameters
//     int threadsPerBlock = 256;
//     int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

//     // 5. Launch the Kernel
//     elementwiseMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

//     // 6. Copy Result back to Host
//     cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

//     // Print a quick validation snippet
//     std::cout << "Result validation (C[0]): " << h_C[0] << " (Expected: 6)" << std::endl;

//     // 7. Cleanup Free Memory
//     cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
//     free(h_A); free(h_B); free(h_C);

//     return 0;
// }
