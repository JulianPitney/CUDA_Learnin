
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <thread>
#include <iostream>

using namespace std;


cudaError_t addWithCuda(int *inputVec1, int *inputVec2, int *outputVec, unsigned int arraySize, dim3 gridDims, dim3 blockDims);

__global__ void addKernel(int *inputVec1, int *inputVec2, int *outputVec)
{
	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
    outputVec[i] = inputVec1[i] + inputVec2[i];
}

int main()
{
	unsigned int arraySize;
	cin >> arraySize;

	dim3 gridDims(65000);
	dim3 blockDims(1024);


	int *inputVec1 = new int[arraySize];
	int *inputVec2 = new int[arraySize];
	int *outputVec = new int[arraySize];

	auto t1 = std::chrono::high_resolution_clock::now();
	for (unsigned int i = 0; i < arraySize; i++)
	{
		inputVec1[i] = rand() % 354876;
		inputVec2[i] = rand() % 234587;
	}
	auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
	cout << "Host Vec Init(ms): " << duration / 1000 << endl;



    cudaError_t cudaStatus = addWithCuda(inputVec1, inputVec2, outputVec, arraySize, gridDims, blockDims);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }


    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t addWithCuda(int *inputVec1, int *inputVec2, int *outputVec, unsigned int arraySize, dim3 gridDims, dim3 blockDims)
{
    int *dev_vec1 = 0;
    int *dev_vec2 = 0;
    int *dev_outputVec = 0;
    cudaError_t cudaStatus;

	auto t1_deviceSet = std::chrono::high_resolution_clock::now();
    cudaStatus = cudaSetDevice(0);
	auto t2_deviceSet = std::chrono::high_resolution_clock::now();
	auto duration_deviceSet = std::chrono::duration_cast<std::chrono::microseconds>(t2_deviceSet - t1_deviceSet).count();
	cout << "Device Set Duration(ms): " << duration_deviceSet / 1000 << endl;;


    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

	auto t1_malloc1 = std::chrono::high_resolution_clock::now();
    cudaStatus = cudaMalloc((void**)&dev_vec1, arraySize * sizeof(int));
	auto t2_malloc1 = std::chrono::high_resolution_clock::now();
	auto duration_malloc1 = std::chrono::duration_cast<std::chrono::microseconds>(t2_malloc1 - t1_malloc1).count();
	cout << "Malloc1 Duration(ms): " << duration_malloc1 / 1000 << endl;;
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	auto t1_malloc2 = std::chrono::high_resolution_clock::now();
    cudaStatus = cudaMalloc((void**)&dev_vec2, arraySize * sizeof(int));
	auto t2_malloc2 = std::chrono::high_resolution_clock::now();
	auto duration_malloc2 = std::chrono::duration_cast<std::chrono::microseconds>(t2_malloc2 - t1_malloc2).count();
	cout << "Malloc2 Duration(ms): " << duration_malloc2 / 1000 << endl;;
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	auto t1_malloc3 = std::chrono::high_resolution_clock::now();
    cudaStatus = cudaMalloc((void**)&dev_outputVec, arraySize * sizeof(int));
	auto t2_malloc3 = std::chrono::high_resolution_clock::now();
	auto duration_malloc3 = std::chrono::duration_cast<std::chrono::microseconds>(t2_malloc3 - t1_malloc3).count();
	cout << "Malloc3 Duration(ms): " << duration_malloc3 / 1000 << endl;;
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	auto t1_memCpy1 = std::chrono::high_resolution_clock::now();
    cudaStatus = cudaMemcpy(dev_vec1, inputVec1, arraySize * sizeof(int), cudaMemcpyHostToDevice);
	auto t2_memCpy1 = std::chrono::high_resolution_clock::now();
	auto duration_memCpy1 = std::chrono::duration_cast<std::chrono::microseconds>(t2_memCpy1 - t1_memCpy1).count();
	cout << "MemCpy1 Duration(ms): " << duration_memCpy1 / 1000 << endl;;
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	auto t1_memCpy2 = std::chrono::high_resolution_clock::now();
    cudaStatus = cudaMemcpy(dev_vec2, inputVec2, arraySize * sizeof(int), cudaMemcpyHostToDevice);
	auto t2_memCpy2 = std::chrono::high_resolution_clock::now();
	auto duration_memCpy2 = std::chrono::duration_cast<std::chrono::microseconds>(t2_memCpy2 - t1_memCpy2).count();
	cout << "MemCpy2 Duration(ms): " << duration_memCpy2 / 1000 << endl;;
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	auto t1_kernelLaunch = std::chrono::high_resolution_clock::now();
    addKernel<<<gridDims, blockDims>>>(dev_vec1, dev_vec2, dev_outputVec);
	auto t2_kernelLaunch = std::chrono::high_resolution_clock::now();
	auto duration_kernelLaunch = std::chrono::duration_cast<std::chrono::microseconds>(t2_kernelLaunch - t1_kernelLaunch).count();
	cout << "Kernel Launch Duration(ms): " << duration_kernelLaunch / 1000 << endl;;

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	auto t3_kernelComplete = std::chrono::high_resolution_clock::now();
	auto duration_kernelRuntime = std::chrono::duration_cast<std::chrono::microseconds>(t3_kernelComplete - t1_kernelLaunch).count();
	cout << "Kernel Runtime(ms): " << duration_kernelRuntime / 1000 << endl;;



	auto t1_devToHost_memCpy = std::chrono::high_resolution_clock::now();
    cudaStatus = cudaMemcpy(outputVec, dev_outputVec, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
	auto t2_devToHost_memCpy = std::chrono::high_resolution_clock::now();
	auto duration_devToHost_memCpy = std::chrono::duration_cast<std::chrono::microseconds>(t2_devToHost_memCpy - t1_devToHost_memCpy).count();
	cout << "devToHost memCpy Duration(ms): " << duration_devToHost_memCpy / 1000 << endl;;
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_outputVec);
    cudaFree(dev_vec1);
    cudaFree(dev_vec2);
    
    return cudaStatus;
}
