#include <cassert>

#include <iostream>

#include <cuda_runtime.h>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(EXIT_FAILURE);
  }
}

__global__ void Hillis_Steele_Scan_Kernel(float *d_out, float *d_intermediate, float *d_in) {
  extern __shared__ float sdata[];
  
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid  = threadIdx.x;
  
  int pout = 0, pin = 1;
  
  sdata[tid] = d_in[myId];
  __syncthreads();
  
  for(unsigned int s = 1; s < blockDim.x; s <<= 1) {
    pout = 1 - pout;
    pin  = 1 - pout;
    if (tid >= s)
      sdata[pout*blockDim.x + tid] = sdata[pin*blockDim.x + tid] + sdata[pin*blockDim.x + tid - s];
    else
      sdata[pout*blockDim.x + tid] = sdata[pin*blockDim.x + tid];
    __syncthreads();
  }
  
  if (tid == blockDim.x - 1) {
    d_intermediate[blockIdx.x] = sdata[blockDim.x - 1];
  }
  d_out[myId] = sdata[tid];
}


__global__ void Blelloch_Scan_Kernel(float *d_out, float *d_intermediate, float *d_in) {
  extern __shared__ float sdata[];
  
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;
  
  sdata[tid] = d_in[myId];
  __syncthreads();
  
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid]
    } 
  }
}

__global__ void Sum_Kernel(float *d_out, float *d_intermediate, float *d_in) {

}

void scan(float *d_out, float *d_in, int size) {
  float *d_intermediate;
  const int maxThreadsPerBlock = 1024;
  int threads = maxThreadsPerBlock;
  int blocks = size/threads;
  
  checkCudaErrors(cudaMalloc(&d_intermediate, blocks * sizeof(float)));
  
  Hillis_Steele_Scan_Kernel<<<blocks, threads, 2 * threads * sizeof(float)>>>
    (d_out, d_intermediate, d_in);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  threads = blocks;
  blocks = 1;
  Blelloch_Scan_Kernel<<<blocks, threads, threads * sizeof(float)>>>
    (d_intermediate, d_intermediate, d_intermediate);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  threads = maxThreadsPerBlock;
  blocks = size/threads;
  Sum_Kernel<<<blocks, threads, threads * sizeof(float)>>>
    (d_out, d_intermediate, d_out);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  cudaFree(d_intermediate);
}

int main(int argc, char** argv) {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    std::cerr << "error: no devices supporting CUDA.\n";
    exit(EXIT_FAILURE);
  }
  int dev = 0;
  cudaSetDevice(dev);
  
  cudaDeviceProp devProps;
  if (cudaGetDeviceProperties(&devProps, dev) == 0) {
    printf("Using device %d:\n", dev);
    printf("%S; compute v%d.%d; clock: %dkHz\n",
           devProps.name, (int)devProps.major,
           (int)devProps.minor, (int)devProps.clockRate);
  }
  
  const int ARRAY_SIZE = 1 << 20;
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
  
  float h_in[ARRAY_SIZE];
  for (int i = 0; i < ARRAY_SIZE; i++) {
    h_in[i] = (float)random();
  }
  
  float *d_in, *d_out;
  
  checkCudaErrors(cudaMalloc(&d_in, ARRAY_BYTES));
  checkCudaErrors(cudaMalloc(&d_out, ARRAY_BYTES));
  
  checkCudaErrors(cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice));
  
  scan(d_out, d_in, ARRAY_SIZE);
  return 0;
}
