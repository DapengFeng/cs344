#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <algorithm>

#include <cuda_runtime.h>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

__global__ void minmax_reduce_kernel(float * d_out, const float * d_in, bool minmax){
  // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmen>>>
  extern __shared__ float sdata[];
  
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;
  
  // load shared mem from global mem
  sdata[tid] = d_in[myId];
  __syncthreads(); // make sure entire block is loaded!
  
  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (minmax) {
        sdata[tid] = (sdata[tid] > sdata[tid + s]) ? sdata[tid] : sdata[tid + s];
      }
      else {
        sdata[tid] = (sdata[tid] < sdata[tid + s]) ? sdata[tid] : sdata[tid + s];
      }
    }
    __syncthreads(); // make sure all compare at one stage are done!
  }
  
  // only thread 0 writes result for this block back to global mem
  if (tid == 0) {
    d_out[blockIdx.x] = sdata[0];    
  }
}

void minmax_reduce(float * d_out, float	* d_intermediate, float	* d_in, int size) {
  // assumes that size is not greater than maxThreadsPerBlock^2
  // and that size is multiple of maxThreadsPerBlock
  const int maxThreadsPerBlock = 1024;
  int threads = maxThreadsPerBlock;
  int blocks = size/maxThreadsPerBlock;
  minmax_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>
    (d_intermediate, d_in,1);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  threads = blocks;
  blocks = 1;
  minmax_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>
    (&d_out[1], d_intermediate, 1);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
  threads = maxThreadsPerBlock;
  blocks = size/maxThreadsPerBlock;
  minmax_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>
    (d_intermediate, d_in,0);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  threads = blocks;
  blocks = 1;
  minmax_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>
    (&d_out[0], d_intermediate, 0);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

int main(int argc, char** argv) {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    fprintf(stderr, "error: no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }
  int dev = 0;
  cudaSetDevice(dev);
  
  cudaDeviceProp devProps;
  if (cudaGetDeviceProperties(&devProps, dev) == 0) {
    printf("Using device %d:\n", dev);
    printf("%s; global mem: %dB; compute v%d.%d; clock: %dkHz\n",
           devProps.name, (int)devProps.totalGlobalMem,
           (int)devProps.major, (int)devProps.minor,
           (int)devProps.clockRate);
  }
  
  const int ARRAY_SIZE = 1 << 20;
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
  
  // generate the input array on the host
  float h_in[ARRAY_SIZE];
  for (int i = 0; i < ARRAY_SIZE; i++) {
    // generate random float in [0.0f ARRAY_SIZE]
    h_in[i] = (float)random()/((float)RAND_MAX/(float)ARRAY_SIZE);
    /*if (i % (1 << 15))
      printf("%f\n",h_in[i]);*/
  }
  
  // declare GPU memory pointers
  float * d_in, * d_intermediate, * d_out;
  
  // allocate GPU memory
  checkCudaErrors(cudaMalloc(&d_in, ARRAY_BYTES));
  checkCudaErrors(cudaMalloc(&d_intermediate, ARRAY_BYTES)); // overallocated
  checkCudaErrors(cudaMalloc(&d_out, 2 * sizeof(float)));
  
  // transfer the input array to the GPU
  checkCudaErrors(cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice));
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  // launch the kernel
  
  cudaEventRecord(start, 0);
  for (int i = 0; i < 1000; i++) {
    minmax_reduce(d_out, d_intermediate, d_in, ARRAY_SIZE);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  elapsedTime /= 1000.0f; // 1000 trials
  
  // copy back the min and max from GPU
  float h_out[2];
  checkCudaErrors(cudaMemcpy(h_out, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost));
  
  printf("average time elapsed: %fms\n", elapsedTime);
  
  float ans[2];
  std::sort(h_in, h_in + ARRAY_SIZE);
  ans[0] = h_in[0];
  ans[1] = h_in[ARRAY_SIZE - 1];
  for (int i = 0; i < 2; i++) {
    assert(h_out[i] == ans[i]);
  }
  printf("Correct!\n");
  printf("Min: %f Max: %f\n", h_out[0], h_out[1]);
  return 0;
}
