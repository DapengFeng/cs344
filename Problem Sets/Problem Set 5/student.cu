/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"

const int K = 128;
const int numStreams = 2;

__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               unsigned int numVals)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible
  __shared__ unsigned int sdata[K];
  
  unsigned int myId = threadIdx.x + blockIdx.x * blockDim.x;
  int id = threadIdx.x;
  
  if (myId >= numVals) {
    return;
  }
  
  sdata[id] = vals[myId];
  __syncthreads();
  
  atomicAdd(&histo[sdata[id]], 1);
  //Although we provide only one kernel skeleton,
  //feel free to use more if it will help you
  //write faster code
}

__global__
void mergeHisto(unsigned int* histo1,
                unsigned int* histo2,
                unsigned int* const histo
                )
{
  __shared__ unsigned int sdata1[K];
  __shared__ unsigned int sdata2[K];
  
  unsigned int myId = threadIdx.x + blockIdx.x * blockDim.x;
  int id = threadIdx.x;
  
  sdata1[id] = histo1[myId];
  sdata2[id] = histo2[myId];
  __syncthreads();
  
  sdata1[id] += sdata2[id];
  histo[myId] = sdata1[id];
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  //TODO Launch the yourHisto kernel
  cudaStream_t s1,s2;
  const int numElemsPerStream =  numElems / numStreams;
  dim3 threads(K);
  dim3 blocks(numElemsPerStream / K);
  
  unsigned int *d_vals1, *d_vals2;
  //unsigned int *d_histo1, *d_histo2;
  
  checkCudaErrors(cudaMalloc(&d_vals1, numElemsPerStream * sizeof(unsigned int)));
  //checkCudaErrors(cudaMalloc(&d_histo1, numElemsPerStream * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_vals2, numElemsPerStream * sizeof(unsigned int)));
  //checkCudaErrors(cudaMalloc(&d_histo2, numElemsPerStream * sizeof(unsigned int)));
  
  checkCudaErrors(cudaStreamCreate(&s1));
  checkCudaErrors(cudaStreamCreate(&s2));
  
  checkCudaErrors(cudaMemcpyAsync(d_vals1, d_vals, numElemsPerStream * sizeof(unsigned int), cudaMemcpyDeviceToDevice, s1));
  yourHisto<<<blocks,threads,0,s1>>>(d_vals1, d_histo, numElemsPerStream);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  checkCudaErrors(cudaMemcpyAsync(d_vals2, d_vals+numElemsPerStream, numElemsPerStream * sizeof(unsigned int), cudaMemcpyDeviceToDevice, s2));
  yourHisto<<<blocks,threads,0,s2>>>(d_vals2, d_histo, numElemsPerStream);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //if you want to use/launch more than one kernel,
  //feel free
  //mergeHisto<<<numBins / K, K>>>(d_histo1, d_histo2,d_histo);
  //cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  //yourHisto<<<numElems/K, threads>>>(d_vals,d_histo, numElems);
  
  checkCudaErrors(cudaStreamDestroy(s1));
  checkCudaErrors(cudaStreamDestroy(s2));
  
  cudaFree(d_vals1);
  cudaFree(d_vals2);
  //cudaFree(d_histo1);
  //cudaFree(d_histo2);
}
