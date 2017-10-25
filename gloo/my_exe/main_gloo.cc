#include <cstdlib>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include "cuda_runtime.h"

#include "gloo/cuda_allreduce_nccl.h"
#include "gloo/mpi/context.h"
#include "gloo/transport/tcp/device.h"

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

int main(int argc, char* argv[])
{
  // read devices from command-line args
  if (argc < 2) {
    printf("Usage: %s device [device ...]\n", argv[0]);
    return 1;
  }
  int numDevices = argc - 1;
  int devices[numDevices];
  for (int i=0; i<numDevices; ++i) {
    devices[i] = atoi(argv[i+1]);
  }

  // get gloo mpi context
  auto dev = gloo::transport::tcp::CreateDevice("localhost");
  auto context = gloo::mpi::Context::createManaged();
  context->connectFullMesh(dev);
  int mpiRank = context->rank;
  int mpiSize = context->size;

  printf("mpiRank=%d devices:", mpiRank);
  for (int i=0; i<numDevices; i++) {
    printf(" %d", devices[i]);
  }
  printf("\n");

  // allocate buffers and streams
  int8_t* sendBuffers[numDevices];
  int8_t* recvBuffers[numDevices];
  cudaStream_t streams[numDevices];
  int dataSize = 32*1024*1024;
  for (int i = 0; i < numDevices; ++i) {
    CUDACHECK(cudaSetDevice(devices[i]));
    CUDACHECK(cudaMalloc(&sendBuffers[i], dataSize * sizeof(int8_t)));
    CUDACHECK(cudaMalloc(&recvBuffers[i], dataSize * sizeof(int8_t)));
    CUDACHECK(cudaMemset(sendBuffers[i], 1, dataSize * sizeof(int8_t)));
    CUDACHECK(cudaMemset(recvBuffers[i], 0, dataSize * sizeof(int8_t)));
    CUDACHECK(cudaStreamCreate(&streams[i]));
  }

  // run NCCL
  std::vector<int8_t*> ptrs;
  for (int i = 0; i < numDevices; ++i) {
    ptrs.push_back(sendBuffers[i]);
  }
  gloo::CudaAllreduceNCCL<int8_t>(context, ptrs, dataSize).run();

  // verify results
  bool success = true;
  for (int i=0; i<numDevices; ++i) {
    int8_t result;
    CUDACHECK(cudaMemcpyAsync(&result, sendBuffers[i],
          sizeof(int8_t), cudaMemcpyDeviceToHost, streams[i]));
    CUDACHECK(cudaStreamSynchronize(streams[i]));
    printf("ncclRank=%d value=%d\n", mpiRank * numDevices + i, result);
    if (result != mpiSize * numDevices) {
      success = false;
    }
  }

  // free buffers and streams
  for (int i=0; i<numDevices; ++i) {
    CUDACHECK(cudaFree(sendBuffers[i]));
    CUDACHECK(cudaFree(recvBuffers[i]));
    CUDACHECK(cudaStreamDestroy(streams[i]));
  }

  printf("[MPI Rank %d] Success: %d\n", mpiRank, success);
  return !success;
}
