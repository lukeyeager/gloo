#include <cstdlib>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include "cuda_runtime.h"
#include "mpi.h"
#include "nccl.h"

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

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
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

  // initialize MPI
  int mpiRank, mpiSize;
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpiSize));

  printf("mpiRank=%d devices:", mpiRank);
  for (int i=0; i<numDevices; i++) {
    printf(" %d", devices[i]);
  }
  printf("\n");

  // send numDevices to other nodes
  int devicesPerRank[mpiSize];
  devicesPerRank[mpiRank] = numDevices;
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
        devicesPerRank, 1, MPI_INT, MPI_COMM_WORLD));

  // calculate rank for first nccl comm
  int ncclRankStart = 0;
  int ncclSize = 0;
  for (int i=0; i<mpiSize; ++i) {
    ncclSize += devicesPerRank[i];
    if (i < mpiRank) {
      ncclRankStart += devicesPerRank[i];
    }
  }

  // generate NCCL unique ID in one process and broadcast to all
  ncclUniqueId ncclID;
  if (mpiRank == 0) ncclGetUniqueId(&ncclID);
  MPICHECK(MPI_Bcast((void *)&ncclID, sizeof(ncclID), MPI_BYTE, 0, MPI_COMM_WORLD));

  // initialize NCCL
  ncclComm_t ncclComms[numDevices];
  ncclGroupStart();
  for (int i=0; i<numDevices; ++i) {
    CUDACHECK(cudaSetDevice(devices[i]));
    NCCLCHECK(ncclCommInitRank(&ncclComms[i], ncclSize, ncclID, ncclRankStart + i));
  }
  ncclGroupEnd();

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

  // call NCCL
  ncclGroupStart();
  for (int i=0; i<numDevices; ++i) {
    NCCLCHECK(ncclAllReduce((const void*)sendBuffers[i], (void*)recvBuffers[i],
          dataSize, ncclInt8, ncclSum, ncclComms[i], streams[i]));
  }
  ncclGroupEnd();

  // synchronize on CUDA stream to complete NCCL communication
  for (int i=0; i<numDevices; ++i) {
    CUDACHECK(cudaStreamSynchronize(streams[i]));
  }

  // verify results
  bool success = true;
  for (int i=0; i<numDevices; ++i) {
    int8_t result;
    CUDACHECK(cudaMemcpyAsync(&result, recvBuffers[i],
          sizeof(int8_t), cudaMemcpyDeviceToHost, streams[i]));
    CUDACHECK(cudaStreamSynchronize(streams[i]));
    printf("ncclRank=%d value=%d\n", ncclRankStart + i, result);
    if (result != ncclSize) {
      success = false;
    }
  }

  // finalize NCCL
  for (int i=0; i<numDevices; ++i) {
    ncclCommDestroy(ncclComms[i]);
  }

  // free buffers and streams
  for (int i=0; i<numDevices; ++i) {
    CUDACHECK(cudaFree(sendBuffers[i]));
    CUDACHECK(cudaFree(recvBuffers[i]));
    CUDACHECK(cudaStreamDestroy(streams[i]));
  }

  // finalize MPI
  MPICHECK(MPI_Finalize());

  printf("[MPI Rank %d] Success: %d\n", mpiRank, success);
  return !success;
}
