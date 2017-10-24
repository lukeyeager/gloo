#pragma once

#include "gloo/config.h"

#if GLOO_USE_NCCL

#include "gloo/algorithm.h"
#include "gloo/cuda.h"
#include "gloo/cuda_workspace.h"
#include "gloo/nccl/nccl.h"

#include <memory>

namespace gloo {

template <typename T, typename W = CudaHostWorkspace<T> >
class CudaAllreduceNCCL : public Algorithm {
 public:
  CudaAllreduceNCCL(
      const std::shared_ptr<Context>& context,
      const std::vector<T*>& ptrs,
      const int count,
      const std::vector<cudaStream_t>& streams = std::vector<cudaStream_t>());

  virtual ~CudaAllreduceNCCL() = default;

  virtual void run() override;

 protected:
  std::vector<CudaStream> streams_;
  std::unique_ptr<nccl::NCCLOp<T>> op_;
};

} // namespace gloo

#endif // GLOO_USE_NCCL
