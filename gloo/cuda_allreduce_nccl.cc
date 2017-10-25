#include "gloo/cuda_allreduce_nccl.h"

#if GLOO_USE_NCCL

#include "gloo/common/error.h"
#include "gloo/cuda_private.h"

namespace gloo {

template <typename T>
CudaAllreduceNCCL<T>::CudaAllreduceNCCL(
    const std::shared_ptr<Context>& context,
    const std::vector<T*>& ptrs,
    const int count,
    const std::vector<cudaStream_t>& streams) : Algorithm(context) {
  // Populate ptrs_ and streams_
  bool newStream = true;
  if (streams.size() > 0) {
    GLOO_ENFORCE_EQ(streams.size(), ptrs.size());
    newStream = false;
  }
  for (auto i = 0; i < ptrs.size(); i++) {
    ptrs_.emplace_back(CudaDevicePointer<T>::create(ptrs[i], count));
    auto& ptr = ptrs_[-1];
    if (newStream) {
      streams_.emplace_back(CudaStream(ptr.getDeviceID()));
    } else {
      streams_.emplace_back(CudaStream(ptr.getDeviceID(), streams[i]));
    }
  }

  // Create NCCLElements
  std::vector<nccl::NCCLElement<T> > elements;
  for (auto i = 0; i < ptrs.size(); i++) {
    elements.push_back(nccl::NCCLElement<T>(
          ptrs_[i].range(0, count), streams_[i],
          ptrs_[i].range(0, count), streams_[i]));
  }

  // Create op
  op_ = make_unique<nccl::AllreduceOp<T>>(
      std::move(elements), CudaReductionFunction<T>::sum, context, true);
}

template <typename T>
void CudaAllreduceNCCL<T>::run() {
  op_->runAsync();
  op_->wait();
}

// Instantiate templates
#define INSTANTIATE_TEMPLATE(T)                                         \
template class CudaAllreduceNCCL<T>;

INSTANTIATE_TEMPLATE(int8_t);
INSTANTIATE_TEMPLATE(int32_t);
INSTANTIATE_TEMPLATE(int64_t);
INSTANTIATE_TEMPLATE(uint64_t);
INSTANTIATE_TEMPLATE(float);
INSTANTIATE_TEMPLATE(double);
INSTANTIATE_TEMPLATE(float16);

} // namespace gloo

#endif // GLOO_USE_NCCL
