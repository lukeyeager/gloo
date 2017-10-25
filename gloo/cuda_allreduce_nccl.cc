#include "gloo/cuda_allreduce_nccl.h"

#if GLOO_USE_NCCL

#include "gloo/common/error.h"
#include "gloo/cuda_private.h"

namespace gloo {

template <typename T>
std::vector<nccl::NCCLElement<T> > toDeviceElements(
    const std::vector<T*>& ptrs, const int count,
    std::vector<CudaStream>& streams) {
  std::vector<nccl::NCCLElement<T> > elements;
  for (auto i = 0; i < ptrs.size(); i++) {
    elements.push_back(nccl::NCCLElement<T>(
          CudaDevicePointer<T>::create(ptrs[i], count), streams[i],
          CudaDevicePointer<T>::create(ptrs[i], count), streams[i]));
  }
  return elements;
}

template <typename T>
CudaAllreduceNCCL<T>::CudaAllreduceNCCL(
    const std::shared_ptr<Context>& context,
    const std::vector<T*>& ptrs,
    const int count,
    const std::vector<cudaStream_t>& streams) : Algorithm(context) {
  // Create streams
  GLOO_ENFORCE(streams.size() == 0, "Setting streams not supported");
  for (auto i = 0; i < ptrs.size(); i++) {
    streams_.push_back(CudaStream(getGPUIDForPointer(ptrs[i])));
  }
  // Create op
  op_ = make_unique<nccl::AllreduceOp<T>>(
      toDeviceElements(ptrs, count, streams_),
      CudaReductionFunction<T>::sum, context, true);
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
