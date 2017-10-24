#include "gloo/nccl/nccl_distributed.h"

#if NCCL_VERSION_MIN(2,0,0)

#include "gloo/broadcast_one_to_all.h"
#include "gloo/cuda_private.h"

#include <unordered_map>

namespace gloo {
namespace nccl {

template <typename T>
NCCLDistributedContext<T>::NCCLDistributedContext(
    const std::vector<int>& devices, const std::shared_ptr<Context>& context)
  : NCCLContext<T>(devices)
{
  // Generate unique ID on root node
  ncclUniqueId id;
  if (context->rank == 0) {
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    ncclGetUniqueId(&id);
  }

  // Broadcast ID to all nodes
  std::vector<int8_t*> ids;
  ids.push_back((int8_t*)id.internal);
  BroadcastOneToAll<int8_t>(context, ids, NCCL_UNIQUE_ID_BYTES).run();

  // Assumes all nodes have the same number of devices
  int ncclWorldSize = context->size * devices.size();
  int ncclLocalRankStart = context->rank * devices.size();

  // Initialize comms
  this->comms.resize(devices.size());
  {
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    NCCL_CHECK(ncclGroupStart());
    for (int i=0; i<devices.size(); i++) {
      CUDA_CHECK(cudaSetDevice(devices[i]));
      NCCL_CHECK(ncclCommInitRank(&this->comms[i], ncclWorldSize, id,
            ncclLocalRankStart + i));
    }
    NCCL_CHECK(ncclGroupEnd());
  }
}

// Initializing NCCL communications is expensive. Allocate context as needed per
// unique (rank, size, devices) and cache for reuse.
template <typename T>
std::shared_ptr<NCCLDistributedContext<T>> NCCLDistributedContext<T>::getCached(
    const NCCLExecution<T>& ex, const std::shared_ptr<Context>& context) {
  static std::unordered_map<std::string,
    std::shared_ptr<NCCLDistributedContext<T>>> contexts;
  const std::string key = "size" + std::to_string(context->size) +
    ",rank" + std::to_string(context->rank) + "," + ex.getKey();
  {
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    if (!contexts[key]) {
      contexts[key] = std::make_shared<NCCLDistributedContext<T>>(ex.getDevices(), context);
    }
  }
  const auto nccl_context = contexts[key];
  GLOO_ENFORCE_NE(nccl_context.get(), (void*)nullptr);
  return nccl_context;
}

#define DEFINE_NCCL_TYPES_AND_OPS(T)                                    \
template class NCCLDistributedContext<T>;

DEFINE_NCCL_TYPES_AND_OPS(int8_t);
DEFINE_NCCL_TYPES_AND_OPS(int32_t);
DEFINE_NCCL_TYPES_AND_OPS(int64_t);
DEFINE_NCCL_TYPES_AND_OPS(uint64_t);
DEFINE_NCCL_TYPES_AND_OPS(float16);
DEFINE_NCCL_TYPES_AND_OPS(float);
DEFINE_NCCL_TYPES_AND_OPS(double);

} // namespace nccl
} // namespace gloo

#endif // NCCL_VERSION
