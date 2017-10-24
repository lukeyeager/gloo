#pragma once

#include "gloo/nccl/nccl.h"

#if NCCL_VERSION_MIN(2,0,0)

namespace gloo {
namespace nccl {

template <typename T>
class NCCLDistributedContext : public NCCLContext<T> {
 public:
  NCCLDistributedContext(const std::vector<int>& devices,
      const std::shared_ptr<Context>& context);

  static std::shared_ptr<NCCLDistributedContext<T>> getCached(
      const NCCLExecution<T>& ex, const std::shared_ptr<Context>& context);
};

} // namespace nccl
} // namespace gloo

#endif // NCCL_VERSION
