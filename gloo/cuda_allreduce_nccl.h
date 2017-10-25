/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "gloo/algorithm.h"
#include "gloo/cuda.h"
#include "gloo/cuda_workspace.h"
#include "gloo/nccl/nccl.h"

namespace gloo {

class NCCLCommList {
 public:
  NCCLCommList(const std::shared_ptr<Context>& context,
      const std::vector<int> localDevices);
  ~NCCLCommList();
  std::vector<ncclComm_t> comms;
};

template <typename T>
class CudaAllreduceNCCL : public Algorithm {
 public:
  CudaAllreduceNCCL(
      const std::shared_ptr<Context>& context,
      const std::vector<T*>& ptrs,
      const int count,
      const std::vector<cudaStream_t>& streams = std::vector<cudaStream_t>());

  virtual void run() override;

 protected:
  std::vector<CudaDevicePointer<T> > ptrs_;
  std::vector<CudaStream> streams_;
  const int count_;

  std::shared_ptr<NCCLCommList> commList_;
};

} // namespace gloo
