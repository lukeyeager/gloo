/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/cuda_allreduce_nccl.h"

#include "gloo/broadcast_one_to_all.h"
#include "gloo/cuda_private.h"

#include <unordered_map>

namespace gloo {

namespace {

// Creating NCCL communicators is expensive. So we cache and reuse them.
static std::shared_ptr<NCCLCommList> getCachedCommList(
    const std::shared_ptr<Context>& context,
    const std::vector<int> localDevices)
{
  static std::unordered_map<std::string, std::shared_ptr<NCCLCommList> >
    commLists;

  // generate key
  const int numDevices = localDevices.size();
  std::string key = std::to_string(context->size) + ' ' +
    std::to_string(context->rank);
  for (auto i = 0; i < numDevices; ++i) {
    key += ' ' + std::to_string(localDevices[i]);
  }

  // get or create CommList
  {
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    if (!commLists[key]) {
      commLists[key] = std::make_shared<NCCLCommList>(context, localDevices);
    }
  }

  const auto commList = commLists[key];
  GLOO_ENFORCE_NE(commList.get(), (void*)nullptr);
  return commList;
}

} // namespace

NCCLCommList::NCCLCommList(const std::shared_ptr<Context>& context,
    const std::vector<int> localDevices) {
  // generate unique ID on root node
  ncclUniqueId id;
  std::vector<int8_t*> ids;
  ids.push_back((int8_t*)id.internal);
  if (context->rank == 0) {
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    ncclGetUniqueId(&id);
  }

  // broadcast ID to other nodes
  BroadcastOneToAll<int8_t>(context, ids, NCCL_UNIQUE_ID_BYTES).run();

  // create comms
  // FIXME currently, we assume all ranks use the same number of devices
  const int numDevices = localDevices.size();
  const int ncclSize = context->size * numDevices;
  const int ncclRankStart = context->rank * numDevices;
  comms.reserve(numDevices);
  {
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    NCCL_CHECK(ncclGroupStart());
    for (auto i = 0; i < numDevices; ++i) {
      CudaDeviceScope scope(localDevices[i]);
      NCCL_CHECK(ncclCommInitRank(&comms[i], ncclSize, id,
            ncclRankStart + i));
    }
    NCCL_CHECK(ncclGroupEnd());
  }
}

NCCLCommList::~NCCLCommList() {
  for (auto i = 0; i < comms.size(); ++i) {
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    ncclCommDestroy(comms[i]);
  }
}

template <typename T>
CudaAllreduceNCCL<T>::CudaAllreduceNCCL(
  const std::shared_ptr<Context>& context,
  const std::vector<T*>& ptrs,
  const int count,
  const std::vector<cudaStream_t>& streams)
    : Algorithm(context),
      count_(count) {
  // populate ptrs_ and streams_
  auto newStream = true;
  if (streams.size() > 0) {
    GLOO_ENFORCE_EQ(streams.size(), ptrs.size());
    newStream = false;
  }
  std::vector<int> localDevices(ptrs.size());
  for (auto i = 0; i < ptrs.size(); i++) {
    auto ptr = CudaDevicePointer<T>::create(ptrs[i], count_);
    const auto device = ptr.getDeviceID();
    localDevices[i] = device;
    if (newStream) {
      streams_.push_back(CudaStream(device));
    } else {
      streams_.push_back(CudaStream(device, streams[i]));
    }
    ptrs_.emplace_back(std::move(ptr));
  }

  // get comms
  commList_ = getCachedCommList(context, localDevices);
}

template <typename T>
void CudaAllreduceNCCL<T>::run() {
  {
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    NCCL_CHECK(ncclGroupStart());
    for (auto i = 0; i < ptrs_.size(); ++i) {
      NCCL_CHECK(ncclAllReduce(
            (const void*)(*ptrs_[i]), (void*)(*ptrs_[i]), count_,
            nccl::ncclTypeWrapper<T>::type, ncclSum,
            commList_->comms[i], *streams_[i]));
    }
    NCCL_CHECK(ncclGroupEnd());
  }

  for (auto i = 0; i < ptrs_.size(); ++i)
    CUDA_CHECK(cudaStreamSynchronize(*streams_[i]));
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
