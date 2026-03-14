/*
 * Helix All-to-All — public C++ header (stub for Phase 1).
 *
 * Real implementations will be ported from TRT-LLM in Phase 2.
 */
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace helix_a2a {
namespace kernels {

struct HelixFieldInfo {
  uint8_t* dataPtr;
  int elementCount;
  int elementSize;
  int stride;
};

struct HelixAllToAllParams {
  HelixFieldInfo sendFields[2];
  HelixFieldInfo recvFields[2];
  int entryCount;
  uint64_t* workspace;
  size_t workspaceStrideInU64;
  int cpRank;
  int cpSize;
  int channelCount;
  int maxChannelCount;
};

int computeHelixMaxChannelCount(int cpSize, int smCount = 0);

size_t computeHelixWorkspaceSizePerRank(int cpSize);

void initializeHelixWorkspace(uint64_t* workspace, int cpSize, cudaStream_t stream);

void launchHelixAllToAll(HelixAllToAllParams const& params, bool allowVariableField1,
                         cudaStream_t stream);

}  // namespace kernels
}  // namespace helix_a2a
