# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-GPU tests for DCP LL128 FIFO All-to-All with MNNVL workspace.

Complements test_dcp_alltoall.py (single-GPU simulation) by running
real multi-process, multi-GPU alltoall via MPI. This catches bugs
that single-GPU tests cannot:
  - MNNVL workspace allocation and communicator grouping
  - Cross-GPU memory visibility (LL128 FIFO writes to peer memory)
  - Workspace shape [cp_size, ws_elems] with real multi-rank segments

Run:
  mpirun -np 2 pytest tests/comm/test_mnnvl_dcp_alltoall.py -v -s
  mpirun -np 4 pytest tests/comm/test_mnnvl_dcp_alltoall.py -v -s
"""

import socket
import traceback

import pynvml
import pytest
import torch
from mpi4py import MPI

from flashinfer.comm import (
    dcp_a2a_alltoall,
    dcp_a2a_allocate_workspace,
    dcp_a2a_init_workspace,
    dcp_a2a_workspace_size,
)
from flashinfer.comm.mapping import Mapping
from flashinfer.comm.mnnvl import MnnvlMemory, MpiComm

from .conftest import mnnvl_available

pynvml.nvmlInit()


# ─── SM90+ gate ──────────────────────────────────────────────────────────


def _sm90_available() -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability(0)
        return major >= 9
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(
        not _sm90_available(),
        reason="Requires SM90+ GPU (Hopper/Blackwell)",
    ),
    pytest.mark.skipif(
        not mnnvl_available(),
        reason="MNNVL not supported on this platform or container lacks SYS_PTRACE",
    ),
]


# ─── MPI helpers ─────────────────────────────────────────────────────────


class MPIExit(Exception):
    pass


def check_any_rank_failed():
    comm = MPI.COMM_WORLD
    if any(comm.allgather(False)):
        raise MPIExit("Another rank failed")


def safe_run(func, *args, **kwargs):
    comm = MPI.COMM_WORLD
    try:
        func(*args, **kwargs)
    except MPIExit:
        raise
    except Exception:
        traceback.print_exc()
        comm.allgather(True)
        raise


# ─── Helper ──────────────────────────────────────────────────────────────


def _to_torch(t):
    """Convert a tvm_ffi.core.Tensor (or any DLPack object) to torch.Tensor."""
    if isinstance(t, torch.Tensor):
        return t
    return torch.from_dlpack(t)


def _setup_rank():
    """Initialize MPI rank and CUDA device. Returns (rank, world_size, comm)."""
    comm = MpiComm()
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    # Get local rank from hostname
    hostname = socket.gethostname()
    all_hostnames = comm.allgather(hostname)
    local_ranks_before_me = sum(1 for i in range(rank) if all_hostnames[i] == hostname)
    local_rank = local_ranks_before_me
    torch.cuda.set_device(local_rank)

    return rank, world_size, comm


def _allocate_mnnvl_workspace(rank, cp_size, comm):
    """Allocate MNNVL workspace for DCP A2A, grouping CP peers.

    Sets MnnvlMemory.comm directly to avoid set_comm_from_config's
    MoE-oriented split (which groups TP peers, not CP peers).
    """
    MnnvlMemory.initialize()
    MnnvlMemory.comm = comm

    mapping = Mapping(
        world_size=cp_size,
        rank=rank,
        cp_size=cp_size,
        tp_size=1,
        pp_size=1,
    )

    ws_bytes = dcp_a2a_workspace_size(cp_size)
    mnnvl_mem = MnnvlMemory(mapping, ws_bytes)
    workspace = mnnvl_mem.as_torch_strided_tensor(torch.int64)
    workspace._mnnvl_mem = mnnvl_mem  # prevent GC

    return workspace


# ─── Tests ───────────────────────────────────────────────────────────────


class TestMnnvlDcpWorkspace:
    """Test MNNVL workspace allocation for DCP A2A."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.rank, self.cp_size, self.comm = _setup_rank()
        torch.manual_seed(0xA2A + self.rank)
        yield

    def test_workspace_shape(self):
        """MNNVL workspace must have shape [cp_size, ws_elems_per_rank]."""
        workspace = _allocate_mnnvl_workspace(self.rank, self.cp_size, self.comm)
        assert workspace.shape[0] == self.cp_size, (
            f"Expected workspace.shape[0] == {self.cp_size}, got {workspace.shape[0]}"
        )

        ws_bytes = dcp_a2a_workspace_size(self.cp_size)
        expected_elems = (ws_bytes + 7) // 8  # int64 elements
        assert workspace.shape[1] == expected_elems
        assert workspace.dtype == torch.int64

        self.comm.Barrier()

    def test_workspace_cross_rank_visible(self):
        """Each rank can write to its own segment and peers can read it."""
        workspace = _allocate_mnnvl_workspace(self.rank, self.cp_size, self.comm)

        # Each rank writes a unique pattern to its own workspace segment
        pattern = torch.full_like(workspace[self.rank], fill_value=self.rank + 1)
        workspace[self.rank].copy_(pattern)
        torch.cuda.synchronize()
        self.comm.Barrier()

        # Each rank reads all segments and verifies the pattern
        for peer in range(self.cp_size):
            expected = peer + 1
            actual = workspace[peer][0].item()
            assert actual == expected, (
                f"Rank {self.rank}: workspace[{peer}][0] = {actual}, "
                f"expected {expected}"
            )

        self.comm.Barrier()


class TestMnnvlDcpAlltoall:
    """Multi-GPU correctness tests for DCP A2A alltoall."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.rank, self.cp_size, self.comm = _setup_rank()
        torch.manual_seed(0xA2A)
        yield

    def _run_alltoall(self, batch_size, head_dim, stats_dim, dtype):
        """Run DCP A2A alltoall on real multi-GPU and verify correctness.

        The transpose property must hold:
          recv_o[rank][.., peer, :] == send_o[peer][.., rank, :]
          recv_s[rank][.., peer, :] == send_s[peer][.., rank, :]
        """
        rank = self.rank
        cp_size = self.cp_size

        workspace = _allocate_mnnvl_workspace(rank, cp_size, self.comm)

        dcp_a2a_init_workspace(workspace, rank, cp_size)
        torch.cuda.synchronize()
        self.comm.Barrier()

        # Generate input with deterministic seed per rank
        torch.manual_seed(0xA2A + rank)
        partial_o = torch.randn(
            batch_size, cp_size, head_dim, dtype=dtype, device="cuda"
        )
        softmax_stats = torch.randn(
            batch_size, cp_size, stats_dim, dtype=torch.float32, device="cuda"
        )

        # Run alltoall
        recv_o, recv_s = dcp_a2a_alltoall(
            partial_o, softmax_stats, workspace, rank, cp_size
        )
        recv_o = _to_torch(recv_o)
        recv_s = _to_torch(recv_s)
        torch.cuda.synchronize()
        self.comm.Barrier()

        # Gather all inputs to all ranks for verification
        all_partial_o = self.comm.allgather(partial_o.cpu())
        all_softmax_stats = self.comm.allgather(softmax_stats.cpu())

        # Verify transpose property
        for peer in range(cp_size):
            expected_o = all_partial_o[peer][..., rank, :].cuda()
            expected_s = all_softmax_stats[peer][..., rank, :].cuda()

            torch.testing.assert_close(
                recv_o[..., peer, :],
                expected_o,
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                recv_s[..., peer, :],
                expected_s,
                atol=0,
                rtol=0,
            )

        self.comm.Barrier()

    @pytest.mark.parametrize(
        "batch_size,head_dim,stats_dim,dtype",
        [
            pytest.param(1, 128, 2, torch.bfloat16, id="B1-D128-S2-bf16"),
            pytest.param(16, 128, 2, torch.bfloat16, id="B16-D128-S2-bf16"),
            pytest.param(128, 128, 2, torch.bfloat16, id="B128-D128-S2-bf16"),
            pytest.param(16, 256, 4, torch.bfloat16, id="B16-D256-S4-bf16"),
            pytest.param(16, 128, 2, torch.float16, id="B16-D128-S2-fp16"),
        ],
    )
    def test_alltoall_correctness(self, batch_size, head_dim, stats_dim, dtype):
        """Verify transpose property across real GPUs."""
        self._run_alltoall(batch_size, head_dim, stats_dim, dtype)

    def test_repeated_alltoall(self):
        """Multiple alltoall calls on the same workspace (FIFO reuse)."""
        rank = self.rank
        cp_size = self.cp_size

        workspace = _allocate_mnnvl_workspace(rank, cp_size, self.comm)
        dcp_a2a_init_workspace(workspace, rank, cp_size)
        torch.cuda.synchronize()
        self.comm.Barrier()

        for round_idx in range(3):
            torch.manual_seed(0xA2A + rank * 100 + round_idx)
            partial_o = torch.randn(
                16, cp_size, 128, dtype=torch.bfloat16, device="cuda"
            )
            softmax_stats = torch.randn(
                16, cp_size, 2, dtype=torch.float32, device="cuda"
            )

            recv_o, recv_s = dcp_a2a_alltoall(
                partial_o, softmax_stats, workspace, rank, cp_size
            )
            recv_o = _to_torch(recv_o)
            recv_s = _to_torch(recv_s)
            torch.cuda.synchronize()
            self.comm.Barrier()

            all_partial_o = self.comm.allgather(partial_o.cpu())
            all_softmax_stats = self.comm.allgather(softmax_stats.cpu())

            for peer in range(cp_size):
                torch.testing.assert_close(
                    recv_o[..., peer, :],
                    all_partial_o[peer][..., rank, :].cuda(),
                    atol=0,
                    rtol=0,
                )
                torch.testing.assert_close(
                    recv_s[..., peer, :],
                    all_softmax_stats[peer][..., rank, :].cuda(),
                    atol=0,
                    rtol=0,
                )

            self.comm.Barrier()


class TestMnnvlDcpDeviceMemoryFallback:
    """Test that non-MNNVL (device memory) path also works multi-GPU.

    Uses dcp_a2a_allocate_workspace without MNNVL mapping. This only
    works when all ranks are on the same GPU (single-GPU simulation)
    or with IPC. Included here to verify the workspace API contract.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.rank, self.cp_size, self.comm = _setup_rank()
        torch.manual_seed(0xA2A)
        yield

    def test_device_workspace_shape(self):
        """Device workspace has correct shape [cp_size, ws_elems]."""
        workspace = dcp_a2a_allocate_workspace(self.cp_size, cp_rank=self.rank)
        assert workspace.shape[0] == self.cp_size

        ws_bytes = dcp_a2a_workspace_size(self.cp_size)
        expected_elems = (ws_bytes + 7) // 8
        assert workspace.shape[1] == expected_elems
        assert workspace.dtype == torch.int64

        self.comm.Barrier()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
