import asyncio
import logging
import math
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from queue import Empty, Queue
from typing import List, Optional, Tuple

import torch

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_embedding_store import (
    MooncakeEmbeddingStore,
)

logger = logging.getLogger(__name__)

TARGET_PAGE_BYTES = 256 * 1024
VISION_POOL_RATIO = 0.8


def _dtype_element_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def compute_page_size_tokens(dim: int, element_size: int = 4) -> int:
    return max(TARGET_PAGE_BYTES // (dim * element_size), 1)


class EntryState(Enum):
    LOADING = auto()
    EVICTABLE = auto()
    PUTTING = auto()


class TokenPageAllocator:
    """LIFO page allocator for fixed-size token pages."""

    def __init__(self, num_pages: int):
        self.free_list: List[int] = list(reversed(range(num_pages)))

    def allocate(self, num_tokens: int, page_size_tokens: int) -> Optional[List[int]]:
        required_pages = math.ceil(num_tokens / page_size_tokens)
        if len(self.free_list) < required_pages:
            return None
        return [self.free_list.pop() for _ in range(required_pages)]

    def free(self, page_ids: List[int]):
        self.free_list.extend(reversed(page_ids))

    @property
    def free_pages(self) -> int:
        return len(self.free_list)


@dataclass
class EmbeddingPool:
    modality: str
    dim: int
    dtype: torch.dtype
    page_size_tokens: int
    tensor: torch.Tensor
    num_pages: int
    allocator: TokenPageAllocator
    page_bytes: int
    pool_size_bytes: int
    pin_memory: bool = True

    @classmethod
    def create(
        cls,
        modality: str,
        dim: int,
        pool_size_bytes: int,
        dtype: torch.dtype = torch.float32,
        pin_memory: bool = True,
    ) -> "EmbeddingPool":
        element_size = _dtype_element_size(dtype)
        page_size_tokens = compute_page_size_tokens(dim, element_size)
        capacity_tokens = pool_size_bytes // (dim * element_size)
        num_pages = capacity_tokens // page_size_tokens
        total_tokens = num_pages * page_size_tokens
        tensor = torch.empty(
            (total_tokens, dim),
            dtype=dtype,
            pin_memory=pin_memory,
        )
        page_bytes = page_size_tokens * dim * element_size
        return cls(
            modality=modality,
            dim=dim,
            dtype=dtype,
            page_size_tokens=page_size_tokens,
            tensor=tensor,
            num_pages=num_pages,
            allocator=TokenPageAllocator(num_pages),
            page_bytes=page_bytes,
            pool_size_bytes=pool_size_bytes,
            pin_memory=pin_memory,
        )

    def reinit(self, new_dim: int):
        fresh = EmbeddingPool.create(
            self.modality,
            new_dim,
            self.pool_size_bytes,
            dtype=self.dtype,
            pin_memory=self.pin_memory,
        )
        self.dim = fresh.dim
        self.page_size_tokens = fresh.page_size_tokens
        self.page_bytes = fresh.page_bytes
        self.tensor = fresh.tensor
        self.num_pages = fresh.num_pages
        self.allocator = fresh.allocator


@dataclass
class EmbeddingCacheEntry:
    hash: str
    modality: object
    num_tokens: int
    dim: int
    page_ids: List[int]
    state: EntryState
    ref_count: int = 0


def build_transfer_buffers(
    entry: EmbeddingCacheEntry, pool: EmbeddingPool
) -> Tuple[List[int], List[int]]:
    """Coalesce physically adjacent pages while preserving logical page order.

    Returns:
        (ptrs, sizes) - two parallel lists of buffer pointers and byte sizes.
    """
    if not entry.page_ids:
        return [], []

    ptrs: List[int] = []
    sizes: List[int] = []
    remaining_tokens = entry.num_tokens
    element_size = _dtype_element_size(pool.dtype)

    def flush(run_start: int, run_len: int):
        nonlocal remaining_tokens
        if remaining_tokens <= 0:
            return
        valid_tokens = min(pool.page_size_tokens * run_len, remaining_tokens)
        ptr = pool.tensor[run_start * pool.page_size_tokens].data_ptr()
        size_bytes = valid_tokens * entry.dim * element_size
        ptrs.append(ptr)
        sizes.append(size_bytes)
        remaining_tokens -= valid_tokens

    run_start = entry.page_ids[0]
    prev = run_start
    run_len = 1
    for page_id in entry.page_ids[1:]:
        if page_id == prev + 1:
            run_len += 1
        else:
            flush(run_start, run_len)
            run_start = page_id
            run_len = 1
        prev = page_id
    flush(run_start, run_len)
    return ptrs, sizes


class EmbeddingPrefetchOperation:
    """Groups all missing images of a request for a single batch GET."""

    def __init__(
        self,
        req_id: str,
        keys: List[str],
        ptrs: List[List[int]],
        sizes: List[List[int]],
    ):
        self.req_id = req_id
        self.keys = keys
        self.ptrs = ptrs
        self.sizes = sizes
        self.is_finished = False
        self.success = False
        self._lock = threading.Lock()

    def mark_done(self, success: bool):
        with self._lock:
            self.success = success
            self.is_finished = True


class EmbeddingInsertOperation:
    """Groups all newly computed images of a request for a single batch PUT."""

    def __init__(self, keys: List[str], ptrs: List[List[int]], sizes: List[List[int]]):
        self.keys = keys
        self.ptrs = ptrs
        self.sizes = sizes


class EmbeddingCacheController:
    def __init__(
        self,
        tp_rank,
        tp_size,
        max_pool_size_gb=4.0,
        hidden_dims: dict = None,
        tp_group=None,
        all_rank_get=False,
        enable_eviction: bool = True,
        max_eviction_batch: int = 100,
    ):
        self.tp_world_size = tp_size
        self.tp_group = tp_group
        self.tp_rank = tp_rank
        self.all_rank_get = all_rank_get
        self.hidden_dims = hidden_dims or {}
        self.dtype = torch.float32
        self.element_size = _dtype_element_size(self.dtype)
        self.enable_eviction = enable_eviction
        self.max_eviction_batch = max_eviction_batch

        self.mooncake_store = MooncakeEmbeddingStore()
        self.total_pool_size_bytes = int(max_pool_size_gb * 1024**3)
        self.vision_pool, self.audio_pool = self._create_pools(pin_memory=True)
        self.pools = {
            "vision": self.vision_pool,
            "audio": self.audio_pool,
        }
        self._retired_pool_tensors = []
        self._register_pool_buffer(self.vision_pool)
        self._register_pool_buffer(self.audio_pool)

        self.entries = {}
        self.access_order = {}
        self.access_lock = threading.Lock()

        self.stats = {
            "total_allocated": 0,
            "total_evicted": 0,
            "eviction_count": 0,
            "allocation_failures": 0,
        }

        self.ongoing_prefetch = {}
        self.prefetch_queue = Queue()
        self.insert_queue = Queue()

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.io_thread = threading.Thread(target=self._io_loop, daemon=True)
        self.io_thread.start()

        if self.tp_world_size > 1:
            if self.tp_group is None:
                raise ValueError("tp_group must be provided when tp_size > 1")
            from sglang.srt.distributed.parallel_state import (
                create_custom_parallel_group,
            )

            group_ranks = torch.distributed.get_process_group_ranks(self.tp_group)
            self.prefetch_tp_group = create_custom_parallel_group(
                group_ranks=group_ranks, backend="gloo"
            )
        else:
            self.prefetch_tp_group = None

    def _create_pools(self, pin_memory: bool) -> Tuple[EmbeddingPool, EmbeddingPool]:
        # vision pool uses IMAGE dim (IMAGE == VIDEO dim in all supported models)
        vision_dim = (
            self.hidden_dims.get(Modality.IMAGE)
            or self.hidden_dims.get(Modality.VIDEO)
            or 1
        )
        audio_dim = self.hidden_dims.get(Modality.AUDIO) or vision_dim
        vision_bytes = int(self.total_pool_size_bytes * VISION_POOL_RATIO)
        audio_bytes = self.total_pool_size_bytes - vision_bytes
        return (
            EmbeddingPool.create(
                "vision", vision_dim, vision_bytes, self.dtype, pin_memory
            ),
            EmbeddingPool.create(
                "audio", audio_dim, audio_bytes, self.dtype, pin_memory
            ),
        )

    def _register_pool_buffer(self, pool: EmbeddingPool):
        if pool.tensor.numel() == 0:
            logger.warning(
                f"[Rank {self.tp_rank}] {pool.modality} embedding pool has zero pages; "
                f"dim={pool.dim}, budget={pool.pool_size_bytes} bytes"
            )
            return
        self.mooncake_store.register_buffer(pool.tensor)
        logger.info(
            f"[Rank {self.tp_rank}] Registered {pool.modality} embedding pool: "
            f"dim={pool.dim}, pages={pool.num_pages}, "
            f"page_tokens={pool.page_size_tokens}, "
            f"capacity={pool.num_pages * pool.page_bytes / 1024**2:.2f} MB"
        )

    def _get_pool(self, modality: Modality) -> Optional[EmbeddingPool]:
        if modality == Modality.AUDIO:
            return self.audio_pool
        if modality in (Modality.IMAGE, Modality.VIDEO):
            return self.vision_pool
        return None

    def _update_access_time(self, image_hash: str):
        """Update LRU access time for a hash."""
        with self.access_lock:
            # Move to end (most recently used)
            if image_hash in self.access_order:
                del self.access_order[image_hash]
            self.access_order[image_hash] = time.time()

    def _evict_entry(self, image_hash: str):
        """Evict a single entry and free its pages.

        NOTE: Caller must hold self.lock.
        """
        entry = self.entries.get(image_hash)
        if entry is None:
            return
        pool = self._get_pool(entry.modality)
        pool.allocator.free(entry.page_ids)
        self.stats["total_evicted"] += entry.num_tokens * entry.dim * self.element_size
        del self.entries[image_hash]
        with self.access_lock:
            self.access_order.pop(image_hash, None)

    def _evict_for_pool(self, pool: EmbeddingPool, required_pages: int):
        """Evict LRU entries until pool has at least required_pages free.

        NOTE: Caller must hold self.lock.
        """
        if pool.allocator.free_pages >= required_pages:
            return
        with self.access_lock:
            lru_hashes = list(self.access_order.keys())
        evicted = 0
        for image_hash in lru_hashes:
            if pool.allocator.free_pages >= required_pages:
                break
            if evicted >= self.max_eviction_batch:
                break
            entry = self.entries.get(image_hash)
            if entry is None:
                with self.access_lock:
                    self.access_order.pop(image_hash, None)
                continue
            if self._get_pool(entry.modality) is not pool:
                continue
            if entry.state in (EntryState.LOADING, EntryState.PUTTING):
                continue
            if entry.ref_count > 0:
                continue
            self._evict_entry(image_hash)
            evicted += 1
        if evicted > 0:
            self.stats["eviction_count"] += 1
            logger.info(
                f"[Rank {self.tp_rank}] Evicted {evicted} embeddings from "
                f"{pool.modality} pool"
            )

    def _allocate_with_eviction(
        self, pool: EmbeddingPool, num_tokens: int
    ) -> Optional[List[int]]:
        """Allocate pages, evicting LRU entries from the same pool if needed.

        NOTE: Caller must hold self.lock.
        """
        required_pages = math.ceil(num_tokens / pool.page_size_tokens)
        if required_pages > pool.num_pages:
            self.stats["allocation_failures"] += 1
            return None

        if self.enable_eviction:
            self._evict_for_pool(pool, required_pages)

        page_ids = pool.allocator.allocate(num_tokens, pool.page_size_tokens)
        if page_ids is not None:
            self.stats["total_allocated"] += num_tokens * pool.dim * self.element_size
        else:
            self.stats["allocation_failures"] += 1
            logger.warning(
                f"[Rank {self.tp_rank}] Cannot allocate {required_pages} pages "
                f"in {pool.modality} pool: free={pool.allocator.free_pages}"
            )
        return page_ids

    # TODO(cya): check how to reinit pool
    def _drop_pool_entries(self, pool: EmbeddingPool):
        for image_hash, entry in list(self.entries.items()):
            if self._get_pool(entry.modality) is not pool:
                continue
            del self.entries[image_hash]
            with self.access_lock:
                self.access_order.pop(image_hash, None)

    def _reinit_pool_for_dim(self, pool: EmbeddingPool, actual_dim: int) -> bool:
        active = [
            entry.hash
            for entry in self.entries.values()
            if self._get_pool(entry.modality) is pool
            and (entry.ref_count > 0 or entry.state != EntryState.EVICTABLE)
        ]
        if active:
            logger.warning(
                f"[Rank {self.tp_rank}] Cannot reinitialize {pool.modality} pool "
                f"for dim={actual_dim}; active entries={len(active)}"
            )
            return False

        logger.warning(
            f"[Rank {self.tp_rank}] Reinitializing {pool.modality} embedding pool: "
            f"expected dim={pool.dim}, actual dim={actual_dim}"
        )
        self._drop_pool_entries(pool)
        if not hasattr(self, "_retired_pool_tensors"):
            self._retired_pool_tensors = []
        self._retired_pool_tensors.append(pool.tensor)
        pool.reinit(actual_dim)
        self._register_pool_buffer(pool)
        return True

    # TODO(cya): change to hicahe copy
    def _copy_tensor_to_pages(
        self, tensor: torch.Tensor, entry: EmbeddingCacheEntry, pool: EmbeddingPool
    ):
        src = tensor.detach()
        if src.ndim != 2:
            src = src.reshape(-1, src.shape[-1])
        if src.device.type != "cpu":
            src = src.cpu()
        if not src.is_contiguous():
            src = src.contiguous()

        copied = 0
        for page_id in entry.page_ids:
            valid_tokens = min(pool.page_size_tokens, entry.num_tokens - copied)
            if valid_tokens <= 0:
                break
            start = page_id * pool.page_size_tokens
            pool.tensor[start : start + valid_tokens].copy_(
                src[copied : copied + valid_tokens]
            )
            copied += valid_tokens

    def _copy_entry_to_tensor(
        self,
        entry: EmbeddingCacheEntry,
        dst_tensor: torch.Tensor,
        dst_token_offset: int,
    ):
        pool = self._get_pool(entry.modality)
        copied = 0
        non_blocking = dst_tensor.device.type == "cuda"
        for page_id in entry.page_ids:
            valid_tokens = min(pool.page_size_tokens, entry.num_tokens - copied)
            if valid_tokens <= 0:
                break
            src_start = page_id * pool.page_size_tokens
            dst_start = dst_token_offset + copied
            dst_tensor[dst_start : dst_start + valid_tokens].copy_(
                pool.tensor[src_start : src_start + valid_tokens],
                non_blocking=non_blocking,
            )
            copied += valid_tokens

    def prefetch(
        self,
        req_id: str,
        image_hashes: List[str],
        expected_tokens: List[int],
        modality=None,
    ):
        """Issues ONE batch GET for cache-hit embeddings that are not local yet."""
        pool = self._get_pool(modality)
        if pool is None:
            logger.warning(f"insert_batch: unknown modality {modality}; skipping.")
            return

        keys, all_ptrs, all_sizes = [], [], []

        with self.lock:
            # TODO(cya): check raw_hash, 保留还是让调用方都转str
            for image_hash, num_tokens in zip(image_hashes, expected_tokens):
                entry = self.entries.get(image_hash)
                if entry is not None:
                    if entry.state in (EntryState.EVICTABLE, EntryState.PUTTING):
                        self._update_access_time(image_hash)
                    # TODO(cya): check how to deal with LOADING entry
                    else:
                        logger.debug(
                            f"Req {req_id}: {image_hash} is already LOADING; "
                            f"treating as miss."
                        )
                    continue

                page_ids = self._allocate_with_eviction(pool, int(num_tokens))
                if page_ids is None:
                    logger.warning(
                        f"Req {req_id}: Failed to allocate {num_tokens} tokens "
                        f"in {pool.modality} pool; falling back to encoder."
                    )
                    continue

                entry = EmbeddingCacheEntry(
                    hash=image_hash,
                    modality=modality,
                    num_tokens=int(num_tokens),
                    dim=pool.dim,
                    page_ids=page_ids,
                    state=EntryState.LOADING,
                    ref_count=0,
                )
                self.entries[image_hash] = entry
                self._update_access_time(image_hash)
                keys.append(image_hash)
                entry_ptrs, entry_sizes = build_transfer_buffers(entry, pool)
                all_ptrs.append(entry_ptrs)
                all_sizes.append(entry_sizes)

            if not keys:
                return

            logger.info(
                f"Req {req_id}: Starting global fetch for {len(keys)} "
                f"embeddings from Mooncake."
            )

            op = EmbeddingPrefetchOperation(req_id, keys, all_ptrs, all_sizes)
            self.ongoing_prefetch[req_id] = op
            self.prefetch_queue.put(op)

    def insert_batch(
        self,
        image_hashes: List[str],
        embedding_tensors: List[torch.Tensor],
        modality: Modality = None,
    ):
        """Issues ONE batch PUT for all embeddings computed by this request.

        Note: Even if the embedding exists locally, we still push to Mooncake
        to ensure multi-node cache consistency. Mooncake's batch_put has
        built-in deduplication to avoid redundant transfers.
        """
        pool = self._get_pool(modality)
        if pool is None:
            logger.warning(f"insert_batch: unknown modality {modality}; skipping.")
            return

        keys, all_ptrs, all_sizes = [], [], []
        local_hit_count = 0
        new_count = 0
        skipped_count = 0

        with self.lock:
            for image_hash, tensor in zip(image_hashes, embedding_tensors):
                if tensor.ndim != 2:
                    tensor = tensor.reshape(-1, tensor.shape[-1])
                num_tokens, actual_dim = int(tensor.shape[0]), int(tensor.shape[1])
                # TODO(CYA): check how to reinit
                if actual_dim != pool.dim and not self._reinit_pool_for_dim(
                    pool, actual_dim
                ):
                    skipped_count += 1
                    continue

                entry = self.entries.get(image_hash)
                if entry is not None:
                    if entry.state == EntryState.LOADING:
                        skipped_count += 1
                        logger.debug(
                            f"Skipping insert for {image_hash}: GET is in flight."
                        )
                        continue
                    if entry.dim != actual_dim or entry.num_tokens != num_tokens:
                        if entry.ref_count > 0 or entry.state == EntryState.PUTTING:
                            skipped_count += 1
                            continue
                        self._evict_entry(image_hash)
                        entry = None
                    else:
                        self._update_access_time(image_hash)
                        if entry.state == EntryState.PUTTING:
                            local_hit_count += 1
                            continue
                        entry.state = EntryState.PUTTING
                        keys.append(image_hash)
                        entry_ptrs, entry_sizes = build_transfer_buffers(entry, pool)
                        all_ptrs.append(entry_ptrs)
                        all_sizes.append(entry_sizes)
                        local_hit_count += 1
                        continue

                if entry is None:
                    page_ids = self._allocate_with_eviction(pool, num_tokens)
                    if page_ids is None:
                        logger.warning(
                            f"Failed to allocate {num_tokens} tokens in "
                            f"{pool.modality} pool for insert; skipping."
                        )
                        skipped_count += 1
                        continue

                    entry = EmbeddingCacheEntry(
                        hash=image_hash,
                        modality=modality,
                        num_tokens=num_tokens,
                        dim=actual_dim,
                        page_ids=page_ids,
                        state=EntryState.PUTTING,
                        ref_count=0,
                    )
                    self._copy_tensor_to_pages(tensor, entry, pool)
                    self.entries[image_hash] = entry
                    self._update_access_time(image_hash)
                    keys.append(image_hash)
                    entry_ptrs, entry_sizes = build_transfer_buffers(entry, pool)
                    all_ptrs.append(entry_ptrs)
                    all_sizes.append(entry_sizes)
                    new_count += 1

            if keys:
                logger.info(
                    f"Global Cache: Inserting {len(keys)} embeddings into "
                    f"Mooncake cluster ({new_count} new, {local_hit_count} existing, "
                    f"{skipped_count} skipped)"
                )
                self.insert_queue.put(
                    EmbeddingInsertOperation(keys, all_ptrs, all_sizes)
                )

    def _finish_get(self, op: EmbeddingPrefetchOperation, results: List[bool]):
        with self.lock:
            for image_hash, success in zip(op.keys, results):
                entry = self.entries.get(image_hash)
                if entry is None:
                    continue
                if success:
                    if entry.state == EntryState.LOADING:
                        entry.state = EntryState.EVICTABLE
                else:
                    pool = self._get_pool(entry.modality)
                    pool.allocator.free(entry.page_ids)
                    del self.entries[image_hash]
                    with self.access_lock:
                        self.access_order.pop(image_hash, None)
        op.mark_done(all(results))

    def _finish_put(self, op: EmbeddingInsertOperation, results: List[bool]):
        with self.lock:
            for image_hash, success in zip(op.keys, results):
                entry = self.entries.get(image_hash)
                if entry is None:
                    continue
                if not success:
                    logger.warning(
                        f"[Rank {self.tp_rank}] Mooncake PUT failed for "
                        f"{image_hash}; keeping local cache entry."
                    )
                if entry.state == EntryState.PUTTING:
                    entry.state = EntryState.EVICTABLE

    def _io_loop(self):
        """Asynchronous worker handling both Batch GET and Batch PUT."""
        while not self.stop_event.is_set():
            processed_any = False

            try:
                op = self.prefetch_queue.get_nowait()
                try:
                    results = self.mooncake_store.batch_get_into_multi_buffers(
                        op.keys, op.ptrs, op.sizes
                    )
                except Exception:
                    logger.exception("Mooncake multi-buffer GET failed")
                    results = [False] * len(op.keys)
                success_count = sum(results)
                logger.info(
                    f"Mooncake GET Finished: Req {op.req_id}, "
                    f"Successfully fetched {success_count}/{len(op.keys)} embeddings."
                )
                self._finish_get(op, results)
                self.prefetch_queue.task_done()
                processed_any = True
            except Empty:
                pass

            try:
                op = self.insert_queue.get_nowait()
                try:
                    results = self.mooncake_store.batch_put_from_multi_buffers(
                        op.keys, op.ptrs, op.sizes
                    )
                except Exception:
                    logger.exception("Mooncake multi-buffer PUT failed")
                    results = [False] * len(op.keys)
                self._finish_put(op, results)
                logger.info(
                    f"Mooncake PUT Finished: Stored {sum(results)}/{len(op.keys)} "
                    f"embeddings in cluster."
                )
                self.insert_queue.task_done()
                processed_any = True
            except Empty:
                pass

            if not processed_any:
                time.sleep(0.001)

    def check_prefetch_progress(self, req_id: str) -> bool:
        """TP-Group barrier: ensures all cards have the request batch ready."""
        local_ready = False
        with self.lock:
            if req_id not in self.ongoing_prefetch:
                local_ready = True
            else:
                op = self.ongoing_prefetch[req_id]
                if op.is_finished:
                    local_ready = True

        if self.all_rank_get and self.tp_world_size > 1:
            ready_tensor = torch.tensor(
                [1 if local_ready else 0], dtype=torch.int, device="cpu"
            )
            torch.distributed.all_reduce(
                ready_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.prefetch_tp_group,
            )
            local_ready = ready_tensor.item() == 1

        if local_ready:
            with self.lock:
                self.ongoing_prefetch.pop(req_id, None)
            return True
        return False

    def copy_embedding_to(
        self, image_hash: str, dst_tensor: torch.Tensor, dst_token_offset: int
    ) -> bool:
        # Pin the entry to prevent eviction during copy (including async DMA).
        with self.lock:
            entry = self.entries.get(image_hash)
            if entry is None:
                logger.warning(f"Hash {image_hash} not found in local cache")
                return False
            if entry.state not in (EntryState.EVICTABLE, EntryState.PUTTING):
                logger.warning(
                    f"Hash {image_hash} is not ready; state={entry.state.name}"
                )
                return False
            entry.ref_count += 1
            self._update_access_time(image_hash)

        try:
            pool = self._get_pool(entry.modality)
            self._copy_entry_to_tensor(entry, dst_tensor, dst_token_offset)
            # Wait for non-blocking H2D copy to finish before unpinning,
            # so the source pages are not reused while DMA is in flight.
            if dst_tensor.device.type == "cuda":
                torch.cuda.current_stream(dst_tensor.device).synchronize()
            return True
        finally:
            with self.lock:
                entry.ref_count -= 1

    def has_local_embedding(self, image_hash: str) -> bool:
        with self.lock:
            entry = self.entries.get(image_hash)
            return entry is not None and entry.state in (
                EntryState.EVICTABLE,
                EntryState.PUTTING,
            )

    def store_to_pool(
        self,
        image_hashes: List[str],
        tensors: List[torch.Tensor],
        modality=None,
    ) -> List[bool]:
        """Store GPU/CPU tensors into host paged pool with blocking copy.

        Allocates pages and performs synchronous device-to-host copy.
        Does NOT initiate storage PUT. Entries are marked EVICTABLE so
        that subsequent insert_batch() can reuse them for async PUT.

        Returns: per-item success (False if page allocation failed).
        """
        pool = self._get_pool(modality)
        if pool is None:
            return [False] * len(image_hashes)

        results = []
        with self.lock:
            for image_hash, tensor in zip(image_hashes, tensors):
                if tensor.ndim != 2:
                    tensor = tensor.reshape(-1, tensor.shape[-1])
                num_tokens, actual_dim = int(tensor.shape[0]), int(tensor.shape[1])

                if actual_dim != pool.dim and not self._reinit_pool_for_dim(
                    pool, actual_dim
                ):
                    results.append(False)
                    continue

                entry = self.entries.get(image_hash)
                if entry is not None:
                    if entry.state in (EntryState.EVICTABLE, EntryState.PUTTING):
                        self._update_access_time(image_hash)
                        results.append(True)
                        continue
                    if entry.state == EntryState.LOADING:
                        results.append(False)
                        continue
                    self._evict_entry(image_hash)

                page_ids = self._allocate_with_eviction(pool, num_tokens)
                if page_ids is None:
                    results.append(False)
                    continue

                entry = EmbeddingCacheEntry(
                    hash=image_hash,
                    modality=modality,
                    num_tokens=num_tokens,
                    dim=actual_dim,
                    page_ids=page_ids,
                    state=EntryState.EVICTABLE,
                    ref_count=0,
                )
                self._copy_tensor_to_pool_sync(tensor, entry, pool)
                self.entries[image_hash] = entry
                self._update_access_time(image_hash)
                results.append(True)
        return results

    def _copy_tensor_to_pool_sync(
        self, tensor: torch.Tensor, entry: EmbeddingCacheEntry, pool: EmbeddingPool
    ):
        """Blocking copy from device/host tensor to pinned pool pages."""
        src = tensor.detach()
        if src.ndim != 2:
            src = src.reshape(-1, src.shape[-1])
        if not src.is_contiguous():
            src = src.contiguous()

        needs_sync = src.device.type == "cuda"
        copied = 0
        for page_id in entry.page_ids:
            valid_tokens = min(pool.page_size_tokens, entry.num_tokens - copied)
            if valid_tokens <= 0:
                break
            start = page_id * pool.page_size_tokens
            pool.tensor[start : start + valid_tokens].copy_(
                src[copied : copied + valid_tokens]
            )
            copied += valid_tokens

        if needs_sync:
            torch.cuda.current_stream(src.device).synchronize()

    def get_embedding_dim(self, modality=None) -> int:
        return self._get_pool(modality).dim

    def get_stats(self) -> dict:
        """Return cache statistics."""
        with self.lock:
            allocated_bytes = sum(
                len(entry.page_ids) * self._get_pool(entry.modality).page_bytes
                for entry in self.entries.values()
            )
            free_bytes = sum(
                pool.allocator.free_pages * pool.page_bytes
                for pool in self.pools.values()
            )
            return {
                **self.stats,
                "num_cached": len(self.entries),
                "num_protected": sum(
                    1 for entry in self.entries.values() if entry.ref_count > 0
                ),
                "allocated_mb": allocated_bytes / 1024**2,
                "free_mb": free_bytes / 1024**2,
                "total_mb": sum(
                    pool.num_pages * pool.page_bytes for pool in self.pools.values()
                )
                / 1024**2,
                "vision_free_pages": self.vision_pool.allocator.free_pages,
                "audio_free_pages": self.audio_pool.allocator.free_pages,
            }

    async def batch_is_exist(self, image_hashes: List[str]) -> List[bool]:
        with self.lock:
            # TODO(cya): check it，正在load的算不算命中
            local_results = [
                (
                    (entry := self.entries.get(h)) is not None
                    and entry.state in (EntryState.EVICTABLE, EntryState.PUTTING)
                )
                for h in image_hashes
            ]
        local_hit_count = sum(local_results)

        global_hit_count = 0
        if not all(local_results):
            missing_indices = [i for i, res in enumerate(local_results) if not res]
            missing_hashes = [image_hashes[i] for i in missing_indices]

            global_exists = await asyncio.to_thread(
                self.mooncake_store.batch_is_exist, missing_hashes
            )
            global_hit_count = sum(global_exists)

            for i, exists in zip(missing_indices, global_exists):
                local_results[i] = exists

        total = len(image_hashes)
        miss_count = total - local_hit_count - global_hit_count
        logger.info(
            f"=== Multi-Level Cache Check === "
            f"Total: {total} | "
            f"Local Hits: {local_hit_count} | "
            f"Global Hits: {global_hit_count} | "
            f"Misses (GPU Work): {miss_count}"
        )
        return local_results
