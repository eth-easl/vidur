from math import ceil
from typing import List

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)


class DisaggregationReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._preempted_requests: List[Request] = []
        self._num_running_batches = 0
        # For vLLM and its derivatives, we only need to set a loose max batch size
        # Memory requirements are handled explicitly by the scheduler
        self._max_micro_batch_size = self._config.batch_size_cap // self._num_stages
        # The threshold used for memory swapping.
        self._watermark_blocks = int(
            self._config.watermark_blocks_fraction * self._config.num_blocks
        )

        self.restart_requests: List[Request] = []
        self.kvcache_transfer_mode = None

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1
        # iteration-level scheduling
        if self._replica_type == "prompt":
            return

        for request in batch.requests:
            if request.completed:
                self.free(request.id)
            else:
                self._preempted_requests.append(request)

    def _can_allocate_request(self, request: Request) -> bool:
        if self._replica_type == "prompt":
            num_required_blocks = ceil(
                (request.num_prefill_tokens) / self._config.block_size
            )
            return (
                self._config.num_blocks
                - self._num_allocated_blocks
                - num_required_blocks
                >= self._watermark_blocks
            )
        else:
            new_blocks_required = 0
            if request.id not in self._allocation_map:
                # new request
                # space for kv cache + first token
                new_blocks_required = ceil(
                    (request.num_prefill_tokens + 1) / self._config.block_size
                )

            # vllm requires at least one block to be available
            # for decode stage
            return self._config.num_blocks - self._num_allocated_blocks - new_blocks_required >= 1

    def _allocate_request(self, request: Request) -> None:
        if request.id not in self._allocation_map:
            # new request
            if self._replica_type == "prompt":
                num_required_blocks = ceil(
                    (request.num_prefill_tokens) / self._config.block_size
                )
                self.allocate(request.id, num_required_blocks)
                return
            else:
                num_required_blocks = ceil(
                    (request.num_prefill_tokens + 1) / self._config.block_size
                )
                self.allocate(request.id, num_required_blocks)
                return

        num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size
        num_tokens_required = max(0, request.num_processed_tokens - num_tokens_reserved)
        assert (
            num_tokens_required == 0 or num_tokens_required == 1
        ), f"num_tokens_required: {num_tokens_required}; machine: {self.replica_type}"

        if num_tokens_required == 0:
            return

        self.allocate(request.id, 1)

    def _get_next_batch(self) -> Batch:
        requests = []
        num_tokens = []
        num_batch_tokens = 0
        requests_without_kvcache = []
        if self._replica_type == "prompt":
            while self._request_queue:
                request = self._request_queue[0]

                next_num_tokens = self._get_request_next_num_tokens(request)

                if not self._can_allocate_request(request):
                    break

                new_num_tokens = num_tokens + [next_num_tokens]
                new_num_batch_tokens = len(new_num_tokens) * max(new_num_tokens)
                if new_num_batch_tokens > self._config.max_tokens_in_batch:
                    break

                if len(self._allocation_map) == self._config.batch_size_cap:
                    break

                if len(requests) == self._max_micro_batch_size:
                    break

                request = self._request_queue.pop(0)

                self._allocate_request(request)
                requests.append(request)
                num_tokens.append(next_num_tokens)
                num_batch_tokens += next_num_tokens

            if requests:
                return Batch(self._replica_id, requests, num_tokens)
        else:
            # Safer to sort preempted_requests to maintain FIFO order
            self._preempted_requests.sort(key=lambda r: r.arrived_at)
            # all preempted_requests will have prefill completed
            while self._preempted_requests:
                if len(requests) == self._max_micro_batch_size:
                    break

                request = self._preempted_requests.pop(0)

                # if cannot run the next preempted request
                while not self._can_allocate_request(request):
                    if self._preempted_requests:
                        # select a victim (the latest one) from preempted requests
                        victim_request = self._preempted_requests.pop(-1)
                        victim_request.restart()
                        self.free(victim_request.id)
                        # put at the first of the request queue
                        # self._request_queue = [victim_request] + self._request_queue
                        self.restart_requests.append(victim_request)
                    else:
                        # if no more preempted requests
                        request.restart()
                        self.free(request.id)
                        # self._request_queue = [request] + self._request_queue
                        self.restart_requests.append(request)
                        break
                else:
                    self._allocate_request(request)
                    next_num_tokens = self._get_request_next_num_tokens(request)
                    requests.append(request)
                    num_tokens.append(next_num_tokens)
            kvcache_transfers_in_progress = []
            while self._request_queue:
                request = self._request_queue[0]
                if self.kvcache_transfer_mode == "serialized-push" and not request.kvcache_transfered:
                    request = self._request_queue.pop(0)
                    kvcache_transfers_in_progress.append(request)
                    continue

                next_num_tokens = self._get_request_next_num_tokens(request)

                if not self.kvcache_transfer_mode == "serialized-push" and not self._can_allocate_request(request):
                    break

                new_num_tokens = num_tokens + [next_num_tokens]
                new_num_batch_tokens = len(new_num_tokens) * max(new_num_tokens)
                if new_num_batch_tokens > self._config.max_tokens_in_batch:
                    break

                if len(self._allocation_map) == self._config.batch_size_cap:
                    break

                if len(requests) == self._max_micro_batch_size:
                    break

                request = self._request_queue.pop(0)

                if not self.kvcache_transfer_mode == "serialized-push":
                    self._allocate_request(request)
                    requests_without_kvcache.append(request)

                requests.append(request)
                num_tokens.append(next_num_tokens)
                num_batch_tokens += next_num_tokens
            if self.kvcache_transfer_mode == "serialized-push":
                self._request_queue = kvcache_transfers_in_progress + self._request_queue
            if requests:
                return Batch(self._replica_id, requests, num_tokens, requests_without_kvcache)

    def clear_restart_requests(self):
        self.restart_requests = []
