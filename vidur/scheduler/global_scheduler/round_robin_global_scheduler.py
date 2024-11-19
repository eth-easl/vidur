from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class RoundRobinGlobalScheduler(BaseGlobalScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._request_counter = 0

    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()

        request_mapping = []
        if self._prompt_pool_ratio:
            while self._request_queue:
                request = self._request_queue.pop(0)
                prompt_replica_id = self._request_counter % self._prompt_pool_size
                token_replica_id = self._request_counter % (self._num_replicas - self._prompt_pool_size) + self._prompt_pool_size
                request.assign({"prompt": prompt_replica_id, "token": token_replica_id})
                self._request_counter += 1
                request_mapping.append((prompt_replica_id, request))
        else:
            while self._request_queue:
                request = self._request_queue.pop(0)
                replica_id = self._request_counter % self._num_replicas
                request.assign({"prompt": replica_id, "token": replica_id})
                self._request_counter += 1
                request_mapping.append((replica_id, request))

        return request_mapping
