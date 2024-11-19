from random import randint
from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class RandomGlobalScheduler(BaseGlobalScheduler):
    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()

        request_mapping = []
        if self._prompt_pool_ratio:
            while self._request_queue:
                request = self._request_queue.pop(0)
                prompt_replica_id = randint(0, self._prompt_pool_size - 1)
                token_replica_id = randint(self._prompt_pool_size, self._num_replicas - 1)
                request.assign({"prompt": prompt_replica_id, "token": token_replica_id})
                request_mapping.append((prompt_replica_id, request))
        else:
            while self._request_queue:
                request = self._request_queue.pop(0)
                replica_id = randint(1, self._num_replicas) - 1
                request.assign({"prompt": replica_id, "token": replica_id})
                request_mapping.append((replica_id, request))
        return request_mapping
