from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class LORGlobalScheduler(BaseGlobalScheduler):
    """
    Least outstanding requests (LOR) global scheduler.
    """

    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()

        request_mapping = []
        # keep a map of replica_id -> replica_scheduler
        # this is used to find the replica with the least outstanding requests
        if self._prompt_pool_ratio:
            pending_prompt_requests_map = {
                replica_scheduler.replica_id: replica_scheduler.num_pending_requests
                for replica_scheduler in self._replica_schedulers.values()
                if replica_scheduler.replica_type == "prompt"
            }
            pending_token_requests_map = {
                replica_scheduler.replica_id: replica_scheduler.num_pending_requests
                for replica_scheduler in self._replica_schedulers.values()
                if replica_scheduler.replica_type == "token"
            }
            while self._request_queue:
                request = self._request_queue.pop(0)
                prompt_replica_id = min(pending_prompt_requests_map.items(), key=lambda x: x[1])[0]
                pending_prompt_requests_map[prompt_replica_id] += 1
                token_replica_id = min(pending_token_requests_map.items(), key=lambda x: x[1])[0]
                pending_token_requests_map[token_replica_id] += 1
                request.assign({"prompt": prompt_replica_id, "token": token_replica_id})

                request_mapping.append((prompt_replica_id, request))
        else:
            # Single pending requests map for non-disaggregated setup
            pending_requests_map = {
                replica_scheduler.replica_id: replica_scheduler.num_pending_requests
                for replica_scheduler in self._replica_schedulers.values()
            }
            while self._request_queue:
                request = self._request_queue.pop(0)
                replica_id = min(pending_requests_map.items(), key=lambda x: x[1])[0]
                pending_requests_map[replica_id] += 1
                request_mapping.append((replica_id, request))

        return request_mapping
