from typing import List

from vidur.entities.request import Request
from vidur.events import BaseEvent
from vidur.events.kvcache_transfer_end_event import KVCacheTransferEndEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class RequestPrefillEndEvent(BaseEvent):
    def __init__(self, time: float, request: Request):
        super().__init__(time, EventType.REQUEST_PREFILL_END)

        self._request = request

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.replica_schedule_event import ReplicaScheduleEvent
        assert self._request.is_prefill_complete
        if self._request.completed:
            scheduler.get_replica_scheduler(self._request.assigned_replicas["prompt"]).free(self._request.id)
            return []
        token_replica_id = self._request.assigned_replicas["token"]
        token_replica_scheduler = scheduler.get_replica_scheduler(token_replica_id)
        assert token_replica_scheduler.replica_type == "token"
        token_replica_scheduler.add_request(self._request)
        if scheduler._kvcache_transfer_mode == "layer-wise" and token_replica_scheduler.is_allocated(self._request._id):
            return [
                KVCacheTransferEndEvent(self._request.kvcache_transfer_time, self._request, "gpu-gpu"),
                ReplicaScheduleEvent(self._request.kvcache_transfer_time, token_replica_id)
            ]

        if scheduler._kvcache_transfer_mode == "serialized-push":
            if token_replica_scheduler._can_allocate_request(self._request):
                # if serialized-push mode and memory available
                execution_time_predictor = scheduler.get_execution_time_predictor()
                token_replica_scheduler._allocate_request(self._request)
                kv_cache_transfer_time = execution_time_predictor.get_kvcache_transfer_time(self._request)
                return [
                    KVCacheTransferEndEvent(self.time + kv_cache_transfer_time, self._request, "gpu-gpu"),
                    ReplicaScheduleEvent(self.time + kv_cache_transfer_time, token_replica_id)
                ]

        return [
            ReplicaScheduleEvent(self.time, token_replica_id)
        ]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "request_id": self._request.id,
        }