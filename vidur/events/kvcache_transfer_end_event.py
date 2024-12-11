from typing import List

from vidur.entities.request import Request
from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class KVCacheTransferEndEvent(BaseEvent):
    def __init__(self, time: float, request: Request, type: str):
        super().__init__(time, EventType.REQUEST_PREFILL_END)

        self._request = request
        self._type = type

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.replica_schedule_event import ReplicaScheduleEvent
        assert self._request.is_prefill_complete

        if self._type == "gpu-gpu":
            self._request.kvcache_transfered = True
            prompt_replica_id = self._request.assigned_replicas["prompt"]
            prompt_replica_scheduler = scheduler.get_replica_scheduler(prompt_replica_id)
            assert prompt_replica_scheduler.replica_type == "prompt"
            prompt_replica_scheduler.free(self._request.id)

            return [
                ReplicaScheduleEvent(self.time, prompt_replica_id)
            ]

        if self._type == "gpu-cpu":
            prompt_replica_id = self._request.assigned_replicas["prompt"]
            prompt_replica_scheduler = scheduler.get_replica_scheduler(prompt_replica_id)
            prompt_replica_scheduler.free(self._request.id)
            # check whether the cpu in token machine has memory
            token_replica_id = self._request.assigned_replicas["token"]
            token_replica_scheduler = scheduler.get_replica_scheduler(token_replica_id)
            self._request.ready_for_transfer = True
            if token_replica_scheduler._can_cpu_allocate_request(self._request):
                token_replica_scheduler._cpu_allocate_request(self._request)
                execution_time_predictor = scheduler.get_execution_time_predictor()
                kvcache_transfer_time = execution_time_predictor.get_kvcache_transfer_time(self._request, "cpu-cpu")
                self._request.kvcache_transfer_time["cpu-cpu"] = kvcache_transfer_time
                return [
                    KVCacheTransferEndEvent(self.time + kvcache_transfer_time, self._request, "cpu-cpu")
                ]

        if self._type == "cpu-cpu":
            self._request.kvcache_transfered = True
            prompt_replica_id = self._request.assigned_replicas["prompt"]
            prompt_replica_scheduler = scheduler.get_replica_scheduler(prompt_replica_id)
            prompt_replica_scheduler.free_cpu(self._request.id)
            token_replica_id = self._request.assigned_replicas["token"]
            return [
                ReplicaScheduleEvent(self.time, token_replica_id)
            ]
        return []
    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "request_id": self._request.id,
        }
