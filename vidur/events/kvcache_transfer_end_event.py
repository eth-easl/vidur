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
        assert not self._request.kvcache_transfered

        self._request.kvcache_transfered = True
        if self._type == "gpu-gpu" or "gpu-cpu":
            prompt_replica_id = self._request.assigned_replicas["prompt"]
            prompt_replica_scheduler = scheduler.get_replica_scheduler(prompt_replica_id)
            assert prompt_replica_scheduler.replica_type == "prompt"
            prompt_replica_scheduler.free(self._request.id)

            return [
                ReplicaScheduleEvent(self.time, prompt_replica_id)
            ]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "request_id": self._request.id,
        }
