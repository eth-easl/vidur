from typing import List

from vidur.entities.request import Request
from vidur.events import BaseEvent
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
        scheduler.get_replica_scheduler(token_replica_id).add_request(self._request)

        return [
            ReplicaScheduleEvent(self.time, token_replica_id)
        ]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "request_id": self._request.id,
        }