from typing import List

from vidur.entities import Batch
from vidur.events import BaseEvent
from vidur.events.request_prefill_end_event import RequestPrefillEndEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class BatchEndEvent(BaseEvent):
    def __init__(self, time: float, replica_id: int, batch: Batch):
        super().__init__(time, EventType.BATCH_END)

        self._replica_id = replica_id
        self._batch = batch

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.replica_schedule_event import ReplicaScheduleEvent

        self._batch.on_batch_end(self.time)
        replica_scheduler = scheduler.get_replica_scheduler(self._replica_id)
        replica_scheduler.on_batch_end(self._batch)

        memory_usage_percent = replica_scheduler.memory_usage_percent
        metrics_store.on_batch_end(
            self.time, self._batch, self._replica_id, memory_usage_percent
        )

        if replica_scheduler.replica_type == "prompt":
            events = [RequestPrefillEndEvent(self.time, request) for request in self._batch.requests]
            return events + [ReplicaScheduleEvent(self.time, self._replica_id)]
        if replica_scheduler.replica_type == "token" and \
            (scheduler._kvcache_transfer_mode == "layer-wise-cpu" or scheduler._kvcache_transfer_mode == "serialized-cpu"):
            for request in self._batch.requests:
                if request.completed:
                    replica_scheduler.free_cpu(request.id)

        return [ReplicaScheduleEvent(self.time, self._replica_id)]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "batch_id": self._batch.id,
        }
