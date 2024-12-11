from typing import List

from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class ReplicaStageScheduleEvent(BaseEvent):
    def __init__(self, time: float, replica_id: int, stage_id: int):
        super().__init__(time, EventType.REPLICA_STAGE_SCHEDULE)

        self._replica_id = replica_id
        self._stage_id = stage_id

        self._batch = None
        self._batch_stage = None
        self._is_last_stage = None

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.batch_stage_end_event import BatchStageEndEvent

        stage_scheduler = scheduler._replica_schedulers[
            self._replica_id
        ]._replica_stage_schedulers[self._stage_id]

        self._batch, self._batch_stage, execution_time = stage_scheduler.on_schedule()

        if not (self._batch and self._batch_stage):
            return []

        self._batch_stage.on_schedule(self.time)
        metrics_store.on_replica_stage_schedule(
            self.time,
            self._replica_id,
            self._stage_id,
            self._batch_stage,
            execution_time,
        )

        self._is_last_stage = stage_scheduler.is_last_stage
        replica_scheduler = scheduler._replica_schedulers[self._replica_id]
        if replica_scheduler._replica_type == "prompt":
            if (scheduler._kvcache_transfer_mode == "layer-wise"):
                for request in self._batch._requests:
                    execution_time_predictor = scheduler.get_execution_time_predictor()
                    kvcache_transfer_time = execution_time_predictor.get_kvcache_transfer_time(request, "gpu-gpu")
                    request.kvcache_transfer_time["gpu-gpu"] = kvcache_transfer_time
            if scheduler._kvcache_transfer_mode == "layer-wise-cpu":
                for request in self._batch._requests:
                    execution_time_predictor = scheduler.get_execution_time_predictor()
                    kvcache_transfer_time = execution_time_predictor.get_kvcache_transfer_time(request, "gpu-cpu")
                    request.kvcache_transfer_time["gpu-cpu"] = kvcache_transfer_time
        return [
            BatchStageEndEvent(
                self.time + self._batch_stage.execution_time,
                self._replica_id,
                self._stage_id,
                self._is_last_stage,
                self._batch,
                self._batch_stage,
            ),
        ]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "replica_id": self._replica_id,
            "stage_id": self._stage_id,
            "batch_id": self._batch.id if self._batch else None,
            "batch_stage_id": self._batch_stage.id if self._batch_stage else None,
            "is_last_stage": self._is_last_stage,
        }
