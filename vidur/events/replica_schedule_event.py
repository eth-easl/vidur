from typing import List

from vidur.events import BaseEvent
from vidur.events.kvcache_transfer_end_event import KVCacheTransferEndEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class ReplicaScheduleEvent(BaseEvent):
    def __init__(self, time: float, replica_id: int):
        super().__init__(time, EventType.REPLICA_SCHEDULE)

        self._replica_id = replica_id

        self._batches = []

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.batch_stage_arrival_event import BatchStageArrivalEvent

        replica_scheduler = scheduler.get_replica_scheduler(self._replica_id)
        self._batches = replica_scheduler.on_schedule()

        if not self._batches:
            return []

        memory_usage_percent = replica_scheduler.memory_usage_percent
        metrics_store.on_replica_schedule(
            self.time, self._replica_id, memory_usage_percent
        )

        for batch in self._batches:
            batch.on_schedule(self.time)

        if replica_scheduler.replica_type == "token":
            events = []
            execution_time_predictor = scheduler.get_execution_time_predictor()
            # handle restarted requests
            for request in replica_scheduler.restart_requests:
                prompt_replica_id = request.assigned_replicas["prompt"]
                scheduler.get_replica_scheduler(prompt_replica_id).add_request(request)
                events.append(ReplicaScheduleEvent(self.time, prompt_replica_id))
            replica_scheduler.clear_restart_requests()
            if scheduler._kvcache_transfer_mode == "serialized" or scheduler._kvcache_transfer_mode == "layer-wise":
                # load kvcache from prompt machine
                for request in replica_scheduler.request_queue:
                    if not replica_scheduler.is_allocated(request.id) and replica_scheduler._can_allocate_request(request):
                        replica_scheduler._allocate_request(request)
                        kvcache_transfer_time = execution_time_predictor.get_kvcache_transfer_time(request, "gpu-gpu")
                        request.kvcache_transfer_time["gpu-gpu"] = kvcache_transfer_time
                        events.append(KVCacheTransferEndEvent(self.time + kvcache_transfer_time, request, "gpu-gpu"))
                        events.append(ReplicaScheduleEvent(self.time + kvcache_transfer_time, self._replica_id))
                events.append(
                    BatchStageArrivalEvent(
                        self.time,
                        self._replica_id,
                        0,  # stage_id
                        batch,
                ))
                return events
            if scheduler._kvcache_transfer_mode == "serialized-cpu" or scheduler._kvcache_transfer_mode == "layer-wise-cpu":
                for request in replica_scheduler.request_queue:
                    if not replica_scheduler.is_cpu_allocated(request.id) and replica_scheduler._can_cpu_allocate_request(request) and request.ready_for_transfer:
                        replica_scheduler._cpu_allocate_request(request)
                        kvcache_transfer_time = execution_time_predictor.get_kvcache_transfer_time(request, "cpu-cpu")
                        events.append(KVCacheTransferEndEvent(self.time + kvcache_transfer_time, request, "cpu-cpu"))
                        events.append(ReplicaScheduleEvent(self.time + kvcache_transfer_time, self._replica_id))
                for batch in self._batches:
                    max_kvcache_transfer_time = 0.0
                    for request in batch.requests_without_kvcache:
                        kvcache_transfer_time = execution_time_predictor.get_kvcache_transfer_time(request, "cpu-gpu")
                        request.kvcache_transfer_time["cpu-gpu"] = kvcache_transfer_time
                        max_kvcache_transfer_time = max(max_kvcache_transfer_time, kvcache_transfer_time)
                    events.append(
                        BatchStageArrivalEvent(
                            self.time + max_kvcache_transfer_time,
                            self._replica_id,
                            0,  # stage_id
                            batch,
                    ))
            if scheduler._kvcache_transfer_mode == "pull":
                for batch in self._batches:
                    max_kvcache_transfer_time = 0.0
                    for request in batch.requests_without_kvcache:
                        kvcache_transfer_time = execution_time_predictor.get_kvcache_transfer_time(request, "gpu-gpu")
                        request.kvcache_transfer_time["gpu-gpu"] = kvcache_transfer_time
                        max_kvcache_transfer_time = max(max_kvcache_transfer_time, kvcache_transfer_time)
                        events.append(
                            KVCacheTransferEndEvent(
                                self.time + kvcache_transfer_time,
                                request,
                                "gpu-gpu"
                            )
                        )
                    events.append(
                        BatchStageArrivalEvent(
                            self.time + max_kvcache_transfer_time,
                            self._replica_id,
                            0,  # stage_id
                            batch,
                    ))
            return events

        if replica_scheduler.replica_type == "prompt" and scheduler._kvcache_transfer_mode == "layer-wise":
            for batch in self._batches:
                for request in batch._requests:
                    token_replica_id = request.assigned_replicas["token"]
                    token_replica_scheduler = scheduler.get_replica_scheduler(token_replica_id)
                    if token_replica_scheduler._can_allocate_request(request):
                        # if layer-wise and memory available
                        token_replica_scheduler._allocate_request(request)

        if replica_scheduler.replica_type == "prompt" and scheduler._kvcache_transfer_mode == "layer-wise-cpu":
            for batch in self._batches:
                for request in batch._requests:
                    if replica_scheduler._can_cpu_allocate_request(request):
                        replica_scheduler._cpu_allocate_request(request)
        return [
            BatchStageArrivalEvent(
                self.time,
                self._replica_id,
                0,  # stage_id
                batch,
            )
            for batch in self._batches
        ]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "replica_id": self._replica_id,
            "batch_ids": [batch.id for batch in self._batches],
        }
