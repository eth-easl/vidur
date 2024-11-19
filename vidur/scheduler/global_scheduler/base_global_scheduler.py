from abc import ABC, abstractmethod
import math
from typing import Dict, List, Tuple

from vidur.config import SimulationConfig
from vidur.entities import Replica, Request
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
from vidur.scheduler.replica_scheduler.replica_scheduler_registry import (
    ReplicaSchedulerRegistry,
)


class BaseGlobalScheduler(ABC):
    def __init__(self, config: SimulationConfig, replicas: Dict[int, Replica]):
        self._config = config
        self._replicas = replicas

        self._num_replicas = len(self._replicas)

        self._execution_time_predictor = ExecutionTimePredictorRegistry.get(
            config.execution_time_predictor_config.get_type(),
            predictor_config=config.execution_time_predictor_config,
            replica_config=config.cluster_config.replica_config,
            replica_scheduler_config=config.cluster_config.replica_scheduler_config,
            metrics_config=config.metrics_config,
        )
        self._replica_schedulers = {
            replica_id: ReplicaSchedulerRegistry.get(
                config.cluster_config.replica_scheduler_config.get_type(),
                replica_config=config.cluster_config.replica_config,
                replica_scheduler_config=config.cluster_config.replica_scheduler_config,
                request_generator_config=config.request_generator_config,
                replica=replica,
                num_stages=replica.num_pipeline_stages,
                execution_time_predictor=self._execution_time_predictor,
            )
            for replica_id, replica in replicas.items()
        }
        self._request_queue = []
        if config.cluster_config.prompt_pool_ratio:
            assert 0 < config.cluster_config.prompt_pool_ratio < 1, "prompt_pool_ratio must be between 0 and 1 (exclusive)."
            self._prompt_pool_ratio = config.cluster_config.prompt_pool_ratio
            self._prompt_pool_size = max(1, math.floor(self._num_replicas * self._prompt_pool_ratio))
            self._kvcache_transfer_mode = config.cluster_config.kvcache_transfer_mode
            for _, replica_scheduler in self._replica_schedulers.items():
                replica_scheduler.kvcache_transfer_mode = self._kvcache_transfer_mode
        else:
            self._prompt_pool_ratio = None
            self._kvcache_transfer_mode = None

    def sort_requests(self) -> None:
        self._request_queue.sort(key=lambda request: request._arrived_at)

    def add_request(self, request: Request) -> None:
        self._request_queue.append(request)

    def get_replica_scheduler(self, replica_id: int):
        return self._replica_schedulers[replica_id]

    def get_replica_stage_scheduler(self, replica_id: int, stage_id: int):
        return self._replica_schedulers[replica_id].get_replica_stage_scheduler(
            stage_id
        )

    def is_empty(self) -> bool:
        return len(self._request_queue) == 0 and all(
            replica_scheduler.is_empty()
            for replica_scheduler in self._replica_schedulers.values()
        )

    def get_execution_time_predictor(self):
        return self._execution_time_predictor

    @abstractmethod
    def schedule(self) -> List[Tuple[int, Request]]:
        pass
