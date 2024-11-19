import json
import math

from vidur.config import BaseRequestGeneratorConfig, ClusterConfig, MetricsConfig
from vidur.entities.base_entity import BaseEntity
from vidur.entities.replica import Replica
from vidur.logger import init_logger

logger = init_logger(__name__)


class Cluster(BaseEntity):
    def __init__(
        self,
        cluster_config: ClusterConfig,
        metrics_config: MetricsConfig,
        generator_config: BaseRequestGeneratorConfig,
    ) -> None:
        self._id = Cluster.generate_id()
        self._config = cluster_config

        # get metrics config
        self._output_dir = metrics_config.output_dir

        # Init replica object handles
        self._replicas = {}

        if cluster_config.prompt_pool_ratio:
            assert 0 < cluster_config.prompt_pool_ratio < 1, "prompt_pool_ratio must be between 0 and 1 (exclusive)."
            self._prompt_pool_ratio = cluster_config.prompt_pool_ratio
            self._prompt_pool_size = max(1, math.floor(cluster_config.num_replicas * self._prompt_pool_ratio))
            for i in range(self._config.num_replicas):
                if i < self._prompt_pool_size:
                    replica_type = "prompt"
                else:
                    replica_type = "token"
                replica = Replica(self._config.replica_config, generator_config, replica_type)
                self._replicas[replica._id] = replica
        else:
            self._prompt_pool_ratio = None
            for _ in range(self._config.num_replicas):
                replica = Replica(self._config.replica_config, generator_config)
                self._replicas[replica.id] = replica

        if metrics_config.write_json_trace:
            self._write_cluster_info_to_file()

    @property
    def replicas(self):
        return self._replicas

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "num_replicas": len(self._replicas),
        }

    def _write_cluster_info_to_file(self) -> None:
        replica_dicts = [replica.to_dict() for replica in self._replicas.values()]
        cluster_info = {"replicas": replica_dicts}

        cluster_file = f"{self._output_dir}/cluster.json"
        with open(cluster_file, "w") as f:
            json.dump(cluster_info, f)
