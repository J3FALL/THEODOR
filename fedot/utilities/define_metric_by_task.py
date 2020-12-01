from fedot.core.composer.metrics import RmseMetric, RocAucMetric
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.models.data import InputData, OutputData


class MetricByTask:
    __metric_by_task = {TaskTypesEnum.regression: RmseMetric.metric,
                        TaskTypesEnum.classification: RocAucMetric.metric,
                        TaskTypesEnum.clustering: None,
                        TaskTypesEnum.ts_forecasting: RmseMetric.metric,
                        }

    def __init__(self, task_type):
        self._metric = self.__metric_by_task.get(task_type)

    def get_value(self, true: InputData, predicted: OutputData, round_up_to: int = 6):
        """Returns the value of metric defined by task"""
        try:
            return round(self._metric(reference=true, predicted=predicted), round_up_to)
        # TODO or raise ValueError?
        except ValueError:
            return None
