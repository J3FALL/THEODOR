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

    def get_value(self, true: InputData, predicted: OutputData):
        """Returns the value of metric defined by task"""
        try:
            return self._metric(reference=true, predicted=predicted)
        # TODO or raise ValueError?
        except ValueError:
            return None

    def get_round_value(self, true: InputData, predicted: OutputData):
        """Returns the value of metric defined by task but round up to 3 signs"""
        try:
            return round(self.get_value(true, predicted), 3)
        # TODO or raise ValueError?
        except ValueError:
            return None
