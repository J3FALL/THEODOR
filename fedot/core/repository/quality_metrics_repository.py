from typing import Callable

from fedot.core.composer.metrics import F1Metric, MaeMetric, RmseMetric, RocAucMetric, SilhouetteMetric, \
    StructuralComplexityMetric
from fedot.core.utils import ComparableEnum as Enum


class MetricsEnum(Enum):
    pass


class QualityMetricsEnum(MetricsEnum):
    pass


class ComplexityMetricsEnum(MetricsEnum):
    node_num = 'node_number'
    structural = 'structural'


class ClusteringMetricsEnum(QualityMetricsEnum):
    silhouette = 'silhouette'


class ClassificationMetricsEnum(QualityMetricsEnum):
    ROCAUC = 'roc_auc'
    ROCAUC_penalty = 'roc_auc_pen'
    precision = 'precision'
    f1 = 'f1'


class RegressionMetricsEnum(QualityMetricsEnum):
    RMSE = 'rmse'
    RMSE_penalty = 'roc_auc_pen'
    MAE = 'mae'


class MetricsRepository:
    __metrics_implementations = {
        ClassificationMetricsEnum.ROCAUC: RocAucMetric.get_value,
        ClassificationMetricsEnum.ROCAUC_penalty: RocAucMetric.get_value_with_penalty,
        RegressionMetricsEnum.MAE: MaeMetric.get_value,
        RegressionMetricsEnum.RMSE: RmseMetric.get_value,
        RegressionMetricsEnum.RMSE_penalty: RmseMetric.get_value_with_penalty,
        ClassificationMetricsEnum.f1: F1Metric.get_value,
        ComplexityMetricsEnum.structural: StructuralComplexityMetric.get_value,
        ClusteringMetricsEnum.silhouette: SilhouetteMetric.get_value
    }

    def metric_by_id(self, metric_id: MetricsEnum) -> Callable:
        return self.__metrics_implementations[metric_id]
