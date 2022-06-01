

from scipy import stats

from common.constant import CommonVars


class CorCoefficient:

    @staticmethod
    def measure(x1: list, x2: list, measure_metrics: str = CommonVars.AvgCorrelation) -> float:
        """
        Measure the correlation coefficient between x1 and x2
        It requires that each dataset be normally distributed.
        :param x1: list1
        :param x2: list2
        :param measure_metrics: str
        :return: correlation，
            Like other correlation coefficients, this one varies between -1 and +1 with 0 implying no correlation.
            Correlations of -1 or +1 imply an exact linear relationship
        """

        if measure_metrics == CommonVars.KendallTau:
            correlation, p_value = stats.kendalltau(x1, x2)
        elif measure_metrics == CommonVars.KendallTau:
            correlation, p_value = stats.spearmanr(x1, x2)
        elif measure_metrics == CommonVars.KendallTau:
            correlation, p_value = stats.pearsonr(x1, x2)
        elif measure_metrics == CommonVars.AvgCorrelation:
            # calculate average over all
            correlation1, p_value = stats.kendalltau(x1, x2)
            correlation2, p_value = stats.spearmanr(x1, x2)
            correlation3, p_value = stats.pearsonr(x1, x2)
            correlation = (correlation1+correlation2+correlation3) / 3
        else:
            raise NotImplementedError(measure_metrics + " is not implemented")

        return correlation

