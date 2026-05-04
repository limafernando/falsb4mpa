from numpy import mean, where
from falsb4mpa.evaluation.baseline_metrics import FPR, PR, TPR
from falsb4mpa.evaluation.grouping_functions import intersectional_subgroup


def wc_spd(y_hat, a1, a1dim, a2, a2dim):
    if a1dim == 1 and a2dim == 1:
        sg1_rate = intersectional_subgroup(PR, a1, a2, y_hat)
        sg2_rate = intersectional_subgroup(PR, 1 - a1, a2, y_hat)
        sg3_rate = intersectional_subgroup(PR, a1, 1 - a2, y_hat)
        sg4_rate = intersectional_subgroup(PR, 1 - a1, 1 - a2, y_hat)

        sg_rates = [sg1_rate, sg2_rate, sg3_rate, sg4_rate]

        return max(sg_rates) - min(sg_rates)

    else:
        pass


def opt_wc_spd(wc_spd):
    return 1 - wc_spd


def wc_aod(y_real, y_hat, a1, a1dim, a2, a2dim):
    if a1dim == 1 and a2dim == 1:
        sg1_tpr_rate = intersectional_subgroup(TPR, a1, a2, y_real, y_hat)
        sg2_tpr_rate = intersectional_subgroup(TPR, 1 - a1, a2, y_real, y_hat)
        sg3_tpr_rate = intersectional_subgroup(TPR, a1, 1 - a2, y_real, y_hat)
        sg4_tpr_rate = intersectional_subgroup(TPR, 1 - a1, 1 - a2, y_real, y_hat)

        sg1_fpr_rate = intersectional_subgroup(FPR, a1, a2, y_real, y_hat)
        sg2_fpr_rate = intersectional_subgroup(FPR, 1 - a1, a2, y_real, y_hat)
        sg3_fpr_rate = intersectional_subgroup(FPR, a1, 1 - a2, y_real, y_hat)
        sg4_fpr_rate = intersectional_subgroup(FPR, 1 - a1, 1 - a2, y_real, y_hat)

        sg_rates = [
            sg1_tpr_rate + sg1_fpr_rate,
            sg2_tpr_rate + sg2_fpr_rate,
            sg3_tpr_rate + sg3_fpr_rate,
            sg4_tpr_rate + sg4_fpr_rate,
        ]

        return (max(sg_rates) - min(sg_rates)) * 0.5

    else:
        pass


def opt_wc_aod(wc_aod):
    return 1 - wc_aod


def wc_eod(y_real, y_hat, a1, a1dim, a2, a2dim):

    if a1dim == 1 and a2dim == 1:
        sg1_rate = intersectional_subgroup(TPR, a1, a2, y_real, y_hat)
        sg2_rate = intersectional_subgroup(TPR, 1 - a1, a2, y_real, y_hat)
        sg3_rate = intersectional_subgroup(TPR, a1, 1 - a2, y_real, y_hat)
        sg4_rate = intersectional_subgroup(TPR, 1 - a1, 1 - a2, y_real, y_hat)

        sg_rates = [sg1_rate, sg2_rate, sg3_rate, sg4_rate]

        return max(sg_rates) - min(sg_rates)

    else:
        pass


def opt_wc_eod(wc_eod):
    return 1 - wc_eod
