from pymoo.algorithms.moo.nsga2 import NSGA2
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score, accuracy_score
import fomo

shared_args = dict(
    # accuracy_metrics=[make_scorer(roc_auc_score, greater_is_better=False, needs_proba=True)],
    accuracy_metrics=[make_scorer(accuracy_score, greater_is_better=False, needs_proba=False)],
    # accuracy_metrics=[make_scorer(fomo.metrics.FPR, needs_proba=True)],
    fairness_metrics=[fomo.metrics.subgroup_FNR_scorer], 
    verbose=True,
    algorithm = NSGA2(pop_size = 50),
    n_jobs = 1
)