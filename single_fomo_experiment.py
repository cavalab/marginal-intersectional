import pandas as pd
from read_file import read_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, make_scorer, balanced_accuracy_score,
    f1_score
)
from fomo.metrics import subgroup_FPR_loss, subgroup_FNR_loss
import fomo.metrics
import os
import json
import json_numpy
import uuid

json_numpy.patch()
import time

################################################################################
# define experiment functions
################################################################################
def categorize(y_true, y_pred_proba, X, encodings, all_groups):
    """calculate TP, TN, FP, FN, for y_pred_proba"""
    # calculate overall, ethnicity, gender, and intersection false negative rate for each scenario
    label_bool = y_true.astype(bool)
    prediction_bool = y_pred_proba > 0.5

    out = pd.DataFrame(
        {"ind": X.index, "y": y_true, "prediction": prediction_bool.astype(int)}
    )

    for g in all_groups:
        out[g] = X[g].apply(lambda x: encodings[g][x])

    out["TP"] = (label_bool & prediction_bool).astype(int)
    out["TN"] = (~label_bool & ~prediction_bool).astype(int)
    out["FP"] = (~label_bool & prediction_bool).astype(int)
    out["FN"] = (label_bool & ~prediction_bool).astype(int)
    return out


def groupwise_evaluation(y_true, y_pred_proba, X, encodings, all_groups):
    """Function that takes prediction data ane summarizes the overall, marginal, and intersection
    TP, TN, FP, and FN rate and calculates the marginal and intersectionl roup prevalences.
    Commented out text is deprecated post-processing.
    """
    temp = categorize(y_true, y_pred_proba, X, encodings, all_groups)
    intergroup = ",".join(all_groups)
    temp[intergroup] = temp[all_groups[0]]
    for g in all_groups[1:]:
        temp[intergroup] += "," + temp[g]
        # temp['ethnicity']+","+temp['gender']+","+temp['race']
    n = len(y_true)
    pred_metrics = {
        "grouping_overall": temp[["TP", "TN", "FP", "FN"]].mean(),
        "grouping_intersection": temp.groupby(intergroup)[
            ["TP", "TN", "FP", "FN"]
        ].mean(),
        "intersection_prevalence": temp.groupby(intergroup).size() / n,
        "intersection_outcome_rate": temp.groupby(intergroup)["y"].mean(),
    }
    for g in all_groups:
        pred_metrics[f"grouping_{g}"] = temp.groupby([g])[
            ["TP", "TN", "FP", "FN"]
        ].mean()
        pred_metrics[f"{g}_prevalence"] = temp.groupby([g]).size() / n
        pred_metrics[f"{g}_outcome_rate"] = temp.groupby([g])["y"].mean()

    # include 2-way groupings if there are more than 2 intersecting attributes.
    if len(all_groups)>2:
        import itertools
        for g1,g2 in itertools.combinations(all_groups,2):
            intergroup = ','.join([g1,g2])
            temp[intergroup] = [','.join([tg1,tg2]) for tg1,tg2 in zip(temp[g1],temp[g2])]
            tg = temp.groupby(intergroup)
            pred_metrics[f"grouping_{g1}-{g2}"] = (
                tg[ ["TP", "TN", "FP", "FN"] ].mean()
            )
            pred_metrics[f"{g1}-{g2}_prevalence"] = tg.size() / n
            pred_metrics[f"{g1}-{g2}_outcome_rate"] = tg["y"].mean()

    pred_metrics = {k: v.to_dict() for k, v in pred_metrics.items()}
    pred_metrics["grouping_overall"]["subgroup_FPR"] = subgroup_FPR_loss(
        y_true,
        y_pred_proba,
        X[all_groups],
        grouping="intersectional",
        abs_val=False,
        gamma=True,
    )
    pred_metrics["grouping_overall"]["subgroup_FNR"] = subgroup_FNR_loss(
        y_true,
        y_pred_proba,
        X[all_groups],
        grouping="intersectional",
        abs_val=False,
        gamma=True,
    )
    pred_metrics["grouping_overall"]["subgroup_FPR_abs"] = subgroup_FPR_loss(
        y_true,
        y_pred_proba,
        X[all_groups],
        grouping="intersectional",
        abs_val=True,
        gamma=True,
    )
    pred_metrics["grouping_overall"]["subgroup_FNR_abs"] = subgroup_FNR_loss(
        y_true,
        y_pred_proba,
        X[all_groups],
        grouping="intersectional",
        abs_val=True,
        gamma=True,
    )

    return pred_metrics


def setup_fomo(
    base_est, scenario, problem, all_groups, fairness_metric, accuracy_metric
):
    """Choose a fomo estimator and define grouping, groups based on args"""

    if base_est == "lr":
        print(base_est)
        fomo_est_MLP = fomo_lr_MLP
        fomo_est_lin = fomo_lr_lin
        fomo_est_interlin = fomo_lr_interlin
    else:
        print(base_est)
        fomo_est_MLP = fomo_rf_MLP
        fomo_est_lin = fomo_rf_lin
        fomo_est_interlin = fomo_rf_interlin
    if scenario == "Base":
        est = clone(lr) if base_est == "lr" else clone(rf)
        grouping = None
        groups = None
    elif scenario in ["Gender", "Ethnicity", "Race"]:
        if problem == "MLP":
            est = clone(fomo_est_MLP)
        else:
            est = clone(fomo_est_lin)
        grouping = "marginal"
        groups = [scenario.lower()]
    else:
        if problem == "MLP":
            est = clone(fomo_est_MLP)
        elif scenario == "Marginal":
            est = clone(fomo_est_lin)
        else:
            est = clone(fomo_est_interlin)
        print("scenario")
        grouping = scenario.lower()
        groups = all_groups

    # customize fairness metrics
    if fairness_metric is not None:
        if fairness_metric == "FNR":
            est.fairness_metrics = [fomo.metrics.subgroup_FNR_scorer]
        elif fairness_metric == "FPR":
            est.fairness_metrics = [fomo.metrics.subgroup_FPR_scorer]
        else:
            raise ValueError(f"{fairness_metric} is not a valid fairness_metric")

    # customize accuracy metrics
    if accuracy_metric is not None:
        if accuracy_metric == "AUROC":
            est.accuracy_metrics = [
                make_scorer(roc_auc_score, greater_is_better=False, needs_proba=True)
            ]
        elif accuracy_metric == "FPR":
            est.accuracy_metrics = [make_scorer(fomo.metrics.FPR, needs_proba=True)]
        elif accuracy_metric == "accuracy":
            est.accuracy_metrics = [
                make_scorer(accuracy_score, greater_is_better=False, needs_proba=False)
            ]
        elif accuracy_metric == "balanced_accuracy":
            est.accuracy_metrics = [
                make_scorer(balanced_accuracy_score, greater_is_better=False, needs_proba=False)
            ]
        elif accuracy_metric == "f1":
            est.accuracy_metrics = [
                make_scorer(f1_score, greater_is_better=False, needs_proba=False)
            ]
        else:
            raise ValueError(f"{accuracy_metric} is not a valid accuracy_metric")

    return est, grouping, groups


def experiment(
    scenario=None,
    base_est=None,
    gamma=None,
    problem=None,
    fairness_metric=None,
    accuracy_metric=None,
    seed=0,
    rdir="",
    all_groups=None,
    dataset=None,  # 'data/bch_cleaned.r1.csv', 'data/mimic 'data/mimic4_admissions.csv',
    label="y",
    text_features=["chiefcomplaint"],
    one_hot_encode=[True],
    ohc_min_frac=0.01,
    n_gen=50,
    job_name=None,
):
    """
    scenario : which scenario of the experiment is being run. one of:
        Base
        Race
        Ethnicity
        Gender
        Marginal
        Intersectional
    base_est: 'lr' or 'rf'
        the underlying ML model for fomo
    gamma: True or False
        whether to account for group probability in fomo
    problem: fomo specification of how to encode weights
    fairness_metric: FNR or FPR
    seed: random state
    rdir: where to save results
    all_groups: column names to group by for fairness
    dataset: filename of dataset
    label: target name in dataset
    text_features: features to be encoded
    one_hot_encode: whether to one hot encode categorical features
    ohc_min_frac: minimum fraction of samples of category to be one hot encoded
    """
    start = time.time()
    runid = uuid.uuid4()

    if job_name is None:
        filename = f"{rdir}/runid-{runid}"
    else:
        filename = f"{rdir}/{job_name}"

    # manage all_groups
    BCH = "bch" in dataset
    if all_groups is None:
        if BCH:
            all_groups = ["gender", "race", "ethnicity"]
        else:
            all_groups = ["gender", "ethnicity"]
    elif type(all_groups) is not list:
        all_groups = list(all_groups)

    ########################################
    # setup data
    features, label, encodings = read_file(
        dataset,
        label=label,
        text_features=text_features,
        one_hot_encode=one_hot_encode,
        ohc_min_frac=ohc_min_frac,
    )
    stratify = (
        features["gender"].astype(str)
        + features["ethnicity"].astype(str)
        + label.astype(str)
    )
    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        label,
        train_size=0.1 if "mimic" in dataset else 0.5,
        test_size=0.5,
        random_state=seed,
        stratify=stratify,
    )
    ########################################

    est, grouping, groups = setup_fomo(
        base_est, scenario, problem, all_groups, fairness_metric, accuracy_metric
    )

    if fairness_metric is None:
        fairness_metric = est.fairness_metrics[0].__name__
    if accuracy_metric is None:
        if hasattr(est.accuracy_metrics[0],'_score_func'):
            accuracy_metric = est.accuracy_metrics[0]._score_func.__name__
        else:
            accuracy_metric = est.accuracy_metrics[0].__name__

    print(
        "scenario:",
        scenario,
        "grouping:",
        grouping,
        # 'fairness_metric:',est.fairness_metric,
        # 'accuracy_metric:',est.fairness_metric,
        # 'base_est:', base_est,
        # 'gamma:', gamma,
        # 'problem:', problem
    )
    print("Estimator settings:")
    for k, v in vars(est).items():
        print("\t", k, "=", v)

    ########################################
    # train
    if scenario == "Base":
        est.fit(X_train, y_train)
    else:
        est.fit(
            X_train,
            y_train,
            gamma=gamma,
            grouping=grouping,
            protected_features=groups,
            termination=("n_gen", n_gen),
        )
        pareto_df = est.get_pareto_points()
        pareto_df.to_csv(f"{filename}_pareto_front.csv", index=False)
    ########################################

    y_pred_proba_train = est.predict_proba(X_train)[:, 1]
    y_pred_proba_test = est.predict_proba(X_test)[:, 1]

    # save predictions for AUROC curves
    df_preds = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred_proba_test": y_pred_proba_test,
            "runid": runid,
            "dataset": "BCH" if BCH else "MIMIC-IV",
            "seed": seed,
        }
    )
    # add patient group info to predictions for convenience
    for g in all_groups:
        df_preds.loc[:, g] = X_test[g]

    os.makedirs(rdir, exist_ok=True)
    df_preds.to_csv(f"{filename}_preds.csv")

    auc = roc_auc_score(y_test, y_pred_proba_test)

    # calculate results and store

    res = dict(
        runid=str(runid),
        job_name=job_name,
        scenario=scenario,
        grouping=grouping,
        seed=seed,
        metric=fairness_metric,  # compatibility
        fairness_metric=fairness_metric,
        accuracy_metric=accuracy_metric,
        AUC=auc,
        base_est=base_est,
        gamma=gamma,
        problem=problem,
        dataset=dataset,
        time=time.time() - start,
    )

    res["train"] = groupwise_evaluation(
        y_train, y_pred_proba_train, X_train, encodings, all_groups
    )
    res["test"] = groupwise_evaluation(
        y_test, y_pred_proba_test, X_test, encodings, all_groups
    )

    print("saving results to", filename + ".json")

    with open(filename + ".json", "w") as of:
        json.dump(res, of)
    return res


################################################################################
# set experiment parameters
################################################################################

from ml.lr import est as lr
from ml.fomo_lr import est_MLP as fomo_lr_MLP
from ml.fomo_lr import est_lin as fomo_lr_lin
from ml.fomo_lr import est_interlin as fomo_lr_interlin
from ml.fomo_rf import est_MLP as fomo_rf_MLP
from ml.fomo_rf import est_lin as fomo_rf_lin
from ml.fomo_rf import est_interlin as fomo_rf_interlin
from ml.rf import est as rf
from sklearn.base import clone

################################################################################
# run experiment
################################################################################
# run the experiment
# experiment(scenario = 'Intersectional', metric='FNR', seed=215, base_est='lr')
import fire

if __name__ == "__main__":
    fire.Fire(experiment)
