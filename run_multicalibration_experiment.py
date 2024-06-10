#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from read_file import read_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pmc.metrics import multicalibration_loss as mc_loss
import os
import sys
import random
import json
import json_numpy
import uuid
json_numpy.patch()

################################################################################
# arguments
################################################################################

################################################################################
# global variables
################################################################################
whichdataset = sys.argv[1]
assert whichdataset in ['bch','mimic']

BCH = whichdataset == 'bch'

if whichdataset == 'bch':
    dataset = 'data/bch_cleaned.r2.csv'
else:
    dataset = 'data/mimic4_admissions.csv'


FEATURES,LABEL,ENCODINGS=read_file(
    dataset,
    label='y',
    text_features=['chiefcomplaint'],
    one_hot_encode=[True],
    ohc_min_frac=0.01
)
if BCH:
    ALL_GROUPS = ['gender','race','ethnicity']
else:
    ALL_GROUPS = ['gender','ethnicity']


################################################################################
# load data
################################################################################

if len(sys.argv) > 2:
    rdir = sys.argv[2]
else:
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M%S")
    rdir = f'results_{dt_string}'
    print('rdir: ',rdir)
os.makedirs(rdir,exist_ok=False)

################################################################################
# define experiment functions
################################################################################

def calibration_error(y_true, y_pred_proba, n_bins = 10, agg='mean'):
    """Return the expected or max calibration error. """
    bins = np.linspace(0.0, 1.0+1e-8, n_bins+1) 
    bin_idx = np.digitize(y_pred_proba, bins)
    errors = []
    probs = []
    for bin in np.unique(bin_idx):
        mask = (bin_idx==bin)
        yt = y_true[mask].mean()
        yp = y_pred_proba[mask].mean()
        errors.append(np.abs(yt-yp))
        probs.append(mask.sum()/len(y_true))
    if agg == 'mean':
        return np.sum([e*p for e,p in zip(errors,probs)])
    elif agg == 'max':
        return np.max(errors)

def max_calibration_error(df, n_bins=10):
    return calibration_error(df['y'], df['prediction'], agg='max',n_bins=n_bins)

def expected_calibration_error(df, n_bins=10):
    return calibration_error(df['y'], df['prediction'], agg='mean',n_bins=n_bins)


def categorize(y_true, y_pred_proba, X):
    """calculate MSE and MAE for each sample given a prediction on data"""
    #calculate overall, ethnicity, gender, and intersection false negative rate 
    # for each scenario
    out=pd.DataFrame({
        'ind':X.index,
        'ethnicity':X['ethnicity'].apply(lambda x: ENCODINGS['ethnicity'][x]),
        'gender':X['gender'].apply(lambda x: ENCODINGS['gender'][x]),
        'y':y_true,
        'prediction':y_pred_proba
    })
    if BCH:
        out['race'] = X['race'].apply(lambda x: ENCODINGS['race'][x])
    out['MSE']=(y_true - y_pred_proba)**2
    out['MAE']=np.abs(y_true - y_pred_proba)
    return out

def groupwise_evaluation(y_true, y_pred_proba, X, n_bins):
    """Function that takes prediction data ane summarizes the overall, marginal,
    and intersectional MSE and MAE and calculates the marginal and 
    intersectional group prevalences.
    """
    temp=categorize(y_true, y_pred_proba, X)
    if BCH:
        temp['race,ethnicity,gender'] = temp['race']+','+temp['ethnicity']+","+temp['gender']
    else:
        temp['ethnicity,gender'] = temp['ethnicity']+","+temp['gender']
    n=len(y_true)   
    res = {
        'grouping_overall':{},
        'grouping_ethnicity':{},
        'grouping_gender':{},
        'grouping_intersection':{},
        'ethnicity_prevalence':{},
        'gender_prevalence':{},
        'intersection_prevalence':{}
    }
    if BCH:
        res['grouping_race']  = {}
    for fn in [max_calibration_error, expected_calibration_error]:
        m = fn.__name__
        def fnl(x):
            return fn(x, n_bins=n_bins)
        res['grouping_overall'][m] = fnl(temp)
        res['grouping_ethnicity'][m] = temp.groupby(['ethnicity']).apply(fnl).to_dict()
        res['grouping_gender'][m] = temp.groupby(['gender']).apply(fnl).to_dict()
        if BCH:
            res['grouping_race'][m] = temp.groupby(['race']).apply(fnl).to_dict()
            res['grouping_intersection'][m] = temp.groupby('race,ethnicity,gender').apply(fnl).to_dict()
        else:
            res['grouping_intersection'][m] = temp.groupby('ethnicity,gender').apply(fnl).to_dict()
    res['ethnicity_prevalence'] = (temp.groupby(['ethnicity']).size()/n).to_dict()
    res['gender_prevalence'] = (temp.groupby(['gender']).size()/n).to_dict()
    if BCH:
        res['race_prevalence'] = (temp.groupby(['race']).size()/n).to_dict()
        res['intersection_prevalence'] = (temp.groupby('race,ethnicity,gender').size()/n).to_dict()
    else:
        res['intersection_prevalence'] = (temp.groupby('ethnicity,gender').size()/n).to_dict()
    return res

def experiment(alpha=None,gamma=None,scen=None,grouping=None,groups=None,
               est=None, group_label="",random_state=None,n_bins=10,**kwargs):
    """
    alpha : alpha for that run
    gamma : gamma for that run
    scen : which scenario of the experiment is being run
           
    group_label : used to name the group speciiced in scenario 2, for all 
        others, it is ""
    grouping : if the estimator itself is a marginal (scenarios 2 and 3) or 
        intersectional fairness model  (scenario 
    groups : which groups are protected (for scenario 1 this is none, for 
        scenario 2, it will be gender or ethnicity, and for scenario 3 and 4 
        it will be a list of gender and ethnicity
    estimator : the estimator object
    """
    runid = uuid.uuid4() 
    print('scenario:',scen,
        'grouping:',grouping, 
        'groups:', groups, 
        'alpha:', alpha,
        'gamma:', gamma,
        'est:', type(est).__name__, 
        'random_state:',random_state
        )

    # split data, stratifying by groups and label
    if BCH:
        strat_groups = ['gender','ethnicity']
    else:
        strat_groups = ALL_GROUPS
    stratify = ""
    for f in strat_groups:
        stratify += FEATURES[f].astype(str)
    stratify += LABEL.astype(str)

    X_train, X_test, y_train, y_test=train_test_split(
        FEATURES,
        LABEL,
        train_size=0.5,
        test_size=0.5,
        random_state=random_state,
        stratify=stratify

    )
    # set multicalibrator parameters
    for k,v in [('alpha',alpha),
                ('gamma',gamma),
                ('random_state',random_state),
                ('n_bins',n_bins)
                ]:
        if hasattr(est, k):
            setattr(est, k, v) 
    if hasattr(est, 'auditor_type'):
        setattr(est.auditor_type, 'groups', groups) 
        setattr(est.auditor_type, 'grouping', grouping) 

    for k,v in kwargs.items():
        if hasattr(est, k):
            setattr(est, k, v) 

    # train model
    est.fit(X_train,y_train)
    y_pred_proba_train = est.predict_proba(X_train)[:,1]
    y_pred_proba_test = est.predict_proba(X_test)[:,1]

    scenario=f'Scenario {scen} {group_label}'
    auc=roc_auc_score(y_test, y_pred_proba_test) 
    mc_intersection=mc_loss(
        estimator=est,
        X=X_test,
        y_true=y_test,
        groups=ALL_GROUPS,
        grouping='intersectional',
        alpha=alpha,
        gamma=gamma,
        n_bins=n_bins
    )
    if scen==1:
        mc_train = None
        mc_test = None
    else:
        mc_train=mc_loss(
            estimator=est,
            X=X_train,
            y_true=y_train,
            groups=groups,
            grouping=grouping,
            alpha=alpha,
            gamma=gamma,
            n_bins=n_bins
        )
        mc_test=mc_loss(
            estimator=est,
            X=X_test,
            y_true=y_test,
            groups=groups,
            grouping=grouping,
            alpha=alpha,
            gamma=gamma,
            n_bins=n_bins
        )

    res = {'alpha': alpha,
           'gamma':gamma,
           'n_bins':n_bins,
           "Scenario": scenario,
           "AUC":auc,
           # todo: add accuracy, precision recall
           "MC_train":mc_train,
           "MC_test":mc_test,
           "mc_intersection":mc_intersection,
           'runid':str(runid),
           'dataset':'BCH' if BCH else 'MIMIC-IV',
           'seed':int(random_state)
          }
    res['train'] = groupwise_evaluation(y_train, y_pred_proba_train, X_train, n_bins)
    res['test'] = groupwise_evaluation(y_test, y_pred_proba_test, X_test, n_bins)
    res.update(kwargs)
    filename = rdir + '/'
    descriptors = [
        # f'alpha-{alpha}',
        # f'gamma-{gamma}',
        # f'{scenario}'.replace(' ','-'),
        # f'random_state-{random_state}',
        f'runid-{runid}'
    ]
    filename += '_'.join(descriptors) + '.json'
    print('saving',filename)
    with open(filename, 'w') as of:
        json.dump(res, of)

    # save predictions for AUROC curves
    df_preds = pd.DataFrame({
        'y_true':y_test,
        'y_pred_proba_test':y_pred_proba_test,
        'runid':runid,
        'dataset':'BCH' if BCH else 'MIMIC-IV',
        'seed':random_state
    })
    # add patient group info to predictions for convenience
    for g in ALL_GROUPS:
        df_preds.loc[:,g] = X_test[g]

    df_preds.to_csv(f'{rdir}/runid-{runid}_preds.csv')

    return res



################################################################################
# set experiment parameters
################################################################################

from ml.lr import est as lr
from ml.mc_lr import est as mc_lr
from ml.mc_rf import est as mc_rf
from ml.rf import est as rf
from sklearn.base import clone
scen_params = []
for base_est in ['rf','lr']:
    if base_est == 'lr':
        scen1_est = lr
        mc_est = mc_lr
    else:
        scen1_est = rf
        mc_est = mc_rf
    scen1_params={
        'scen':1,
        'grouping': None,
        'groups': None,
        'est': clone(scen1_est),
        'base_est': base_est
    }
    if BCH:
        scen2_race_params={
            'scen':2,
            'grouping': 'marginal',
            'groups': ['race'],
            'est': clone(mc_est),
            'group_label': 'race',
            'base_est': base_est
        }
    scen2_gender_params={
        'scen':2,
        'grouping': 'marginal',
        'groups': ['gender'],
        'est': clone(mc_est),
        'group_label': 'Gender',
        'base_est': base_est
    }
    scen2_ethnicity_params={
        "scen":2,
        'grouping': 'marginal',
        'groups': ['ethnicity'],
        'est': clone(mc_est),
        'group_label': 'Ethnicity',
        'base_est': base_est
    }
    scen3_params={
        "scen":3,
                  'grouping':'marginal',
                  'groups':ALL_GROUPS,
                  'est':clone(mc_est),
                  'base_est':base_est
                 }
    scen4_params={
        "scen":4,
                  'grouping':'intersectional',
                  'groups':ALL_GROUPS, 
                  'est':clone(mc_est),
                  'base_est':base_est
                 }
    scen_params.extend([
        scen1_params,
        scen2_gender_params,
        scen2_ethnicity_params,
        scen3_params,
        scen4_params
        ]
    )
    if BCH: 
        scen_params.extend([scen2_race_params])


################################################################################
# run experiment
################################################################################

from pqdm.processes import pqdm
from tqdm import tqdm
import itertools as it

random.seed(11301991)
alphas=[0.001,0.01,0.05,0.1,0.2]
gammas=[0.001,0.01,0.1]
n_bins=[5]
n_trials=100
seeds = np.random.randint(1,2**15,n_trials) 
results=[]
args=[]

# construct a list of arguments
for s,a,g,p,nb in it.product(seeds, alphas, gammas, scen_params, n_bins):
    args.append(
        {'alpha':a,
         'gamma':g,
         'random_state':s,
         'n_bins':nb
        }
    )
    args[-1].update(p)
print('running',len(args),'experiments, saving to',rdir)
DEBUG=False
N_JOBS=48
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    if DEBUG:
        #################
        # serial version
        results=[]
        for a in tqdm(args):
            results.append(experiment(**a)) 
        #################
    else:
        #################
        # run in parallel. set n_jobs to the number of parallel jobs you want to run 
        results = pqdm(
            args, 
            experiment, 
            n_jobs=min(len(args),N_JOBS), 
            argument_type='kwargs'
        ) 