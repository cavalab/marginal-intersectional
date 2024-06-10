import pandas as pd

def get_groupwise_results(data, include_outcome_rates=True):
    """Processes the results of `groupwise_evaluation` and return."""
    res = {}
    for fold in ['train','test']:
        # reformat grouping_overall to match other dicts
        data[fold]['grouping_overall'] = {
            k:{'OVERALL':v} 
            for k,v in data[fold]['grouping_overall'].items()
        }
        # add grouping metrics
        grouping_keys = [k for k in data[fold].keys() if 'grouping_' in k]
        res[fold] = pd.concat([ 
            pd.DataFrame(data[fold][g]) for g in grouping_keys 
        ])
        # add columns contain prevalence

        prevalence_keys = [k for k in data[fold].keys() if '_prevalence' in k]
        for g in prevalence_keys:
            if g not in data[fold]:
                continue
            df = pd.DataFrame({g:data[fold][g]})
            res[fold].loc[df.index,'prevalence'] = df[g]
        res[fold].loc['OVERALL','prevalence'] = 1.0
        # train/test fold indicator
        res[fold]['fold'] = fold
        res[fold]['group_identity'] = res[fold].index

        if include_outcome_rates:
            out_keys = [k for k in data[fold].keys() if '_outcome_rate' in k] 
            # add columns contain outcome rates 
            for g in out_keys:
                df = pd.DataFrame({g:data[fold][g]})
                res[fold].loc[df.index,'outcome_rate'] = df[g]
            res[fold].loc['OVERALL','outcome_rate'] = 1.0
            # train/test fold indicator
            res[fold]['fold'] = fold
            res[fold]['group_identity'] = res[fold].index
    # concatenate into one dataframe
    df = pd.concat([res['train'], res['test']],ignore_index=True)
    return df

########################################
# clean up results data 

def renamer(k1,k2): 
    d = {
        'MIMIC-IV':{
            'Scenario 1 ': 'Base',
            'Scenario 2 Gender': 'Gender',
            'Scenario 2 Ethnicity': 'Ethnoracial',
            'Ethnicity': 'Ethnoracial',
            'Scenario 2 race': 'Race',
            'Scenario 3 ': 'Marginal',
            'Scenario 4 ': 'Intersectional'
        },
        'BCH': {
            'Scenario 1 ': 'Base',
            'Scenario 2 Gender': 'Gender',
            'Scenario 2 Ethnicity': 'Ethnicity',
            'Scenario 2 race': 'Race',
            'Scenario 3 ': 'Marginal',
            'Scenario 4 ': 'Intersectional'
        }
    }
    if k2 in d[k1]:
        return d[k1][k2]
    else:
        return k2
# harmonized intersectional group names
intersectional_groups = {
    'MIMIC-IV': [
        'AI/AN,F',
        'AI/AN,M',
        'ASIAN,F',
        'ASIAN,M',
        'BLACK/AA,F',
        'BLACK/AA,M',
        'HL,F',
        'HL,M',
        'WHITE,F',
        'WHITE,M'
    ],
    'BCH':[
        # 3 way
        'AI/AN,HL,F',
        'AI/AN,HL,M',
        'AI/AN,NHL,F',
        'AI/AN,NHL,M',
        'ASIAN,HL,F',
        'ASIAN,HL,M',
        'ASIAN,NHL,F',
        'ASIAN,NHL,M',
        'BLACK/AA,HL,F',
        'BLACK/AA,HL,M',
        'BLACK/AA,NHL,F',
        'BLACK/AA,NHL,M',
        'NHPI,HL,F',
        'NHPI,HL,M',
        'NHPI,NHL,F',
        'NHPI,NHL,M',
        'WHITE,HL,F',
        'WHITE,HL,M',
        'WHITE,NHL,F',
        'WHITE,NHL,M',
        # 2 way
        'AI/AN,F',
        'AI/AN,M',
        'ASIAN,F',
        'ASIAN,M',
        'BLACK/AA,F',
        'BLACK/AA,M',
        'NHPI,F',
        'NHPI,M',
        'WHITE,M',
        'WHITE,F',
        'HL,F',
        'HL,M',
        'NHL,F',
        'NHL,M',
        'AI/AN,HL',
        'AI/AN,NHL',
        'ASIAN,HL',
        'ASIAN,NHL',
        'BLACK/AA,HL',
        'BLACK/AA,NHL',
        'NHPI,HL',
        'NHPI,NHL',
        'WHITE,NHL',
        'WHITE,HL',
    ],
    'BCH-simple':[
        'AI/AN,F',
        'AI/AN,M',
        'ASIAN,F',
        'ASIAN,M',
        'BLACK/AA,F',
        'BLACK/AA,M',
        'NHPI,F',
        'NHPI,M',
        'WHITE,F',
        'WHITE,M',
    ]
}
scenario_order = [
    'Base',
    'Gender',
    'Ethnicity',
    'Ethnoracial',
    'Race',
    'Marginal',
    'Intersectional'
]

def relabel_group(x):
    return (x
            .replace('AMERICAN INDIAN/ALASKA NATIVE','AI/AN')
            .replace('Asian', 'ASIAN')
            .replace('BLACK/AFRICAN AMERICAN','BLACK/AA')
            .replace('Black','BLACK/AA')
            .replace('white','WHITE')
            .replace('HISPANIC/LATINO','HL')
            .replace('AI','AI/AN')
            .replace('AI/AN/AN','AI/AN')
           )
def reorder_group(x):
    #return ','.join(x.split(',')[2,0,1])
    print(x,'->',x.split(',')[1,2,0])
    return ','.join(x.split(',')[1,2,0])


def clean_results(df_results):
    """Standardize scenario names, result labels, and group names"""
    for dataset in df_results.dataset.unique():
        df_results.loc[df_results.dataset==dataset,'Scenario'] = (
            df_results.loc[df_results.dataset==dataset,'Scenario']
            .apply(lambda x: renamer(dataset,x))
        )

    df_results = df_results.rename(columns={
        'mc_intersection':'Multicalibration Intersectional Loss',
        'base_est':'Base Model'
    })
    df_results['Base Model'] = df_results['Base Model'].apply(
        lambda x: x.upper()
    )

    if 'group_identity' in df_results.columns:
        df_results['group_identity'] = df_results['group_identity'].apply(
            lambda x: relabel_group(x)
        )

    return df_results
