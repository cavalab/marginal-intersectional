#https://github.com/cavalab/popp_fairness/issues/6
import os
import numpy as np
import pandas as pd
import argparse
import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict

COLUMNS_TO_DROP = [
    'Unnamed: 0', 
    'Contact Serial Number',
    'ED Checkin Dt Tm',
    'ED Checkout Dt Tm', 
    'MRN',
    'ED Derived Disposition',
    'Race Line',
    'Patient Death Ind',
    'ED Room Number',
    'CASE_NUM',
    'ED LOS'
]
# demographics
A = ['Race','Ethnicity','Gender','Age']

A_rename = {
    'Hispanic Yes No': 'Ethnicity',
    'age_group': 'Age'
}

A_options = {
    'Race':defaultdict(lambda: 'Other'),
    'Ethnicity': {
       'Yes':'HL',
       'No':'NHL'
    },
    'Gender': {
        'M':'M',
        'F':'F'
    },
# leave age as is
#    'Age': {
#        '>5Y':'older than 5Y',
#        '18M-3Y':'5Y or younger', 
#        '3-5Y':'5Y or younger', 
#        '12-18M':'5Y or younger', 
#        '0-3M':'5Y or younger', 
#        '6-12M':'5Y or younger', 
#        '3-6M':'5Y or younger'
#    }
}

A_options['Race'].update({
        'Black or African American':'Black', 
        'White':'white', 
        'Asian':'Asian',
        'American Indian or Alaska Native':'AI',
        'Native Hawaiian or Other Pacific Islander':'NHPI', 
})

# Need to use the data file, might need to merge dem with data file
def clean_dems(df):
    """Re-codes demographics according to dictionaries above"""
    df = df.rename(columns = A_rename)
    # if patient's race is "hispanic or latino", make sure their ethnicity is too
    df.loc[df['Race']=='Hispanic or Latino','Ethnicity'] = 'Yes'
    for a in A:
        print('original encoding:\n',df[a].value_counts()/len(df[a]))
        # a_to_drop = [c for c in df[a].unique() if c not in A_options[a].keys()]
        # print('On the chopping block:',a_to_drop)
        if a in A_options.keys():
            if a != 'Race':
                df = df.loc[df[a].isin(list(A_options[a].keys())),:]
            df[a] = df[a].apply(lambda x: A_options[a][x])
        print('new encoding:\n',df[a].value_counts()/len(df[a]))
    return df

def process_data(data,dem,results_path):
    print('loading and processing BCH files...')
    df = pd.read_csv(dem, index_col=None).drop(columns=['Miles Traveled'])
    


    print('merging covariates')
    df_data = pd.read_csv(data, index_col=None).drop(columns=['Unnamed: 0'])
    overlap_cols = [c for c in df.columns if c in df_data.columns] 
    df = df.merge(df_data,on=overlap_cols)
    print('merged data size:',df.shape)
    print('# patients',df['MRN'].nunique())

    print('Renaming the categorical features.')
    df = clean_dems(df)
    print('data size:',df.shape)
    print('# patients',df['MRN'].nunique())
    # rename data
    df = df.rename(columns={"Gender": "gender", 
                         "Ethnicity": "ethnicity",
                         "Race": "race",
                         "isCase": "y", 
                         "ED Complaint 1": 
                         "chiefcomplaint"})

    # remove deceased patients
    print('removing deceased')
    df = df.loc[~df["Patient Death Ind"]]
    print('data size:',df.shape)
    print('# patients',df['MRN'].nunique())

    print('remove patients with unkown Triage acuity')
    df = df.loc[df['ED Triage Acuity']!='Unknown']
    print('data size:',df.shape)
    print('# patients',df.groupby('y')['MRN'].nunique())

    print('dropping columns...')
    df = df.drop(COLUMNS_TO_DROP,axis = 1)
    print('data size:',df.shape)

    # remove features with very low prevalence to reduce feature dimensionality
    is_bool = df.dtypes == 'bool'
    bool_features = np.asarray(df.columns)[is_bool]
    print(len(bool_features),'boolean features')
    drop_features = []
    for b in bool_features: 
        prev = df[b].sum()/len(df)
        if prev < 0.01:
            drop_features.append(b)
            
    print('dropping',len(drop_features),'boolean features with < 1% prevalence')
    df = df.drop(columns=drop_features)
    print('data size:',df.shape)

    print('remove any samples with an unknown outcome')
    df = df.dropna(subset = ['y'])
    print('data size:',df.shape)

    print('finished processing dataset.')
    print(f'size: {df.shape}, cases: {df.y.sum()/len(df)}')
    print('dataset columns:',df.columns)
    print(df)
    print(df.describe())

    print(f'saving to {results_path}...')
    df.to_csv(results_path,index= False)
    print('done.')

# add more options here later
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Input the file location for BCH files", add_help=False)
    parser.add_argument('-bch_path', action='store', type=str,
                        default='./popp/model.data.60mins_forBill/',
                        help='Path for admission file')
    parser.add_argument('-Data_File', action='store', type=str,
                        default='model.data.60mins_forBill.csv',
                        help='Path for Full Data')
    parser.add_argument('-Dem_File', action='store', type=str,
                        default='demographics_forBill.csv',
                        help='Path for Dem Data')    
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-p', action='store',
                        dest='PATH',default='data/bch_cleaned.r2.csv',type=str,
            help='Path of Saved final fire')
    args = parser.parse_args()

    process_data(
        os.path.join(args.bch_path, args.Data_File), 
        os.path.join(args.bch_path, args.Dem_File), 
        args.PATH
    ) 
 
