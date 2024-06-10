import os
import subprocess
import itertools as it
import time
import datetime
from datetime import date, timedelta
day = date.today()
ymdhms = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
import numpy as np
from tqdm import tqdm
import ipdb

def get_num_pending_jobs():
    sacct =( 
        subprocess
        .check_output(
            ['sacct', '-X', '--format=State'],
            # f'sacct -X --format=State | grep "PEND\|RUN" | wc -l',
            # shell=True)
        )
        .decode()
        .replace(' ','')
        .split('\n')
        )
    unique_states = np.unique(sacct)
    # print(unique_states)
    n = (len([s for s in sacct if 'PEND' in s]) 
         + len([s for s in sacct if 'RUN' in s])
    )
    # import pdb
    # pdb.set_trace()
    return n

def get_job_names():
    job_names =( 
        subprocess
        .check_output(
            ['sacct','-X','--format=JobName%100','--state','RUNNING,PENDING']
        )
        .decode()
        .replace(' ','')
        .split('\n')
        )
    return job_names
    # unique_states = np.unique(sacct)
    # # print(unique_states)
    # n = (len([s for s in sacct if 'PEND' in s]) 
    #      + len([s for s in sacct if 'RUN' in s]))
    # # import pdb
    # # pdb.set_trace()
    # return n
   
def fix_arg_type(v):
    if type(v) is str and ',' in v:
        v = v.split(',')
    if type(v) is not list:
        if type(v) is tuple:
            v = list(v)
        else:
            v = [v]
    return v

def run(
    rdir=f"results_fomo_{ymdhms}",
    ntrials=100, 
    # base_ests=("lr","rf"),
    base_ests=["rf"],
    fairness_metrics=["FNR"],
    accuracy_metrics=["AUROC"],
    scenarios: list[str]=(
        "Base",
        "Race",
        "Ethnicity",
        "Gender",
        "Marginal",
        "Intersectional"
    ),
    gammas :list[bool]=[True],
    all_groups: list[str]=None,
    problems :list[str]=["linear"],
    datasets :list[str]=['data/mimic4_admissions.csv','data/bch_cleaned.r1.csv'],
    max_time='05:00:00', # Running time (in hours-minutes-seconds)
    nodes=1,    # number of compute nodes
    ntasks=1,   # number of cpu cores on one node
    queue='bch-compute', # queue to be used
    memory=8192,
    job_submit_limit=2500,
    filter_matching_jobs=True,
    starting_trial=0
):
    """Submits cluster jobs based on the fomo experiment parameters specified."""
    seeds = []
    count = 1
    for i,line in enumerate(open('seeds.txt', 'r').readlines()):
        if i > starting_trial:
            seeds.append(int(line.strip()))
            count += 1
        if count == ntrials:
            break
    print('seeds:',seeds)

    datasets = fix_arg_type(datasets)
    scenarios = fix_arg_type(scenarios)
    accuracy_metrics = fix_arg_type(accuracy_metrics)
    gammas = fix_arg_type(gammas)
    base_ests = fix_arg_type(base_ests)

    job_names, job_files, job_cmds = [], [], []

    for (s, accuracy_metric, fairness_metric, scenario, base_est, gamma, problem, dataset) in it.product(
        seeds,
        accuracy_metrics,
        fairness_metrics,
        scenarios,
        base_ests,
        gammas,
        problems,
        datasets
        ):
        if scenario == 'Race' and 'mimic' in dataset.lower():
            continue

        datasetname = 'mimic' if 'mimic' in dataset else 'bch'
        job_name = (
            f'{datasetname}_{scenario}_{s}_{base_est}_gamma-{gamma}_{problem}_{accuracy_metric}_fair-{fairness_metric}'
        )

        job_cmd = (f'python -u single_fomo_experiment.py '
            f' --dataset {dataset}'
            f' --seed {s}'
            f' --scenario {scenario}'
            f' --base_est {base_est}'
            f' --gamma {gamma}'
            f' --problem {problem}'
            f' --rdir {rdir} '
            f' --job_name {job_name} '
            f' --accuracy_metric {accuracy_metric} '
            f' --fairness_metric {fairness_metric} '
        )
        if all_groups:
            job_cmd += f" --all_groups {','.join(list(all_groups))}"

        job_names.append(job_name)
        job_cmds.append(job_cmd)

    # os.makedirs(f'jobs_{ymdhms}', exist_ok=True)

    for jc in job_cmds[-10:]: print(jc)

    os.makedirs(rdir, exist_ok=True)
    print(f'There are {len(job_cmds)} total possible jobs')
    slurm_job_names = get_job_names()
    batch_files = []
    existing_jobs = 0
    queued_jobs = 0
    ########################################
    # write batch files
    ########################################
    pbar = tqdm(zip(job_names, job_cmds), total=len(job_names))
    for job_name,cmd in pbar:
        if filter_matching_jobs:
            if os.path.exists(f'{rdir}/{job_name}.json'):
                pbar.set_description(f'{job_name}.json already exists, skipping')
                existing_jobs += 1
                continue
            elif job_name in slurm_job_names:
                pbar.set_description(f'{job_name} is queued, skipping')
                queued_jobs += 1
                continue
        batch_script = '\n'.join([
            f'#!/bin/bash',
            # '#SBATCH --account=lacava',
            # '#SBATCH --qos=lacava',
            f'#SBATCH -J {job_name}',
            f'#SBATCH -o {rdir}/{job_name}.log',
            f'#SBATCH --nodes={nodes} ',
            f'#SBATCH --ntasks={ntasks} ',
            f'#SBATCH --partition={queue} ',
            f'#SBATCH --ntasks-per-node=1 ',
            f'#SBATCH --time={max_time} ',
            f'#SBATCH --mem-per-cpu={memory} ',
            # f'conda activate marg-int',
            f'conda info',
            #f'conda run -n marg-int {cmd}',
            f'ulimit -n 10000', # trying to stop "too many files" error
            f'{cmd}',
            f'sstat  -j   $SLURM_JOB_ID.batch   --format=JobID,MaxVMSize'
        ])
        # with open('tmp_script','w') as f:
        #     f.write(batch_script)
        batch_file = os.path.join(rdir,f'{job_name}.sh')
        with open(batch_file,'w') as of:
            of.write(batch_script)

        pbar.set_description(f'wrote {job_name}.sh')
        # store batch file
        batch_files.append(batch_file)

    ########################################
    # submit batch files, chaining submissions if job limit is reached
    ########################################
    print(f'Filtered {existing_jobs} jobs with existing results and '
          f'{queued_jobs} jobs that are already queued')
    c = input(f'About to submit {len(batch_files)} jobs. Continue? [Y/n]')
    if c.lower() == 'n':
        return

    submission_limit = job_submit_limit - get_num_pending_jobs()
    if not submission_limit > 0:
        print('there are',get_num_pending_jobs(), 'jobs in queue;submission limit is',submission_limit)
        print('cant submit any jobs at this time')
        return 1

    if len(batch_files) >= submission_limit:
        chain_submissions=True
        print(f'chaining submissions and restricting to {submission_limit} at a time')

        batch_file_batches = [[bf] for bf in batch_files[:submission_limit]]
        leftovers = batch_files[submission_limit:]
        i = 0
        # ipdb.set_trace()
        while len(leftovers) > 0:
            if i >= len(batch_file_batches):
                i = 0
            batch_file_batches[i].append(
                leftovers.pop()
            )
            i += 1 

        assert len(batch_file_batches) == submission_limit

        # append next submission to batch files
        submit_list = []
        for bfb in batch_file_batches:
            for i in range(len(bfb)-1):
                with open(bfb[i], 'a') as of:
                    of.write(f'\n\nsbatch {bfb[i+1]}')
            submit_list.append(bfb[0])
        assert len(submit_list) == submission_limit
    
    else:
        submit_list = batch_files

    ########################################
    # submit jobs
    pbar = tqdm(submit_list)
    for batch_file in pbar:
        sbatch_response = subprocess.check_output(['sbatch', batch_file]).decode()     
        pbar.set_description(sbatch_response.strip().replace('\n',''))
        # print(sbatch_response)


import fire
if __name__ == '__main__':
    fire.Fire(run)
