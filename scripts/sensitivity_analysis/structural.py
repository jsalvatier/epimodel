import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import pymc3 as pm

from epimodel import EpidemiologicalParameters
from epimodel.preprocessing.data_preprocessor import preprocess_data

import argparse

from scripts.sensitivity_analysis.utils import *

argparser = argparse.ArgumentParser()
add_argparse_arguments(argparser)
# this is a hack to make this work easily.
argparser.add_argument('--model_structure', dest='model_structure', type=str)
args = argparser.parse_args()

if __name__ == '__main__':
    data = preprocess_data('notebooks/double-entry-data/double_entry_final.csv', last_day='2020-05-30')
    data.mask_reopenings()

    ep = EpidemiologicalParameters()
    model_class = get_model_class_from_str(args.model_structure)

    bd = ep.get_model_build_dict()

    if args.model_structure == 'discrete_renewal_fixed_gi':
        # posterior means from a full model run
        bd['gi_mean_mean'] = 5.12
        bd['gi_sd_mean'] = 2.20

    with model_class(data) as model:
        model.build_model(**bd)

    with model.model:
        model.trace = pm.sample(args.n_samples, tune=500, chains=args.n_chains, cores=args.n_chains, max_treedepth=14,
                                target_accept=0.95)

    save_cm_trace(f'{args.model_structure}.txt', model.trace.CMReduction, args.exp_tag, args.model_type)
