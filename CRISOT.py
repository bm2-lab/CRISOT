#!/usr/bin/python

import argparse
import pandas as pd
import numpy as np
from crisot_modules import *
from utils import *
import os
import pickle

__version__ = 'v0.4'
pwd = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(pwd, 'models/crisot_score_param.pkl'), 'rb') as f:
    paramread = pickle.load(f)

# with open(os.path.join(pwd, 'models/crisot_fingerprint_encoding.pkl'), 'rb') as f:
#     featread = pickle.load(f)

GENOME = os.path.join(pwd, 'script/hg38.na')
model = CRISOT(param=paramread, ref_genome=GENOME)

def cal_score(sgr, tar):
    return model.single_score_(sgr, tar)

def cal_scores(df_in, On='On', Off='Off'):
    df_in['CRISOT_Score'] = model.score(data_df=df_in, On=On, Off=Off)
    return df_in

def cal_spec(df_in, On='On', Off='Off'):
    spec = model.spec(data_df=df_in, On=On, Off=Off, out_df=False)
    return spec

def cal_casoffinder_spec(sgr, tar, ref_genome, mm=6, dev='G0'):
    model.ref_genome = ref_genome
    spec = model.CasoffinderSpec_(sgr, tar, mm=mm, dev=dev)
    return spec

def rescore_chopchop(chop_df, ref_genome, top=None, mm=6, dev='G0'):
    model.ref_genome = ref_genome
    tsv_df = model.rescore_chopchop(chop_df=chop_df, top=top, mm=mm, dev=dev)
    return tsv_df

def opti(tar, ref_genome, threshold=0.8, percent_activity_df=None, cd33cut=0.6, mm=6, dev='G0'):
    model.ref_genome = ref_genome
    model.opti_th = threshold
    csv_df = model.opti(target=tar, accepted_mutate=percent_activity_df, cd33cut=cd33cut, mm=mm, dev=dev)
    return csv_df

def cal_crisot_fp(df_in, xgb_model):
    df_in['CRISOT_FP'] = CRISOT_FP(model_path=xgb_model, dataread=df_in)
    return df_in

# setup the argument parser
def get_parser():
    parser = argparse.ArgumentParser(description="CRISOT Suite "+__version__)
    module_input = parser.add_argument_group("# CRISOT modules")
    module_input.add_argument("method", metavar="<method>", type=str, default='score', 
        help="Select a method to calculate the score(s): \n \
        score: CRISOT-Score of sgRNA-DNA, requires [--sgr, --tar]; \n \
        scores: Batch calculation of CRISOT-Score, requires [--csv], options [--on_item, --off_item, --out]; \n \
        spec: Calculate CRISOT-Spec, on-target sequence must be in the first line, requires [--csv], options [--on_item, --off_item]; \n \
        off_spec: Perform Cas-Offinder search and calculate CRISOT-Spec, requires [--sgr, --tar, --genome], options [--mm, --dev]; \n \
        rescore: Rescoring CHOPCHOP results by CRISOT-Score and CRISOT-Spec, requires [--tsv, --genome], options [--mm, --dev, --out]; \n \
        opti: CRISOT-Opti optimization by mutation, requires [--tar, --genome], options [--threshold, --percent_activity_file, --percent_activity, --mm, --dev, --out]; \n \
        crisot_fp: CRISOT-FP XGBoost machine learning prediction, requires [--csv], options [--xgb_model, --out] ")
    
    key_settings = parser.add_argument_group('# Key Settings')
    key_settings.add_argument("--sgr", metavar="<seq>", type=str, default=None, 
        help="sgRNA sequence to analyse (23nt, 20nt+PAM)")
    key_settings.add_argument("--tar", metavar="<seq>", type=str, default=None, 
        help="Target DNA sequence to analyse (23nt, 20nt+PAM)")
    key_settings.add_argument("--csv", metavar="<file>", type=str, default=None, 
        help="CSV file containing sgRNA and Target DNA sequences, headers are On and Off, respectively. (spec: On-target sequence must be in the first line)")
    key_settings.add_argument("--tsv", metavar="<file>", type=str, default=None, 
        help="TSV file of the result of CHOPCHOP sgRNA design, column name of the target sequences must be 'Target sequence'.")
    key_settings.add_argument("--genome", metavar="<file>", type=str, default=os.path.join(pwd, 'script/hg38.fa'), 
        help="Path to the file of reference genome. (default: script/hg38.fa)")
    key_settings.add_argument("--xgb_model", metavar="<pkl file>", type=str, default=os.path.join(pwd, 'models/guideseq_xgbcls_models.pkl'), 
        help="Path to the file of XGBoost models. (default: models/guideseq_xgbcls_models.pkl)")

    out_setting = parser.add_argument_group("# Output Settings")
    out_setting.add_argument('--out', metavar="<file>", type=str, default='default', 
        help="Output file name.")

    other_option = parser.add_argument_group("# Other Options")
    other_option.add_argument('--on_item', metavar="<name>", type=str, default='On', 
        help="sgRNA column name of the csv file (default: On)")
    other_option.add_argument('--off_item', metavar="<name>", type=str, default='Off', 
        help="Off-target column name of the csv file (default: Off)")
    other_option.add_argument('--mm', metavar="<number>", type=int, default=6, 
        help="Mismatch tolerance (default: 6)")
    other_option.add_argument('--dev', metavar="<device>", type=str, default='C', 
        help="GPU/CPU device setting, the same as in the CasOffinder (default: C)")
    other_option.add_argument('--threshold', metavar="<float>", type=float, default=0.8, 
        help="The CRISOT-Score threshold for mutated sgRNAs. (default: 0.8)")
    other_option.add_argument('--percent_activity_file', metavar="<file>", type=str, default=None, 
        help="The percent-activity file for sgRNAs mutation. (default: None)")
    other_option.add_argument('--percent_activity', metavar="<float>", type=float, default=0.6, 
        help="The percent-activity threshold for sgRNAs mutation, works only when percent_activity_file is NOT None. (default: 0.6)")

    parser.add_argument('--version', action='version', version='CRISOT {}'.format(__version__))

    return parser


# MAIN FUNCTION
def main():
    # Get the necessary arguments
    parser = get_parser()
    args = parser.parse_args()

    # Setting CRISOT model
    model.ref_genome = args.genome
    
    # Method 1: CRISOT-Score
    if args.method == 'score':
        assert len(args.sgr) == 23, 'sgRNA sequence must be 23-nt with NGG PAM'
        assert len(args.tar) == 23, 'Target DNA sequence must be 23-nt with NGG PAM'
        score = cal_score(args.sgr, args.tar)
        print('CRISOT-Score: \n' + str(score))
    
    elif args.method == 'scores':
        assert os.path.exists(args.csv), '{} file not exists'.format(args.csv)
        df_csv = pd.read_csv(args.csv, header=0, index_col=None)
        scores = cal_scores(df_csv, On=args.on_item, Off=args.off_item)
        if args.out == 'default':
            scores.to_csv('CRISOT-Score_results.csv', index=False)
        else:
            if args.out[-3:] == 'csv':
                out_name = args.out
            else:
                out_name = args.out + '.csv'
            scores.to_csv(out_name, index=False)
        print('CRISOT-Score calculation done')

    elif args.method == 'spec':
        assert os.path.exists(args.csv), '{} file not exists'.format(args.csv)
        df_csv = pd.read_csv(args.csv, header=0, index_col=None)
        spec = cal_spec(df_csv, On=args.on_item, Off=args.off_item)
        print('CRISOT-Spec: \n' + str(spec))
    
    elif args.method == 'off_spec':
        assert len(args.sgr) == 23, 'sgRNA sequence must be 23-nt with NGG PAM'
        assert len(args.tar) == 23, 'Target DNA sequence must be 23-nt with NGG PAM'
        assert os.path.exists(args.genome), '{}\n reference genome file not exists'.format(args.genome)
        spec = cal_casoffinder_spec(args.sgr, args.tar, ref_genome=args.genome, mm=args.mm, dev=args.dev)
        print('CRISOT-Spec: \n' + str(spec))
    
    elif args.method == 'rescore':
        assert os.path.exists(args.tsv), 'CHOPCHOP tsv file <{}> not exists'.format(args.tsv)
        assert os.path.exists(args.genome), '{}\n reference genome file not exists'.format(args.genome)
        df_tsv = pd.read_csv(args.tsv, sep='\t', header=0, index_col=None)
        tsv_result = rescore_chopchop(chop_df=df_tsv, ref_genome=args.genome, mm=args.mm, dev=args.dev, top=None)
        if args.out == 'default':
            tsv_result.to_csv('CRISOT_rescoring_results.tsv', sep='\t', index=False)
        else:
            if args.out[-3:] == 'tsv':
                out_name = args.out
            else:
                out_name = args.out + '.tsv'
            tsv_result.to_csv(out_name, sep='\t', index=False)
        print('CRISOT-Opti rescoring done')

    elif args.method == 'opti':
        assert len(args.tar) == 23, 'Target DNA sequence must be 23-nt with NGG PAM'
        assert os.path.exists(args.genome), '{}\n reference genome file not exists'.format(args.genome)
        if args.percent_activity_file == 'None':
            percent_activity_df = None
        elif args.percent_activity_file == None:
            percent_activity_df = None
        else:
            percent_activity_df = pd.read_csv(args.percent_activity_file, header=0, index_col=0)
        opti_result = opti(args.tar, ref_genome=args.genome, threshold=args.threshold, percent_activity_df=percent_activity_df, cd33cut=args.percent_activity, mm=args.mm, dev=args.dev)
        if args.out == 'default':
            opti_result.to_csv('CRISOT-Opti_optimization_results.csv', index=False)
        else:
            if args.out[-3:] == 'csv':
                out_name = args.out
            else:
                out_name = args.out + '.csv'
            opti_result.to_csv(out_name, index=False)
        print('CRISOT-Opti optimization done')

    elif args.method == 'crisot_fp':
        assert os.path.exists(args.csv), '{} file not exists'.format(args.csv)
        assert os.path.exists(args.xgb_model), 'XGBoost model file <{}> not exists'.format(args.xgb_model)
        df_csv = pd.read_csv(args.csv, header=0, index_col=None)
        scores = cal_crisot_fp(df_csv, args.xgb_model)
        if args.out == 'default':
            scores.to_csv('CRISOT-FP_results.csv', index=False)
        else:
            if args.out[-3:] == 'csv':
                out_name = args.out
            else:
                out_name = args.out + '.csv'
            scores.to_csv(out_name, index=False)
        print('CRISOT-FP calculation done')

    else:
        print('Please choose a correct method')


if __name__== "__main__":
    main()


