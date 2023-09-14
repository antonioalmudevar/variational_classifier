import argparse
import sys

from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[1])
from src.experiments.ood import test_ood

def parse_args():

    parser = argparse.ArgumentParser(description='Test  OOD')

    parser.add_argument(
        'method', 
        nargs='?', 
        type=str,
        choices=["vanilla", "temperature", "ensembles", "mc_dropout", "ll_dropout", "vc"],
        help='method'
    )

    parser.add_argument(
        'config_data', 
        nargs='?', 
        type=str,
        help='data configuration file'
    )

    parser.add_argument(
        'config_encoder', 
        nargs='?', 
        type=str, 
        help='encoder configuration file'
    )

    parser.add_argument(
        'config_training', 
        nargs='?', 
        type=str, 
        help='training configuration file'
    )

    parser.add_argument(
        '--n_ensembles', 
        nargs='?',
        type=int, 
        help="ensembles in ensembles case"
    )

    parser.add_argument(
        '--dropout_rate', 
        nargs='?',
        type=float, 
        help="dropout rate"
    )

    parser.add_argument(
        '--config_classifier', 
        nargs='?',
        type=str, 
        help="classifier configuration file"
    )

    parser.add_argument(
        '--n_samples', 
        nargs='?',
        type=int, 
        help="number of samples in MC Dropout or Variational Classifier"
    )

    args = parser.parse_args()
    if args.method=="ensemble" and args.n_ensembles is None:
        parser.error("ensemble requires --n_ensembles")
    elif (args.method=="mc_dropout" or args.method=="ll_dropout") and args.dropout_rate is None:
        parser.error(args.method+" requires --dropout_rate")
    elif args.method=="vc" and args.config_classifier is None:
        parser.error("vc requires --config_classifier")
    elif (args.method=="vc" or args.method=="mc_dropout") and args.n_samples is None:
        parser.error(args.method+" requires --n_samples")

    return args


if __name__ == "__main__":
    args = parse_args()
    test_ood(**vars(args))