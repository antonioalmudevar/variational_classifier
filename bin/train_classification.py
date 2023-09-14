import argparse
import sys

from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[1])
from src.experiments.classification import train_classification

def parse_args():

    parser = argparse.ArgumentParser(description='Train Classifier')

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
        'config_classifier', 
        nargs='?', 
        type=str, 
        help='classifier configuration file'
    )

    parser.add_argument(
        'config_training', 
        nargs='?', 
        type=str, 
        help='training configuration file'
    )

    parser.add_argument(
        '--n_iters', 
        nargs='?', 
        type=int,
        default=1,
        help='number of iterations'
    )

    parser.add_argument(
        '--restart', 
        action='store_true',
        help='restart the training with this flag'
    )
    parser.set_defaults(restart=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_classification(**vars(args))