import argparse
import sys

from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[1])
from src.experiments.cone import train_cone

def parse_args():

    parser = argparse.ArgumentParser(description='Train cone')

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

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    train_cone(**vars(args))