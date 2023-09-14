import logging
import csv

import itertools
from pathlib import Path
import torch

from src.helpers import *
from src.utils import accuracy, ece
from .datasets import \
    get_dataset, create_corrupted_dict, CORRUPTION_TYPES, CORRUPTION_LEVELS


def test_vc(
    config_data: str, 
    config_encoder: str,
    config_training: str,
    config_classifier: str,
    n_samples: int,
    **kwargs,
):

    root = Path(__file__).resolve().parents[3]

    results_path = root/"results"/"uncertainity"/"vc"/\
        config_data/config_encoder/config_training/config_classifier
    configs_path = root/"configs"
    
    model_dir = results_path/"models"

    scores_dir = results_path/"scores"

    logs_path = "{}/logs/uncertainity/test_vc_{}_{}_{}_{}_{}-samples.log".format(
        root, config_data, config_encoder, config_training, config_classifier, n_samples
    )
    setup_default_logging(log_path=logs_path, restart=True)
    logger = logging.getLogger('test')

    cfg_data, cfg_encoder, cfg_classifier, cfg_training = read_configs_classifier(
        path=configs_path, 
        model_dir=model_dir,
        config_data=config_data,
        config_encoder=config_encoder,
        config_classifier=config_classifier,
        config_training=config_training,
        save=True,
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    _, n_channels, size, n_classes = get_dataset(train=False, **cfg_data)

    classifier = get_classifier(
        cfg_encoder=cfg_encoder,
        cfg_classifier=cfg_classifier,
        ch_in=n_channels,
        size_in=size,
        n_classes=n_classes,
        device=device,
    )
    load_epoch_classifier(cfg_training['scheduler']['epochs'], device, model_dir, classifier)

    logger.info("Number of parameters: {}".format(count_parameters(classifier)))

    results = {
        "acc_top1": create_corrupted_dict(),
        "acc_top5": create_corrupted_dict(),
        "ece":      create_corrupted_dict()
    }

    for ctype, clevel in itertools.product(CORRUPTION_TYPES, [0]+CORRUPTION_LEVELS):

        test_dataset, n_channels, size, n_classes = get_dataset(
            corruption_type=ctype, 
            corruption_level=clevel, 
            train=False,
            **cfg_data
        )

        test_loader, n_iters_test = get_test_loader(
            test_dataset, batch_size=cfg_training['batch_size'],
        )

        #=====Test Epoch==========
        test_labels, test_preds = test_epoch_classifier(
            logger=logger,
            epoch=1, 
            n_epochs=1,
            n_iters_test=n_iters_test,
            device=device,
            test_loader=test_loader,
            classifier=classifier,
            n_classes=n_classes,
            n_samples=n_samples,
            print_iters=-1,
        )

        #=====Results==========
        acc_c = accuracy(test_preds, test_labels, topk=(1,5))
        ece_c = ece(test_preds, test_labels)

        logger.info(ctype+" - Level "+str(clevel))
        print_results(logger, False, 1, 1, acc_c, ece_c)
        logger.info("")

        results['acc_top1'][ctype][clevel] = acc_c[0].numpy()
        results['acc_top5'][ctype][clevel] = acc_c[1].numpy()
        results['ece'][ctype][clevel] = ece_c


    for result in results:
        with open(scores_dir/("corrupted_"+result+".csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Type"]+CORRUPTION_LEVELS)
            for ctype, info in results[result].items():
                row = [ctype] + [info[i] for i in CORRUPTION_LEVELS]
                writer.writerow(row)