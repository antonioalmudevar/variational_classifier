import logging
import csv

from pathlib import Path
import torch

from src.helpers import *
from .datasets import svhn


def test_vanilla(
    config_data: str, 
    config_encoder: str,
    config_training: str,
    **kwargs,
):

    root = Path(__file__).resolve().parents[3]

    results_path = root/"results"/"uncertainity"/"vanilla"/\
        config_data/config_encoder/config_training
    configs_path = root/"configs"
    
    model_dir = results_path/"models"

    scores_dir = results_path/"scores"

    logs_path = "{}/logs/uncertainity/test_ood_vanilla_{}_{}_{}.log".format(
        root, config_data, config_encoder, config_training
    )
    setup_default_logging(log_path=logs_path, restart=True)
    logger = logging.getLogger('test')

    cfg_data, cfg_encoder, cfg_classifier, cfg_training = read_configs_classifier(
        path=configs_path, 
        model_dir=model_dir,
        config_data=config_data,
        config_encoder=config_encoder,
        config_classifier="fce",
        config_training=config_training,
        save=True,
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test_dataset, n_channels, size, _ = svhn(train=False)
    n_classes = 100 if cfg_data['dataset']=='cifar-100' else 10
    
    test_loader, n_iters_test = get_test_loader(
        test_dataset, batch_size=cfg_training['batch_size'],
    )

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
        n_samples=1,
        print_iters=-1,
    )

    #=====Results==========
    test_ent = -torch.sum(test_preds*torch.log(test_preds), axis=-1).numpy()
    test_conf = (test_preds.max(axis=1)[0]).numpy()
    results = {'entropy': test_ent, 'confidence': test_conf}

    for result in results:
        with open(scores_dir/("ood_"+result+".csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(results[result])