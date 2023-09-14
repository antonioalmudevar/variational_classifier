import logging
import csv

from pathlib import Path
import numpy as np
import torch

from src.helpers import *
from src.utils import accuracy, ece
from src.classifiers import BaseClassifierParallel
from .datasets import get_dataset


def train_ensembles(
    config_data: str, 
    config_encoder: str,
    config_training: str,
    n_ensembles: int,
    **kwargs,
):

    root = Path(__file__).resolve().parents[3]

    results_path = root/"results"/"uncertainity"/"ensembles"/\
        config_data/config_encoder/config_training
    configs_path = root/"configs"
    
    models_dir = results_path/"models"
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    scores_dir = results_path/"scores"
    Path(scores_dir).mkdir(parents=True, exist_ok=True)

    Path(root/"logs"/"uncertainity").mkdir(parents=True, exist_ok=True)
    logs_path = "{}/logs/uncertainity/train_ensembles_{}_{}_{}_{}-ensembles.log".format(
        root, config_data, config_encoder, config_training, n_ensembles
    )
    setup_default_logging(log_path=logs_path, restart=True)
    logger = logging.getLogger('train')

    cfg_data, cfg_encoder, cfg_classifier, cfg_training = read_configs_classifier(
        path=configs_path, 
        model_dir=models_dir,
        config_data=config_data,
        config_encoder=config_encoder,
        config_classifier="fce",
        config_training=config_training,
        save=True,
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dataset, n_channels, size, n_classes = get_dataset(train=True, **cfg_data)
    train_loader, n_iters_train = get_train_loader(
        train_dataset, batch_size=cfg_training['batch_size'],
    )
    
    test_dataset, _, _, _ = get_dataset(train=False, **cfg_data)
    test_loader, n_iters_test = get_test_loader(
        test_dataset, batch_size=cfg_training['batch_size'],
    )

    classifiers, optimizers, schedulers = [], [], []
    for i in range(n_ensembles):
    
        classifier = get_classifier(
            cfg_encoder=cfg_encoder,
            cfg_classifier=cfg_classifier,
            ch_in=n_channels,
            size_in=size,
            n_classes=n_classes,
            device=device,
        )

        optimizer, scheduler, n_epochs = get_optimizer_scheduler(
            params=classifier.parameters(), 
            cfg_optimizer=cfg_training['optimizer'], 
            cfg_scheduler=cfg_training['scheduler'],
        )

        classifier = BaseClassifierParallel(classifier)

        classifiers.append(classifier)
        optimizers.append(optimizer)
        schedulers.append(scheduler)

        Path(models_dir/("ensemble"+str(i+1))).mkdir(parents=True, exist_ok=True)

    logger.info("Number of parameters: {}".format(count_parameters(classifier)))

    mixup = Mixup(**cfg_training['mixup']) if 'mixup' in cfg_training else None

    results_epoch = {
        'train_acc_top1': np.zeros((n_ensembles, n_epochs+1)),
        'train_acc_top5': np.zeros((n_ensembles, n_epochs+1)),
        'test_acc_top1': np.zeros(n_epochs+1), 
        'test_acc_top5': np.zeros(n_epochs+1),
        'train_ece': np.zeros((n_ensembles, n_epochs+1)),
        'test_ece': np.zeros(n_epochs+1),
    }

    for epoch in range(1, n_epochs+1):

        #=====Train Epoch==========
        train_labels, train_preds = train_epoch_classifier_ensemble(
            logger=logger,
            epoch=epoch, 
            error_vals=1-results_epoch['train_acc_top1'][:,epoch-1],
            n_epochs=n_epochs,
            n_iters_train=n_iters_train,
            device=device,
            models_dir=models_dir,
            train_loader=train_loader,
            classifiers=classifiers,
            optimizers=optimizers,
            schedulers=schedulers,
            n_classes=n_classes,
            mixup=mixup,
            save=(epoch==n_epochs),
            print_iters=-1,
        )

        #=====Test Epoch==========
        test_labels, test_preds = test_epoch_classifier_ensemble(
            logger=logger,
            epoch=epoch, 
            n_epochs=n_epochs,
            n_iters_test=n_iters_test,
            device=device,
            test_loader=test_loader,
            classifiers=classifiers,
            n_classes=n_classes,
            n_samples=1,
            print_iters=-1,
        )

        #=====Results==========
        for i in range(n_ensembles):
            train_acc = accuracy(train_preds[i], train_labels, topk=(1,5))
            train_ece = ece(train_preds[i], train_labels)

            results_epoch['train_acc_top1'][i, epoch] = train_acc[0].numpy()
            results_epoch['train_acc_top5'][i, epoch] = train_acc[1].numpy()
            results_epoch['train_ece'][i, epoch] = train_ece

        test_acc = accuracy(test_preds, test_labels, topk=(1,5))
        test_ece = ece(test_preds, test_labels)

        print_results(logger, False, epoch, n_epochs, test_acc, test_ece)
        logger.info("")

        results_epoch['test_acc_top1'][epoch] = test_acc[0].numpy()
        results_epoch['test_acc_top5'][epoch] = test_acc[1].numpy()
        results_epoch['test_ece'][epoch] = test_ece

    for result in results_epoch:
        with open(scores_dir/(result+".csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(results_epoch[result])