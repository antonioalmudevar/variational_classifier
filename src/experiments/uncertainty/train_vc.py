import logging
import csv

from pathlib import Path
import torch
import torch.nn.functional as F

from src.helpers import *
from src.utils import accuracy, ece
from src.classifiers import BaseClassifierParallel
from .datasets import get_dataset


def train_vc(
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
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    scores_dir = results_path/"scores"
    Path(scores_dir).mkdir(parents=True, exist_ok=True)

    Path(root/"logs"/"uncertainity").mkdir(parents=True, exist_ok=True)
    logs_path = "{}/logs/uncertainity/train_vc_{}_{}_{}_{}_{}-samples.log".format(
        root, config_data, config_encoder, config_training, config_classifier, n_samples
    )
    setup_default_logging(log_path=logs_path, restart=True)
    logger = logging.getLogger('train')

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

    train_dataset, n_channels, size, n_classes = get_dataset(train=True, **cfg_data)
    train_loader, n_iters_train = get_train_loader(
        train_dataset, batch_size=cfg_training['batch_size'],
    )
    
    test_dataset, _, _, _ = get_dataset(train=False, **cfg_data)
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
    logger.info("Number of parameters: {}".format(count_parameters(classifier)))

    optimizer, scheduler, n_epochs = get_optimizer_scheduler(
        params=classifier.parameters(), 
        cfg_optimizer=cfg_training['optimizer'], 
        cfg_scheduler=cfg_training['scheduler'],
    )

    classifier = BaseClassifierParallel(classifier)

    mixup = Mixup(**cfg_training['mixup']) if 'mixup' in cfg_training else None

    results_epoch = {
        'train_acc_top1': [0], 'train_acc_top5': [0], 'test_acc_top1': [0], 'test_acc_top5': [0],
        'train_ece': [0], 'test_ece': [0]
    }

    for epoch in range(1, n_epochs+1):

        #=====Train Epoch==========
        train_labels, train_preds = train_epoch_classifier(
            logger=logger,
            epoch=epoch, 
            error_val=1-results_epoch['train_acc_top1'][epoch-1],
            n_epochs=n_epochs,
            n_iters_train=n_iters_train,
            device=device,
            model_dir=model_dir,
            train_loader=train_loader,
            classifier=classifier,
            optimizer=optimizer,
            scheduler=scheduler,
            n_classes=n_classes,
            mixup=mixup,
            save=(epoch==n_epochs),
            print_iters=-1,
        )

        #=====Test Epoch==========
        test_labels, test_preds = test_epoch_classifier(
            logger=logger,
            epoch=epoch, 
            n_epochs=n_epochs,
            n_iters_test=n_iters_test,
            device=device,
            test_loader=test_loader,
            classifier=classifier,
            n_classes=n_classes,
            n_samples=n_samples,
            print_iters=-1,
        )

        #=====Results==========
        train_acc = accuracy(train_preds, train_labels, topk=(1,5))
        test_acc = accuracy(test_preds, test_labels, topk=(1,5))

        train_ece = ece(train_preds, train_labels)
        test_ece = ece(test_preds, test_labels)

        print_results(logger, True, epoch, n_epochs, train_acc, train_ece)
        print_results(logger, False, epoch, n_epochs, test_acc, test_ece)
        logger.info("")

        results_epoch['train_acc_top1'].append(train_acc[0].numpy())
        results_epoch['train_acc_top5'].append(train_acc[1].numpy())
        results_epoch['test_acc_top1'].append(test_acc[0].numpy())
        results_epoch['test_acc_top5'].append(test_acc[1].numpy())
        results_epoch['train_ece'].append(train_ece)
        results_epoch['test_ece'].append(test_ece)

    for result in results_epoch:
        with open(scores_dir/(result+".csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(results_epoch[result])