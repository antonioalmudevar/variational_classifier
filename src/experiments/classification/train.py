import logging
import csv

from pathlib import Path
import torch

from src.helpers import *
from src.utils import *
from src.classifiers import BaseClassifierParallel
from .datasets import get_dataset


def train_classification(
    config_data: str, 
    config_encoder: str,
    config_classifier: str, 
    config_training: str,
    n_iters: int=1,
    restart: bool=False,
):
    restart = restart or (n_iters!=1)

    root = Path(__file__).resolve().parents[3]

    results_path = root/"results"/"classification"/\
        config_data/config_encoder/config_classifier/config_training
    configs_path = root/"configs"
    
    model_dir = results_path/"models"
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    scores_dir = results_path/"scores"
    Path(scores_dir).mkdir(parents=True, exist_ok=True)

    Path(root/"logs"/"classification").mkdir(parents=True, exist_ok=True)
    logs_path = "{}/logs/classification/train_{}_{}_{}_{}.log".format(
        root, config_data, config_encoder, config_classifier, config_training
    )
    setup_default_logging(log_path=logs_path, restart=restart)
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

    device = torch.device('cuda') if torch.cuda.is_available() \
        else torch.device('cpu')

    train_dataset, n_channels, size, n_classes = get_dataset(train=True, **cfg_data)
    train_loader, n_iters_train = get_train_loader(
        train_dataset, batch_size=cfg_training['batch_size'],
    )
    is_imagenet = cfg_data['dataset'].upper()=='IMAGENET'
    
    test_dataset, _, _, _ = get_dataset(train=False, **cfg_data)
    test_loader, n_iters_test = get_test_loader(
        test_dataset, batch_size=cfg_training['batch_size'],
    )

    acc_iters = {
        'train_top1': [], 'train_top5': [], 'test_top1': [], 'test_top5': []
    }

    for iter in range(n_iters):

        Path(model_dir/("iter_"+str(iter))).mkdir(parents=True, exist_ok=True)

        logger.info("Iteration {}".format(iter+1))

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

        ini_epoch = load_last_epoch_classifier(
            model_dir=model_dir/("iter_"+str(iter)),
            classifier=classifier,
            optimizer=optimizer,
            scheduler=scheduler,
            restart=restart,
        )
        print(restart)

        classifier = BaseClassifierParallel(classifier)

        mixup = Mixup(**cfg_training['mixup']) if 'mixup' in cfg_training else None

        acc_epoch = {
            'train_top1': [0 for _ in range(ini_epoch+1)], 
            'train_top5': [0 for _ in range(ini_epoch+1)],
            'test_top1': [0 for _ in range(ini_epoch+1)],
            'test_top5': [0 for _ in range(ini_epoch+1)]
        }

        for epoch in range(ini_epoch+1, n_epochs+1):

            #=====Train Epoch==========
            train_labels, train_preds = train_epoch_classifier(
                logger=logger,
                epoch=epoch, 
                error_val=1-acc_epoch['train_top1'][epoch-1],
                n_epochs=n_epochs,
                n_iters_train=n_iters_train,
                device=device,
                model_dir=model_dir/("iter_"+str(iter)),
                train_loader=train_loader,
                classifier=classifier,
                optimizer=optimizer,
                scheduler=scheduler,
                n_classes=n_classes,
                mixup=mixup,
                save=(epoch%cfg_training['save_epochs']==0) or epoch==n_epochs,
                save_labels_preds=not is_imagenet,
                print_iters=1000,
            )

            if is_imagenet:
                train_acc = [torch.tensor(1),torch.tensor(1)]
            else:
                train_acc = accuracy(train_preds, train_labels, topk=(1,5))

            print_accuracy(logger, True, epoch, n_epochs, train_acc)

            acc_epoch['train_top1'].append(train_acc[0].numpy())
            acc_epoch['train_top5'].append(train_acc[1].numpy())


            for n_samples in [0]:
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
                    print_iters=1000,
                )

                test_acc = accuracy(test_preds, test_labels, topk=(1,5))

                print_accuracy(logger, False, epoch, n_epochs, test_acc)
                

                if n_samples==0:
                    acc_epoch['test_top1'].append(test_acc[0].numpy())
                    acc_epoch['test_top5'].append(test_acc[1].numpy())

            logger.info("")
            
        for acc in acc_iters:
            acc_iters[acc].append(acc_epoch[acc][1:])
    
    for acc in acc_iters:
        with open(scores_dir/(acc+".csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(acc_iters[acc])