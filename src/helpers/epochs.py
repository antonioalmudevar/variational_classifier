from typing import Union, List
from logging import Logger

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from src.helpers.training import print_loss, Mixup
from src.helpers.classifiers import save_epoch_classifier
from src.utils import plot_tsne

__all__ = [
    "train_epoch_classifier", 
    "test_epoch_classifier",
    "train_epoch_classifier_ensemble",
    "test_epoch_classifier_ensemble"
]


def train_epoch_classifier(
    logger: Logger,
    epoch: int, 
    error_val: float,
    n_epochs: int,
    n_iters_train: int,
    device: torch.device,
    model_dir: str,
    train_loader: DataLoader,
    classifier: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    n_classes: int,
    n_samples: int=None,
    mixup: Mixup=None,
    save: bool=False,
    save_labels_preds: bool=True,
    print_iters: int=1000,
):
    
    train_loss, train_labels, train_preds = 0, [], []
    classifier.train()

    for iter, (input, labels) in enumerate(train_loader, start=1):

        #======Data preparation=======
        if save_labels_preds: 
            train_labels.extend(labels.cpu())
        input = Variable(input).to(
            device=device, dtype=torch.float, non_blocking=True
        )
        labels = labels.to(device=device, non_blocking=True)
        if mixup is not None:
            input, labels = mixup(input, labels)
        else:
            labels = torch.nn.functional.one_hot(labels, n_classes).to(dtype=torch.float)

        #======Forward=======
        output = classifier(input, n_samples)
        loss = torch.mean(
            classifier.calc_loss(**output, labels=labels)
        )
        train_loss += loss.detach()
        if save_labels_preds: 
            train_preds.extend(output['preds'].cpu().detach())

        #======Backward=======
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #======Logs=======
        print_loss(
            logger=logger,
            train=True, 
            epoch=epoch, 
            n_epochs=n_epochs, 
            iter=iter, 
            n_iters=n_iters_train, 
            loss=train_loss,
            print_iters=print_iters,
        )

    scheduler.step(epoch, error_val)

    save_epoch_classifier(
        epoch, model_dir, classifier, optimizer, scheduler, save
    )
    
    if save_labels_preds: 
        train_labels = torch.stack(train_labels)
        train_preds = torch.stack(train_preds)

    return train_labels, train_preds



def test_epoch_classifier(
    logger: Logger,
    epoch: int,
    n_epochs: int,
    n_iters_test: int,
    device: torch.device,
    test_loader: DataLoader,
    classifier: torch.nn.Module,
    n_classes: int,
    n_samples: int=1,
    print_iters: int=1000,
    preds_dir: str=None,
    save_embeds: bool=False,
    save_tsne: bool=False,
):
    
    test_loss, test_labels, test_preds, test_embeds = 0, [], [], []
    classifier.eval()

    for iter, (input, labels) in enumerate(test_loader, start=1):

        #======Data preparation=======
        test_labels.extend(labels.cpu())
        input = Variable(input).to(
            device=device, dtype=torch.float, non_blocking=True
        )
        labels = torch.nn.functional.one_hot(labels, n_classes).to(
            device=device, non_blocking=True, dtype=torch.float
        )

        #======Forward=======
        with torch.no_grad():
            output = classifier(input, n_samples)
            test_loss += torch.mean(
                classifier.calc_loss(**output, labels=labels)
            )
            test_preds.extend(output['preds'].cpu().detach())
            if save_tsne or save_embeds:
                if 'embeds' in output:
                    test_embeds.extend(output['embeds'].cpu().detach())
                else:
                    test_embeds.extend(output['mu'].cpu().detach())

        #======Logs=======
        print_loss(
            logger=logger,
            train=False, 
            epoch=epoch, 
            n_epochs=n_epochs, 
            iter=iter, 
            n_iters=n_iters_test, 
            loss=test_loss,
            print_iters=print_iters,
        )
    
    test_labels = torch.stack(test_labels)
    test_preds = torch.stack(test_preds)
    
    #======Plot t-SNE=======
    if save_tsne:
        plot_tsne(
            x=torch.stack(test_embeds).numpy(), 
            y=test_labels.numpy(), 
            save_path=preds_dir/("epoch_"+str(epoch)+".png")
        )

    #======Save embeddings=======
    if save_embeds:
        np.savez(
            preds_dir/("predictions"), 
            embeds=torch.stack(test_embeds).numpy(),  
            labels=test_labels.numpy(), 
        )

    return test_labels, test_preds


#==========Ensembles===============
def train_epoch_classifier_ensemble(
    logger: Logger,
    epoch: int, 
    error_vals: Union[float, List[float]],
    n_epochs: int,
    n_iters_train: int,
    device: torch.device,
    models_dir: str,
    train_loader: DataLoader,
    classifiers: Union[torch.nn.Module, List[torch.nn.Module]],
    optimizers: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]],
    schedulers: Union[torch.optim.lr_scheduler._LRScheduler, List[torch.optim.lr_scheduler._LRScheduler]],
    n_classes: int,
    n_samples: int=None,
    mixup: Mixup=None,
    save: bool=False,
    print_iters: int=1000,
):
    
    classifiers = [classifiers] if not isinstance(classifiers, list) else classifiers
    optimizers = [optimizers] if not isinstance(optimizers, list) else optimizers
    schedulers = [schedulers] if not isinstance(schedulers, list) else schedulers

    for classifier in classifiers:
        classifier.train()

    train_labels = []
    train_loss = [0 for _ in range(len(classifiers))]
    train_preds = [[] for _ in range(len(classifiers))]

    for iter, (input, labels) in enumerate(train_loader, start=1):

        #======Data preparation=======
        train_labels.extend(labels.cpu())
        input = Variable(input).to(
            device=device, dtype=torch.float, non_blocking=True
        )
        labels = labels.to(device=device, non_blocking=True)
        if mixup is not None:
            input, labels = mixup(input, labels)
        else:
            labels = torch.nn.functional.one_hot(labels, n_classes).to(dtype=torch.float)

        for i, classifier in enumerate(classifiers):

            #======Forward=======
            output = classifier(input, n_samples)
            loss = torch.mean(
                classifier.calc_loss(**output, labels=labels)
            )
            train_loss[i] += loss.detach()
            train_preds[i].extend(output['preds'].cpu().detach())

            #======Backward=======
            optimizers[i].zero_grad()
            loss.backward()
            optimizers[i].step()

            #======Logs=======
            print_loss(
                logger=logger,
                train=True, 
                epoch=epoch, 
                n_epochs=n_epochs, 
                iter=iter, 
                n_iters=n_iters_train, 
                loss=train_loss[i],
                print_iters=print_iters,
            )

    for i, classifier in enumerate(classifiers):

        schedulers[i].step(epoch, error_vals[i])

        model_dir = models_dir/("ensemble"+str(i+1))
        save_epoch_classifier(
            epoch, model_dir, classifier, optimizers[i], schedulers[i], save
        )
    
    for i in range(len(classifiers)):
        train_preds[i] = torch.stack(train_preds[i])
    
    train_labels = torch.stack(train_labels)
    train_preds = torch.stack(train_preds)

    return train_labels, train_preds



def test_epoch_classifier_ensemble(
    logger: Logger,
    epoch: int,
    n_epochs: int,
    n_iters_test: int,
    device: torch.device,
    test_loader: DataLoader,
    classifiers: torch.nn.Module,
    n_classes: int,
    n_samples: int=1,
    print_iters: int=1000,
):
    classifiers = [classifiers] if not isinstance(classifiers, list) else classifiers

    for classifier in classifiers:
        classifier.eval()

    test_labels = []
    test_loss = [0 for _ in range(len(classifiers))]
    test_preds = [[] for _ in range(len(classifiers))]

    for iter, (input, labels) in enumerate(test_loader, start=1):

        #======Data preparation=======
        test_labels.extend(labels.cpu())
        input = Variable(input).to(
            device=device, dtype=torch.float, non_blocking=True
        )
        labels = torch.nn.functional.one_hot(labels, n_classes).to(
            device=device, non_blocking=True, dtype=torch.float
        )

        for i, classifier in enumerate(classifiers):

            #======Forward=======
            with torch.no_grad():
                output = classifier(input, n_samples)
                test_loss[i] += torch.mean(
                    classifier.calc_loss(**output, labels=labels)
                )
                test_preds[i].extend(output['preds'].cpu().detach())

            #======Logs=======
            print_loss(
                logger=logger,
                train=False, 
                epoch=epoch, 
                n_epochs=n_epochs, 
                iter=iter, 
                n_iters=n_iters_test, 
                loss=test_loss,
                print_iters=print_iters,
            )

    for i in range(len(classifiers)):
        test_preds[i] = torch.stack(test_preds[i])
    
    test_labels = torch.stack(test_labels)
    test_preds = torch.stack(test_preds).mean(axis=0)

    return test_labels, test_preds