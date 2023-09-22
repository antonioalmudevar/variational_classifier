# Variational Classifier
This code allows to replicate the experiments presented in the article. The present directories are explained below:


## To begin with
To create the working environment to run the different experiments you must execute the following chunk of code:
```
git clone https://github.com/antonioalmudevar/variational_classifier.git
cd variational_classifier
python3 -m venv venv
source ./venv/bin/activate
python setup.py develop
pip install -r requirements.txt
```


## Launch experiments
The following explains how to launch the experiments proposed in the paper. To do so, each of the scripts found in the **bin** folder is used. 
In all of them, some of their input arguments are of the form *config_x*. In this case, the different values that *config_x* can take are those found in the **configs/x** directory. In the case where **x=data**, not all scripts are prepared to be launched with all databases in the **configs/data** folder. Only those presented in the paper.
The results of all experiments are stored in the **results** folder which will be created in case it do not exist.
We do not provide the trained models or the results. Except for ImageNet, the experiments are quickly trainable and reproducible.


### Classification performance
All the results associated with this section are obtained with the **train_classification.py** script. It is used as follows:
```
python train_classification.py config_data config_encoder config_classifier config_training --n_iters n
```
For example, in case we want to train a vanilla classifier with CIFAR-10 and ResNet-20 for 5 iterations according to the given formula, we would have:
```
python train_classification.py cifar10 resnet20 vanilla sgd-128-1e-1 --n_iters 5
```
The results of this training are the models of each iteration in **results/.../models** and the Top-1 and Top-5 Accuracy of the train and test datasets in **results/.../scores**.

where **/.../** means **config_data/config_encoder/config_classifier/config_training/** (here and in the next experiments)


### Calibrated uncertainty
To train the models needed for the experiments in this section, the script **train_uncertainty.py** must be used. Since the arguments depend on the chosen calibration method, here we give directly the example of how to launch the training of each method according to the parameters given in the article.
```
python train_uncertainty.py vanilla cifar100 resnet56 sgd-128-1e-1
python train_uncertainty.py temperature cifar100 resnet56 sgd-128-1e-1
python train_uncertainty.py ensembles cifar100 resnet56 sgd-128-1e-1 --n_ensembles 10
python train_uncertainty.py mc_dropout cifar100 resnet56 sgd-128-1e-1 --dropout_rate 0.5 --n_samples 128
python train_uncertainty.py ll_dropout cifar100 resnet56 sgd-128-1e-1 --dropout_rate 0.5 --n_samples 128
python train_uncertainty.py vc cifar100 resnet56 sgd-128-1e-1 fvvc-orth-10 --n_samples 128
```
Here, the result of this training are the models in **results/.../models** and the Top-1 and Top-5 Accuracy and ECE of the uncorrupted train and test datasets in **results/.../scores**.

To obtain the Corrupted Data scores, the **test_uncertainty.py** script must be used. The use of the script is identical to the previous one. For example, in the case of the Variational Classifier:
```
python test_uncertainty.py vc cifar100 resnet56 sgd-128-1e-1 fvvc-orth-10 --n_samples 128
```
The results are saved back to **results/.../scores**.

Finally, to obtain the data results of the OOD-related experiments, the **test_ood.py** script must be used, whose operation is again identical to the previous case. As an example:
```
python test_ood.py vc cifar100 resnet56 sgd-128-1e-1 fvvc-orth-10 --n_samples 128
```
And the results are stored in **results/.../scores**.


### Better Space Utilization
For this last experiment, the **train_cone.py** script must be used. Its use follows the following formula:
```
python train_cone.py config_data config_encoder config_classifier config_training
```
And the results of this experiment are the model (**results/.../scores**) and the test embeddings in **results/.../predictions**.
