# Prototypical Networks for Few-shot in TensorFlow 2.0
Implementation of Prototypical Networks for Few-shot Learning paper (https://arxiv.org/abs/1703.05175) in TensorFlow 2.14. Model has been tested on Omniglot and miniImagenet datasets with the same splitting as in the paper. Also, adapted to use as encoder, as a backbone, the EfficientNet V2 variant b0.

<img width="896" alt="Screenshot 2019-04-02 at 9 53 06 AM" src="https://user-images.githubusercontent.com/23639048/55438102-5d9e4c00-55a9-11e9-86e2-b4f79f880b83.png">

### Dependencies and Installation
* The code has been tested on Ubuntu 22.04 with Python 3.10.12 and TensorFflow 2.14.0
* The two main dependencies are TensorFlow (2.14.0) and Pillow package (9.5.0)
* In this version there is no need to install `prototf` lib. Just execute the code like downbelow.
* Run `bash data/download_omniglot.sh` from repo's root directory to download Omniglot dataset. But warning, as long as EfficientNet B0 is build for 3 channel images, omniglot is not available with this arquitecture.
* miniImagenet was downloaded from brilliant repo from `renmengye` (https://github.com/renmengye/few-shot-ssl-public) and placed into `data/mini-imagenet/data` folder. 

### Repository Structure

The repository organized as follows. `data` directory contains scripts for dataset downloading and used as a default directory for datasets. `prototf` is the library containing the model itself (`prototf/models`) and logic for datasets loading and processing (`prototf/data`). `scripts` directory contains scripts for launching the training. `train/run_train.py` and `eval/run_eval.py` launch training and evaluation respectively. `tests` folder contains basic training procedure on small-valued parameters to check general correctness. `results` folder contains .md file with current configuration and details of conducted experiments.

### Training

* Training and evaluation configurations are specified through config files, each config describes single train+eval evnironment.
* Run `python scripts/train/run_train.py --config scripts/config_omniglot.conf` to run training on Omniglot with default parameters.
* Run `python scripts/train/run_train.py --config scripts/config_miniimagenet.conf` to run training on miniImagenet with default parmeters

### Evaluating

* Run `python scripts/eval/run_eval.py --config scripts/config_omniglot.conf` to run evaluation on Omniglot
* Run `python scripts/eval/run_eval.py --config scripts/config_miniimagenet.conf` to run evaluation on miniImagenet

### Tests

* Run `python -m unittest tests/test_omniglot.py` from repo's root to test Omniglot
* Run `python -m unittest tests/test_mini_imagenet.py` from repo's root test miniImagenet 