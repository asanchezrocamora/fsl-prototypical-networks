# Prototypical Networks with EfficientNetV2-B0 Encoder for Few-Shot Learning

This repository implements **Prototypical Networks** for Few-Shot Learning (FSL) tasks using the **EfficientNetV2-B0** encoder. The architecture has been adapted to enhance Few-Shot Learning performance, as detailed in our paper [Few-shot Learning for Presentation Attack Detection in ID Cards](https://arxiv.org/pdf/2409.06842v1). EfficientNetV2-B0 is a state-of-the-art convolutional neural network, as described in the original [EfficientNetV2 paper](https://arxiv.org/pdf/2104.00298), known for its efficiency and accuracy.

Our research demonstrates the applicability of this structure for detecting **Presentation Attacks on ID Cards** in remote verification systems, extending to new countries and novel attack types. By leveraging meta-learning principles and the EfficientNetV2-B0 encoder, we achieve competitive results with as few as five unique identities and under 100 images per new country.

![Prototypical Networks Architecture](https://user-images.githubusercontent.com/23639048/55438102-5d9e4c00-55a9-11e9-86e2-b4f79f880b83.png)

---

## Dependencies and Installation

1. The code has been tested on **Ubuntu 22.04** with **Python 3.10.12** and **TensorFlow 2.14.0**.
2. Required dependencies:
   - **TensorFlow 2.14.0**
   - **Pillow 9.5.0**
3. The `prototf` library is no longer required; execute the scripts as shown below.
4. Datasets:
   - **miniImageNet**: Downloaded from the repository of [renmengye](https://github.com/renmengye/few-shot-ssl-public) and placed in the `data/mini-imagenet/data` folder.

---

## Repository Structure

The repository is structured as follows:

- **`data/`**: Scripts for dataset downloading and storage.
- **`prototf/`**: Contains the core model implementation in `prototf/models` and dataset processing logic in `prototf/data`.
- **`scripts/`**: Includes scripts to train (`train/run_train.py`) and evaluate (`eval/run_eval.py`) the model.
- **`tests/`**: Basic scripts to validate the training pipeline.
- **`results/`**: Contains markdown files with details about experiments and configurations.

---

### Training

* Training and evaluation configurations are specified through config files, each config describes single train+eval evnironment.
* Run `python scripts/train/run_train.py --config scripts/config_miniimagenet.conf` to run training on miniImagenet with default parmeters

### Evaluating

* Run `python scripts/eval/run_eval.py --config scripts/config_miniimagenet.conf` to run evaluation on miniImagenet

### Tests

* Run `python -m unittest tests/test_mini_imagenet.py` from repo's root test miniImagenet 

## Research Paper

This repository supports the experiments conducted in our paper:  
**"[Few-shot Learning for Presentation Attack Detection in ID Cards](https://arxiv.org/pdf/2409.06842v1)"**

Our research focuses on using Few-Shot Learning to detect **Presentation Attacks** on ID cards in remote verification systems. By employing **Prototypical Networks** with an **EfficientNetV2-B0 encoder**, we achieve robust generalization across countries like **Spain**, **Chile**, **Argentina**, and **Costa Rica**. 

Our work specifically targets the challenge of **screen display presentation attacks** and achieves the following results:
- **Competitive performance** with as few as **five unique identities** per country.
- **High efficacy** with under **100 images** for unseen data.

This research highlights the potential of novel Few-Shot Learning architectures for **generalized Presentation Attack Detection** in scenarios with limited training data.


### Citation

If you use this repository or find it helpful, please consider citing our papers:

> Sánchez, Álvaro, Espín, Juan M., Tapia, Juan E.  
> **Few-Shot Learning: Expanding ID Cards Presentation Attack Detection to Unknown ID Countries**  
> *2024 IEEE International Joint Conference on Biometrics (IJCB)* (2024).  
> [DOI: 10.1109/IJCB62174.2024.10744501](https://doi.org/10.1109/IJCB62174.2024.10744501)

BibTeX citation:

```bibtex
@INPROCEEDINGS{10744501,
  author={Sanchez, Alvaro and Espín, Juan M. and Tapia, Juan E.},
  booktitle={2024 IEEE International Joint Conference on Biometrics (IJCB)}, 
  title={Few-Shot Learning: Expanding ID Cards Presentation Attack Detection to Unknown ID Countries}, 
  year={2024},
  volume={},
  number={},
  pages={1-9},
  keywords={Metalearning;Measurement;Biometrics;Few shot learning},
  doi={10.1109/IJCB62174.2024.10744501}
}```