<br />
<p align="center">
  <a href=" ">
    <img src="data/figures/misc/logo.png" alt="Logo" width="500"> 
  </a>

  <h1 align="center">Large-scale Attribution & Attention Evaluation in Computer Vision</h1>

  <p align="center">
      <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.10-3776AB?&logo=python&logoColor=white"></a>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.12-EE4C2C?logo=pytorch&logoColor=white"></a>
    <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Pytorch Lightning 1.8-792EE5?logo=pytorchlightning&logoColor=white"></a>
    <a href="https://black.readthedocs.io/en/stable"><img alt="L: Hydra" src="https://img.shields.io/badge/Code Style-Black-black" ></a>
    <br>
    <a href="https://openreview.net/pdf?id=HRkwnZewLC"><strong>Read the paper »</strong></a>
    <br />

  </p>
</p>


LATEC is a benchmark for large-scale generation and evaluation of saliency maps across diverse computer vision modalities, datasets and model architectures. It contains the code for the paper "[*Navigating the Maze of Explainable AI: A Systematic Approach to Evaluating Methods and Metrics*](https://openreview.net/pdf?id=HRkwnZewLC)".


## Introduction
Explainable AI (XAI) is a rapidly growing domain with a myriad of methods as well as metrics aiming to evaluate their efficacy. However, current literature is often of limited scope, examining only a handful of XAI methods and employing one or a few metrics. Furthermore, pivotal factors for performance, such as the underlying architecture or the nature of input data, remain largely unexplored. This lack of comprehensive analysis hinders the ability to make generalized and robust conclusions about XAI performance, which is crucial for directing scientific progress but also for trustworthy real-world application of XAI. In response, we introduce LATEC, a large-scale benchmark that critically evaluates 17 prominent XAI methods using 20 distinct metrics. Our benchmark systematically incorporates vital elements like varied architectures and diverse input types, resulting in 7,560 examined combinations. Using this benchmark, we derive empirically grounded insights into areas of current debate, such as the impact of Transformer architectures and a comparative analysis of traditional attribution methods against novel attention mechanisms. To further solidify LATEC's position as a pivotal resource for future XAI research, all auxiliary data—from trained model weights to over 326k saliency maps and 378k metric scores—are made publicly available.

<br>
<p align="center">
    <img src="data/figures/misc/image.png" width="600"> <br>
    <img src="data/figures/misc/volume.gif" width="600"> <br>
    <img src="data/figures/misc/pc.gif" width="600">
</p>

## 🧭&nbsp;&nbsp;Table of Contents
* [Installation](#Installation)
* [Project Structure](#project-structure)
* [LATEC Dataset](#latec-dataset)
* [Getting started](#getting-started)
  * [Reproducing the Results](#reproducing-the-results)
  * [Run your own Experiments](#run-your-own-experiments)
* [Citation](#citation)  
* [Acknowledgements](#acknowledgements)

## ⚙️&nbsp;&nbsp;Installation

LATEC requires Python version 3.9 or later. All essential libraries for the execution of the code are installed when installing this repository:

```bash
git clone https://github.com/IML-DKFZ/latec
cd latec
pip install .
```
Depending on your GPU, you need to install an appropriate version of PyTorch and torchvision separately. All scripts run also on CPU, but can take substantially longer depending on the experiment. Testing and development were done with the Pytorch version using CUDA 11.6. Note that the packages [Captum](https://github.com/pytorch/captum) and [Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus) are not the official versions but forks to adapt the XAI methods and metrics to 3D modalities and the benchmark.

## 🗃&nbsp;&nbsp;Project Structure


```
├── configs                   - Hydra config files
│   ├── callbacks
│   ├── data
│   ├── eval_metric
│   ├── experiment
│   ├── explain_method
│   ├── extras
│   ├── hydra
│   ├── logger
│   └── paths                 
├── data                      - Data storage and ouput folders
│   ├── datasets              - Datasets for all modalities
│   ├── evaluation            - Evaluation scores as .npz
│   ├── saliency_mapss        - Saliency maps output as .npz
│   ├── figures               - Output of figures and gifs
│   └── model_weights         - Model weights as .ckpt files
├── logs                      - Log files             
├── notebooks                 - Notebooks for visualizations
├── scripts                   - Bash scripts for multi-runs
└── src                       
    ├── data                  - Datamodule scripts
    ├── main                  - Main experiment scripts
    │   ├── main_eval.py      - Runs evaluation pipeline
    │   ├── main_explain.py   - Runs explanation pipeline
    │   └── main_rank.py      - Runs ranking pipeline
    ├── modules               
    │   ├── components        - Various submodules
    │   ├── registry          - Object registries for methods
    │   ├── eval_methods.py   - Loads evaluation metrics
    │   ├── models.py         - Loads deep learning models
    │   └── xai_methods.py    - Loads XAI methods
    └── utils                 - Various utility scripts
```

## 💾&nbsp;&nbsp;LATEC Dataset
If you want to reproduce only certain results or use our provided model weights, saliency maps, or evaluation scores for your own experiments, please download them here:

If you would like to reproduce specific results or utilize our provided model weights, saliency maps, or evaluation scores for your own experiments, please follow the instructions below:

- **Model Weights**: [Download](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/674630/latec_model_weights.zip?sequence=8&isAllowed=y) and unzip the files into the `./data/` directory.
  
- **Saliency Maps (Per Dataset)**: [Download](https://libdrive.ethz.ch/index.php/s/4tm0gxcvBqvMlRA), move them to the respective modality folder, and unzip them at `./data/*modality*/`.
  
- **Evaluation Scores**: [Download](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/674331/latec_evaluation_scores.zip?sequence=1&isAllowed=y) and unzip the files into the `./data/` directory.

## 🚀&nbsp;&nbsp;Getting Started

### ♻️&nbsp;Reproducing the Results

For the [CoMA](https://coma.is.tue.mpg.de/) and [RESISC45](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html) datasets, please download them directly from their respective websites. All other datasets will be automatically downloaded into the `./data/datasets/` folder when running the experiment for the first time.

#### **Generating Saliency Maps**

To generate saliency maps, select the appropriate `.yaml` configuration file for the dataset from `./config/data/` and the modality for the XAI method configuration from `./config/explain_method/`. Then, run the following command, specifying both configurations:

```bash
latec-explain data=vesselmnist3d.yaml explain_method=volume.yaml
```

#### **Evaluating Saliency Maps**

For score computation, in addition to specifying `data` and `explain_method`, define the evaluation method configuration from `./config/eval_metric/` and provide the `.npz` file containing the saliency maps (located at `./data/saliency_maps/*modality*/`). Run the following command with all the required configurations:

```bash
latec-eval data=vesselmnist3d.yaml explain_method=volume.yaml eval_metric=volume_vessel.yaml attr_path='saliency_maps_vesselmnist3d.npz'
```

#### **Ranking Evaluation Scores**

To generate ranking tables, run the following command. Ensure that the paths in `./config/rank.yaml` point to the correct evaluation score `.npz` files and that the appropriate ranking schema is selected:

```bash
latec-rank
```

<br>

To run all three steps in sequence, use the provided bash script `./scripts/run_all_steps.sh`, ensuring that the respective configuration files are correctly filled out. Please note that this process can be time-consuming, even with GPU resources.

<br>


### 🧪&nbsp;Run Your Own Experiments

#### Using Your Own **Dataset** and **Model Weights**

1. Add your dataset to the `./data/datasets/` folder and place your model weights as a `.ckpt` file in the `./data/model_weights/` folder.
2. Add a *LightningDataModule* file for your dataset to `./src/data/` and a corresponding *config.yaml* file to `./config/data/`. Ensure the YAML file includes the `*_target_*` specification.
3. Initialize the model and load the weights in the *ModelsModule.__init__* function (from `./src/modules/models.py`) for the appropriate modality, and append the model to the `self.models` list.
4. Add the necessary layer for CAM methods and the Relational Representation Stability metric to both functions in `./src/utils/hidden_layer_selection.py`.

#### Using Your Own **XAI Method**

1. Add the XAI method parameters to the relevant config file located at `./config/explain_methods/*modality*.yaml`.
2. Add the method to the method registry as a `config()` function similar to other methods in `./src/modules/registry/xai_methods_registry.py`.
3. Ensure that your XAI method object contains a `.attribute(input, target, **hparams)` method that takes an observation, target, and parameters as input, and returns the saliency map as a NumPy array or PyTorch tensor.

#### Using Your Own **Evaluation Metric**

1. Add the evaluation metric parameters to the relevant config file located at `./config/eval_methods/*dataset*.yaml`.
2. Add the metric to the metric registry as a `config()` function similar to other metrics in `./src/modules/registry/eval_metrics_registry.py`.
3. Ensure that your metric's `__call__(x_batch, y_batch, a_batch, device, **kwargs)` function accepts the observation batches (`x_batch`), targets (`y_batch`), saliency maps (`a_batch`), and device as inputs, and outputs the scores as a NumPy array. These scores will be appended to `eval_scores`. Depending on the experiment, a `custom_batch` of data and the XAI method might be applied as well. If your metric requires them, include them in the `**kwargs` input.

## 📝&nbsp;&nbsp;Citation

**Bibtex:**

```bibtex
@misc{klein2024navigating,
      title={Navigating the Maze of Explainable AI: A Systematic Approach to Evaluating Methods and Metrics}, 
      author={Lukas Klein and Carsten T. Lüth and Udo Schlegel and Till J. Bungert and Mennatallah El-Assady and Paul F. Jäger},
      year={2024},
      eprint={2409.16756},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.16756}, 
}
```

## 📣&nbsp;&nbsp;Acknowledgements

The code is developed by the authors of the paper. However, it does also contain pieces of code from the following packages:

- Pytorch EfficientNet 3D by Shi, Jian: https://github.com/shijianjian/EfficientNet-PyTorch-3D
- Pytorch Point Cloud Transformer by Guo, Meng-Hao et al.: https://github.com/Strawberry-Eat-Mango/PCT_Pytorch
- Pytorch Transformer-Explainability by Chefer, Hila et al.: https://github.com/hila-chefer/Transformer-Explainability
- Image Classification by Ziegler, Sebastian: https://github.com/MIC-DKFZ/image_classification

____

<br>

<p align="center">
  <img src="https://polybox.ethz.ch/index.php/s/I6VJEPrCDW9zbEE/download" width="190"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://polybox.ethz.ch/index.php/s/kqDrOTTIzPFYPU7/download" width="91"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://img.in-part.com/thumbnail?stripmeta=true&noprofile=true&quality=95&url=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fassets.in-part.com%2Funiversities%2F227%2FGdzZTi4yThyBhFzWlOxu_DKFZ_Logo-3zu-Research_en_Black-Blue_sRGB.png&width=750" width="120"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="data/figures/misc/eth_logo_kurz_pos.png" width="250">
</p>

LATEC is developed and maintained by the Interactive Machine Learning Group of [Helmholtz Imaging](https://www.helmholtz-imaging.de/) and the [DKFZ](https://www.dkfz.de/de/index.html), as well as the Institute for Machine Learning of the [ETH Zürich](https://ml.inf.ethz.ch/).