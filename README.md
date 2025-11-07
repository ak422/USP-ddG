### USP-ddG network

---

#### Description

This repo contains code for **USP-ddG: Unified Structural Paradigm with Data Efficacy and MoE Serves as an Effective Predictor for Mutational Effects on Protein-Protein Interactions** by Guanglei Yu, Qichang Zhao, Xuehua Bi, and Jianxin Wang.

In this work, we introduce a unified structural encoder scenario, dubbed USP-ddG, that integrates CATH-guided Folding Ordering (CFO) for data organization. It uses a dual-channel architecture combining the Feed-Forward Network (FFN) and Mixture of Experts (MoE) for capturing domain-invariant and specific features of protein structures, respectively. Under different structural assumptions, $\Delta \Delta G$ is decomposed into structure-fixed and structure-relaxed components. The structure-fixed part is estimated based on the negative log-likelihood scores of the inverse folding model, while synergistically guided by the energy-based inductive bias. The structure-relaxed part is derived via the geometric encoder that takes atomic coordinates with added Gaussian noise as input. Furthermore, self-supervised learning (SSL) is employed to enhance embedding alignment across diverse protein domains. 

Overview of our **USP-ddG** architecture is shown below.

<img src="./assets/cover.png" alt="cover" style="width:70%;" />



## Installation

To install USP-ddG, first clone this repository

```
git clone https://github.com/ak422/USP-ddG.git
```

To install the necessary python dependencies, then use the file `env.yml to create the environment

```bash
conda env create -f env.yml
conda activate USP-ddG
```

The default PyTorch version is 2.3.1, python version is 3.10, and cudatoolkit version is 11.3.1. They can be changed in [`env.yml`](./env.yml).

## Preparation of processed dataset

To generate mutant structures and prepare the processed datasets for SKEMPI v2.0, CR6261, HER2, and S285, respectively, execute the following commands from the code directory:

```bash
python skempi_parallel.py --reset --subset skempi_v2
python skempi_parallel.py --reset --subset CR6261
python skempi_parallel.py --reset --subset HER2
python skempi_parallel.py --reset --subset S285
```

#### Trained Weights

***

1. The 5 trained model weights of USP-ddG for the hold-out CATH test set are available at: [USP-ddG-sampler](https://drive.google.com/file/d/1A0G4bce1I_CN-M8vvJQ2WJcKSvW-wmJh/view?usp=sharing);
3. The  trained weights for case study is located in: [case_study](https://drive.google.com/file/d/19hZRsNzwDTDYrKhsg_5BwDTKorVtQ1g_/view?usp=sharing)

#### Demo example

1. Download the  trained weights from [case_study](https://drive.google.com/file/d/19hZRsNzwDTDYrKhsg_5BwDTKorVtQ1g_/view?usp=sharing) folder and the Demo dataset from [Demo_cache];
2. Execute the following command:

---

```
python case_study.py ./configs/inference/demo.yml --device cuda:0
```

#### Usage of Inference and Training Models

***

##### Evaluate USP-ddG on hold-out CATH test set

```bash
python test.py ./configs/train/CATH-MoE.yml  --device cuda:0
```

##### Case Study 1: CR6261

```python
python case_study.py ./configs/inference/case_study_CR6261.yml --device cuda:0
```

##### Case Study 2: HER2

```bash
python case_study.py ./configs/inference/case_study_HER2.yml --device cuda:0
```

##### Case Study 3: S285

```
python case_study.py ./configs/inference/case_study_S285.yml --device cuda:0
```

##### Evaluate USP-ddG in a zero-shot setting using the provided PDB file and mutation list

```
python zero_shot.py ./configs/inference/zero_shot.yml  --device cuda:0
```

##### Train USP-ddG on the hold-out CATH test set splitting

```bash
python train.py ./configs/train/CATH-MoE.yml --device cuda:0
```
