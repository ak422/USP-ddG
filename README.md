### USP-ddG network

---

#### Description

This repo contains code for **USP-ddG:  A Unified Structural Paradigm with Data Efficacy and Mixture-of-Experts for Predicting Mutational Effects on Protein-Protein Interactions ** by Guanglei Yu, Xuehua Bi, Qichang Zhao, and Jianxin Wang.

In this work, we present USP-ddG, a unified structural paradigm for $\Delta \Delta G$ prediction. USP-ddG integrates a dual-channel architecture with three complementary components: (i) inverse folding-based log-odds ratio, (ii) empirical energy terms from FoldX, and (iii) a geometric encoder with Gaussian noise to capture relaxed conformations. To enhance representation power, we introduce a framework through the integration of feed-forward network (FFN) and Mixture-of-Experts (MoE) adapters to model domain-invariant and domain-specific features. We further propose CATH-guided Folding Ordering (CFO), a data efficacy strategy that organizes samples to mitigate catastrophic forgetting and bias. USP-ddG consistently outperforms existing state-of- the-art (SoTA) methods on the SKEMPI v2.0 benchmark, including the challenging hold-out CATH test set. It achieves superior accuracy on both single- and multi-point mutations and demonstrates strong performance in antibody affinity optimization against H1N1, HER2, and SARS-CoV-2.  

Overview of our **USP-ddG** architecture is shown below.

<div align="center"><img src="./assets/cover.png" alt="cover" style="width:70%;" /></div>



#### Installation

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

#### Preparation of processed dataset

---

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

---

1. Download the trained weights from [case_study](https://drive.google.com/file/d/19hZRsNzwDTDYrKhsg_5BwDTKorVtQ1g_/view?usp=sharing) and place  `case_study.pt`  in the `./trained_models/` directory.
2. Download the Demo dataset from [Demo_cache](https://drive.google.com/file/d/1_xw_JL-ck_90ghWAHKHOhZ0NKHxiODG5/view?usp=sharing) and place `Demo_cache` in the `./data/SKEMPI2/` directory. 
3. Execute the following command:

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

##### Train USP-ddG on the hold-out CATH test set splitting

```bash
python train.py ./configs/train/CATH-MoE.yml --device cuda:0
```
