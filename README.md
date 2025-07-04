# BECAME  
Official code for the paper **"BECAME: BayEsian Continual Learning with Adaptive Model MErging"** (ICML 2025).

## Setup

### Dependencies
To set up the Python environment, run:

```bash
conda env create -f environment.yaml
```

### Datasets

**CIFAR-100**  
The CIFAR-100 dataset will be automatically downloaded when running the code.

**MiniImageNet**  
Please manually download the following files and place them in the appropriate directory:
- [Training data](https://drive.google.com/file/d/1fm6TcKIwELbuoEOOdvxq72TtUlZlvGIm/view)
- [Test data](https://drive.google.com/file/d/1RA-MluRWM4fqxG9HQbQBBVVjDddYPCri/view)

**TinyImageNet**  
Download the dataset from its [official website](http://cs231n.stanford.edu/).

## Usage

The code supports two types of experiments: **GPM-based** and **NSCL-based**.  
Navigate to the corresponding directory and run:

```bash
cd GPM-based  # or cd NSCL-based
bash scripts/run.sh
```


## Citation
```bibtex
@inproceedings{li2025became,
  title     = {BECAME: BayEsian Continual Learning with Adaptive Model MErging},
  author    = {Li, Mei and Lu, Yuxiang and Dai, Qinyan and Huang, Suizhi and Ding, Yue and Lu, Hongtao},
  booktitle = {International Conference on Machine Learning},
  year      = {2025}
}
```