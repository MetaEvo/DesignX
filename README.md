# An example of DesignX inference

This project provides the example inference code of generating optimizers and controlling their hyper-parameters for the 20 problem instances in the jupyter notebook.

Firstly, create the conda environment with python 3.9.18 and torch 2.3.1:

```bash
conda create -n DesignX python=3.9.18
conda activate DesignX
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

Then after **selecting the jupyter notebook kernel as DesignX**, we can execute the [inference code](inference.ipynb) step by step and see how DesignX solves the problems.
