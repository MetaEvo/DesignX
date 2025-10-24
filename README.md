# DesignX: Human-Competitive Algorithm Designer for Black-Box Optimization


This project provides the sourcecodes of DesignX, which has been recently accpeted by NeurIPS 2025

## Citation

The PDF version of the paper is available [here](https://arxiv.org/abs/2505.17866). If you find our DesignX useful, please cite it in your publications or projects.

```latex
@inproceedings{guo2025designx,
  title={DesignX: Human-Competitive Algorithm Designer for Black-Box Optimization},
  author={Guo, Hongshu and Ma, Zeyuan and Ma, Yining and Zhang, Xinglin and Chen, Wei-Neng and Gong, Yue-Jiao},
  booktitle={Proceedings of the 39th Conference on Neural Information Processing Systems},
  year={2025}
}
```
A sister projest we also recommend you to read and refer to is [ConfigX](https://github.com/MetaEvo/ConfigX), which is the basis of DesignX and the origin of the modular evolutionary operator space.

## Requirements

Create the conda environment with python 3.9.18 and torch 2.3.1, then install packages:

```bash
conda create -n DesignX python=3.9.18
conda activate DesignX
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Train

To train the Agent-1, run:

```bash
python main.py
```

To train the Agent-2, run:
```bash
python main.py --train_cfx True
```

For more adjustable settings, please refer to `config.py` and `config_cfg.py` for details.

Recording results: Log files will be saved to `./logs`, the file structure is as follow:
```
logs
|--run_name
   |--logging files
   |--...
```
The saved checkpoints will be saved to `./outputs`, the file structure is as follow:
```
outputs
|--run_name
   |--epoch-0.pt
   |--epoch-1.pt
   |--...
```

## Rollout

Modify `load_path` (The checkpoint saving directory, default to be "./outputs"), `load_name` (The run_name of the target Agent-1 model) and `load_epoch` (The epoch of the model) in `config.py` to assign the Agent-1 model.

Modify `load_path` (The checkpoint saving directory, default to be "./outputs"), `load_name` (The run_name of the target Agent-2 model) and `load_epoch` (The epoch of the model) in `config_cfg.py` to assign the Agent-2 model.

Then run:

```bash
python main.py --test
```

to rollout the assigned Agent-1 & 2 models.
