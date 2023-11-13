# A Survey on Federated Unlearning
This repository is the official repository of the paper 
"A Survey on Federated Unlearning".

citation

## Preliminaries
The simulation code in this repository mainly leverages Flower 
and TensorFlow (TF). Python virtual env is managed via Poetry.
See `unlearning_fl/pyproject.toml`. To reproduce our virtual env,
follow the instructions in the Environment Setup section of this readme.




## Environment Setup
By default, Poetry will use the Python version in your system. 
In some settings, you might want to specify a particular version of Python 
to use inside your Poetry environment. You can do so with `pyenv`. 
Check the documentation for the different ways of installing `pyenv`,
but one easy way is using the automatic installer:

```bash
curl https://pyenv.run | bash
```
You can then install any Python version with `pyenv install <python-version>`
(e.g. `pyenv install 3.9.17`) and set that version as the one to be used. 
```bash
# cd to your unlearning_fl directory (i.e. where the `pyproject.toml` is)
pyenv install 3.10.12

pyenv local 3.10.12

# set that version for poetry
poetry env use 3.10.12
```
To build the Python environment as specified in the `pyproject.toml`, use the following commands:
```bash
# cd to your unlearning_fl directory (i.e. where the `pyproject.toml` is)

# install the base Poetry environment
poetry install

# activate the environment
poetry shell
```

## Results of pretrained transformers on `imagenet-1k`
The code in `eval_imagenet1k.py` evaluates pretrained ViT and MiT models 
on `imagenet-1k` validation set.


| Model | Params | Accuracy | 
| ------------- | ------------- | ------------- |
| ViT-T | 5.7M | 72.13 |
| ViT-S | 22M | 78.92 |
| MiT-B0 | 3.6M | 69.27 |
| MiT-B1 | 13.7M | 78.01 |
| MiT-B2 | 24.7M | 81.54 |

ViT architectures and pretraining from [[1]](https://arxiv.org/abs/2012.12877).

MiT architectures and pretraining from [[2]](https://arxiv.org/abs/2105.15203).

[1] Touvron, Hugo, et al. "Training data-efficient image transformers & distillation through attention." 
International conference on machine learning. PMLR, 2021.

[2] Xie, Enze, et al. "SegFormer: Simple and efficient design for semantic segmentation with transformers." 
Advances in Neural Information Processing Systems 34 (2021): 12077-12090.