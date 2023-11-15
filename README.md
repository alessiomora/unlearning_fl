# A Survey on Federated Unlearning
This repository is the official repository of the paper 
"A Survey on Federated Unlearning".

citation

## Preliminaries
The simulation code in this repository mainly leverages Flower 
and TensorFlow (TF). Python virtual env is managed via Poetry.
See `unlearning_fl/pyproject.toml`. To reproduce our virtual env,
follow the instructions in the Environment Setup section of this readme.


The code in this repository has been tested on Ubuntu 22.04.3,
and with Python version `3.10.13`.



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

## Results from the Paper
To intuitively understand the impact of a client's data on the global model
we performed a set of experiments, shown in Table below (Table I in the paper),
where we compared the performance in accuracy of a global model trained including 
or excluding a specific client $u$ in the federation. While the test accuracy is 
very similar among such two versions of the global models, when the federation
includes client *u* the global model exhibits significantly higher accuracy on 
client *u*'s train data. This means that the global model trained with client 
$u$ among the federation may leak information about client *u*'s private data 
and that may be susceptible to, at least, membership attacks. 
Table I below reports the accuracy of a global model
*w* trained on a federation of participants including or excluding a specific
client *u*. Three datasets for image classification were considered 
(CIFAR-100, Birds and Aircrafts), and a visual transformer architecture
was used (MiT-B0), starting from a pretrained
checkpoint and performing FL for 100 rounds. Further detail about the 
experimental settings are provided in section Experimental Setting.
The *test* column refers to the global model's accuracy on a test 
set that clients never see during training. The *train* column refers 
to the global model's accuracy on the train set of client *u*.

<table>
  <tr>
    <td></td>
    <td style="text-align: center", colspan="2">Retrained Model</td>
    <td style="text-align: center", colspan="2">Model with client <i>u</i> </td>
  </tr>
  <tr>
    <td> </td>
    <td> Test </td>
    <td> Train </td>
    <td> Test </td>
    <td> Train </td>
  </tr>
  <tr>
    <td> CIFAR-100 </td>
    <td> 78.99 </td>
    <td> 76.80 </td>
    <td> 78.52 </td>
    <td> <b>94.00</b></td>
  </tr>
<tr>
    <td> Aircrafts </td>
    <td> 67.72 </td>
    <td> 67.00 </td>
    <td> 67.66 </td>
    <td> <b>93.00</b> </td>
  </tr>
  <tr>
    <td> Birds </td>
    <td> 70.81 </td>
    <td> 63.50 </td>
    <td> 70.99 </td>
    <td> <b>95.50</b> </td>
  </tr>
</table>

### Reproducing Results

### Experimental Setting
This section describes the experimental setting for the results reported 
in Table I.

#### Model Architecture and Hyperparameters
We used a visual transformer, i.e. MiT-B0, 
with approximately 3.6M parameters, initialized from a pre-trained 
model checkpoint trained on ImageNet-1k (69.27\% accuracy on test data).
We adapted the one-layer classification head to the specific task, 
initializing such a layer from scratch. We employed the AdamW optimizer 
with a client learning rate of 3e-4, with a round-wise exponential
decay of 0.998, 5 local epochs, batch size of 32, and weight decay 
regularization of 1e-3.

#### Datasets
We performed the experiments on three datasets, i.e. CIFAR-100, 
Caltech-2011 (birds), and FGVC-Aircraft (aircrafts).


**CIFAR-100.** CIFAR-100 consists of 60,000 examples of 32x32 color images 
-- 50,000 for training and 10,000 for testing -- belonging to 100 classeDs. 
To match the transformer models' input size, we resized the images to a 
resolution of 224x224 pixels; we also preprocessed the training images with
random crop and horizontal flip layers. We partitioned the training set to 
simulate 100 clients in the federation; we set 100 as the number of clients 
so that, in the IID setting, each client can have at least five per-class 
(unique) examples. At each round, 10 clients out of 100 were randomly 
selected to participate.

**Caltech-2011.** Caltech 2011 (birds) consists of 11,788 examples
of color images -- 5,994 for training and 5,794 for testing -- belonging
to 200 classes. To match the input size of the transformer models we resized 
the images to a resolution of 224x224 pixels; we also preprocessed the training
images with random crop and horizontal flip layers, similarly to the work in
\cite{hu2023federated}. We partitioned the training set to simulate 29 clients
in the federation; we set 29 as the number of clients so that, 
in the IID setting, each client can have at least one per-class (unique) example.
At each round, 5 clients out of 29 were randomly selected to participate.


**FGVC-Aircraft.** The FGVC-Aircraft dataset contains 10,000
images of aircraft -- 6,667 for training and validation and 3,333 for testing. 
The aircraft labels are organized in a four-level hierarchy, i.e., model, variant,
family, and manufacturer (from finer to coarser). We consider a classification 
at the variant level (e.g. Boeing 737-700). A variant collapses all the models
that are visually indistinguishable into one class. The dataset contains 100
images for each of the 100 different aircraft model variants. We removed the 
copyright banner from the images by cutting off the bottom 20 pixels in height. 
To match the input size of the transformer models we resized the images to a 
resolution of 224x224 pixels; we also preprocessed the training images with 
random crop and horizontal flip layers, similarly to the work in 
\cite{hu2023federated}. We partitioned the training set to simulate 
65 clients in the federation; we set 65 as the number of clients so 
that, in the IID setting, each client can have at least one per-class
(unique) example. At each round, 7 clients out of 65 were randomly
selected to participate.


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