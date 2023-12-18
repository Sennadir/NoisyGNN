# A Simple and Yet Fairly Effective Defense for Graph Neural Networks

## Overview

This repository contains python codes and datasets necessary to run the proposed NoisyGNN approach. NoisyGNN is a defense approach designed to defend GNNs while maintaining a low complexity in term of time and operations. The main idea of the paper consists of injecting noise during training in the hidden representations. Please refer to our paper for additional specifications.


## Requirements

Code is written in Python 3.6 and requires:
- PyTorch
- Torch Geometric
- NetworkX

## Datasets
For node classification, the used datasets are as follows:
- Cora
- CiteSeer
- PubMed
- PolBlogs

All these datasets are part of the torch_geometric datasets and are directly downloaded when running the code.


## Training and Evaluation
To use our code, the user should first download the [DeepRobust](https://github.com/DSE-MSU/DeepRobust) package ( https://github.com/DSE-MSU/DeepRobust). Since we are using the [GNNGuard](https://github.com/mims-harvard/GNNGuard/tree/master) as a baseline, we are using the provided code from their official GitHub repository (https://github.com/mims-harvard/GNNGuard/tree/master).

As explained in the GNNGuard's original code, some files need to be substituted in the original DeepRobust implementation in the folder "deeprobust/graph/defense" by the one provided in our implementation. The file "noisy_gcn.py" contains our proposed framework.


To train and evaluate the model in the paper, the user should specify the following :

- Dataset : The dataset to be used
- hidden_dimension: The hidden dimension used in the model (if desired, otherwise default will be used)
- learning rate and epochs
- Budget: The budget of the attack
- beta_max/beta_min: the range of the hyper-parameters related to the noise ratio injected into the underlying GNN (referred to in the paper as \beta).

To run a normal code of NoisyGCN with the Mettack approach using the Cora dataset and using the default parameters for a 10% budget:

```bash
python main_mettack.py --dataset cora --ptb_rate 0.1
```

## Citing

If you find our proposed NoisyGNN useful for your research, please consider citing our paper.

For any additional questions/suggestions you might have about the code and/or the proposed approach to <ennadir@kth.se>.

## License

NoisyGNN is licensed under the MIT License.
