# ECMGD
Implementation of ECMGD in our paper: Towards Multi-view Consistent Graph Diffusion, ACM MM 2024.


====
This is the Pytorch implementation of IMvGCN proposed in our paper:

![framework](./framework.png)

## Requirement

  * Python == 3.9.12
  * PyTorch == 2.2.2
  * Numpy == 1.24.1
  * Scikit-learn == 1.4.1
  * Scipy == 1.12.0
  * Texttable == 1.7.0
  * Tqdm == 4.64.2

## Usage

```
python main.py
```

  * --device: gpu number or 'cpu'.
  * --path: path of datasets.
  * --dataset: name of datasets.
  * --seed: random seed.
  * --fix_seed: fix the seed or not.
  * --n_repeated: number of repeat times.
  * --lr: learning rate.
  * --weight_decay: weight decay.
  * --ratio: label ratio.
  * --num_epoch: number of training epochs.
  * --layer: hyperparameter $K$.
  * --alpha: hyperparameter $\alpha$.

All the configs are set as default, so you only need to set dataset.
For example:

 ```
 python main.py --dataset 3Sources
 ```

## Dataset

Please unzip the datasets folders first.

Saved in ./data/datasets/datasets.7z

*Please feel free to email me for the four large datasets or any questions.*
