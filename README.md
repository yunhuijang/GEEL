# A Simple and Scalable Representation for Graph Generation (ICLR 2024)

In this repository, we implement [GEEL: Gap Encoded Edge List in the paper: A Simple and Scalable Representation for Graph Generation](https://openreview.net/pdf?id=nO344avRib).

## Dependencies

ALT is built in Python 3.10.0, PyTorch 1.12.1, and PyTorch Geometric 2.2.0 . Use the following commands to install the required python packages.

```sh
conda env create --file environment.yaml
```

## Running experiments

### 1. Configurations

The configurations are given in `config/trans/` directory. Note that max_len denotes the maximum length of the sequence representation in generation. We set max_len as the maximum number of edges of training and test graphs.

### 2. Training and evaluation

You can train ALT model and generate samples by running:
```sh
CUDA_VISIBLE_DEVICES=${gpu_id} bash script/trans/{script_name}.sh
```

For example, 
```sh
CUDA_VISIBLE_DEVICES=0 bash script/trans/com_small.sh
```

Then the generated samples are saved in  `samples/` directory and the metrics are reported on WANDB.
