# Setup the Development Environment

## 1. Use `conda` to Manage the Python Environment

You can create the `conda` environment using the environment file if your CUDA version is 11.x and skip the rest part of this section:
```shell
$ conda env create --file gleap_env.yml
```

```shell
$ conda create --name gleap_env
$ conda activate gleap_env
$ conda install pip
$ pip install scipy
$ pip install matplotlib
$ pip install jupyterlab
$ pip install emoji
$ pip install tqdm
$ pip install -U scikit-learn
$ pip install mplcairo
```

**Install the `transformers` library**

First, install PyTorch according to the official documentation. For the server I used, execute the following command:
```shell
$ pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Then install transformers:
```shell
$ pip install transformers
```

## 2. Start JupyterLab

```shell
$ bash ./start_jupyterlab.sh
```
