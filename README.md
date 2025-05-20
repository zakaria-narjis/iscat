# iscat
### Setting up the Conda Environment

To create a Conda environment named `iscat` with Python 3.11, install the required packages from `requirements.txt`, and include OpenJDK, use the following command:

```bash
conda create -n iscat python=3.11 -y && \
conda activate iscat && \
pip install -r requirements.txt && \
conda install -c conda-forge openjdk=8 maven -y
```
### training semantic segmentation
Adapt default config on configs/seg_config.yaml
```bash
python train.py 
```
Or:
```bash
python train.py --config=<your_config_path>
```

### training EV semantic segmentation
Adapt default config on configs/ev_seg_config.yaml
```bash
python train_EV.py 
```
Or:
```bash
python train_EV.py --config=<your_config_path>
```

### training Size prediction 
Adapt default config on configs/size_pred_config.yaml
```bash
python train_size_pred.py 
```
Or:
```bash
python train_size_pred.py  --config=<your_config_path>
```

### training Size prediction (Parallel option)
Adapt default config on configs/size_pred_config.yaml and specifiy wich cuda devices: One single node with 3 gpus for example:
```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nnodes=1 --nproc_per_node=3 train_size_pred_.py 
```
Or:
```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nnodes=1 --nproc_per_node=3 train_size_pred_.py  --config=<your_config_path>
```