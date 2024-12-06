# iscat
### Setting up the Conda Environment

To create a Conda environment named `iscat` with Python 3.11, install the required packages from `requirements.txt`, and include OpenJDK, use the following command:

```bash
conda create -n iscat python=3.11 -y && \
conda activate iscat && \
pip install -r requirements.txt && \
conda install -c conda-forge openjdk=8 maven -y
```