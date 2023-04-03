

## Repository Setup
1. Create a fresh conda environment, and install all dependencies.
    ```
    conda create -n env_MACA python=3.7
    conda activate env_MACA
    git clone https://github.com/xxayt/MACA.git
    cd MACA
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
    ```

2. Install pytorch
    ```
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==1.7.1 torchvision==0.8.2
    ```