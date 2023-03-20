

# Create new environment for it.

conda config --set ssl_verify false
conda create -n "firmest" python=3.6.13
conda activate firmest
pip install -r requirements.txt  --trusted-host pypi.org --trusted-host files.pythonhosted.org

pip install  tqdm==4.47.0 --trusted-host pypi.org --trusted-host files.pythonhosted.org
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html --trusted-host pypi.org --trusted-host files.pythonhosted.org

# config the python path

source init_env

# run one example

    python main.py