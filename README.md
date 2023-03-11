

# Create new environment for it.

conda create -n "firmest" python=3.6.13
conda activate firmest
pip install -r requirements.txt


# config the python path

    export PYTHONPATH=$PYTHONPATH:./src
    export PATH="$PATH:./src"
    export PYTHONPATH=$PYTHONPATH:./main/
    export PATH="$PATH:./main/"
    export PYTHONPATH=$PYTHONPATH:./main/0_local_api/
    export PATH="$PATH:./main/0_local_api/"
    export PYTHONPATH=$PYTHONPATH:./main/2_verify_sampler/
    export PATH="$PATH:./main/2_verify_sampler/"
    export PYTHONPATH=$PYTHONPATH:./main/3_benchmark_sampler/
    export PATH="$PATH:./main/3_benchmark_sampler/"
    export PYTHONPATH=$PYTHONPATH:./main/statistic_lib/
    export PATH="$PATH:./main/statistic_lib/"
    export PYTHONPATH=$PYTHONPATH:./main/apiserver/
    export PATH="$PATH:./main/apiserver/"

# run one example

    python main.py