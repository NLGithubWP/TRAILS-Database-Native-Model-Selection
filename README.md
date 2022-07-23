

# Requirement

torch              1.11.0
torchaudio         0.11.0
torchvision        0.12.0
TensorFlow 1.15
python 3.60

connect to ciidaa first:
ssh shaofeng@ciidaa.d2.comp.nus.edu.sg
pwd: shaofengtmp

then connect to ncrs:
ssh xingnaili@10.0.0.125
xingnailincrs

When run in server, add path to env
export PATH=”$PATH:/home/xingnaili/Fast-AutoNAS/src”
export PYTHONPATH=$PYTHONPATH:/home/xingnaili/Fast-AutoNAS/src
export PYTHONPATH=$PYTHONPATH:/Users/kevin/project_python/Fast-AutoNAS/src

export PATH=”$PATH:/home/naili/Fast-AutoNAS/src”
export PYTHONPATH=$PYTHONPATH:/home/naili/Fast-AutoNAS/src

# Pip install
conda env export --no-builds > env.yml
conda env create -f env.yml

# env
scp env.yml shaofeng@ciidaa.d2.comp.nus.edu.sg:/users/shaofeng/nl/Fast-AutoNAS/
scp env.yml xingnaili@10.0.0.125:/home/xingnaili/Fast-AutoNAS

scp main/benchmark.py shaofeng@ciidaa.d2.comp.nus.edu.sg:/users/shaofeng/nl/Fast-AutoNAS/main/
scp main/benchmark.py xingnaili@10.0.0.125:/home/xingnaili/Fast-AutoNAS/main/

scp -r src shaofeng@ciidaa.d2.comp.nus.edu.sg:/users/shaofeng/nl/Fast-AutoNAS/
scp -r src xingnaili@10.0.0.125:/home/xingnaili/Fast-AutoNAS/

scp -r xingnaili@10.0.0.125:/home/xingnaili/Fast-AutoNAS/Logs .
scp -r shaofeng@ciidaa.d2.comp.nus.edu.sg:/users/shaofeng/nl/Fast-AutoNAS/Logs .

scp -r src naili@pandax2.d2.comp.nus.edu.sg:/home/naili/Fast-AutoNAS/
scp -r main naili@pandax2.d2.comp.nus.edu.sg:/home/naili/Fast-AutoNAS/
scp -r data naili@pandax2.d2.comp.nus.edu.sg:/home/naili/Fast-AutoNAS/

scp naili@pandax2.d2.comp.nus.edu.sg:/home/naili/Fast-AutoNAS/Logs/fast_auto_nas_log.log .

export PATH=”$PATH:/Users/kevin/project_python/Fast-AutoNAS/src”
export PYTHONPATH=$PYTHONPATH:/Users/kevin/project_python/Fast-AutoNAS/src
export PYTHONPATH=$PYTHONPATH://Users/kevin/project_python/Fast-AutoNAS/src

# Sampling Algorithms

    1. https://github.com/automl/SMAC3
    1. https://github.com/microsoft/FLAML
3. https://github.com/automl/HpBandSter, https://automl.github.io/HpBandSter/build/html/auto_examples/example_5_mnist.html

[Sampling summary](https://github.com/huawei-noah/vega/blob/master/docs/cn/algorithms/hpo.md)

![image-20220512213253528](documents/img.png)

# Problems

1. 多分类，求∇0 f(xi), 是不是只考虑这个样本的true label 的那以个维度的输出 f ?

1. NASI 什么时候算一个epoch， 怎么判断是否满足了一个epoch， 怎么定义T 从而看出满足了一个epoch?

   ![image-20220528205050861](documents/image-20220528205049492.png)

