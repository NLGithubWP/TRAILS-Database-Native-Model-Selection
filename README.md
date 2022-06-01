

# Requirement

torch              1.11.0
torchaudio         0.11.0
torchvision        0.12.0

TensorFlow 1.15

python 3.60


When run in server, add path to env
# export PATH=”$PATH:/home/xingnaili/Fast-AutoNAS/src”
# export PYTHONPATH=$PYTHONPATH:/home/xingnaili/Fast-AutoNAS/src

export PYTHONPATH=$PYTHONPATH:/Users/kevin/project_python/Fast-AutoNAS/src




# Sampling Algorithms

    1. https://github.com/automl/SMAC3
    1. https://github.com/microsoft/FLAML
3. https://github.com/automl/HpBandSter, https://automl.github.io/HpBandSter/build/html/auto_examples/example_5_mnist.html

[Sampling summary](https://github.com/huawei-noah/vega/blob/master/docs/cn/algorithms/hpo.md)

![image-20220512213253528](documents/img.png)

# Problems

1. NASI 什么时候算一个epoch， 怎么判断是否满足了一个epoch， 怎么定义T 从而看出满足了一个epoch?

   ![image-20220528205050861](documents/image-20220528205049492.png)

