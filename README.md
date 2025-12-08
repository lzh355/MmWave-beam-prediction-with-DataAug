# Efficient Training of mmWave Beam Prediction Models: A Data Augmentation Approach

This folder includes deep learning codes and corresponding results.

The folder is free to use, including dataset utilization, simulation result reproduction, model improvement, etc.

To start simulations of conventional supervised learning:
~~~
python train.py --aug_type=none
~~~

Reference: K. Ma, et al., Deep Learning Assisted Calibrated Beam Training for Millimeter-Wave Communication Systems.

To start simulations of of our proposed approach:
~~~
python train.py --aug_type=All
~~~

To start simulations of the ablation study:
~~~
python train.py --aug_type=Cyclic_shift
python train.py --aug_type=Flip
python train.py --aug_type=Noise --sigma=0.3
python train.py --aug_type=Label --Z0=2.0 --Zf=8.0 --k=6.0
~~~


Environment: Pytorch, Python 3.8.0., MATLAB R2023b.
