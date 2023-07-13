This is an unofficial implementation of [SURROGATE GAP MINIMIZATION IMPROVES SHARPNESS-AWARE TRAINING](https://openreview.net/pdf?id=edONMAnhLu-)
for keras and tensorflow 2

## Introduction
The proposed [Sharpness-Aware Minimization](https://arxiv.org/abs/2010.01412) (SAM) improves generalization by minimizing a perturbed loss defined as the maximum loss within a neighborhood in the parameter space. [Surrogate Gap Guided Sharpness-Aware Minimization](https://arxiv.org/pdf/2203.08065.pdf) (GSAM) is a novel improvement over SAM with negligible computation overhead. Conceptually, GSAM consists of two steps: 1) a gradient descent like SAM to minimize the perturbed loss, and 2) an ascent step in the orthogonal direction (after gradient decomposition) to minimize the surrogate gap and yet not affect the perturbed loss. Empirically, GSAM consistently improves generalization (e.g., +3.2% over SAM
and +5.4% over AdamW on ImageNet top-1 accuracy for ViT-B/32). [Official implementation in JAX](https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/gsam/gsam.py)
<img src="https://github.com/mortfer/keras-gsam/blob/master/gsam_algo.png" width="850"/>

## Installation
`pip install git+https://github.com/mortfer/keras-gsam.git`
## How to use
```diff
from gsam import GSAM
# Wrap keras.model instance and specify rho and alpha hyperparameters
gsam_model = GSAM(model, rho=0.05, alpha=0.01)
```
You can use rho schedulers similar to learning rate schedulers 
```diff
from gsam.callbacks import RhoScheduler, CosineAnnealingScheduler
from tensorflow.keras.callbacks import LearningRateScheduler

callbacks = [
    LearningRateScheduler(CosineAnnealingScheduler(T_max=max_epochs, eta_max=1e-3, eta_min=0), verbose=1), 
    RhoScheduler(CosineAnnealingScheduler(T_max=max_epochs, eta_max=0.1, eta_min=0.01), verbose=1), 
]

gsam_model.fit(
    x_train, 
    y_train,
    callbacks=callbacks,
    batch_size=batch_size, 
    epochs=max_epochs
)
```
## Results
An example of how to use gsam can be found in [gsam_comparison.ipynb](https://github.com/mortfer/keras-gsam/blob/master/examples/gsam_comparison.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mortfer/keras-gsam/blob/master/examples/gsam_comparison.ipynb) <br> 
Results obtained:
|              |Val accuracy (%)|
| -------------|:--------------:|
| Vanilla      | 80.52          |
| SAM          | 82.33          |
| GSAM         | 83.04          |
## Acknowledgements
* [Sayak Paul's SAM implementation](https://github.com/sayakpaul/Sharpness-Aware-Minimization-TensorFlow/tree/main)
* [Juntang zhuang's Pytorch Implementation](https://github.com/juntang-zhuang/GSAM)
