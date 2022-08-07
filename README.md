# PDO-s3DCNNs
The implementation of the paper "[PDO-s3DCNNs: Partial Differential Operator Based Steerable 3D CNNs](https://proceedings.mlr.press/v162/shen22c/shen22c.pdf)" (ICML2022).
Please contact shenzhy@pku.edu.cn if you have any question.

## Requirements

Pytorch-1.7

albumentations-1.0.0

tifffile-2021.4.8

scipy-1.6.3

elasticdeform-0.4.9

## Datasets

**3D Tetris**: constructed in tetris/run_tetris.py

**SHREC'17**: https://shapenet.cs.stanford.edu/shrec17/
The 3d models can be converted into voxels of size 64 × 64 × 64 with the following code https://github.com/mariogeiger/obj2voxel.

**ISBI 2012**:  http://brainiac2.mit.edu/isbi_challenge/home

## Training and Evaluation

To train and evaluate the model(s) in the paper, run following commands:

**3D Tetris**: \
**For cubical transformations:** python3 tetris/run_tetris.py --model S4 --discretization_mode FD\
**For arbitrary rotations:** python3 tetris/run_tetris.py --model SO3 --discretization_mode Gaussian --drop_rate 0.2

**SHREC'17**: python3 shrec17/train.py --log_dir S4 --model_path model/S4_Net.py

**ISBI 2012**: python3 ISBI2012/train.py


## Results

Our model achieves the following performance on :

### [3D Tetris]

The test accuracy of the $\mathcal{O}$- and $SO(3)$- steerable CNNs discretized by FD on 3D Tetris with cubic rotations.

|Group | Feature fields  | Test acc. (\%)  | Params | Time |
|-------------| --------- |---------------- | -------------- | -------------- |
|$\mathcal{O}$| Regular   |     100.0         |      31k       | 14.3s |
|$\mathcal{O}$| $\mathcal{V}$-quotient   |     100.0         |      5.5k       | 2.3s |
|$\mathcal{O}$| $\mathcal{T}$-quotient   |     100.0         |      2.2k       | 1.3s|
|$SO(3)$| Irreducible   |     100.0         |      22.8k       | 66.7s|

The test accuracy of the SO(3)-steerable CNNs on 3D Tetris with arbitrary rotations.

|  Discretization  | Kernel size | Test acc. | Time |
| ------------------ |---------------- | -------------- | -------------- |
| FD   |  $3\times 3\times 3$ |      $18.20\pm 3.13$       | 66.7s |
| Gaussian   |      $3\times 3\times 3$         |    $29.99\pm 4.98$       | 67.3s |
| Gaussian   |     $5\times 5\times 5$        |     $99.04\pm 0.14$      | 109.5s|

### [SHREC'17 Retrieval]
The retrieval performance of $\mathcal{V}$-, $\mathcal{T}$-, $\mathcal{O}$-, $\mathcal{I}$- and $SO(3)$-steerable CNNs, tested on SHREC17. Score is the average value of mAP of micro-average version and macro-average version.
|Group| Discretization | Feature field | Score |
|-|-|-|-|
|$\mathcal{V}$| FD | Regular| 52.7 |
|$\mathcal{T}$| FD | Regular| 57.6 |
|$\mathcal{O}$| FD | Regular| 58.6 |
|$\mathcal{I}$| Gaussian | Regular| 55.5 |
|$SO(3)$| FD | Irreducible | 57.4 |
|$SO(3)$| Gaussian | Irreducible | 58.3 |

SHREC’ 17 perturbed dataset results. Mixed features mean that the features are composed of regular and $\mathcal{V}$-quotient features.
|  Method  | Score  | P@N | R@N | mAP | P@N | R@N | mAP | Params |
| ------------------ |---------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| SE3CNN   |   55.5 |70.4| 70.6 |66.1| 49.0 |54.9| 44.9 |0.14M|
| PDO-s3DCNN w.r.t. $SO(3)$   |   58.3 | 73.1 | **73.4** | 69.3 | **52.5** | 55.4 | 47.3 | 0.15M |
| PDO-s3DCNN w.r.t. $\mathcal{O}$ with regular features   |    58.6 | 72.9| 73.0 | 68.8 | 51.9 | 57.7 | 48.3 | 0.15M |
| PDO-s3DCNN w.r.t. $\mathcal{O}$ with V-quotient features   |  55.5 | 69.2 | 69.6 | 65.0 | 48.0 | 56.3 | 46.0 | 0.15M|
| PDO-s3DCNN w.r.t. $\mathcal{O}$ with mixed features   |     **59.1**| **73.2**| 73.3| **69.3** |51.7 | **57.8** | **48.8**| 0.15M|

### [ISBI 2012 Segmentation]

|  Method  | $V_{\text{rand}}$ | $V_{\text{info}}$|
| ------------------ |---------------- | -------------- | 
| PDO-s3DCNN w.r.t. $\mathcal{V}$   |     0.98415         |     0.99031 |
| PDO-s3DCNN w.r.t. $\mathcal{O}$   |     0.98727         |      0.99089  |

## Citation
If you found this package useful, please cite
```
@inproceedings{shen2022pdo,
  title={PDO-s3DCNNs: Partial Differential Operator Based Steerable 3D CNNs},
  author={Shen, Zhengyang and Hong, Tao and She, Qi and Ma, Jinwen and Lin, Zhouchen},
  booktitle={International Conference on Machine Learning},
  pages={19827--19846},
  year={2022},
  organization={PMLR}
}
```
