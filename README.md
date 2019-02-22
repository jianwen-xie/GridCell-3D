# GridCell-3D

This repository contains a tensorflow implementation of 3D grid cell, which is in the supplemental material of paper "[Learning Grid Cells as Vector Representation of Self-Position Coupled with Matrix Representation of Self-Motion](https://openreview.net/pdf?id=Syx0Mh05YQ)".

## Reference
    @article{DG,
        author = {Gao, Ruiqi  and Xie, Jianwen and and Zhu, Song-Chun and Wu, Ying Nian},
        title = {Learning Grid Cells as Vector Representation of Self-Position Coupled with Matrix Representation of Self-Motion},
        journal={Seventh International Conference on Learning Representations (ICLR)},
        year = {2019}
    }
  
 ## Requirements
- Python 2.7 
- [Tensorflow r1.0+](https://www.tensorflow.org/install/)


## Usage

### (i) Path integral

#### Training with Gaussian kernel

    $ python main.py --mode 0 --GandE 1

#### Testing path integral

    $ python main.py --mode 2 --GandE 1
    
### (ii) Path planning

#### Training with exponential kernel

    $ python main.py --mode 0 --GandE 0
    
#### Testing path planning

    $ python path_planning.py
    
####  Testing path planning with obstacle

    $ python path_planning_obstacle.py
