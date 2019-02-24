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


## How to run

### (0) Learning 3D units 

#### Training with single block setting

    $ python ./code_single_group/main.py --single_block True --num_group 1 --block_size 8
    
#### Visualization

    $ python ./code_single_group/visualize_3d_grid_cells.py
    
<p align="center">
<img src="https://github.com/jianwen-xie/GridCell-3D/blob/master/demo/3D_patterns/heatmap_0.png" width="200px"/>
<img src="https://github.com/jianwen-xie/GridCell-3D/blob/master/demo/3D_patterns/heatmap_1.png" width="200px"/>
<img src="https://github.com/jianwen-xie/GridCell-3D/blob/master/demo/3D_patterns/heatmap_2.png" width="200px"/>
<img src="https://github.com/jianwen-xie/GridCell-3D/blob/master/demo/3D_patterns/heatmap_3.png" width="200px"/>
<img src="https://github.com/jianwen-xie/GridCell-3D/blob/master/demo/3D_patterns/heatmap_4.png" width="200px"/>
<img src="https://github.com/jianwen-xie/GridCell-3D/blob/master/demo/3D_patterns/heatmap_5.png" width="200px"/>
<img src="https://github.com/jianwen-xie/GridCell-3D/blob/master/demo/3D_patterns/heatmap_6.png" width="200px"/>
<img src="https://github.com/jianwen-xie/GridCell-3D/blob/master/demo/3D_patterns/heatmap_7.png" width="200px"/></p>  


### (1) Path integral

#### Training with Gaussian kernel

    $ python ./code/main.py --mode 0 --GandE 1

#### Testing path integral

    $ python ./code/main.py --mode 2 --GandE 1 --ckpt model.ckpt-7999
    
 <p align="center">
 <img src="https://github.com/jianwen-xie/GridCell-3D/blob/master/demo/path_integral/1.png" width="260px"/>
 <img src="https://github.com/jianwen-xie/GridCell-3D/blob/master/demo/path_integral/2.png" width="260px"/>
 <img src="https://github.com/jianwen-xie/GridCell-3D/blob/master/demo/path_integral/3.png" width="260px"/>
 <img src="https://github.com/jianwen-xie/GridCell-3D/blob/master/demo/path_integral/4.png" width="260px"/>
 <img src="https://github.com/jianwen-xie/GridCell-3D/blob/master/demo/path_integral/5.png" width="260px"/>
 <img src="https://github.com/jianwen-xie/GridCell-3D/blob/master/demo/path_integral/6.png" width="260px"/></p>
    
### (2) Path planning

#### Training with exponential kernel

    $ python ./code/main.py --mode 0 --GandE 0
    
#### Testing path planning

    $ python ./code/path_planning.py    

    
 <p align="center">
 <img src="https://github.com/jianwen-xie/GridCell-3D/blob/master/demo/path_planning/1.png" width="260px"/>
 <img src="https://github.com/jianwen-xie/GridCell-3D/blob/master/demo/path_planning/2.png" width="260px"/>
 <img src="https://github.com/jianwen-xie/GridCell-3D/blob/master/demo/path_planning/3.png" width="260px"/>
 <img src="https://github.com/jianwen-xie/GridCell-3D/blob/master/demo/path_planning/4.png" width="260px"/>
 <img src="https://github.com/jianwen-xie/GridCell-3D/blob/master/demo/path_planning/5.png" width="260px"/>
 <img src="https://github.com/jianwen-xie/GridCell-3D/blob/master/demo/path_planning/6.png" width="260px"/></p>
 
 ####  Testing path planning with obstacle

    $ python ./code/path_planning_obstacle.py
 
 <p align="center">
 <img src="https://github.com/jianwen-xie/GridCell-3D/blob/master/demo/path_planning_obstacle/test00.png" width="260px"/>
 <img src="https://github.com/jianwen-xie/GridCell-3D/blob/master/demo/path_planning_obstacle/test01.png" width="260px"/>
 <img src="https://github.com/jianwen-xie/GridCell-3D/blob/master/demo/path_planning_obstacle/test02.png" width="260px"/>
 <img src="https://github.com/jianwen-xie/GridCell-3D/blob/master/demo/path_planning_obstacle/test03.png" width="260px"/>
 <img src="https://github.com/jianwen-xie/GridCell-3D/blob/master/demo/path_planning_obstacle/test04.png" width="260px"/>
 <img src="https://github.com/jianwen-xie/GridCell-3D/blob/master/demo/path_planning_obstacle/test05.png" width="260px"/></p>
 
 ### Q & A
For any questions, please contact Jianwen Xie (jianwen@ucla.edu) and Ruiqi Gao (ruiqigao@ucla.edu) 
 
