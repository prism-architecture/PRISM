
## 💻 Installation

PRISM is built on top of imitation learning frameworks [GNFactor](https://github.com/YanjieZe/GNFactor/tree/main), [PerAct](https://github.com/peract/peract), [RLBench](https://github.com/stepjam/RLBench). For trajectory generation, we follow VoxPoser \cite{huang2023voxposer} to map subtask descriptions to 3D poses and motion plans.

It is recommended to have a conda envirionment for running the code.

# create python env

```
conda create -n prism-env python=3.9
conda activate prism-env
```

# download coppeliasim 
```
wget https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz --no-check-certificate

tar -xvf CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz

rm CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz
```

# add following lines to your `~/.bashrc` file. 
Remember to source your bashrc (source ~/.bashrc) and reopen a new terminal then.

You should replace the path here with your own path to the coppeliasim installation directory.
```
export COPPELIASIM_ROOT=EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT

export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

# install PyRep and RLBench
For installing PyRep and RLBench, please refer to the [instructions](https://github.com/stepjam/RLBench#install).

# install VoxPoser
For installing PyRep and RLBench, please refer to the [instructions](https://github.com/huangwl18/VoxPoser).

# install Grounded_sam2
For installing grounded_sam2_hf_model, please refer to the [instructions](https://github.com/IDEA-Research/Grounded-SAM-2).

# install PRISM

git clone https://github.com/prism-architecture/PRISM.git

```
cd prism/code/prism
pip install -r requirements.txt
```

# VLM
Obtain an [OpenAI](https://github.com/stepjam/RLBench#install) OpenAI API key.


## Quickstart

```
Test RLBench-COLLAB Tasks with test_code_all_task.py
Run a single episode of a task with all_task_runner.py
```

