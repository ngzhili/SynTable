# SynTable - A Synthetic Data Generation Pipeline for Cluttered Tabletop Scenes

SynTable is a robust data generation pipeline that creates photorealistic synthetic datasets of Cluttered Tabletop Scenes. It includes 6 DOF grasping annotations, object amodal masks, object visible masks, object invisible masks, object occlusion rate, scene's occlusion ordering adjacency matrix (OOAM) annotations for each scene.


# **Installation**
1. Change Directory to isaac_sim-2022.1.0\tools directory
``` bash
cd \home\<username>\.local\share\ov\pkg\isaac_sim-2022.1.0\tools
```

2. Clone the repo 
``` bash
git clone https://github.com/ngzhili/SynTable.git
```

3. Change Directory to isaac_sim-2022.1.0 directory
``` bash
cd \home\<username>\.local\share\ov\pkg\isaac_sim-2022.1.0
```

4. Install Dependencies into isaac sim's python
- Get isaac sim python path
    - Get issac sim source code directory path in command line.
    ``` bash
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    echo $SCRIPT_DIR
    ```
    - Get isaac sim python path
    ``` bash
    python_exe=${PYTHONEXE:-"${SCRIPT_DIR}/kit/python/bin/python3"}
    echo $python_exe
    ```
    - Run isaac sim python
    ``` bash
    $python_exe
    ```
- while running isaac sim in bash, install pycocotools and opencv-python into isaac sim python
    ``` bash
    import pip
    package_names=['pycocotools', 'opencv-python'] #packages to install
    pip.main(['install'] + package_names + ['--upgrade'])
    ```
    

# **Generate Synthetic Dataset**

1. Change Directory to Isaac SIM source code
``` bash
cd \home\<username>\.local\share\ov\pkg\isaac_sim-2022.1.0\tools
```
2. Run Syntable Pipeline
``` bash
./python.sh tools/syntable/src/main1.py --input */parameters/default1.yaml --output */dataset/train --mount <mount_directory> --num_scenes 3 --num_views 3 --overwrite --save_segmentation_data
```

### **Types of Flag**
| Flag           | Description |
| :---           |    :----:   |
| ```--input```  | Path to input parameter file.       |
| ```--mount```   | Path to mount symbolized in parameter files via '*'.        |
| ```--headless```   | Will not launch Isaac SIM window.        |
| ```--nap```   | Will nap Isaac SIM after the first scene is generated.        |
| ```--overwrite```   | Overwrites dataset in output directory.        |
| ```--output```   | Output directory. Overrides 'output_dir' param.        |
| ```--num_scenes```  | Number of scenes in dataset. Overrides 'num_scenes' param.       |
| ```--num_views```  | Number of views to generate per scene. Overrides 'num_views' param.      |
| ```--save_segmentation_data```  | Saves visualisation of annotations into output directory. False by default.      |

# **Visualise Annotations**
(to be filled)


# **Sample Annotations**
![RGB](./readme_images/RGB.png) 
![OODAG](./readme_images/OODAG.png) ![OODAG2](./readme_images/OODAG2.png)
![OOAM](./readme_images/OOAM.png) 

## MetaGraspNet README

- Conda name：metagraspnet_env，Python version：3.8。

### Sample **Parallel-Jaw Grasps**

```bash
python ./grasps_sampling/scripts/sample_grasps.py --mesh_root ./models/models_ifl/065/ --paralleljaw --max_grasps 10
```

### **Visualize Parallel-Jaw Grasp Label**

```bash
python ./Scripts/visualize_labels.py --root ./models --dataset_name models_ifl --object 065 --parallel_grasps --analytical --max_grasps 500
```

### To Read .hdf5 File

```bash
# read data structure
h5ls models/models_ifl/065/textured.obj.hdf5

# read data details
h5dump models/models_ifl/065/textured.obj.hdf5
```

### **How to generate quality_score_simulation**

- For generating parallel grasps based on physics simulation, please fullfill installation process from [IsaacGym](https://developer.nvidia.com/isaac-gym). Scripts are tested for **isaac gym version 1.0.preview2**.
    1. Install Issac Gym, tutorials: [https://learningreinforcementlearning.com/setting-up-isaac-gym-on-an-ubuntu-laptop-785b5a15e5a9](https://learningreinforcementlearning.com/setting-up-isaac-gym-on-an-ubuntu-laptop-785b5a15e5a9)
    2. create conda environment:
        
        ```bash
        cd isaacgym
        ./create_conda_env_rlgpu.sh       #takes a while
        conda activate rlgpu
        ```
        
    3. update `LD_LIBRARY_PATH` **🚨**
        
        ```bash
        export LD_LIBRARY_PATH=/home/haozeh/anaconda3/envs/rlgpu/lib:$LD_LIBRARY_PATH
        ```
        
    4. copy file isaacgym (`/home/haozhe/Documents/isaacgym/python/isaacgym`) from folder isaacgym.
    5. install h5py dependency: `pip install h5py`
    6. After you have set up a working isaac gym environment, start simulating with: (existing simulation data will be overwritten!)
        
        ```bash
        (rlgpu) python ./physics_simulation/paralleljaw_simulation.py --root ./models/models_ifl/ --visualize --num_envs 16 --categories 008
        ```
        

### Outstanding Issues with Issac Gym

- when I start run Isaac Gym examples (for example joint_monkey.py) use Isaac Gym preview 4/2 version, my computer will automatically restart.
