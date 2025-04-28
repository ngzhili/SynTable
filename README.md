# SynTable - A Synthetic Data Generation Pipeline for Cluttered Tabletop Scenes

This repository contains the official implementation of the paper **"SynTable: A Synthetic Data Generation Pipeline for Unseen Object Amodal Instance Segmentation of Cluttered Tabletop Scenes"**.

Zhili Ng*, Haozhe Wang*, Zhengshen Zhang*, Francis Eng Hock Tay,  Marcelo H. Ang Jr.
*equal contributions

[![arXiv](https://img.shields.io/badge/arXiv-2307.07333-b31b1b.svg)](https://arxiv.org/pdf/2307.07333.pdf)
[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10565517.svg)](https://doi.org/10.5281/zenodo.10565517)
[![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white)](https://www.youtube.com/watch?v=zHM8H58Kn3E)
[[Website]](https://sites.google.com/view/syntable/home)
[[Modified UOAIS-v2]](https://github.com/ngzhili/uoais-v2?tab=readme-ov-file)



![teaser](./readme_images/teaser.png)

SynTable is a robust custom data generation pipeline that creates photorealistic synthetic datasets of Cluttered Tabletop Scenes. For each scene, it includes metadata such as 
- [x] RGB image of scene
- [x] depth image of Scene
- [x] scene instance segmentation masks
- [x] object amodal (visible + invisible) rgb
- [x] object amodal (visible + invisible) masks
- [x] object modal (visible) masks
- [x] object occlusion (invisible) masks
- [x] object occlusion rate
- [x] object visible bounding box
- [x] tabletop visible masks
- [x] background visible mask (background excludes tabletop and objects)
- [x] occlusion ordering adjacency matrix (OOAM) of objects on tabletop

## **Installation**
1. Install [NVIDIA Isaac Sim 2022.1.1 version](https://developer.nvidia.com/isaac-sim) on Omniverse

2. Change Directory to isaac_sim-2022.1.1 directory
    ``` bash
    cd '/home/<username>/.local/share/ov/pkg/isaac_sim-2022.1.1/tools'
    ```

3. Clone the repo 
    ``` bash
    git clone https://github.com/ngzhili/SynTable.git
    ```

4. Install Dependencies into isaac sim's python
    - Get issac sim source code directory path in command line.
        ``` bash
        SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
        echo $SCRIPT_DIR
        ```
    - Get isaac sim's python path
        ``` bash
        python_exe=${PYTHONEXE:-"${SCRIPT_DIR}/kit/python/bin/python3"}
        echo $python_exe
        ```
    - Run isaac sim's python
        ``` bash
        $python_exe
        ```
    - while running isaac sim's python in bash, install pycocotools and opencv-python into isaac sim's python
        ``` bash
        import pip
        package_names=['pycocotools', 'opencv-python'] #packages to install
        pip.main(['install'] + package_names + ['--upgrade'])
        ```

5. Copy the mount_dir folder to your home directory (anywhere outside of isaac sim source code)
    ``` bash
    cp -r SynTable/mount_dir /home/<username>
    ```

## Docker Installation
Please refer to the [README.docker.md](https://github.com/ngzhili/SynTable/blob/master/README.docker.md).

## **Adding object models to nucleus**
1. You can download the .USD object models to be used for generating the tabletop datasets [here](https://mega.nz/folder/1nJAwQxA#1P3iUtqENKCS66uQYXk1vg).

2. Upload the downloaded syntable_nucleus folder into Omniverse Nucleus into /Users directory.

3. Ensure that the file paths in the config file are correct before running the generate dataset commands.

## **Generate Synthetic Dataset**
Note: Before generating the synthetic dataset, please ensure that you uploaded all object models to isaac sim nucleus and their paths in the config file is correct.

1. Change Directory to Isaac SIM source code
    ``` bash
    cd /home/<username>/.local/share/ov/pkg/isaac_sim-2022.1.1
    ```
2. Run Syntable Pipeline (non-headless)
    ``` bash
    ./python.sh SynTable/syntable_composer/src/main1.py --input */parameters/train_config_syntable1.yaml --output */dataset/train --mount '/home/<username>/mount_dir' --num_scenes 3 --num_views 3 --overwrite --save_segmentation_data
    ```

### **Types of Flags**
| Flag           | Description |
| :---           |    :----:   |
| ```--input```  | Path to input parameter file.       |
| ```--mount```   | Path to mount symbolized in parameter files via '*'.        |
| ```--headless```   | Will not launch Isaac SIM window.        |
| ```--nap```   | Will nap Isaac SIM after the first scene is generated.        |
| ```--overwrite```   | Overwrites dataset in output directory.        |
| ```--output```   | Output directory. Overrides 'output_dir' param.        |
| ```--num-scenes```  | Number of scenes in dataset. Overrides 'num_scenes' param.       |
| ```--num-views```  | Number of views to generate per scene. Overrides 'num_views' param.      |
| ```--save-segmentation-data```  | Saves visualisation of annotations into output directory. False by default.      |

Note: To save Object Amodal RGB Instances, set `save_segmentation_data: True` in parameter file.

### Example Parameter (Configuration) File

```
# Define Objects
object_1:
  obj_class_id: 0
  obj_model: /Users/syntable_nucleus/bop_challenge/hb/models/obj_000001_converted/hb_obj_000001.usd
  obj_count: 1
  obj_size_enabled: false
  obj_scale: 0.001
  obj_coord_camera_relative: false
  obj_rot_camera_relative: false
  obj_coord: Uniform((-4,-2,10), (4,2,15))
  obj_rot: Uniform((0, 0, 0), (360, 360, 360))
  obj_physics: true

object_2:
    ...

# Define Room Ceiling Light Configuration 
ceilinglights:
  light_intensity: Uniform(100,1500)
  light_radius: 15
  light_color: (255, 255, 255)
  light_temp_enabled: false
  light_temp: Uniform(2000,6500)
  light_directed: true
  light_directed_focus: 20
  light_directed_focus_softness: 0
  light_height: 15
  light_width: 15
  light_distant: false
  light_camera_relative: false
  light_rot: (0, 0, 0)
  light_coord: (0, 0, 5)
  light_count: 1
  light_distance: Uniform(3, 8)
  light_coord_camera_relative: false
  light_rot_camera_relative: false
  light_vel: (0, 0, 0)
  light_rot_vel: (0, 0, 0)
  light_accel: (0, 0, 0)
  light_rot_accel: (0, 0, 0)
  light_movement_light_relative: false

# Define Source Light Configuration
lights:
  light_intensity: Uniform(0,30000)
  light_radius: Uniform(0.3,0.7)
  light_color: (255, 255, 255)
  light_temp_enabled: false
  light_temp: Uniform(2000,6500)
  light_directed: false
  light_directed_focus: 20
  light_directed_focus_softness: 0
  light_distant: false
  light_camera_relative: false
  light_rot: (0, 0, 0)
  light_coord: (0, 0, 15)
  light_count: 1
  light_distance: Uniform(3, 8)
  light_horiz_fov_loc: Uniform(-1, 1)
  light_vert_fov_loc: Uniform(-1, 1)
  light_coord_camera_relative: false
  light_rot_camera_relative: false
  light_vel: (0, 0, 0)
  light_rot_vel: (0, 0, 0)
  light_accel: (0, 0, 0)
  light_rot_accel: (0, 0, 0)
  light_movement_light_relative: false

# Define Source Light Radii Bounds
spherelight_hemisphere_radius_min: 1.5
spherelight_hemisphere_radius_max: 2.5

# Define Camera Radii Bounds
cam_hemisphere_radius_min: 0.7
cam_hemisphere_radius_max: 1.4
auto_hemisphere_radius: true

# Define Room Parameters
scenario_room_enabled: true
scenario_class_id: 0
floor_size: 15
wall_height: 15
floor_material: Choice('*/assets/materials/materials.txt')
wall_material: Choice('*/assets/materials/materials.txt')
floor_texture_rot: Uniform(0, 360)
wall_texture_rot: Uniform(0, 360)

# Define Initial Camera Parameters
camera_coord: (-2, 0, 2)
camera_rot: (-20, 0, 0)
focus_distance: 1

# Define number of objects in each scene
max_obj_in_scene: Range(1, 40)
randomise_num_of_objs_in_scene: true

# Annotations Configuration
save_segmentation_data: false # Set True to enable saving Object Amodal RGB Instances
save_background: false # Set True to save background RGB, Depth, amodal/visible instance segmentations (for foreground/background segmentation use case)
output_dir: dataset 
num_scenes: 10
num_views: 2
img_width: 640
img_height: 480
rgb: true
depth: true
instance_seg: true
groundtruth_visuals: true
physics_simulate_time: 5
path_tracing: false
sky_light_intensity: 0
horiz_aperture: 2.6327803436685087
vert_aperture: 1.9573321100745658
focal_length: 1.88
```


## Generated dataset
- SynTable data generation pipeline generates dataset in COCO - Common Objects in Context format.

## **Folder Structure of Generated Synthetic Dataset**

    .
    ├── ...
    ├── SynTable-Sim                  # Generated dataset
    │   ├── data                      # folder to store RGB, Depth, OOAM 
    │   │   └── mono
    │   │       ├── rgb
    │   │       │   ├── 0_0.png       # file naming convention follows sceneNum_viewNum.png
    │   │       │   └── 0_1.png 
    │   │       ├── depth
    │   │       │   ├── 0_0.png  
    │   │       │   └── 0_1.png 
    │   │       └── occlusion order
    │   │           ├── 0_0.npy  
    │   │           └── 0_1.npy 
    │   ├── parameters                # parameters used for generation of annotations
    │   └── train.json                # Annotation COCO.JSON
    └── ...


## **Visualise Annotations**
1. Create python venv and install dependencies
    ```
    python3.8 -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    ```
2. Visualise sample annotations (creates a visualise_dataset directory in dataset directory, then saves annotation visualisations there)
    ```
    python ./visualize_annotations.py --dataset './sample_data' --ann_json './sample_data/annotation_final.json'
    ```

## **Sample Visualisation of Annotations**
![sample_annotations1](./readme_images/1.png)
![sample_annotations2](./readme_images/2.png)

## **References**
We have heavily modified the Python SDK source code from NVIDA Isaac Sim's Replicator Composer.

## **Citation**
If you find our work useful for your research, please consider citing the following BibTeX entry:
```
@misc{ng2023syntable,
      title={SynTable: A Synthetic Data Generation Pipeline for Unseen Object Amodal Instance Segmentation of Cluttered Tabletop Scenes}, 
      author={Zhili Ng and Haozhe Wang and Zhengshen Zhang and Francis Tay Eng Hock and Marcelo H. Ang Jr au2},
      year={2023},
      eprint={2307.07333},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

