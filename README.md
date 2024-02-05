# SynTable - A Synthetic Data Generation Pipeline for Cluttered Tabletop Scenes

This repository contains the official implementation of the paper **"SynTable: A Synthetic Data Generation Pipeline for Unseen Object Amodal Instance Segmentation of Cluttered Tabletop Scenes"**.

Zhili Ng, Haozhe Wang, Zhengshen Zhang, Francis Eng Hock Tay,  Marcelo H. Ang Jr.

[[arXiv]](https://arxiv.org/abs/2307.07333)
[[Website]](https://sites.google.com/view/syntable/home)
[![Dataset](https://zenodo.org/badge/DOI/10.5281/zenodo.10565517.svg)](https://doi.org/10.5281/zenodo.10565517)
[[Demo Video]](https://youtu.be/zHM8H58Kn3E)
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

