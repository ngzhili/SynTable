# Installation and Running SynTable within docker container (Isaac Sim 2022.1.1)

Hardware: RTX 3090, Ubuntu 20.04 and installed an older version of the NVIDIA Driver(535.183.01) using the .run file. It seems like newer versions of NVIDIA drivers within 550 series were incompatible. 
The recommended version 535.129.03 (referenced in nvidia container installation guide) also did not seem to work as well.

1. If required, uninstall the prior NVIDIA related packages and clean the environment before different driver version installation:
    ```
    sudo apt-get remove --purge '^nvidia-.*'
    sudo apt autoremove
    sudo apt autoclean
    sudo apt-get -y install cuda
    sudo reboot
    ```

2. Run the following commands within the directory, in which the [NVIDIA Driver .run file](https://www.nvidia.com/en-us/drivers/details/226764/) was stored (Make sure nouveau is disabled before running):
    ```
    chmod +x NVIDIA-Linux-x86_64-535.183.01.run
    sudo ./NVIDIA-Linux-x86_64-535.183.01.run
    ```

3. Follow the steps in the [NVIDIA container installation page](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html#) to install and deploy the docker container.
- Container Setup: Steps 1-3
- Container Deployment: Steps 1-3
- Within the container deployment, instead of running step 4 with volume caches, run the command without the volume caches to solve GLFW initialization-related errors.
  ```
  Ex: docker run --name isaac-sim --entrypoint bash -it --runtime=nvidia --gpus all -e "ACCEPT_EULA=Y" --rm --network=host
  nvcr.io/nvidia/isaac-sim:2022.1.1
  ```

4.  solve python segmentation errors related to vulkan, by running the command:
    ```
    echo '' > vulkan_check.sh
    ```

5. Now you should have successfully installed the following packages within the activated docker container without facing any python segmentation fault related errors:

- To solve omni.kit.usd related error, install the opencv-python-headless version:
  ```
  ./python.sh -m pip install opencv-python-headless==4.9.0.80
  ./python.sh -m pip install pycocotools==2.0.7
  ```

- Install git within the docker container:
  ```
  apt update && apt install -y git wget vim
  ```
  
- Install nano (text editor) within the docker container:
  ```
  apt-get update && apt-get install nano
  ```
  
6. Changed the given paths (using nano) within the train_config_syntable1.yaml and mount_dir to match your local omniverse host paths.

7. Git clone the SynTable directory directly into the tools folder within isaac-sim docker.

8. Stored the outputs within the docker container and copied the folder to your local machine.

9. Execute SynTable in docker (include the headless parameter to avoid GLFW initialization related errors):
  ```
  ./python.sh /isaac-sim/tools/SynTable/syntable_composer/src/main1.py --input /isaac-sim/tools/SynTable/mount_dir/parameters/train_config_syntable1.yaml --output /isaac-sim/tools/SynTable/output_2/ --mount '/isaac-sim/tools/SynTable/mount_dir' --num_scenes 3 --num_views 3 --overwrite --save_segmentation_data --headless
  ```
