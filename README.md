# hot_fair_utilities ( Utilities for AI Assisted Mapping fAIr )
Initial lib was developed during Open AI Challenge with [Omdena](https://omdena.com/). Learn more about the challenge at [here](https://www.hotosm.org/tech-blog/hot-tech-talk-open-ai-challenge/)  

## This repo
This is a fork from the original (and on-going) [HOT](https://www.hotosm.org/) project, created with the aim to analyse its performance and to work on improving it.

The workflow and notes behind this project is in [this](https://hackmd.io/@annazan/H1PkFnRz6) HackMD (ask for access if needed).

---

## `hot_fair_utilities` Installation

hot_fair_utilities is a collection of utilities which contains core logic for model data preparation, training and postprocessing. It can support multiple models, currently ramp is supported. 

1. To get started clone this repo first : 
    ```
    git clone https://github.com/hotosm/fAIr-utilities.git
    ```
2. Setup your virtualenv with ```python 3.8``` ( Ramp is tested with python 3.8 )

3. Install tensorflow ```2.9.2``` from [here] (https://www.tensorflow.org/install/pip) According to your os

#### Setup Ramp : 

4. Copy your basemodel : Basemodel is derived from ramp basemodel 
    ```
    git clone https://github.com/radiantearth/model_ramp_baseline.git
    ```

5. Clone ramp working dir 

    ```
    git clone https://github.com/kshitijrajsharma/ramp-code-fAIr.git ramp-code
    ```

6. Copy base model to ramp-code 
    ```
    cp -r model_ramp_baseline/data/input/checkpoint.tf ramp-code/ramp/checkpoint.tf
    ```

7. Install native bindings 
    - Install Numpy 
        ```
        pip install numpy==1.23.5
        ```
    - Install [gdal](https://gdal.org/index.html) .

        for eg : on Ubuntu 
        ```
        sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update
        sudo apt-get install gdal-bin
        sudo apt-get install libgdal-dev
        export CPLUS_INCLUDE_PATH=/usr/include/gdal
        export C_INCLUDE_PATH=/usr/include/gdal
        pip install --global-option=build_ext --global-option="-I/usr/include/gdal" GDAL==`gdal-config --version`        
        ```
        on conda : 
        ```
        conda install -c conda-forge gdal
        ```
    - Install rasterio 

        for eg: on ubuntu : 
        ```
        sudo apt-get install -y python3-rasterio
        ```
        on conda : 
        ```
        conda install -c conda-forge rasterio
        ```

8. Install ramp requirements 

    Install necessary requirements for ramp  and hot_fair_utilites
         
    ```
    cd ramp-code && cd colab && make install  && cd ../.. && pip install -e .
    ```



## Conda Virtual Environment
Create from env fle 

```
conda env create -f environment.yml
```
Create your own

```
conda create -n fAIr python=3.8
conda activate fAIr
conda install -c conda-forge gdal
conda install -c conda-forge geopandas
pip install pyogrio rasterio tensorflow
pip install -e hot_fair_utilities
```

## Test Installation and workflow 

You can run ```package_test.ipynb``` to your pc to test the installation and workflow with sample data provided 
