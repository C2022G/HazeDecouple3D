# HazeDecouple3D
**End to End 3D Scene Reconstruction via Haze Scene Disentanglement**
![Overview of our method](https://github.com/C2022G/HazeDecouple3D/blob/main/readme/1.png)

The implementation of our code is referenced in [kwea123-npg_pl](https://github.com/kwea123/ngp_pl)。The hardware and software basis on which our model operates is described next
 - Ubuntu 18.04
 -  NVIDIA GeForce RTX 3090 ,CUDA 11.3


## Setup
Let's complete the basic setup before we run the model。

 
+ Clone this repo by
```python
git clone https://github.com/C2022G/HazeDecouple3D.git
```
+  Create an anaconda environment
```python
conda create -n dcpnerf python=3.7
``` 
+ cuda code compilation dependency.
	- Install pytorch by
	```python
	conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
	```
	- Install torch-scatter following their [instruction](https://github.com/rusty1s/pytorch_scatter#installation) like
	```python
	pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
	```
	- Install tinycudann following their [instrucion](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension)(pytorch extension) like
	```python
	pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
	```
	- Install apex following their [instruction](https://github.com/NVIDIA/apex#linux) like
	```python
	git clone https://github.com/NVIDIA/apex 
	cd apex 
	pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
	```
	- Install core requirements by
	```python
	pip install -r requirements.tx
	```
  
+ Cuda extension:please run this each time you pull the code.``.
 	```python
	pip install models/csrc/
	# (Upgrade pip to >= 22.1)
	```

## Datasets
The dataset can be obtained from https://pan.baidu.com/s/1tS4q59IPoezKPQ0ho1L0CQ?pwd=evnf password: evnf


## Training
Our configuration file is available in config/opt.py, check it out for details.These profiles include the location of the dataset, the name of the experiment, the number of training sessions, and the loss function parameters. The above parameters may be different depending on the dataset.To train a Lego model, run

```python
python run.py  \
	--root_dir /data/data/Synthetic_NeRF_Haz/lego
	--exp_name lego_0.2_0.8_5
	--haz_dir_name 0.2_0.8_5
	--split train
	--num_epochs 5
	--composite_weight 1e-0
	--distortion_weight 1e-2
	--opacity_weight 1e-3
	--dcp_weight 6e-2
	--foggy_weight 1e-3
```
## parameters
According to the better parameters for scene reconstruction obtained from the experiment, from left to right, they are $\lambda_1,\lambda_2,\lambda_3,\lambda_4$.
![Overview of our method](https://github.com/C2022G/HazeDecouple3D/blob/main/readme/5.png)

## Result

![Overview of our method](https://github.com/C2022G/HazeDecouple3D/blob/main/readme/2.png)
![Overview of our method](https://github.com/C2022G/HazeDecouple3D/blob/main/readme/3.png)
![Overview of our method](https://github.com/C2022G/HazeDecouple3D/blob/main/readme/4.png)



