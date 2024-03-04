# DHSNeRF
**Multi-Image Decoupling of Haze and Scenes using Neural Radiance Fields**
![Overview of our method](https://github.com/C2022G/dhsnerf/blob/main/readme/2.png)

The implementation of our code is referenced in [kwea123-npg_pl](https://github.com/kwea123/ngp_pl)。The hardware and software basis on which our model operates is described next
 - Ubuntu 18.04
 -  NVIDIA GeForce RTX 3090 ,CUDA 11.3


## Setup
Let's complete the basic setup before we run the model。

 
+ Clone this repo by
```python
git clone https://github.com/C2022G/dhsnerf.git
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
Due to the lack of a dedicated single-haze scene image dataset, we employed virtual scenes as the experimental subjects.   We utilized Blender 3D models provided in NeRF to render realistic 360° panoramic images and depth maps while maintaining consistent camera poses and intrinsic parameters. Under the assumption of uniformly distributed haze particles in the virtual scenes, we endowed eight virtual scenes with uniform atmospheric light and the same haze density, achieved by applying fog to the rendered original clear images using the ASM formula.

**The dataset can be obtained from [Baidu net disk](https://pan.baidu.com/s/10vo99AKu6sAAfWD2ZYQL7w?pwd=2022) or [Google Drive](https://drive.google.com/file/d/1GeC3HEzEnf0yyYcxEUdlNLr1GDO6LbAD/view?usp=sharing)**


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
After the training, three types of files will be generated in the current location, such as: log files are generated in the logs folder, model weight files are generated in ckpts, and rendering results are generated in the results folder, including test rendering images and videos of haze scenes and clean scenes

![The optimal hyperparameters of each scene are obtained by experiments.](https://github.com/C2022G/dhsnerf/blob/main/readme/table3.png)



Similarly, we can adjust dcp_weight and foggy_weight if the default parameters don't apply to a particular dataset.


## Visualization/Evaluation
By specifying the split, ckpt_path parameters, the run.py script supports rendering new camera trajectories, including test and val, from the pre-trained weights.To render test images,run

```python
python run.py  \
	--root_dir /devdata/chengan/Synthetic_NeRF/lego \
	--split test
	--exp_name lego_0.14_0.8_5 \
	--haz_dir_name 0.14_0.8_5 
	--ckpt_path /ckpts/..
```

## result
![{\bf Qualitative comparisons were performed on LLFF hazy dataset with atmospheric light 0.8 and scattering coefficient 0.14.} The novel viewpoint images rendered by baseline methods fail to faithfully restore the haze-free scene and are plagued by numerous black artifacts. In contrast, DHSNeRF effectively removes substantial haze and faithfully reconstructs the scene.](https://github.com/C2022G/dhsnerf/blob/main/readme/llff.png)

![The optimal hyperparameters of each scene are obtained by experiments.](https://github.com/C2022G/dhsnerf/blob/main/readme/table1.png)

![{\bf Qualitative comparisons were performed on Synthetic hazy dataset with atmospheric light 0.8 and scattering coefficient 0.2.} Both DCP and FFANet methods exhibit noticeable haze artifacts that cannot be removed. DehazFormer and DHSNeRF can faithfully reconstruct the 3D shapes, but DehazFormer shows some color deviations, leading to overfitting.](https://github.com/C2022G/dhsnerf/blob/main/readme/nerf.png)


![The optimal hyperparameters of each scene are obtained by experiments.](https://github.com/C2022G/dhsnerf/blob/main/readme/table2.png)

![{\bf Erosion Study of soft density guided weight ($W^\alpha$).} As shown in (a), we select a ray passing through the calibrated red point in the fern scene, recording the weights of $P\mbox{-}Field$ and $C\mbox{-}Field$ sampling points along with their distances from the ray origin.  In (b), in the absence of soft density-guided weight , there is a noticeable increase in sampling points, and $P\mbox{-}Field$ weights tend to approach 0.  In (c), the weight distribution of DHSNeRF exhibits a uniform distribution of haze particles, and $C\mbox{-}Field$'s ray display a unimodal termination distribution.](https://github.com/C2022G/dhsnerf/blob/main/readme/guide.png)





https://github.com/C2022G/dhsnerf/assets/151046579/369a46a3-b2bf-4029-a561-7995cab9303a



https://github.com/C2022G/dhsnerf/assets/151046579/a68a46a4-f3e9-48cf-b8c3-0241d16681f0





