# Modeling the Shape of the Brain Connectome via Deep Neural Networks
### [Paper](https://arxiv.org/pdf/2203.06122.pdf) | [Slides](https://users.cs.utah.edu/~haocheng/slides/ipmi2023.pdf)
PyTorch implementation of estimating a Riemannian manifold accommodating brain connectome faithfully.<br><br>
 :desert: [Haocheng Dai](https://users.cs.utah.edu/~haocheng/),
 :crocodile: [Martin Bauer](https://www.math.fsu.edu/~bauer/),
 :ocean: [P. Thomas Fletcher](https://scholar.google.com/citations?user=7pRRhkkAAAAJ&hl=en),
 :desert: [Sarang Joshi](https://scholar.google.com/citations?user=GyqdQTEAAAAJ&hl=en) <br>
 :desert:University of Utah, :crocodile:Florida State University, :ocean:University of Virginia <br>
 <br>
International Conference on Information Processing in Medical Imaging (IPMI), 2023 :tent:

<img src='Figures/architecture.png' alt="drawing" width="800"/>
<img src='Figures/performance.png' alt="drawing" width="800"/>

## TL;DR quickstart

To setup a conda environment, begin the training process, and inference:
```
conda env create -f environment.yml
conda activate metcnn
cd Metric-Cnn-2D-IPMI/Scripts/
bash runMetricCnnTrainingInference.sh
```

## Setup

Python 3 dependencies:
```
itk==5.2.0
lazy_import==0.2.2
matplotlib==3.3.1
numba==0.55.1
numpy==1.19.5
PyYAML==6.0
scikit_image==0.18.3
SimpleITK==2.2.1
skimage==0.0
torch==1.10.2
tqdm==4.55.0
```

We provide a conda environment setup file including all of the above dependencies. Create the conda environment `metcnn` by running:
```
conda env create -f environment.yml
```

## What is a Metric CNN?

A metric CNN is a simple convolutional encoder-decoder neural network (CEDNN) trained to estimating a Riemannian manifold that represents the brain connectome faithfully using a covariant derivative loss. The network directly maps from multiple vector fields (3D output) to a Riemannian metric field (4D output), acting as the manifold so we can run geodesic tractography on it and also construct connectome Riemannian manifold atlas from tractography data to statistically quantify the geometric variability of structural connectivity across a population.


Estimating a Riemannian metric takes less than a few hours and only requires a single GPU, depending on resolution. Inference a Riemannian metric from an optimized MetricCNN takes less than a second, again depending on resolution.

## 3D implementaion

You can find the 3D implementation [here](https://github.com/aarentai/Metric-Cnn-3D-IPMI).


## Citation

```
@article{dai2022deep,
  title={Deep Learning the Shape of the Brain Connectome},
  author={Dai, Haocheng and Bauer, Martin and Fletcher, P Thomas and Joshi, Sarang C},
  journal={arXiv preprint arXiv:2203.06122},
  year={2022}
}
```