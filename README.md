# Meta-Embedded Clustering (MEC)

## A new method for improving clustering quality in unlabeled bird sound datasets

In recent years, ecoacoustics has offered an alternative to traditional biodiversity monitoring techniques with the development of passive acoustic monitoring (PAM) systems allowing, among others, to automatically detect and identify species, such as crepuscular and nocturnal tropical birds that are difficult to be tracked by human observers. PAM systems allow generating large audio datasets, but these monitoring techniques still face the challenge to infer ecological information that can be transferred to conservationists. In most cases, several thousand hours of recordings need to be manually labeled by an expert limiting the operability of the systems. Based on the advancement of meta-learning algorithms and unsupervised learning techniques, we propose Meta-Embedded Clustering (MEC), a new method to improve the quality of clustering in unlabeled bird sound datasets. 

|![Meta-Embedded Clustering (MEC)](https://github.com/ear-team/darksound/blob/main/docs/figure.png)| 
|:--:| 
| **Meta Embedded Clustering (MEC)**|

The MEC method is organized in two main steps, with: (a) fine-tuning of a pretrained convolutional neural network (CNN) backbone with different meta-learning algorithms using pseudo-labeled data, and (b) clustering of manually-labeled bird sounds in the latent space based on vector embeddings extracted from the fine-tuned CNN. The MEC method significantly enhanced clustering performance, achieving a 85% improvement over the traditional approach of solely employing CNN features extracted from a general dataset. However, this enhanced performance came with the trade-off of excluding a portion of the data categorized as noise. By improving the quality of clustering in unlabeled bird sound datasets, The MEC method is here designed to facilitate the work of ecoacousticians in managing acoustic units of bird song/call clustered according to their similarities, and in identifying potential clusters of unknown species.

## Installation
Download [Anaconda](https://www.anaconda.com/products/distribution) and prepare your environment using the command line.
```
conda create --name darksound python=3.10
conda activate darksound
```

Install the required libraires using the package installer [pip](https://pip.pypa.io/en/stable/).
```
pip install -r requirements.txt
# If you cannot build wheels for hdbscan, install it with conda
# conda install -c conda-forge hdbscan
```
Install darksound package. First go to the root directory of the package darksound (where is the config file pyproject.toml). Then execute the command line.
```
pip install .
```

## Usage
### Darksound Dataset 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10512103.svg)](https://zenodo.org/records/10512103)

Darksound is an open-source and code-based dataset for the evaluation of unsupervised meta-learning algorithms in the context of ecoacoustics. This dataset is composed of regions of interest (ROIs) of nocturnal and crepuscular bird species living in tropical environments that were automatically segmented using the Python package Bambird ([Michaud et al., 2023](https://www.sciencedirect.com/science/article/abs/pii/S1574954122004022)). The dataset is split into two sets, with a training set and a test set. 

The dataset is easily accessible and downloadable on [Zenodo](https://zenodo.org/records/10512103) or can be directly downloaded using Python:

```python
from torchvision.models import DenseNet121_Weights
from torchvision import transforms
# Import the Darksound class
from dataset import Darksound 

# Load DenseNet weights for transformation
weights = DenseNet121_Weights.IMAGENET1K_V1 
# Download the Darksound training set
train_set = Darksound(split='train', transform=transforms.Compose([weights.transforms()]), download=True)
```

### a) Fine-tuning pretrained CNN backbones with meta-learning algorithms
Specific emphasis is placed on meta "metric" learning based algorithms that are used for performing the experiments. More precisely, it is possible to choose between three meta-learning algorithms ([Matching Networks](https://arxiv.org/pdf/1606.04080.pdf), [Prototypical Networks](https://arxiv.org/pdf/1703.05175.pdf) and [Relation Networks](https://arxiv.org/pdf/1711.06025.pdf)) for fine-tuning four different CNN backbones ([ResNet18](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html), [VGG16](https://arxiv.org/abs/1409.1556), [DenseNet121](https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html), [AlexNet](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)). This requires to indicate the desired parameters in the `config.yaml` file and to run the following command:

```
python fine-tuning.py
```

### b) Clustering vector embeddings extracted from the fine-tuned CNN
Clustering of the vector embeddings is performed using the Meta Embedded Clustering (MEC) method. The objective of the MEC method is to improve the quality of clustering of unlabeled bird sounds datasets in order to determine a number of clusters close to the ground truth. MEC method can be performed on the Darksound dataset by indicating the path to the fine-tuned CNN and running the following command:

```
python clustering.py --path "embeddings/prototypical-networks-5way-1shot-densenet.pt"
```
An example of the evaluation of the clustering performances of the MEC method is accessible from this [notebook](https://github.com/ear-team/darksound/blob/main/notebooks/clustering-evaluation.ipynb).

## Citing this work
[![Citation Badge](https://api.juleskreuer.eu/citation-badge.php?doi=10.1016/j.ecoinf.2024.102687)](https://juleskreuer.eu/citation-badge/)
If you find the MEC method useful for your research, please consider citing it as:

- Poutaraud, J., Sueur, J., Thébaud, C., Haupert, S., [Meta-Embedded Clustering (MEC): A new method for improving clustering quality in unlabeled bird sound datasets](https://doi.org/10.1016/j.ecoinf.2024.102687), Ecological Informatics, Volume 82, September 2024.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Please make sure to update tests as appropriate.

## Acknowledgements
We would like to thank authors from [EasyFSL](https://github.com/sicara/easy-few-shot-learning) for open-sourcing their code and publicly releasing checkpoints, and contributors to [Bambird](https://github.com/ear-team/bambird) for their excellent work in creating labelling function to build cleaner bird song recording dataset.
