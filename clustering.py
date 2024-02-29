"""
Copyright [2023] [Poutaraud]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Import the librairies
import os, random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import umap
from hdbscan.validity import validity_index

import torch
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18, VGG16_Weights, vgg16, DenseNet121_Weights, densenet121, AlexNet_Weights, alexnet
from torchvision.models.feature_extraction import create_feature_extractor

from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from kneed import KneeLocator

from config import load_config
from dataset import Darksound

from networks.protonet import PrototypicalNetworks
from networks.matchnet import MatchingNetworks
from networks.relatnet import RelationNetworks

import warnings
warnings.filterwarnings("ignore")

class Relation(torch.nn.Module):
    def __init__(self, backbone):
        super(Relation, self).__init__()
        self.backbone = backbone
        self.additional_layers = torch.nn.Sequential(torch.nn.AdaptiveMaxPool2d((1, 1)),torch.nn.Flatten())
    def forward(self, x):
        x = self.backbone(x)
        embedding = list(x.values())[0]
        x = self.additional_layers(embedding)
        return x

def get_features(model, spectrogram, device, params):
    # Extract the features from the model
    if params['PARAMS_MODEL']['NETWORK'] == 'relation':
        relation_features = Relation(model.backbone).to(device)
        features = relation_features(spectrogram.to(device).unsqueeze(dim=0))
    else:
        features = model.backbone.forward(spectrogram.to(device).unsqueeze(dim=0)).squeeze(dim=0)
    # Detach and convert to numpy array 
    return features.detach().cpu().numpy()

def evaluation(features, true_labels, clusterer):
    # Remove noisy samples indexes from pseudo and true labels
    pseudo_labels = clusterer.labels_
    clustered = pseudo_labels >= 0
    # Compute evaluation metrics
    ari = adjusted_rand_score(true_labels[clustered], pseudo_labels[clustered])
    ami = adjusted_mutual_info_score(true_labels[clustered], pseudo_labels[clustered])
    dbcv = validity_index(features.astype(np.float64), pseudo_labels, metric='euclidean')
    return ari, ami, dbcv 

if __name__ == '__main__':
    # Import config file and load parameters
    CONFIG_FILE = 'config.yaml'
    params = load_config(CONFIG_FILE)
    
    # Set the seed for all random packages that could possibly be used
    random_seed = params['RANDOM_SEED']
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Use cuda if available for faster computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='Path to the pretrained model')
    parser.add_argument('-v', '--verbose', type=bool, default=True)
    args = parser.parse_args()
    
    # -------------------------------------------------------------------------
    # LOAD THE CNN BACKBONE                
    # -------------------------------------------------------------------------

    # Load the backbones and the weights
    if params['PARAMS_MODEL']['BACKBONE'] == 'resnet18':
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights).to(device)
    if params['PARAMS_MODEL']['BACKBONE'] == 'vgg16':
        weights = VGG16_Weights.IMAGENET1K_V1
        model = vgg16(weights=weights).to(device)
    if params['PARAMS_MODEL']['BACKBONE'] == 'densenet':
        weights = DenseNet121_Weights.IMAGENET1K_V1
        model = densenet121(weights=weights).to(device)
    if params['PARAMS_MODEL']['BACKBONE'] == 'alexnet':   
        weights = AlexNet_Weights.IMAGENET1K_V1
        model = alexnet(weights=weights).to(device)
   
    # -------------------------------------------------------------------------
    # LOAD THE DARKSOUND DATASET                
    # -------------------------------------------------------------------------

    test_set = Darksound(split='test', 
                        hpss=params['PARAMS_MODEL']['HPSS'], 
                        remove_background=params['PARAMS_MODEL']['REMOVE_BG'], 
                        transform=transforms.Compose([weights.transforms()]),
                        n_classes=30,
                        download=True,
                        )
    # Get the ground truth labels
    true_labels = np.array(test_set.__getlabel__())
    
    # -------------------------------------------------------------------------
    # LOAD THE META-LEARNING MODEL                
    # -------------------------------------------------------------------------
    
    # Get features shape and flatten classifier or fully connected layers
    if args.path.split("-")[-1] == 'resnet18':
        features = np.zeros(shape=(len(test_set), model.fc.in_features))
        model.fc = torch.nn.Flatten()
    elif args.path.split("-")[-1] == 'vgg16':
        features = np.zeros(shape=(len(test_set), 512))
        # Reduce vector size and computation in the network
        model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = torch.nn.Flatten()
    elif args.path.split("-")[-1] == 'densenet':
        features = np.zeros(shape=(len(test_set), model.classifier.in_features))
        model.classifier = torch.nn.Flatten()
    elif args.path.split("-")[-1] == 'alexnet':
        features = np.zeros(shape=(len(test_set), 256))
        # Reduce vector size and computation in the network
        model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = torch.nn.Flatten()

    # Load the meta-learning algorithms
    if args.path.split("-")[0] == 'matching':
        model = MatchingNetworks(model, use_softmax=True).to(device)
    if args.path.split("-")[0] == 'prototypical':
        model = PrototypicalNetworks(model, use_softmax=True).to(device)
    if args.path.split("-")[0] == 'relation':
        if args.path.split("-")[-1] == 'resnet18':
            model = create_feature_extractor(model, return_nodes=['layer4.1.bn2']) 
        if args.path.split("-")[-1] == 'vgg16':
            model = create_feature_extractor(model, return_nodes=['features.30'])
        if args.path.split("-")[-1] == 'densenet':
            model = create_feature_extractor(model, return_nodes=['features.norm5'])
        if args.path.split("-")[-1] == 'alexnet':
            model = create_feature_extractor(model, return_nodes=['features.12'])  
        model = RelationNetworks(model, use_softmax=False).to(device)

    # -------------------------------------------------------------------------
    # EXTRACT FEATURES FROM CNN BACKBONE                
    # -------------------------------------------------------------------------
    if args.verbose:
        print("Extracting features from CNN backbone...") 
        
    # Load backbone
    model.load_state_dict(torch.load(args.path, map_location=device)) 

    # Extract the features from the model
    for i in tqdm(range(len(test_set)), desc='Extracting features'):
        # Extracting features from the model
        features[i] = get_features(model, test_set[i][0], device, params) 
    # Save the vector embeddings to disk
    np.save(f'embeddings/features/{os.path.splitext(os.path.basename(args.path))[0]}.npy', features)
    
    # -------------------------------------------------------------------------
    # DIMENSIONALITY RECUCTION AND CLUSTERING               
    # -------------------------------------------------------------------------
    if args.verbose:
        print("Parameter estimation for dimensionality reduction and clustering...")
    
    df = pd.DataFrame() # create empty dataframe
    
    for n_components in [5,10,15,20,25]: # range of UMAP n_components
        # Reduce dimensionality of the latent space
        embedding = umap.UMAP(densmap=True, n_components=n_components, n_jobs=-1, random_state=random_seed).fit_transform(features)

        # Determine the parameters for DBSCAN (Sander et al., 1998)
        k = embedding.shape[1] * 2 - 1 
        # Calculate average distance between each point in the data set and its k-nearest neighbors (k corresponds to min_points).
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(embedding)
        distances, indices = neighbors_fit.kneighbors(embedding)
        # Sort distance values by ascending value and plot
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1] 
        # Find the knee (curvature inflexion point)
        kneedle = KneeLocator(
            x = np.arange(0, len(distances), 1),
            y = distances,
            interp_method = "polynomial",
            curve = "convex",
            direction = "increasing")
        eps = float(kneedle.knee_y)  

        # Perform DBSCAN clustering from embedding array
        clusterer = DBSCAN(min_samples=k, eps=eps).fit(embedding)
        # Get the predicted labels and the number of clusters
        pseudo_labels = clusterer.labels_ 
        clusters = len(np.unique(pseudo_labels[pseudo_labels >= 0])) # remove noisy samples
        percentage = int(round(np.sum(pseudo_labels >= 0) / features.shape[0], 2) * 100)
        ari, ami, dbcv = evaluation(embedding, true_labels, clusterer)
        
        # Add new row to dataframe
        new_row = pd.DataFrame({
                                'AMI': [ami], 
                                'ARI': [ari], 
                                'DBCV': [dbcv], 
                                'n_clusters': [clusters], 
                                'n_components': [n_components], 
                                'min_samples': [k], 
                                'noisy_samples': [len(pseudo_labels[pseudo_labels < 0])]
                                })
        
        df = pd.concat([df, new_row], ignore_index=True)
        
        if args.verbose:
            print(f'\nClustering {percentage}% of the data: Found {clusters} clusters with n_components={n_components}.')
            print(f'ARI: {ari}\nAMI: {ami}\nDBCV: {dbcv}')

    # Write pandas dataframe to csv file
    df.to_csv(f'embeddings/{os.path.splitext(os.path.basename(args.path))[0]}.csv')
    
    if args.verbose:
        # Get highest DBCV score
        best_dbcv = df['DBCV'].max()
        idx = df.index[df['DBCV'] == best_dbcv].tolist()[0]
        best_results = df.iloc[idx]
        print('\nBest clustering performances:\n')
        print(f'Found {best_results["n_clusters"]} clusters with n_components={best_results["n_components"]}.\n')
        print(f'ARI: {best_results["ARI"]}\nAMI: {best_results["AMI"]}\nDBCV: {best_results["DBCV"]}\n')