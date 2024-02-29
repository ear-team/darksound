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

import os
import random
import numpy as np
import torch

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18, VGG16_Weights, vgg16, DenseNet121_Weights, densenet121, AlexNet_Weights, alexnet
from torchvision.models.feature_extraction import create_feature_extractor

from config import load_config
from trainer import trainer
from dataset import Darksound 
from sampler import TaskSampler

from networks.protonet import PrototypicalNetworks
from networks.matchnet import MatchingNetworks
from networks.relatnet import RelationNetworks

import warnings
warnings.filterwarnings("ignore")

def dataloader(train_set, params):  
    train_sampler = TaskSampler(
                                train_set, 
                                n_way=params['PARAMS_MODEL']['N_WAY'], 
                                n_shot=params['PARAMS_MODEL']['N_SHOT'], 
                                n_query=params['PARAMS_MODEL']['N_QUERY'], 
                                n_tasks=params['PARAMS_MODEL']['N_TASKS'],
                               )

    train_loader = DataLoader(
                              train_set,
                              batch_sampler=train_sampler,
                              num_workers=params['PARAMS_MODEL']['N_WORKERS'],
                              pin_memory=True,
                              collate_fn=train_sampler.episode,
                             )
    return train_loader


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
        
    # -------------------------------------------------------------------------
    # LOAD THE BACKBONE AND WEIGHTS                
    # -------------------------------------------------------------------------
    
    if params['PARAMS_MODEL']['BACKBONE'] == 'resnet18':
        if params['PARAMS_MODEL']['PRETRAINED']:
            weights = ResNet18_Weights.IMAGENET1K_V1
            model = resnet18(weights=weights).to(device)
        else:
            model = resnet18().to(device)
        model.fc = torch.nn.Flatten()
        
    if params['PARAMS_MODEL']['BACKBONE'] == 'vgg16':
        if params['PARAMS_MODEL']['PRETRAINED']:
            weights = VGG16_Weights.IMAGENET1K_V1
            model = vgg16(weights=weights).to(device)
        else:
            model = vgg16().to(device)
        # Reduce vector size and computation in the network 
        model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = torch.nn.Flatten()
        
    if params['PARAMS_MODEL']['BACKBONE'] == 'densenet':
        if params['PARAMS_MODEL']['PRETRAINED']:
            weights = DenseNet121_Weights.IMAGENET1K_V1
            model = densenet121(weights=weights).to(device)
        else:
            model = densenet121().to(device)
        model.classifier = torch.nn.Flatten()
        
    if params['PARAMS_MODEL']['BACKBONE'] == 'alexnet':
        if params['PARAMS_MODEL']['PRETRAINED']:
            weights = AlexNet_Weights.IMAGENET1K_V1
            model = alexnet(weights=weights).to(device)
        else:
            model = alexnet().to(device)
        # Reduce vector size and computation in the network
        model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = torch.nn.Flatten()
   
    # -------------------------------------------------------------------------
    # LOAD THE DARKSOUND DATASET                
    # -------------------------------------------------------------------------

    train_set = Darksound(split='train', 
                          hpss=params['PARAMS_MODEL']['HPSS'], 
                          remove_background=params['PARAMS_MODEL']['REMOVE_BG'],
                          transform=transforms.Compose([weights.transforms()]),
                          n_samples=60, # best trade-off according to Sbai et al., 2020 
                          download=True,
                         )
    
    # -------------------------------------------------------------------------
    # TRAIN THE META-LEARNING MODEL                
    # -------------------------------------------------------------------------
    
    # Load the meta-learning algorithm
    if params['PARAMS_MODEL']['NETWORK'] == 'matching':
        few_shot_classifier = MatchingNetworks(model, use_softmax=True).to(device)
    if params['PARAMS_MODEL']['NETWORK'] == 'prototypical':
        few_shot_classifier = PrototypicalNetworks(model, use_softmax=True).to(device)
    if params['PARAMS_MODEL']['NETWORK'] == 'relation':
        # Get the last feature map of the backbone
        if params['PARAMS_MODEL']['BACKBONE'] == 'resnet18':
            model = create_feature_extractor(model, return_nodes=['layer4.1.bn2']) 
        if params['PARAMS_MODEL']['BACKBONE'] == 'vgg16':
            model = create_feature_extractor(model, return_nodes=['features.30'])
        if params['PARAMS_MODEL']['BACKBONE'] == 'densenet':
            model = create_feature_extractor(model, return_nodes=['features.norm5'])
        if params['PARAMS_MODEL']['BACKBONE'] == 'alexnet':
            model = create_feature_extractor(model, return_nodes=['features.12'])  
        few_shot_classifier = RelationNetworks(model, use_softmax=False).to(device)
        
#     # Freeze the layers of the backbone except the last one
#     for param in few_shot_classifier.backbone.parameters():
#         param.requires_grad = False
#     for param in few_shot_classifier.backbone.layer4.parameters():
#         param.requires_grad = True
        
    # -------------------------------------------------------------------------
    # LOAD THE OPTIMIZER AND LOSS FUNCTION                
    # -------------------------------------------------------------------------
    
    optimizer = Adam(few_shot_classifier.parameters(), lr=params['PARAMS_MODEL']['LEARNING_RATE'])

    if params['PARAMS_MODEL']['NETWORK'] == 'relation':
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(params['PARAMS_MODEL']['N_EPOCHS']):
        print(f"Epoch {epoch}")
        train_loader = dataloader(train_set, params) # load different train sampler for each epoch
        training_loss, training_accuracy = trainer(few_shot_classifier, train_loader, optimizer, criterion, train=True, device=device)

    # -------------------------------------------------------------------------
    # SAVE THE MODEL                
    # -------------------------------------------------------------------------
    
    if params['PARAMS_MODEL']['SAVE']:
        torch.save(few_shot_classifier.state_dict(), 
                   os.path.join(params['PARAMS_MODEL']['OUTPUT_DIR'], f"{params['PARAMS_MODEL']['NETWORK']}-networks-{str(params['PARAMS_MODEL']['N_WAY'])}way-{str(params['PARAMS_MODEL']['N_SHOT'])}shot-{str(params['PARAMS_MODEL']['BACKBONE'])}.pt"))