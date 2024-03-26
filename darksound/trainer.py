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

import numpy as np
from tqdm import tqdm
from sklearn import metrics
from typing import Optional, Callable
from codecarbon import track_emissions

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from darksound.networks.core import FewShotClassifier

def compute_score(model, images, labels, loss_fn, optimizer, device):
    # Zero the gradient at each iteration
    optimizer.zero_grad()
    # Get the correct prediction scores
    scores = model(images.to(device))
    correct = (torch.argmax(scores.detach().data, 1) == labels.to(device)).sum().item()

    if type(loss_fn) == torch.nn.modules.loss.MSELoss or type(loss_fn) == torch.nn.modules.loss.KLDivLoss:
        # Compute one-hot encoded vector 
        one_hot = torch.nn.functional.one_hot(labels.to(device))
        loss = loss_fn(scores.to(torch.float64), one_hot.to(torch.float64))
    else:
        loss = loss_fn(scores, labels.to(device))

    # Backward propagation for calculating gradients
    loss.backward()
    # Update the weights
    optimizer.step()
    return loss, correct

@track_emissions(project_name="darksound", log_level="error") # track the carbon emissions of the algorithms
def trainer(model: FewShotClassifier,
    data_loader: DataLoader,
    optimizer: Optimizer = None,
    loss_fn: Optional[Callable] = None,
    train: bool = True,
    verbose: bool = False,
    device: str = 'cpu'):
    
    # -------------------------------------------------------------------------
    # TRAINING                
    # -------------------------------------------------------------------------
    
    if train:
        train_loss = []
        train_accuracy = []
        total_predictions = 0
        correct_predictions = 0

        model.train()
        with tqdm(enumerate(data_loader), total=len(data_loader), disable=not True, desc="Episodic Training") as tqdm_train:
            for i, (support_images, support_labels, query_images, query_labels, _) in tqdm_train:
                model.process_support_set(support_images.to(device), support_labels.to(device))
                # Compute score and correct classifications
                loss, correct = compute_score(model, query_images, query_labels, loss_fn, optimizer, device)
                # Append accuracy and loss to lists
                total_predictions += len(query_labels)
                correct_predictions += correct
                train_accuracy.append(correct_predictions / total_predictions)
                train_loss.append(loss.item())
                # Log loss in real time
                tqdm_train.set_postfix(loss=np.mean(train_loss), acc=np.mean(train_accuracy))
        return np.mean(train_loss), np.mean(train_accuracy)
    
    # -------------------------------------------------------------------------
    # EVALUATING                
    # -------------------------------------------------------------------------
    
    else: 
        labels = []
        predictions = []
        test_loss = []
        test_accuracy = []
        total_predictions = 0
        correct_predictions = 0

        model.eval()
        with tqdm(enumerate(data_loader), total=len(data_loader), disable=not True, desc="Evaluating") as tqdm_eval:
            for i, (support_images, support_labels, query_images, query_labels, _) in tqdm_eval:
                model.process_support_set(support_images.to(device), support_labels.to(device)) 
                scores = model(query_images.to(device)).detach()
                # Compute loss
                if type(loss_fn) == torch.nn.modules.loss.MSELoss:
                    # Compute one-hot encoded vector for calculating MSE loss
                    one_hot = torch.nn.functional.one_hot(query_labels.to(device))
                    loss = loss_fn(scores.to(torch.float64), one_hot.to(torch.float64))
                else:
                    loss = loss_fn(scores, query_labels.to(device))
                    
                correct = (torch.max(scores.detach().data, 1)[1] == query_labels.to(device)).sum().item()
                # Get the predicted labels
                predicted_labels = torch.max(scores.data, 1)[1]
                labels += query_labels.tolist()
                predictions += predicted_labels.tolist()
                total_predictions += len(query_labels)
                correct_predictions += correct
                test_accuracy.append(correct_predictions / total_predictions)
                test_loss.append(loss.item())
                # Log accuracy in real time
                tqdm_eval.set_postfix(acc=correct_predictions / total_predictions)
        if verbose:
            performance = metrics.classification_report(labels, predictions, digits=3, output_dict=True)
            return performance, np.mean(test_accuracy)
        return np.mean(test_loss), np.mean(test_accuracy)