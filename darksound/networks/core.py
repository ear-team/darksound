"""
Adapted from
https://github.com/sicara/easy-few-shot-learning/blob/5baddc36620b907723ea69bf850560b34b4cf1dd/easyfsl/methods/few_shot_classifier.py 
"""

import copy
from abc import abstractmethod
from typing import Tuple

import torch
from torch import nn, Tensor

class FewShotClassifier(nn.Module):
    """
    Abstract class providing methods usable by all few-shot classification algorithms
    """

    def __init__(self, backbone: nn.Module, use_softmax: bool = True):
        """
        Initialize the Few-Shot Classifier
        Args:
            backbone: the embedding extractor used by the method. Must output a tensor of the
                appropriate shape (depending on the method)
            use_softmax: whether to return predictions as soft probabilities
        """
        super().__init__()

        self.backbone = backbone
        self.backbone_output_shape = compute_backbone_output_shape(backbone)
        self.embedding_dimension = self.backbone_output_shape[0]

        self.use_softmax = use_softmax

        self.prototypes = None
        self.support_embeddings = None
        self.support_labels = None

    @abstractmethod
    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        Predict classification labels.
        Args:
            query_images: images of the query set
        Returns:
            a prediction of classification scores for query images
        """
        raise NotImplementedError("All few-shot algorithms must implement a forward method.")

    @abstractmethod
    def process_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        Harness information from the support set, so that query labels can later be predicted using
        a forward call
        Args:
            support_images: images of the support set
            support_labels: labels of support set images
        """
        raise NotImplementedError("All few-shot algorithms must implement a process_support_set method.")

    @staticmethod
    def is_transductive() -> bool:
        raise NotImplementedError("All few-shot algorithms must implement a is_transductive method.")

    def softmax_if_specified(self, output: Tensor) -> Tensor:
        """
        If the option is chosen when the classifier is initialized, we perform a softmax on the
        output in order to return soft probabilities.
        Args:
            output: output of the forward method
        Returns:
            output as it was, or output as soft probabilities
        """
        return output.log_softmax(-1) if self.use_softmax else output

    def l2_distance(self, query_samples: Tensor) -> Tensor:
        """
        Compute prediction logits from their euclidean distance to support set prototypes.
        Args:
            query_samples: embeddings of the items to classify
        Returns:
            prediction logits
        """
        return torch.cdist(query_samples, self.prototypes)

    def cosine_distance(self, query_samples) -> Tensor:
        """
        Compute prediction logits from their cosine distance to support set prototypes.
        Args:
            query_samples: embeddings of the items to classify
        Returns:
            prediction logits
        """
        return (nn.functional.normalize(query_samples, dim=1) @ nn.functional.normalize(self.prototypes, dim=1).T)

    def store_support_set_data(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        Extract support embeddings, compute prototypes,
            and store support labels, embeddings, and prototypes
        Args:
            support_images: images of the support set
            support_labels: labels of support set images
        """
        self.support_labels = support_labels
        self.support_embeddings = self.backbone(support_images)
        self.prototypes = compute_prototypes(self.support_embeddings, support_labels)
        
def compute_backbone_output_shape(backbone: nn.Module) -> Tuple[int]:
    """
    Compute the dimension of the embedding space defined by a embedding extractor.
    Args:
        backbone: embedding extractor
    Returns:
        shape of the embedding vector computed by the embedding extractor for an instance
    """
    input_images = torch.ones((4, 3, 224, 224))
    # whether the backbone has a fully connected or classifier layer or not for relation networks
    try: 
        backbone.fc
        # Use a copy of the backbone on CPU, to avoid device conflict
        output = copy.deepcopy(backbone).cpu()(input_images)
    except:
        try:
            backbone.classifier
            output = copy.deepcopy(backbone).cpu()(input_images)
        except:
            output = list(copy.deepcopy(backbone).cpu()(input_images).values())[0]
#             output = copy.deepcopy(backbone).cpu()(input_images)
        
    return tuple(output.shape[1:])

def compute_prototypes(support_embeddings: Tensor, support_labels: Tensor) -> Tensor:
    """
    Compute class prototypes from support embeddings and labels
    Args:
        support_embeddings: for each instance in the support set, its embedding vector
        support_labels: for each instance in the support set, its label
    Returns:
        For each label of the support set, the average embedding vector of instances with this label
    """
    n_way = len(torch.unique(support_labels))
    # Prototype i is the mean of all instances of embeddings corresponding to labels == i
    prototype = [support_embeddings[torch.nonzero(support_labels == i)].mean(0) for i in range(n_way)]
    return torch.cat(prototype)