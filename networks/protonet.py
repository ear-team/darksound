"""
See original implementation at 
https://github.com/jakesnell/prototypical-networks
Adapted from
https://github.com/sicara/easy-few-shot-learning/blob/5baddc36620b907723ea69bf850560b34b4cf1dd/easyfsl/methods/prototypical_networks.py 
"""

import torch
from torch import Tensor

from networks.core import FewShotClassifier, compute_prototypes


class PrototypicalNetworks(FewShotClassifier):
    """
    Jake Snell, Kevin Swersky, and Richard S. Zemel.
    "Prototypical networks for few-shot learning." (2017)
    https://arxiv.org/abs/1703.05175
    Prototypical networks extract feature vectors for both support and query images. Then it
    computes the mean of support features for each class (called prototypes), and predict
    classification scores for query images based on their euclidean distance to the prototypes.
    """

    def __init__(self, *args, **kwargs):
        """
        Raises:
            ValueError: if the backbone is not a feature extractor,
            i.e. if its output for a given image is not a 1-dim tensor.
        """
        super().__init__(*args, **kwargs)

        if len(self.backbone_output_shape) != 1:
            raise ValueError(
                "Illegal backbone for Prototypical Networks. "
                "Expected output for an image is a 1-dim tensor."
            )

    def process_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        Overrides process_support_set of FewShotClassifier.
        Extract embeddings from the support set and store class prototypes.
        Args:
            support_images: images of the support set
            support_labels: labels of support set images
        """

        support_embeddings = self.backbone.forward(support_images)
        self.prototypes = compute_prototypes(support_embeddings, support_labels)
        # self.prototypes = compute_prototypes(support_images, support_labels)

    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Predict query labels based on their distance to class prototypes in the feature space.
        Classification scores are the negative of euclidean distances.
        Args:
            query_images: images of the query set
        Returns:
            a prediction of classification scores for query images
        """
        # Extract the features of support and query images
        query_embeddings = self.backbone.forward(query_images)

        # Compute the euclidean distance from queries to prototypes
        self.distances = torch.cdist(query_embeddings, self.prototypes)
        # Compute the squarred euclidean distance 
        self.distances = torch.square(self.distances)

        # Use it to compute classification scores
        scores = -self.distances

        return self.softmax_if_specified(scores)

    @staticmethod
    def is_transductive() -> bool:
        return False