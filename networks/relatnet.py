"""
See original implementation at
https://github.com/floodsung/LearningToCompare_FSL

Adapted from
https://github.com/sicara/easy-few-shot-learning/blob/5baddc36620b907723ea69bf850560b34b4cf1dd/easyfsl/methods/prototypical_networks.py 
"""

import torch
from torch import nn, Tensor
from typing import Optional

from networks.core import FewShotClassifier, compute_prototypes


class RelationNetworks(FewShotClassifier):
    """
    Sung, Flood, Yongxin Yang, Li Zhang, Tao Xiang, Philip HS Torr, and Timothy M. Hospedales.
    "Learning to compare: Relation network for few-shot learning." (2018)
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Sung_Learning_to_Compare_CVPR_2018_paper.pdf

    In the Relation Networks algorithm, we first extract feature maps for both support and query
    images. Then we compute the mean of support features for each class (called prototypes).
    To predict the label of a query image, its feature map is concatenated with each class prototype
    and fed into a relation module, i.e. a CNN that outputs a relation score. Finally, the
    classification vector of the query is its relation score to each class prototype.

    Note that for most other few-shot algorithms we talk about feature vectors, because for each
    input image, the backbone outputs a 1-dim feature vector. Here we talk about feature maps,
    because for each input image, the backbone outputs a "feature map" of shape
    (n_channels, width, height). This raises different constraints on the architecture of the
    backbone: while other algorithms require a "flatten" operation in the backbone, here "flatten"
    operations are forbidden.

    Relation Networks use Mean Square Error. This is unusual because this is a classification
    problem. The authors justify this choice by the fact that the output of the model is a relation
    score, which makes it a regression problem. See the article for more details.
    """

    def __init__(self, *args, relation_module: Optional[nn.Module] = None, **kwargs):
        """
        Build Relation Networks by calling the constructor of FewShotClassifier.
        Args:
            relation_module: module that will take the concatenation of a query features vector
                and a prototype to output a relation score. If none is specific, we use the default
                relation module from the original paper.

        Raises:
            ValueError: if the backbone doesn't outputs feature maps, i.e. if its output for a
            given image is not a tensor of shape (n_channels, width, height)
        """
        super().__init__(*args, **kwargs)

        if len(self.backbone_output_shape) != 3:
            raise ValueError(
                "Illegal backbone for Relation Networks. Expected output for an image is a 3-dim "
                "tensor of shape (n_channels, width, height)."
            )

        # Here we build the relation module that will output the relation score for each
        # (query, prototype) pair. See the function docstring for more details.
        self.relation_module = (relation_module if relation_module else self.default_relation_module())

    def default_relation_module(self, inner_channels: int = 8):
        """
        Build the relation module that takes as input the concatenation of two feature maps, from
        Sung et al. : "Learning to compare: Relation network for few-shot learning." (2018)
        In order to make the network robust to any change in the dimensions of the input images,
        we made some changes to the architecture defined in the original implementation
        from Sung et al.(typically the use of adaptive pooling).
        Args:
            embedding_dimension: the dimension of the feature space i.e. size of a feature vector
            inner_channels: number of hidden channels between the linear layers of the relation module
        Returns:
            the constructed relation module
        """
        return nn.Sequential(
            nn.Sequential(
                nn.Conv2d(
                    self.embedding_dimension * 2,
                    self.embedding_dimension,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(self.embedding_dimension, momentum=1, affine=True),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d((5, 5)),
            ),
            nn.Sequential(
                nn.Conv2d(
                    self.embedding_dimension,
                    self.embedding_dimension,
                    kernel_size=3,
                    padding=0,
                ),
                nn.BatchNorm2d(self.embedding_dimension, momentum=1, affine=True),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d((1, 1)),
            ),
            nn.Flatten(),
            nn.Linear(self.embedding_dimension, inner_channels),
            nn.ReLU(),
            nn.Linear(inner_channels, 1),
            nn.Sigmoid(),
        )

    def process_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        Overrides process_support_set of FewShotClassifier.
        Extract feature maps from the support set and store class prototypes.

        Args:
            support_images: images of the support set
            support_labels: labels of support set images
        """

        support_features = list(self.backbone.forward(support_images).values())[0]
#         support_features = self.backbone.forward(support_images)
        self.prototypes = compute_prototypes(support_features, support_labels)

    def forward(self, query_images: Tensor) -> Tensor:
        """
        Overrides method forward in FewShotClassifier.
        Predict the label of a query image by concatenating its feature map with each class
        prototype and feeding the result into a relation module, i.e. a CNN that outputs a relation
        score. Finally, the classification vector of the query is its relation score to each class
        prototype.

        Args:
            query_images: images of the query set
        Returns:
            a prediction of classification scores for query images
        """
        query_features = list(self.backbone.forward(query_images).values())[0]
#         query_features = self.backbone.forward(query_images)

        # For each pair (query, prototype), we compute the concatenation of their feature maps
        # Given that query_features is of shape (n_queries, n_channels, width, height), the
        # constructed tensor is of shape (n_queries * n_prototypes, 2 * n_channels, width, height)
        # (2 * n_channels because prototypes and queries are concatenated)
     
        query_prototype_feature_pairs = torch.cat(
            (
                self.prototypes.unsqueeze(dim=0).expand(
                    query_features.shape[0], -1, -1, -1, -1
                ),
                query_features.unsqueeze(dim=1).expand(
                    -1, self.prototypes.shape[0], -1, -1, -1
                ),
            ),
            dim=2,
        ).view(-1, 2 * self.embedding_dimension, *query_features.shape[2:])

        # Each pair (query, prototype) is assigned a relation scores in [0,1]. 
        # Then we reshape the tensor so that relation_scores is of shape (n_queries, n_prototypes).
        relation_scores = self.relation_module(query_prototype_feature_pairs).view(-1, self.prototypes.shape[0])

        return self.softmax_if_specified(relation_scores)

    @staticmethod
    def is_transductive() -> bool:
        return False
