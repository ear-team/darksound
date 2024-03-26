"""
See original implementation at
https://github.com/facebookresearch/low-shot-shrink-hallucinate
Adapted from
https://github.com/sicara/easy-few-shot-learning/blob/5baddc36620b907723ea69bf850560b34b4cf1dd/easyfsl/methods/matching_networks.py 
"""

import torch
from torch import nn, Tensor
from darksound.networks.core import FewShotClassifier


def matchingnetworks_support_encoder(embedding_dimension: int) -> nn.Module:
    return nn.LSTM(
        input_size=embedding_dimension,
        hidden_size=embedding_dimension,
        num_layers=1,
        batch_first=True,
        bidirectional=True,
    )

def matchingnetworks_query_encoder(embedding_dimension: int) -> nn.Module:
    return nn.LSTMCell(embedding_dimension * 2, embedding_dimension)

class MatchingNetworks(FewShotClassifier):
    """
    Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, and Daan Wierstra.
    "Matching networks for one shot learning." (2016)
    https://arxiv.org/pdf/1606.04080.pdf
    Matching networks extract embedding vectors for both support and query images. Then they refine
    these embedding by using the context of the whole support set, using LSTMs. Finally they compute
    query labels using their cosine similarity to support images.
    Be careful: while some methods use Cross Entropy Loss for episodic training, Matching Networks
    output log-probabilities, so you'll want to use Negative Log Likelihood Loss.
    """

    def __init__(
        self,
        *args,
        support_encoder: nn.Module = None,
        query_encoder: nn.Module = None,
        **kwargs
    ):
        """
        Build Matching Networks by calling the constructor of the FewShotClassifier.
        Args:
            support_encoder: module encoding support embeddings. If none is specific, it uses
                the encoder from the original paper.
            query_encoder: module encoding query embeddings. If none is specific, it uses
                the encoder from the original paper.
        Raises:
            ValueError: if the backbone is not a embedding extractor,
            i.e. if its output for a given image is not a 1-dim tensor.
        """
        super().__init__(*args, **kwargs)

        if len(self.backbone_output_shape) != 1:
            raise ValueError(
                "Illegal backbone for Matching Networks. "
                "Expected output for an image is a 1-dim tensor."
            )

        # These modules refine support and query embedding vectors
        # using information from the whole support set
        self.support_embeddings_encoder = (
            support_encoder
            if support_encoder
            else matchingnetworks_support_encoder(self.embedding_dimension)
        )
        self.query_embeddings_encoding_cell = (
            query_encoder
            if query_encoder
            else matchingnetworks_query_encoder(self.embedding_dimension)
        )

        self.softmax = nn.Softmax(dim=1)

        # Create the fields so that the model can store
        # the computed information from one support set
        self.contextualized_support_embeddings = None
        self.one_hot_support_labels = None

    def process_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        Overrides process_support_set of FewShotClassifier.
        Extract embeddings from the support set with full context embedding.
        Store contextualized embedding vectors, as well as support labels in the one hot format.
        Args:
            support_images: images of the support set
            support_labels: labels of support set images
        """
        support_embeddings = self.backbone(support_images)
        self.contextualized_support_embeddings = self.encode_support_embeddings(support_embeddings)
        self.one_hot_support_labels = nn.functional.one_hot(support_labels).float()

    def forward(self, query_images: Tensor) -> Tensor:
        """
        Overrides method forward in the FewShotClassifier.
        Predict query labels based on their cosine similarity to support set embeddings.
        Classification scores are log-probabilities.
        Args:
            query_images: images of the query set
        Returns:
            a prediction of classification scores for query images
        """

        # Refine query embeddings using the context of the whole support set
        contextualized_query_embeddings = self.encode_query_embeddings(self.backbone(query_images))

        # Compute the matrix of cosine similarities between all query images and normalized support images
        similarity_matrix = self.softmax(contextualized_query_embeddings.mm(nn.functional.normalize(self.contextualized_support_embeddings).T))
        # Compute query log probabilities based on cosine similarity to support instances and support labels
        log_probabilities = (similarity_matrix.mm(self.one_hot_support_labels) + 1e-6).log()
        
        return self.softmax_if_specified(log_probabilities)

    def encode_support_embeddings(
        self,
        support_embeddings: Tensor,
    ) -> Tensor:
        """
        Refine support set embeddings by putting them in the context of the whole support set,
        using a bidirectional LSTM.
        Args:
            support_embeddings: output of the backbone
        Returns:
            contextualised support embeddings, with the same shape as input embeddings
        """

        # Since the LSTM is bidirectional, hidden_state is of the shape
        # [number_of_support_images, 2 * embedding_dimension]
        hidden_state = self.support_embeddings_encoder(support_embeddings.unsqueeze(0))[0].squeeze(0)

        # Following the paper, contextualized embeddings are computed by adding original embeddings, and
        # hidden state of both directions of the bidirectional LSTM.
        contextualized_support_embeddings = (
            support_embeddings
            + hidden_state[:, : self.embedding_dimension]
            + hidden_state[:, self.embedding_dimension :]
        )

        return contextualized_support_embeddings

    def encode_query_embeddings(self, query_embeddings: Tensor) -> Tensor:
        """
        Refine query set embeddings by putting them in the context of the whole support set,
        using attention over support set embeddings.
        Args:
            query_embeddings: output of the backbone
        Returns:
            contextualized query embeddings, with the same shape as input embeddings
        """

        hidden_state = query_embeddings
        cell_state = torch.zeros_like(query_embeddings)

        # Do as many iterations through the LSTM cell as there are query instances
        # Check out the paper for more details about this.
        for _ in range(len(self.contextualized_support_embeddings)):
            attention = self.softmax(
                hidden_state.mm(self.contextualized_support_embeddings.T)
            )
            read_out = attention.mm(self.contextualized_support_embeddings)
            lstm_input = torch.cat((query_embeddings, read_out), 1)

            hidden_state, cell_state = self.query_embeddings_encoding_cell(
                lstm_input, (hidden_state, cell_state)
            )
            hidden_state = hidden_state + query_embeddings

        return hidden_state

    @staticmethod
    def is_transductive() -> bool:
        return False