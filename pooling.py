import torch
from torch import nn

class Pooling(nn.Module):

    def __init__(self, word_embedding_dim, pooling_mode):
        super().__init__()

        self.word_embedding_dim = word_embedding_dim
        self.pooling_mode = pooling_mode

        self.pooling_mode_cls_token = "cls" in pooling_mode
        self.pooling_mode_mean = "mean" in pooling_mode
        self.pooling_mode_max = "max" in pooling_mode

    def forward(self, features, attention_mask):
        token_embeddings = features["last_hidden_state"]

        output_vectors = []
        # cls token pooling
        if self.pooling_mode_cls_token:
            cls_token = features["pooler_output"]
            output_vectors.append(cls_token)

        # mean pooling: TODO
        if self.pooling_mode_mean:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.append(sum_embeddings / sum_mask)

        # max pooling: TODO
        if self.pooling_mode_max:
            output_vectors.append(torch.max(token_embeddings))

        output_vector = torch.cat(output_vectors, 1)
        features["sentence_embedding"] = output_vector
        return features