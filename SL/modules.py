import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from flash_attn import flash_attn_qkvpacked_func

# ========== Positional Encoding ==========

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        # the positional encoding implementation from attention is all you need (with dropout)

        super().__init__()
        self.dropout = dropout

        # unsqueeze adds a dimension at the specified index, [max_len] -> [max_len, 1], 0-indexing
        # the first argument here is the end (length) but below its the second argument, this is done through type checking in the function
        position = torch.arange(max_len).unsqueeze(1)

        # The div_term controls the frequencies of the sine/cosine functions and ensures that each dimension in the positional encoding vector varies at a different rate
        # if the div term is higher the frequency is higher, since a change in the position goes farther along the function
        # arange is non inclusive on the end btw (like range in python)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        # all even get sin, right side of expression makes [1, d_model // 2] matrix which is broadcasted to [1, max_len, d_model // 2]
        pe[0, :, 0::2] = torch.sin(position * div_term)
        # all uneven get cos
        pe[0, :, 1::2] = torch.cos(position * div_term)
        # saves the tensor with the module (useful when doing .to(device)), but not as trainable parameters
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1), :] # add positional encoding up to sequence length
        return F.dropout(x, p=self.dropout)
    
# ========== Attention ==========

class attention_block(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, bias = True, dropout=0.0):
        super().__init__()

        assert d_model % num_heads == 0

        self.qkv_projection = nn.Linear(d_model, d_model * 3, bias=bias)
        self.final_projection = nn.Linear(d_model, d_model, bias=bias)

        torch.nn.init.normal_(self.qkv_projection.weight, mean=0.0, std=0.02)

        # scale weights as in gpt2 paper
        # you cant divide weigths directly so we do .data
        torch.nn.init.normal_(self.final_projection.weight, mean=0.0, std=0.02/math.sqrt(2 * num_layers))

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_heads
        self.bias = bias
        self.dropout = dropout
        pass

    def forward(self, token_embeddings):
        # input shape (batch, seq_len, d_model)
        B, S, D = token_embeddings.size()
        # project token_embeddings into q,k,v
        # resulting shape: (batch, seq_len, d_model * 3)
        projection = self.qkv_projection(token_embeddings)
        # cut into q,k and v matrix
        q_matrix = projection[:,:,0:self.d_model]
        k_matrix = projection[:,:,self.d_model:2*self.d_model]
        v_matrix = projection[:,:,2*self.d_model:]
        # cut them up for multi-head (transform [batch, seq_len, d_model] into [batch, seq_len, num_heads, head_dim])
        q_matrix = q_matrix.reshape(B, S, self.num_heads, self.head_dim)
        k_matrix = k_matrix.reshape(B, S, self.num_heads, self.head_dim)
        v_matrix = v_matrix.reshape(B, S, self.num_heads, self.head_dim)
        # permute [batch, seq_len, num_heads, head_dim] into [batch, num_heads, seq_len, head_dim]
        q_matrix = q_matrix.permute(0,2,1,3)
        k_matrix = k_matrix.permute(0,2,1,3)
        v_matrix = v_matrix.permute(0,2,1,3)
        # mask 
        mask = torch.zeros(S, S).fill_(-torch.inf).triu(1)
        mask = mask.unsqueeze(0).unsqueeze(0)
        # compute softmax(q @ k^T / square of dim) @ v (softmax along the sequence dimension)
        # q @ k^T creates matrix (seq_len, seq_len) where each row shows the "compatibility" of each token to every other token
        # this is scaled so that softmax is softer and less argmax
        # mask the attention matrix, its quite wasteful to compute the whole matrix and then throw half of it away - TODO fix that
        attention_matrix = ((q_matrix @ k_matrix.transpose(-2, -1)) / (math.sqrt(self.head_dim))) + mask
        # apply softmax so that each row sums to one
        attention_matrix = torch.softmax(attention_matrix, dim=3)
        # basically weighted sum of each embedding dimension (in v) according to the attention matrix ("compatibility")
        embedding_matrix = attention_matrix @ v_matrix
        # concat into single embedding again (transform [batch, num_heads, seq_len, head_dim] into [batch, seq_len, num_heads, head_dim]) then into [batch, seq_len, d_model])
        embedding_matrix = embedding_matrix.permute(0,2,1,3)
        embedding_matrix = embedding_matrix.reshape(B, S, -1)
        # learned linear projection
        embedding_matrix = self.final_projection(embedding_matrix)

        return embedding_matrix


class flash_attention_block(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, bias = True, dropout=0.0):
        super().__init__()

        assert d_model % num_heads == 0

        self.qkv_projection = nn.Linear(d_model, d_model * 3, bias=bias)
        self.final_projection = nn.Linear(d_model, d_model, bias=bias)

        torch.nn.init.normal_(self.qkv_projection.weight, mean=0.0, std=0.02)

        # scale weights as in gpt2 paper
        # you cant divide weigths directly so we do .data
        torch.nn.init.normal_(self.final_projection.weight, mean=0.0, std=0.02/math.sqrt(2 * num_layers))

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_heads
        self.dropout = dropout

        pass

    def forward(self, token_embeddings):
        # input shape (batch, seq_len, d_model)
        batch_size = token_embeddings.size(0)
        S = token_embeddings.size(1)
        # project token_embeddings into q,k,v
        # resulting shape: (batch, seq_len, d_model * 3)
        projection = self.qkv_projection(token_embeddings)
        projection = projection.reshape(batch_size, S, 3, self.d_model)
        projection = projection.reshape(
            batch_size, S, 3, self.num_heads, self.head_dim)

        # at short sequence lengths (<1k), this isnt actually much faster
        embedding_matrix = flash_attn_qkvpacked_func(
            projection, self.dropout, causal=True)

        embedding_matrix = embedding_matrix.reshape(batch_size, S, -1)
        # learned linear projection
        embedding_matrix = self.final_projection(embedding_matrix)

        return embedding_matrix


class mlp_block(nn.Module):
    def __init__(self, d_model, ff_d, bias, num_layers):
        super().__init__()
        # Linear doesnt takes in size of parameter matrix but in_features and out_features
        # in a standard matrix multiplication this is the same and in the call it also works out to be the same
        # but in the computation the weight matrix is actually transposed before computation and so it actually has dimension (out_features, in_features)
        self.l1 = nn.Linear(d_model, ff_d, bias=bias)
        self.l2 = nn.Linear(ff_d, d_model, bias=bias)

        torch.nn.init.normal_(self.l1.weight, mean=0.0, std=0.02)

        if(bias):
            torch.nn.init.zeros_(self.l1.bias)
            torch.nn.init.zeros_(self.l2.bias)

        # scale weights as in gpt2 paper, decreases variance
        # you cant divide weigths directly so we do .data
        torch.nn.init.normal_(self.l2.weight, mean=0.0, std=0.02/math.sqrt(2 * num_layers))

        pass

    def forward(self, token_embeddings):
        # token embeddings have shape batch, seq_len, d_model
        # send tokens through same mlp individually
        # this is default in pytorch, because thats just how matrix multiplications work (and also because of how batches are applied)
        # matrix multiplications do not mix information between rows (of the first matrix) anyway
        token_embeddings = F.gelu(self.l1(token_embeddings))
        token_embeddings = self.l2(token_embeddings)
        return token_embeddings


class transformer(nn.Module):
    def __init__(self, c):
        super().__init__()

        self.c = c

        # the embedding layer is just a lookup table
        # if input was one hot of shape [batch_size, seq_len, vocab_size]
        # you could just matrix multiply with [vocab_size, d_model]
        # but loss didnt really work there so idk
        # mathematically equivalent to one hot (without bias), gradients and stuff
        self.embedding_matrix = nn.Embedding(self.c["vocab_size"], self.c["d_model"])
        torch.nn.init.normal_(self.embedding_matrix.weight, mean=0.0, std=0.02)
        # gpt2 uses learned positional encoding
        self.pos_encoding = nn.Embedding(self.c["seq_len"], self.c["d_model"])
        torch.nn.init.normal_(self.pos_encoding.weight, mean=0.0, std=0.02)
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(self.c["d_model"], bias=self.c["bias"]) for _ in range((self.c["num_layers"] * 2) + 1)])

        self.attention_blocks = nn.ModuleList(
            [flash_attention_block(self.c["d_model"], self.c["num_heads"], self.c["num_layers"], 
                                   self.c["bias"], self.c["dropout"]) for _ in range(self.c["num_layers"])])
        self.mlp_blocks = nn.ModuleList(
            [mlp_block(self.c["d_model"], self.c["ff_d"], self.c["bias"], self.c["num_layers"]) for _ in range(self.c["num_layers"])])

        # weigth tying
        self.unembedding_matrix = nn.Linear(self.c["d_model"], self.c["vocab_size"], bias=False)
        # works because linear layer weigt matrices are used transposed, embedding matrix layers are not
        self.unembedding_matrix.weight = self.embedding_matrix.weight
        pass

    def forward(self, token_matrix):
        """
        Arguments:
            token_matrix - matrix of size batch, seq_len, vocab_size
        """
        S = token_matrix.size(1)

        pos = torch.arange(0, S, dtype=torch.long, device="cuda") # shape (t)

        # results in dimension batch, seq_len, d_model
        # scaled because apparently we do that
        token_embeddings = self.embedding_matrix(token_matrix)
        # add positional embedding + dropout
        positional = self.pos_encoding(pos)
        token_embeddings = F.dropout(token_embeddings + positional, p=self.c["dropout"])

        for layer in range(self.c["num_layers"]):
            # (layer) normalization
            # per token, over its features
            # embedding is normalized first (to mean 0, std 1)
            # then learned parameters scale each feature up again and add a bias (independent parameters for each feature)
            # stabilizes training

            # (multi-head) attention + residual stream
            # this is pre norm (because you norm before the block)
            # in post norm, you put in block, add residual, then normalize
            residual = token_embeddings
            token_embeddings = self.layer_norms[(layer * 2)](token_embeddings)
            # flash attention has attention dropout
            token_embeddings = self.attention_blocks[layer](token_embeddings)
            token_embeddings = F.dropout(token_embeddings, p=self.c["dropout"])
            token_embeddings = token_embeddings + residual

            # mlp + residual stream
            residual = token_embeddings
            token_embeddings = self.layer_norms[(
                layer * 2) + 1](token_embeddings)
            token_embeddings = self.mlp_blocks[layer](token_embeddings)
            token_embeddings = F.dropout(token_embeddings, p=self.c["dropout"])
            token_embeddings = token_embeddings + residual

        # (layer) normalization
        token_embeddings = self.layer_norms[-1](token_embeddings)
        # linear projection of the final embedding of the last token to vocab_size
        # liefert matrix of shape (batch_size, seq_len, vocab_size)
        # because you are matrix multiplying with the transpose, each embedding at the end is measured in similarity to every token
        # then one number for each represents their "similarity"
        # like a dot product
        # transpose -2, -1 and -1, -2 is the same btw
        logits = self.unembedding_matrix(token_embeddings)

        # no softmax applied yet
        return logits
