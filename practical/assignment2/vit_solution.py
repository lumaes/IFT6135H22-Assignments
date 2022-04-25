import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def forward(self, inputs):
        """Layer Normalization.

        This module applies Layer Normalization, with rescaling and shift,
        only on the last dimension. See Lecture 07 (I), slide 23.

        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(*dims, hidden_size)`)
            The input tensor. This tensor can have an arbitrary number N of
            dimensions, as long as `inputs.shape[N-1] == hidden_size`. The
            leading N - 1 dimensions `dims` can be arbitrary.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(*dims, hidden_size)`)
            The output tensor, having the same shape as `inputs`.
        """

        mean_inputs = (inputs.sum(dim=-1) * (1/self.hidden_size)).unsqueeze(-1)
        var_inputs = ((inputs - mean_inputs).pow(2).sum(dim=-1) * (1/self.hidden_size)).unsqueeze(-1)
        layer_inputs = ((inputs - mean_inputs) / (torch.sqrt(var_inputs) + self.eps)) * self.weight + self.bias
        
        return layer_inputs

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_size, num_heads, sequence_length):
        super(MultiHeadedAttention, self).__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.sequence_length = sequence_length
        #self.W_Q = nn.Parameter(torch.Tensor(self.num_heads*self.head_size, self.num_heads*self.head_size))
        #self.W_K = nn.Parameter(torch.Tensor(self.num_heads*self.head_size, self.num_heads*self.head_size))
        #self.W_V = nn.Parameter(torch.Tensor(self.num_heads*self.head_size, self.num_heads*self.head_size))
        #self.W_Y = nn.Parameter(torch.Tensor(self.num_heads*self.head_size, self.num_heads*self.head_size))
        #self.b_Q = nn.Parameter(torch.Tensor(self.num_heads*self.head_size))
        #self.b_K = nn.Parameter(torch.Tensor(self.num_heads*self.head_size))
        #self.b_V = nn.Parameter(torch.Tensor(self.num_heads*self.head_size))
        #self.b_Y = nn.Parameter(torch.Tensor(self.num_heads*self.head_size))

        self.W_Q = nn.Linear(self.num_heads*self.head_size, self.num_heads*self.head_size, bias=True)
        self.W_K = nn.Linear(self.num_heads*self.head_size, self.num_heads*self.head_size, bias=True)
        self.W_V = nn.Linear(self.num_heads*self.head_size, self.num_heads*self.head_size, bias=True)
        self.W_Y = nn.Linear(self.num_heads*self.head_size, self.num_heads*self.head_size, bias=True)
        #self.b_Q = nn.Linear(torch.Tensor(self.num_heads*self.head_size))
        #self.b_K = nn.Linear(torch.Tensor(self.num_heads*self.head_size))
        #self.b_V = nn.Linear(torch.Tensor(self.num_heads*self.head_size))
        #self.b_Y = nn.Linear(torch.Tensor(self.num_heads*self.head_size))

        #nn.init.xavier_uniform_(self.W_Q)
        #nn.init.xavier_uniform_(self.W_K)
        #nn.init.xavier_uniform_(self.W_V)
        #nn.init.xavier_uniform_(self.W_Y)

        #nn.init.zeros_(self.b_Q)
        #nn.init.zeros_(self.b_K)
        #nn.init.zeros_(self.b_V)
        #nn.init.zeros_(self.b_Y)

    def get_attention_weights(self, queries, keys):
        """Compute the attention weights.

        This computes the attention weights for all the sequences and all the
        heads in the batch. For a single sequence and a single head (for
        simplicity), if Q are the queries (matrix of size `(sequence_length, head_size)`),
        and K are the keys (matrix of size `(sequence_length, head_size)`), then
        the attention weights are computed as

            weights = softmax(Q * K^{T} / sqrt(head_size))

        Here "*" is the matrix multiplication. See Lecture 06, slides 19-24.

        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the queries for all the positions in the sequences
            and all the heads.

        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the keys for all the positions in the sequences
            and all the heads. 

        Returns
        -------
        attention_weights (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, sequence_length)`)
            Tensor containing the attention weights for all the heads and all
            the sequences in the batch.
        """
        return torch.softmax(queries @ keys.transpose(-1,-2) / torch.sqrt(torch.tensor(self.head_size).float()), dim=-1)

    def apply_attention(self, queries, keys, values):
        """Apply the attention.

        This computes the output of the attention, for all the sequences and
        all the heads in the batch. For a single sequence and a single head
        (for simplicity), if Q are the queries (matrix of size `(sequence_length, head_size)`),
        K are the keys (matrix of size `(sequence_length, head_size)`), and V are
        the values (matrix of size `(sequence_length, head_size)`), then the ouput
        of the attention is given by

            weights = softmax(Q * K^{T} / sqrt(head_size))
            attended_values = weights * V
            outputs = concat(attended_values)

        Here "*" is the matrix multiplication, and "concat" is the operation
        that concatenates the attended values of all the heads (see the
        `merge_heads` function). See Lecture 06, slides 19-24.

        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the queries for all the positions in the sequences
            and all the heads. 

        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the keys for all the positions in the sequences
            and all the heads. 

        values (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the values for all the positions in the sequences
            and all the heads. 
        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Tensor containing the concatenated outputs of the attention for all
            the sequences in the batch, and all positions in each sequence. 
        """
        weights = self.get_attention_weights(queries, keys)
        attended_values = weights @ values
        outputs = self.merge_heads(attended_values)
        return outputs

    def split_heads(self, tensor):
        """Split the head vectors.

        This function splits the head vectors that have been concatenated (e.g.
        through the `merge_heads` function) into a separate dimension. This
        function also transposes the `sequence_length` and `num_heads` axes.
        It only reshapes and transposes the input tensor, and it does not
        apply any further transformation to the tensor. The function `split_heads`
        is the inverse of the function `merge_heads`.

        Parameters
        ----------
        tensor (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * dim)`)
            Input tensor containing the concatenated head vectors (each having
            a size `dim`, which can be arbitrary).

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, dim)`)
            Reshaped and transposed tensor containing the separated head
            vectors. Here `dim` is the same dimension as the one in the
            definition of the input `tensor` above.
        """
        return tensor.reshape((tensor.shape[0], tensor.shape[1], self.num_heads, -1)).transpose(1, 2)

    def merge_heads(self, tensor):
        """Merge the head vectors.

        This function concatenates the head vectors in a single vector. This
        function also transposes the `sequence_length` and the newly created
        "merged" dimension. It only reshapes and transposes the input tensor,
        and it does not apply any further transformation to the tensor. The
        function `merge_heads` is the inverse of the function `split_heads`.

        Parameters
        ----------
        tensor (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, dim)`)
            Input tensor containing the separated head vectors (each having
            a size `dim`, which can be arbitrary).

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * dim)`)
            Reshaped and transposed tensor containing the concatenated head
            vectors. Here `dim` is the same dimension as the one in the
            definition of the input `tensor` above.
        """
        tmp = tensor.transpose(1,2)
        return tmp.reshape((tmp.shape[0], tmp.shape[1], -1))

    def forward(self, hidden_states):
        """Multi-headed attention.

        This applies the multi-headed attention on the input tensors `hidden_states`.
        For a single sequence (for simplicity), if X are the hidden states from
        the previous layer (a matrix of size `(sequence_length, num_heads * head_size)`
        containing the concatenated head vectors), then the output of multi-headed
        attention is given by

            Q = X * W_{Q} + b_{Q}        # Queries
            K = X * W_{K} + b_{K}        # Keys
            V = X * W_{V} + b_{V}        # Values

            Y = attention(Q, K, V)       # Attended values (concatenated for all heads)
            outputs = Y * W_{Y} + b_{Y}  # Linear projection

        Here "*" is the matrix multiplication.

        Parameters
        ----------
        hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Input tensor containing the concatenated head vectors for all the
            sequences in the batch, and all positions in each sequence. This
            is, for example, the tensor returned by the previous layer.

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Tensor containing the output of multi-headed attention for all the
            sequences in the batch, and all positions in each sequence.
        """

        #Q = self.split_heads(hidden_states @ self.W_Q + self.b_Q)
        #K = self.split_heads(hidden_states @ self.W_K + self.b_K)
        #V = self.split_heads(hidden_states @ self.W_V + self.b_V)
        Q = self.split_heads(self.W_Q(hidden_states))
        K = self.split_heads(self.W_K(hidden_states))
        V = self.split_heads(self.W_V(hidden_states))
        Y = self.apply_attention(Q, K, V)
        #outputs = Y @ self.W_Y + self.b_Y
        outputs = self.W_Y(Y)
        return outputs

class PostNormAttentionBlock(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, num_heads,sequence_length, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network 
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        
        
        self.layer_norm_1 = LayerNorm(embed_dim)
        self.attn = MultiHeadedAttention(embed_dim//num_heads, num_heads,sequence_length)
        self.layer_norm_2 = LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        
    def forward(self, x):
       
        attention_outputs = self.attn(x)
        attention_outputs = self.layer_norm_1(x + attention_outputs)
        outputs = self.linear(attention_outputs)
        outputs = self.layer_norm_2(outputs+attention_outputs)
        return outputs

class PreNormAttentionBlock(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, num_heads,sequence_length, dropout=0.0):
        """A decoder layer.

        This module combines a Multi-headed Attention module and an MLP to
        create a layer of the transformer, with normalization and skip-connections.
        See Lecture 06, slide 33.

        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network 
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            sequence_length - Length of the sequence
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        
        self.layer_norm_1 = LayerNorm(embed_dim)
        self.attn = MultiHeadedAttention(embed_dim//num_heads, num_heads,sequence_length)
        self.layer_norm_2 = LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        
    def forward(self, x):

        layer_norm_1 = self.layer_norm_1(x)
        attention_features = self.attn(layer_norm_1)
        skip_connections_1 = x + attention_features

        layer_norm_2 = self.layer_norm_2(skip_connections_1)
        linear_ouputs = self.linear(layer_norm_2)
        skip_connections_2 = skip_connections_1 + linear_ouputs

        outputs = skip_connections_2

        return outputs



class VisionTransformer(nn.Module):
    
    def __init__(self, embed_dim=256, hidden_dim=512, num_channels=3, num_heads=8, num_layers=4, num_classes=10, patch_size=4, num_patches=64,block='prenorm', dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            block - Type of attention block
            dropout - Amount of dropout to apply in the feed-forward network and 
                      on the input encoding
            
        """
        super().__init__()
        
        self.patch_size = patch_size
        #Adding the cls token to the sequnence 
        self.sequence_length= 1+ num_patches
        # Layers/Networks
        print(dropout)
        self.input_layer = nn.Linear(num_channels*(patch_size**2), embed_dim)
        if block =='prenorm':
          self.transformer = nn.Sequential(*[PreNormAttentionBlock(embed_dim, hidden_dim, num_heads,self.sequence_length, dropout=dropout) for _ in range(num_layers)])
        else:
          self.transformer = nn.Sequential(*[PostNormAttentionBlock(embed_dim, hidden_dim, num_heads,self.sequence_length, dropout=dropout) for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)
        
        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,self.sequence_length,embed_dim))
    
    def get_patches(self,image, patch_size, flatten_channels=True):
        """
        Inputs:
            image - torch.Tensor representing the image of shape [B, C, H, W]
            patch_size - Number of pixels per dimension of the patches (integer)
            flatten_channels - If True, the patches will be returned in a flattened format
                              as a feature vector instead of a image grid.
        Output : torch.Tensor representing the sequence of shape [B,patches,patch_size*patch_size] for flattened.
        """

        patches = nn.functional.unfold(image,kernel_size=patch_size, dilation=1, padding=0, stride=patch_size).transpose(1,2)

        if not flatten_channels:
          patches = patches.reshape(patches.shape[0], patches.shape[1], image.shape[1], image.shape[2], image.shape[3])

        return patches

    def forward(self, x):
        """ViT

        This is a small version of Vision Transformer

        Parameters
        ----------
        x - (`torch.LongTensor` of shape `(batch_size, channels,height , width)`)
            The input tensor containing the iamges.

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, num_classes)`)
            A tensor containing the output from the mlp_head.
        """
        # Preprocess input
        x = self.get_patches(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)
        
        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:,:T+1]
        
        #Add dropout and then the transformer
        x = self.dropout(x)
        x = self.transformer(x)

        #Take the cls token representation and send it to mlp_head

        outputs = self.mlp_head(x[:, 0, :])
        return outputs    
    
    def loss(self,preds,labels):
        '''Loss function.

        This function computes the loss 
        parameters:
            preds - predictions from the model
            labels- True labels of the dataset

        '''
        criterion = nn.CrossEntropyLoss()
        return criterion(preds, labels)
 