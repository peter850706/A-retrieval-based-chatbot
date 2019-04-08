import torch
import torch.nn as nn
import torch.nn.functional as F
from .dualencoderattentionnet import DualEncoderAttentionNet


class DualEncoderAttentionPlusNet(DualEncoderAttentionNet):
    """DualEncoderAttentionNet
    Args:
        dim_embedding (int): The number of features in the input word embeddings.
        rnn_module (str): The module for rnn (LSTM/GRU).
        hidden_size (int): The number of features in the hidden state h.
        num_layers (int): Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two GRUs together to form a stacked GRU, with the second GRU taking in outputs of the first GRU and computing the final results.
        dropout (int): If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer, with dropout probability equal to dropout
        bidirectional (bool): If True, becomes a bidirectional GRU.
        pooling_mode (str): The pooling mode to condense the rnn output features.
    """
    def __init__(self, dim_embedding, rnn_module='GRU', hidden_size=64, num_layers=1, dropout=0, bidirectional=False, mlp_dropout=0):
        super(DualEncoderAttentionPlusNet, self).__init__(dim_embedding, rnn_module, hidden_size, num_layers, dropout, bidirectional)
        del self.context_to_option
        
        # transform option to "contextned" space for better similarity comparision
        features = 8 * hidden_size if bidirectional else 4 * hidden_size
        if mlp_dropout == 0:
            self.similarity = nn.Sequential(nn.Linear(features, features//2),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(features//2, features//4),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(features//4, 1))            
        else:
            self.similarity = nn.Sequential(nn.Linear(features, features//2),
                                            nn.Dropout(mlp_dropout),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(features//2, features//4),
                                            nn.Dropout(mlp_dropout),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(features//4, 1))
        
    def forward(self, context, context_len, options, option_lens):
        """
        The DialogDataset generate context and options
        Args:
            context (tensor): (batch, padded_context_len, dim_embedding) padded_context_len = max(context_len)
            context_len (tensor): (batch, ) original length of the context
            options (tensor): (batch, n_samples, padded_option_len, dim_embedding) padded_option_len = max(option_lens)
            option_lens (tensor): (batch, n_samples) original length of the options
        """
        # features = num_directions * hidden_size, num_directions = 2 if bidirectional else 1
        context_output = self.packed_forward(self.rnn0, context.transpose(1, 0), context_len) # context_output: (padded_context_len, batch, features)
        attentioned_context = torch.cat([context_output, context_output], dim=-1)
        attentioned_context_output = self.packed_forward(self.rnn1, attentioned_context, context_len) # attentioned_context_output: (padded_context_len, batch, features)
        
        # concatenate two pooling mode features, condensed_context_output: (batch, 2*features)
        condensed_context_output = torch.cat([self.pooling(attentioned_context_output, context_len, pooling_mode='last'),
                                              self.pooling(attentioned_context_output, context_len, pooling_mode='max')], dim=-1)
        
        logits = []
        for option, option_len in zip(options.transpose(1, 0), option_lens.transpose(1, 0)):
            option_output = self.packed_forward(self.rnn0, option.transpose(1, 0), option_len) # option_output: (padded_option_len, batch, features)
            attentioned_option = self.attention(context_output, context_len, option_output, option_len) # attentioned_option: (padded_option_len, batch, features)
            
            # interaction: concatenate the features before  and after the attention
            attentioned_option = torch.cat([option_output, attentioned_option], dim=-1)
            attentioned_option_output = self.packed_forward(self.rnn1, attentioned_option, option_len) # attentioned_option_output: (padded_option_len, batch, features)
            
            # concatenate two pooling mode features, condensed_option_output: (batch, 2*features)
            condensed_option_output = torch.cat([self.pooling(attentioned_option_output, option_len, pooling_mode='last'),
                                                 self.pooling(attentioned_option_output, option_len, pooling_mode='max')], dim=-1)
            
            # compute the similarity by inner product and sigmoid normalization, logit: (batch,)            
            logit = torch.sigmoid(self.similarity(torch.cat([condensed_context_output, condensed_option_output], dim=-1)).squeeze())
            logits.append(logit)
        logits = torch.stack(logits, dim=1) # logits: (batch, n_samples)
        return logits