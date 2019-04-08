import torch
import torch.nn as nn
import torch.nn.functional as F
from .dualencodernet import DualEncoderNet


class DualEncoderAttentionNet(DualEncoderNet):
    """DualEncoderAttentionNet
    Args:
        dim_embedding (int): The number of features in the input word embeddings.
        rnn_module (str): The module for rnn (LSTM/GRU).
        hidden_size (int): The number of features in the hidden state h.
        num_layers (int): Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two GRUs together to form a stacked GRU, with the second GRU taking in outputs of the first GRU and computing the final results.
        dropout (int): If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer, with dropout probability equal to dropout
        bidirectional (bool): If True, becomes a bidirectional GRU.
    """
    def __init__(self, dim_embedding, rnn_module='GRU', hidden_size=64, num_layers=1, dropout=0, bidirectional=False):
        super(DualEncoderAttentionNet, self).__init__(dim_embedding, rnn_module, hidden_size, num_layers, dropout, bidirectional)
        del self.rnn, self.context_to_option
        self.rnn0 = self.rnn_module(input_size=dim_embedding,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    dropout=dropout,
                                    bidirectional=bidirectional)
        self.rnn1 = self.rnn_module(input_size=4*hidden_size if bidirectional else 2*hidden_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    dropout=dropout,
                                    bidirectional=bidirectional)
        self.attention = Attention(hidden_size, bidirectional)
        
        # transform option to "contexted" space for better similarity comparision
        features = 4 * hidden_size if bidirectional else 2 * hidden_size
        self.context_to_option = nn.Sequential(nn.Linear(features, features),
                                               nn.ReLU(inplace=True),
                                               nn.Linear(features, features))
        
        for name, param in self.rnn0.named_parameters():
            if 'weight' in name and param.requires_grad:
                nn.init.orthogonal_(param)            
        for name, param in self.rnn1.named_parameters():
            if 'weight' in name and param.requires_grad:
                nn.init.orthogonal_(param)
    
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
        condensed_context_output = self.context_to_option(condensed_context_output)
        
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
            logit = torch.sigmoid((condensed_context_output * condensed_option_output).sum(dim=-1))
            logits.append(logit)
        logits = torch.stack(logits, dim=1) # logits: (batch, n_samples)
        return logits
    

class Attention(nn.Module):
    """Attention: attention layer proposed by Luong et al. 
    ref: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html#decoder
    Args:        
        hidden_size (int): refer to DualEncoderAttentionNet's argumemt
        bidirectional (bool): refer to DualEncoderAttentionNet's argumemt
    """
    def __init__(self, hidden_size=64, bidirectional=False):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.option_attention_weights = torch.randn([100, 50, 350], requires_grad=True) # the attention map
        
        # transform option to "contextned" space for better similarity comparision
        if bidirectional:
            self.option_to_context = nn.Sequential(nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False),
                                                   nn.ReLU(inplace=True),
                                                   nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False))
        else:
            self.option_to_context = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=False),
                                                   nn.ReLU(inplace=True),
                                                   nn.Linear(hidden_size, hidden_size, bias=False))
            
    def forward(self, context, context_len, option, option_len):
        """
        Args:
            context: (padded_context_len, batch, features)
            context_len (tensor): (batch, ) original length of the context
            option: (padded_option_len, batch, features)
            option_len (tensor): (batch, ) original length of the option
        """
        contexted_option = self.option_to_context(option.transpose(1, 0)) # contexted_option: (batch, padded_option_len, features)
        
        # compute similarity by inner product, context.transpose(1, 0): (batch, padded_context_len, features), contexted_option.transpose(2, 1): (batch, features, padded_option_len), energies: (batch, padded_context_len, padded_option_len)
        energies = torch.bmm(context.transpose(1, 0), contexted_option.transpose(2, 1))
        
        # compute attention weights by applying softmax, option_attention_weights: (batch, padded_option_len, padded_context_len)
        self.option_attention_weights = F.softmax(energies, dim=1).transpose(2, 1)
        
        # attentioned_option: (padded_option_len, batch, features)        
        attentioned_option = torch.bmm(self.option_attention_weights, context.transpose(1, 0)).transpose(1, 0)
        return attentioned_option
