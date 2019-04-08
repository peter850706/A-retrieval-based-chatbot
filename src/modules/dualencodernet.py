import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DualEncoderNet(nn.Module):
    """DualEncoderNet
    Args:
        dim_embedding (int): The number of features in the input word embeddings.
        rnn_module (str): The module for rnn (LSTM/GRU).
        hidden_size (int): The number of features in the hidden state h.
        num_layers (int): Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two GRUs together to form a stacked GRU, with the second GRU taking in outputs of the first GRU and computing the final results.
        dropout (int): If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer, with dropout probability equal to dropout
        bidirectional (bool): If True, becomes a bidirectional GRU.
    """
    def __init__(self, dim_embedding, rnn_module='GRU', hidden_size=64, num_layers=1, dropout=0, bidirectional=False):
        super(DualEncoderNet, self).__init__()
        if rnn_module == 'LSTM':
            self.rnn_module = nn.LSTM
        elif rnn_module == 'GRU':
            self.rnn_module = nn.GRU
        else:
            raise ValueError(f"rnn_module should be 'LSTM' or 'GRU', but got {rnn_module}.")
        self.rnn = self.rnn_module(input_size=dim_embedding,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout,
                                   bidirectional=bidirectional)
        
        # transform context to "optioned" space for better similarity comparision
        self.context_to_option = nn.Linear(2 * hidden_size, 2 * hidden_size) if bidirectional else nn.Linear(hidden_size, hidden_size)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name and param.requires_grad:
                nn.init.orthogonal_(param)
    
    def packed_forward(self, rnn, padded_input, lengths):
        """
        Args:
            rnn: LSTM/GRU layers
            padded_input (tensor): (padded_input_len, batch, features)
            lengths (tensor): (batch, ) original length of the padded_input
        Return:
            padded_output (tensor): (padded_input_len, batch, features)
        """
        lengths, sorted_indexes = torch.sort(lengths, descending=True) # sorted by descending order
        padded_input = padded_input.index_select(dim=1, index=sorted_indexes)
        packed_input = pack_padded_sequence(input=padded_input, lengths=lengths)
        packed_output, _ = rnn(packed_input)
        padded_output, _ = pad_packed_sequence(sequence=packed_output, padding_value=0)
        unsorted_indexes = torch.argsort(sorted_indexes) # recover the original order
        return padded_output.index_select(dim=1, index=unsorted_indexes)
    
    def pooling(self, padded_input, lengths, pooling_mode='last'):
        """
        Args:
            padded_input (tensor): (padded_input_len, batch, features)
            lengths (tensor): (batch, ) original length of the padded_input
            pooling_mode (str): all ignore the padding in the padded_input
                'last': choose the last hidden state of padded_input for every batch
                'max': choose the max value hidden state of padded_input for every batch
                'mean':
        Return:
            output: (batch, features)
        """        
        batch = padded_input.size(1)
        if pooling_mode == 'last':
            output = padded_input[lengths - 1, list(range(batch)), :]
        elif pooling_mode == 'max':
            output = []
            for i in range(batch):
                output.append(torch.max(padded_input[:lengths[i], i, :], dim=0)[0])
            output = torch.stack(output, dim=0)
        elif pooling_mode == 'mean':
            output = []
            for i in range(batch):
                output.append(torch.mean(padded_input[:lengths[i], i, :], dim=0))
            output = torch.stack(output, dim=0)
        else:
            raise ValueError("pooling mode should be 'last', 'max' or 'mean'.")
        return output
    
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
        context_output = self.packed_forward(self.rnn, context.transpose(1, 0), context_len) # context_output: (padded_context_len, batch, features)
        condensed_context_output = self.context_to_option(self.pooling(context_output, context_len, pooling_mode='last')) # condensed_context_output: (batch, features)
        
        logits = []
        for option, option_len in zip(options.transpose(1, 0), option_lens.transpose(1, 0)):
            option_output = self.packed_forward(self.rnn, option.transpose(1, 0), option_len) # option_output: (padded_option_len, batch, features)
            condensed_option_output = self.pooling(option_output, option_len, pooling_mode='last')
            
            # compute the similarity by inner product and sigmoid normalization, logit: (batch,)
            logit = torch.sigmoid((condensed_context_output * condensed_option_output).sum(dim=-1))
            logits.append(logit)
        logits = torch.stack(logits, dim=1) # logits: (batch, n_samples)
        return logits
