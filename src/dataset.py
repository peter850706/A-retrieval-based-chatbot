import random
import torch
from torch.utils.data import Dataset


class DialogDataset(Dataset):
    """DialogDataset
    Args:
        data (list): List of samples.
        padding (int): Index used to pad sequences to the same length.
        n_negative (int): Number of false options used as negative samples to train. Set to -1 to use all false options.
        n_positive (int): Number of true options used as positive samples to train. Set to -1 to use all true options.
        shuffle (bool): Do not shuffle options when sampling. **SHOULD BE FALSE WHEN TESTING**
    """
    def __init__(self, data, padding, special_tokens, 
                 n_negative=4, n_positive=1,
                 context_padded_len=300, option_padded_len=50, shuffle=True):
        self.data = data
        self.n_positive = n_positive
        self.n_negative = n_negative
        self.context_padded_len = context_padded_len
        self.option_padded_len = option_padded_len        
        self.padding = padding
        self.special_tokens = special_tokens
        self.shuffle = shuffle
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = dict(self.data[index])
        positives = data['options'][:data['n_corrects']]
        negatives = data['options'][data['n_corrects']:]
        positive_ids = data['option_ids'][:data['n_corrects']]
        negative_ids = data['option_ids'][data['n_corrects']:]

        if self.n_positive == -1:
            n_positive = len(positives)
        if self.n_negative == -1:
            n_negative = len(negatives)
        else:
            n_positive = min(len(positives), self.n_positive)
            n_negative = min(len(negatives), self.n_negative)

        if self.shuffle:
            # sample positive indices
            positive_indices = random.sample(range(len(positives)), k=n_positive)

            # sample negative indices
            negative_indices = random.sample(range(len(negatives)), k=n_negative)

            # collect sampled options
            data['options'] = ([positives[i] for i in positive_indices] + [negatives[i] for i in negative_indices])
            data['option_ids'] = ([positive_ids[i] for i in positive_indices] + [negative_ids[i] for i in negative_indices])
            data['labels'] = [1] * n_positive + [0] * n_negative
            
            # random shuffle the options
            index_list = list(range(len(data['options'])))
            random.shuffle(index_list)
            data['options'] = [data['options'][i] for i in index_list]
            data['option_ids'] = [data['option_ids'][i] for i in index_list]
            data['labels'] = [data['labels'][i] for i in index_list]
        else:
            data['labels'] = [1] * n_positive + [0] * n_negative
        
        # use all of the utterances and insert special token
        tmp = []
        for i, utterance in enumerate(data['context']):
            tmp.extend([self.special_tokens[i % 2]])
            tmp.extend(utterance)
        data['context'] = tmp
        return data

    def collate_fn(self, datas):
        batch = {}
                
        # collate lists
        batch['id'] = [data['id'] for data in datas]
        batch['speaker'] = [data['speaker'] for data in datas]
        batch['labels'] = torch.tensor([data['labels'] for data in datas])
        batch['option_ids'] = [data['option_ids'] for data in datas]
                
        # build tensor of context
        batch['context_lens'] = torch.tensor([min(len(data['context']), self.context_padded_len)
                                              for data in datas])
        padded_len = batch['context_lens'].max().item()
        batch['context'] = torch.tensor([pad_to_len(data['context'], padded_len, self.padding)
                                         for data in datas])
        
        # build tensor of options
        batch['option_lens'] = torch.tensor([[min(max(len(option), 1), self.option_padded_len)
                                              for option in data['options']]
                                             for data in datas])
        padded_len = batch['option_lens'].max().item()
        batch['options'] = torch.tensor([[pad_to_len(option, padded_len, self.padding)
                                          for option in data['options']]
                                         for data in datas])
        return batch


def pad_to_len(arr, padded_len, padding=0):
    """pad_to_len
    Pad `arr` to `padded_len` with padding if `len(arr) < padded_len`. If `len(arr) > padded_len`, truncate arr to `padded_len`.
    Example:
        pad_to_len([1, 2, 3], 5, -1) == [1, 2, 3, -1, -1]
        pad_to_len([1, 2, 3, 4, 5, 6], 5, -1) == [1, 2, 3, 4, 5]
    Args:
        arr (list): List of int.
        padded_len (int)
        padding (int): Integer used to pad.
    """
    arr_len = len(arr)
    if arr_len < padded_len:
        arr += [padding] * (padded_len - arr_len)
    elif arr_len > padded_len:
        arr = arr[-padded_len:]
    else:
        pass
    return arr