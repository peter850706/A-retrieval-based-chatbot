import re
import torch
import numpy as np
import logging


class Embedding:
    """
    Args:
        words (None or list): If not None, only load embedding of the words in
            the list.
        fasttext_model (fastText model): A pre-trained fastText model for predicting the vectors of OOV
        oov_as_unk (bool): If argument `words` are provided, whether or not
            treat words in `words` but not in embedding file as `<unk>`. If
            true, OOV (out of vocabulary) will be mapped to the index of `<unk>`. Otherwise,
            embedding of those OOV will be randomly initialize and their
            indices will be after non-OOV.
        lower (bool): Whether or not lower the words.
        rand_seed (int): Random seed for embedding initialization.
    """
    def __init__(self, words=None, fasttext_model=None, lower=True, rand_seed=524):
        self.word_dict = {}
        self.vectors = None
        self.fasttext_model = fasttext_model
        self.lower = lower
        self.extend(words)
        
        # padding
        if '</p>' not in self.word_dict:
            self.add('</p>', torch.zeros(self.get_dim()))
        logging.info("'</p>': {}".format(self.word_dict['</p>']))
        
        # special token for speakers
        if '</s1>' not in self.word_dict:
            self.add('</s1>')        
        if '</s2>' not in self.word_dict:
            self.add('</s2>')
        logging.info("'</s1>': {}".format(self.word_dict['</s1>']))
        logging.info("'</s2>': {}".format(self.word_dict['</s2>']))
        
        # OOV
        if '<unk>' not in self.word_dict:
            self.add('<unk>')
        logging.info("'<unk>': {}".format(self.word_dict['<unk>']))
        
        torch.manual_seed(rand_seed)        
        
    def to_index(self, word):
        """
        word (str)

        Return:
             index of the word. If the word is not in `words` and not in the
             embedding file, then index of `<unk>` will be returned.
        """
        if self.lower:
            word = word.lower()

        if word not in self.word_dict:
            return self.word_dict['<unk>']
        else:
            return self.word_dict[word]

    def get_dim(self):
        return self.vectors.shape[1]

    def get_vocabulary_size(self):
        return self.vectors.shape[0]

    def add(self, word, vector=None):
        if self.lower:
            word = word.lower()

        if vector is not None:
            vector = vector.view(1, -1)
        else:
            vector = torch.empty(1, self.get_dim())
            torch.nn.init.uniform_(vector)
        self.vectors = torch.cat([self.vectors, vector], 0)
        self.word_dict[word] = len(self.word_dict)
    
    def extend(self, words):
        assert words is not None
        
        # use fastText pre-trained model to predict word vectors the dataset and those OOV, ref: https://github.com/facebookresearch/fastText/issues/475
        assert self.fasttext_model is not None
        vectors = []
        for word in words:
            if self.lower:
                word = word.lower()
            if word not in self.word_dict:
                vectors.append(self.fasttext_model.get_word_vector(word))
                self.word_dict[word] = len(self.word_dict)
        vectors = np.array(vectors)
        self.vectors = torch.from_numpy(vectors)