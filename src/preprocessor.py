import json
import logging
from multiprocessing import Pool
from dataset import DialogDataset
from tqdm import tqdm
import nltk


class Preprocessor:
    """Preprocessor
    Args:
        embedding (Embedding): the Embedding object defined in /src/embedding.py.
    """
    def __init__(self, embedding):
        self.embedding = embedding
        if embedding is not None:
            self.lower = embedding.lower
        self.logging = logging.getLogger(name=__name__)
        nltk.download('punkt')

    def tokenize(self, sentence):
        """ Tokenize a sentence.
        Args:
            sentence (str): One string.
        Return:
            tokens (list of str): List of tokens in a sentence.
        """
        tokens = nltk.word_tokenize(sentence)
        return tokens

    def sentence_to_indices(self, sentence):
        """ Convert sentence to its word indices.
        Args:
            sentence (str): One string.
        Return:
            indices (list of int): List of word indices.
        """
        words = self.tokenize(sentence)
        indices = [self.embedding.to_index(word.lower()) for word in words] if self.lower else [self.embedding.to_index(word) for word in words]
        return indices

    def collect_words(self, data_path, n_workers=4):
        with open(data_path) as f:
            data = json.load(f)

        utterances = []
        for sample in data:
            utterances += (
                [message['utterance']
                 for message in sample['messages-so-far']]
                + [option['utterance']
                   for option in sample['options-for-next']]
            )
        utterances = list(set(utterances))
        
        """
        # speed up the tokenizing procedure by concatenating several utterances into one sentence, but will lead to the inconsistency when processing every utterance alone.
        chunks = [
            ' '.join(utterances[i:i + len(utterances) // n_workers])
            for i in range(0, len(utterances), len(utterances) // n_workers)
        ]
        with Pool(n_workers) as pool:
            chunks = pool.map_async(self.tokenize, chunks)
            words = set(sum(chunks.get(), []))
        """
        
        words = []
        with Pool(processes=n_workers) as pool:
            for i in range(n_workers):
                batch_start = (len(utterances) // n_workers) * i                
                batch_end = len(utterances) if i == n_workers - 1 else (len(utterances) // n_workers) * (i + 1)
                for utterance in utterances[batch_start: batch_end]:
                    words.extend(pool.apply_async(self.tokenize, [utterance]).get())
            pool.close()
            pool.join()
        return words

    def get_dataset(self, data_path, n_workers=4, dataset_args={}):
        """ Load data and return Dataset objects for training and validating.
        Args:
            data_path (str): Path to the data.
        """
        self.logging.info('loading dataset...')
        with open(data_path) as f:
            dataset = json.load(f)

        self.logging.info('preprocessing data...')
        
        results = [None] * n_workers
        with Pool(processes=n_workers) as pool:
            for i in range(n_workers):
                batch_start = (len(dataset) // n_workers) * i
                batch_end = len(dataset) if i == n_workers - 1 else (len(dataset) // n_workers) * (i + 1)
                batch = dataset[batch_start: batch_end]
                results[i] = pool.apply_async(self.preprocess_samples, [batch])
            pool.close()
            pool.join()        
        processed = []
        for result in results:
            processed += result.get()
            
        padding = self.embedding.to_index('</p>')
        special_tokens = [self.embedding.to_index('</s1>'), self.embedding.to_index('</s2>')]
        return DialogDataset(processed, padding=padding, special_tokens=special_tokens, **dataset_args)

    def preprocess_samples(self, dataset):
        """ Worker function.
        Args:
            dataset (list of dict)
        Returns:
            list of processed dict.
        """
        processed = []
        for sample in tqdm(dataset):
            processed.append(self.preprocess_sample(sample))
        return processed

    def preprocess_sample(self, data):
        """
        Args:
            data (dict)
        Returns:
            dict
        """
        processed = {}
        processed['id'] = data['example-id']

        # process messages-so-far
        processed['context'] = []
        processed['speaker'] = []
        for message in data['messages-so-far']:
            processed['context'].append(
                self.sentence_to_indices(message['utterance'])
            )

        # process options
        processed['options'] = []
        processed['option_ids'] = []

        # process correct options
        if 'options-for-correct-answers' in data:
            processed['n_corrects'] = len(data['options-for-correct-answers'])
            for option in data['options-for-correct-answers']:
                processed['options'].append(
                    self.sentence_to_indices(option['utterance'])
                )
                processed['option_ids'].append(option['candidate-id'])
        else:
            processed['n_corrects'] = 0

        # process the other options
        for option in data['options-for-next']:
            if option['candidate-id'] in processed['option_ids']:
                continue

            processed['options'].append(
                self.sentence_to_indices(option['utterance'])
            )
            processed['option_ids'].append(option['candidate-id'])

        return processed
