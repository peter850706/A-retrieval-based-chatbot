import argparse
import logging
import os
import pdb
import sys
import traceback
import pickle
import json
from embedding import Embedding
from preprocessor import Preprocessor
import fastText as fasttext
from collections import Counter


def main(args):
    config_path = os.path.join(args.dest_dir, 'config.json')
    logging.info('Loading configuration from {}'.format(config_path))
    with open(config_path) as f:
        config = json.load(f)

    preprocessor = Preprocessor(embedding=None)
    
    # collect words appear in the data
    words_pkl_path = os.path.join(args.dest_dir, 'words.pkl')
    try:
        logging.info('Loading words from {}'.format(words_pkl_path))
        with open(words_pkl_path, 'rb') as f:
            words = pickle.load(f)
    except:
        logging.info('Loading words Failed!! Reconstructing words.')
        logging.info('Collecting words from {}'.format(config['train_json_path']))
        train_words = preprocessor.collect_words(config['train_json_path'],
                                            n_workers=args.n_workers)
        logging.info('Collecting words from {}'.format(config['valid_json_path']))
        valid_words = preprocessor.collect_words(config['valid_json_path'],
                                            n_workers=args.n_workers)    
        logging.info('Collecting words from {}'.format(config['test_json_path']))
        test_words = preprocessor.collect_words(config['test_json_path'],
                                            n_workers=args.n_workers)
        words = Counter(train_words + valid_words + test_words)        
        logging.info('Saving words to {}'.format(words_pkl_path))        
        with open(words_pkl_path, 'wb') as f:
            pickle.dump(words, f)
            
    logging.info('The length of words before filtering is {}'.format(len(words)))
    words = {key: value for key, value in words.items() if value > 9}
    logging.info('The length of words after filtering is {}'.format(len(words)))
    
    # load embedding only for words in the data
    embedding_pkl_path = os.path.join(args.dest_dir, 'embedding.pkl')
    try:
        logging.info('Loading embedding from {}'.format(embedding_pkl_path))
        with open(embedding_pkl_path, 'rb') as f:
            embedding = pickle.load(f)
    except:        
        logging.info('Loading embedding Failed!! Reconstructing embedding.')
        fasttext_model_path = config['fasttext_model_path']
        logging.info(f"Loading pre-trained fastText model bin file from {fasttext_model_path}.")
        embedding = Embedding(words=words, 
                              fasttext_model=fasttext.load_model(fasttext_model_path))
        del embedding.fasttext_model # fasttext model can't be pickled
        logging.info('Saving embedding to {}'.format(embedding_pkl_path))
        with open(embedding_pkl_path, 'wb') as f:
            pickle.dump(embedding, f)
            
    # update embedding used by preprocessor
    preprocessor = Preprocessor(embedding=embedding)
    
    # context_padded_len: the sum of all the utterances in the context plus speakers token, and choose the longest one
    # option_padded_len: the sum of the words in the option, and choose the longest one
    train_context_padded_len = 1831
    train_option_padded_len = 334        
    valid_context_padded_len = 1203
    valid_option_padded_len = 227
    test_context_padded_len = 885
    test_option_padded_len = 233
    
    # train
    logging.info('Processing train from {}'.format(config['train_json_path']))
    config['dataset_args'].update({'context_padded_len': train_context_padded_len, 
                                   'option_padded_len': train_option_padded_len})
    train = preprocessor.get_dataset(config['train_json_path'], args.n_workers, config['dataset_args'])
    train_pkl_path = os.path.join(args.dest_dir, 'train.pkl')
    logging.info('Saving train to {}'.format(train_pkl_path))
    with open(train_pkl_path, 'wb') as f:
        pickle.dump(train, f)
        
    # valid
    logging.info('Processing valid from {}'.format(config['valid_json_path']))
    config['dataset_args'].update({'context_padded_len': valid_context_padded_len, 
                                   'option_padded_len': valid_option_padded_len,
                                   'n_positive': -1, 
                                   'n_negative': -1, 
                                   'shuffle': False})
    valid = preprocessor.get_dataset(config['valid_json_path'], args.n_workers, config['dataset_args'])
    valid_pkl_path = os.path.join(args.dest_dir, 'valid.pkl')
    logging.info('Saving valid to {}'.format(valid_pkl_path))
    with open(valid_pkl_path, 'wb') as f:
        pickle.dump(valid, f)
        
    # test
    logging.info('Processing test from {}'.format(config['test_json_path']))
    config['dataset_args'].update({'context_padded_len': test_context_padded_len, 
                                   'option_padded_len': test_option_padded_len,
                                   'n_positive': -1, 
                                   'n_negative': -1, 
                                   'shuffle': False})
    test = preprocessor.get_dataset(config['test_json_path'], args.n_workers, config['dataset_args'])
    test_pkl_path = os.path.join(args.dest_dir, 'test.pkl')
    logging.info('Saving test to {}'.format(test_pkl_path))
    with open(test_pkl_path, 'wb') as f:
        pickle.dump(test, f)
        
    # check
    logging.info('train | max context_length: {}, max option_length: {}'.format(max([sum([len(utterance) 
                                                                                          for utterance in data['context']]) + 
                                                                                     len(data['context'])
                                                                                     for data in train.data]),
                                                                                max([len(option) 
                                                                                     for data in train.data 
                                                                                     for option in data['options']])
                                                                               )
                )
    logging.info('valid | max context_length: {}, max option_length: {}'.format(max([sum([len(utterance) 
                                                                                          for utterance in data['context']]) + 
                                                                                     len(data['context'])
                                                                                     for data in valid.data]),
                                                                                max([len(option) 
                                                                                     for data in valid.data
                                                                                     for option in data['options']])
                                                                               )
                )
    logging.info('test  | max context_length: {}, max option_length: {}'.format(max([sum([len(utterance) 
                                                                                          for utterance in data['context']]) + 
                                                                                     len(data['context'])
                                                                                     for data in test.data]),
                                                                                max([len(option) 
                                                                                     for data in test.data 
                                                                                     for option in data['options']])
                                                                               )
                )

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess and generate preprocessed pickle.")
    parser.add_argument('dest_dir', type=str,
                        help='[input] Path to the directory that .')
    parser.add_argument('--n_workers', type=int, default=4)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s: %(message)s',
        level=logging.INFO, datefmt='%m-%d %H:%M:%S'
    )
    args = _parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)