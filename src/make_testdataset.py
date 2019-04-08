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
        
    # load embedding only for words in the data
    embedding_pkl_path = os.path.join(args.dest_dir, 'embedding.pkl')
    logging.info('Loading embedding from {}'.format(embedding_pkl_path))
    with open(embedding_pkl_path, 'rb') as f:
        embedding = pickle.load(f)
        
    # update embedding used by preprocessor
    preprocessor = Preprocessor(embedding=embedding)
    
    # context_padded_len & option_padded_len: the sum of all the utterances in context plus speakers token    
    test_context_padded_len = 885
    test_option_padded_len = 233    
    
    # test
    logging.info('Processing test from {}'.format(args.test_json_dir))
    config['dataset_args'].update({'context_padded_len': test_context_padded_len, 
                                   'option_padded_len': test_option_padded_len,
                                   'n_positive': -1, 
                                   'n_negative': -1, 
                                   'shuffle': False})
    test = preprocessor.get_dataset(args.test_json_dir, args.n_workers, config['dataset_args'])
    test_pkl_path = os.path.join(args.dest_dir, 'test.pkl')
    logging.info('Saving test to {}'.format(test_pkl_path))
    with open(test_pkl_path, 'wb') as f:
        pickle.dump(test, f)
        
    # check    
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
    parser.add_argument('dest_dir', type=str, help='[input] Path to the directory.')
    parser.add_argument('test_json_dir', type=str, help='[input] Path to the test.json file.')
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
