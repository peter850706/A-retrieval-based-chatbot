import argparse
import logging
import os
import pdb
import pickle
import sys
import traceback
import json
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from metrics import Recall
from predictor import Predictor


def main(args):
    for input_dir in args.input_dir:
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f'{input_dir} does not exist.')
    
    predicts_list = []
    for i, input_dir in enumerate(args.input_dir):
        print()
        logging.info(f'Loading config from {input_dir}')
        config_path = os.path.join(input_dir, 'config.json')
        with open(config_path) as f:
            config = json.load(f)       
        
        # load test data and word embedding once
        if i == 0:
            logging.info('Loading test data...')
            with open(config['test_path'], 'rb') as f:
                test = pickle.load(f)
                assert test.shuffle is False
                
            logging.info('Loading preproprecessed word embedding...')
            with open(config['embedding_path'], 'rb') as f:
                embedding = pickle.load(f)
                
        test.context_padded_len = config['test_context_padded_len']
        test.option_padded_len = config['test_option_padded_len']
        config['model_parameters']['embedding'] = embedding

        predictor = Predictor(training=False,
                              metrics=[], 
                              device=args.device,
                              **config['model_parameters'])
        model_path = os.path.join(input_dir, 'model.{}.pkl'.format(args.epoch))
        logging.info('Loading model from {}'.format(input_dir))
        predictor.load(model_path)

        logging.info('Start predicting.')
        start = time.time()
        predicts = predictor.predict_dataset(test, test.collate_fn)
        end = time.time()
        total = end - start
        hrs, mins, secs = int(total // 3600), int((total % 3600) // 60), int(total % 60)
        logging.info('End predicting.')
        logging.info(f'Total time: {hrs}hrs {mins}mins {secs}secs.')
        predicts_list.append(predicts)        
        del predictor        
    
    predicts_list = torch.stack(predicts_list, dim=-1)
    predicts = torch.mean(predicts_list, dim=-1) # the ensemble predictions
    
    output_path, filename = os.path.split(args.predict_csv_dir[:-1]) if args.predict_csv_dir[-1] == '/' else os.path.split(args.predict_csv_dir)
    write_predict_csv(predicts, test, output_path, filename)

def write_predict_csv(predicts, data, output_path, filename, n=10):
    outputs = []
    for predict, sample in zip(predicts, data):
        candidate_ranking = [{'candidate-id': oid,
                              'confidence': score.item()}
                             for score, oid in zip(predict, sample['option_ids'])]
        candidate_ranking = sorted(candidate_ranking, key=lambda x: -x['confidence'])
        best_ids = [candidate_ranking[i]['candidate-id']
                    for i in range(n)]
        outputs.append(''.join(['1-' if oid in best_ids else '0-' 
                                for oid in sample['option_ids']]))
        
    filepath = os.path.join(output_path, filename)
    print()
    logging.info(f'Writing output to {filepath}')
    with open(filepath, 'w') as f:
        f.write('Id,Predict\n')
        for output, sample in zip(outputs, data):
            f.write('{},{}\n'.format(sample['id'], output))

def _parse_args():
    parser = argparse.ArgumentParser(description="Script to predict.")
    parser.add_argument('input_dir', nargs='+', type=str, help='Directories to the model checkpoints and log.')
    parser.add_argument('predict_csv_dir', type=str, help='Path to the predict.csv file.')
    parser.add_argument('--device', default=None, help='Device used to train. Can be cpu or cuda:0, cuda:1, etc.')
    parser.add_argument('--not_load', action='store_true', help='Do not load any model.')
    parser.add_argument('--epoch', type=str, default='best')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)