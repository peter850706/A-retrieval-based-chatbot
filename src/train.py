import argparse
import logging
import os
import pdb
import pickle
import sys
import traceback
import json
import time
from callbacks import ModelCheckpoint, MetricsLogger
from metrics import Recall
from predictor import Predictor


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model_dir, exp_dir = os.path.split(args.output_dir[:-1]) if args.output_dir[-1] == '/' else os.path.split(args.output_dir)
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)
    logging.info(f'Save config file to {args.output_dir}.')    
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    logging.info('Loading training data...')
    with open(config['train_path'], 'rb') as f:
        train = pickle.load(f)    
    train.context_padded_len = config['train_context_padded_len']
    train.option_padded_len = config['train_option_padded_len']
    train.n_negative = config['train_n_negative']
    
    logging.info('Loading validation data...')
    with open(config['valid_path'], 'rb') as f:
        valid = pickle.load(f)
        config['model_parameters']['valid'] = valid    
    valid.context_padded_len = config['valid_context_padded_len']
    valid.option_padded_len = config['valid_option_padded_len']
        
    logging.info('Loading preproprecessed word embedding...')
    with open(config['embedding_path'], 'rb') as f:
        embedding = pickle.load(f)
        config['model_parameters']['embedding'] = embedding
        
    metric = Recall(at=10)
    predictor = Predictor(training=True,
                          metrics=[metric],
                          device=args.device,
                          **config['model_parameters'])
    model_checkpoint = ModelCheckpoint(filepath=os.path.join(args.output_dir, 'model'),
                                       monitor=metric.name, 
                                       mode='max', 
                                       all_saved=False)
    metrics_logger = MetricsLogger(log_dest=os.path.join(args.output_dir, 'log.json'))    
    
    if args.load_dir is not None:
        predictor.load(args.load_dir)
    
    logging.info('Start training.')
    start = time.time()
    predictor.fit_dataset(data=train,
                          collate_fn=train.collate_fn,
                          callbacks=[model_checkpoint, metrics_logger],
                          output_dir=args.output_dir)
    end = time.time()
    total = end - start
    hrs, mins, secs = int(total // 3600), int((total % 3600) // 60), int(total % 60)
    logging.info('End training.')
    logging.info(f'Total time: {hrs}hrs {mins}mins {secs}secs.')
    
def _parse_args():
    parser = argparse.ArgumentParser(description="Script to train.")
    parser.add_argument('output_dir', type=str, help='Directory to the model checkpoints and log.')
    parser.add_argument('--device', default=None, help='Device used to train. Can be cpu or cuda:0, cuda:1, etc.')
    parser.add_argument('--load_dir', default=None, type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)