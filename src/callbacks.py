import math
import json
import logging


class Callback:
    def __init__():
        raise NotImplementedError
    
    def save(self):
        raise NotImplementedError
        
    def load(self):        
        raise NotImplementedError

    def on_epoch_end(log_train, log_valid, model):
        raise NotImplementedError
        
        
class MetricsLogger(Callback):
    def __init__(self, log_dest):
        self.history = {'train': [],
                        'valid': []}
        self.log_dest = log_dest
    
    def save(self, path):
        torch.save({'epoch': self.epoch + 1,
                    'best': self.best,
                    'model': self.model.state_dict(),
                    'embedding': self.embedding.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        if self.filepath != checkpoint['filepath']:
            logging.warning(f"The saved ModelCheckpoint filepath ({checkpoint['filepath']}) is different from the filepath in config.json ({self.filepath}). Use the saved one.")
            self.filepath = checkpoint['filepath']
        self.monitor = checkpoint['monitor']
        self.best = checkpoint['best']
        self.mode = checkpoint['mode']
        self.all_saved =checkpoint['all_saved']
    
    def on_epoch_end(self, log_train, log_valid, model):
        log_train['epoch'] = model.epoch
        log_valid['epoch'] = model.epoch
        self.history['train'].append(log_train)
        self.history['valid'].append(log_valid)
        with open(self.log_dest, 'w') as f:
            json.dump(self.history, f, indent='    ')


class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='loss', mode='min', all_saved=False):
        assert mode in ['min', 'max']
        self.filepath = filepath
        self.monitor = monitor
        self.best = math.inf if mode == 'min' else -math.inf
        self.mode = mode
        self.all_saved = all_saved
    
    def save(self, path):
        torch.save({'epoch': self.epoch + 1,
                    'best': self.best,
                    'model': self.model.state_dict(),
                    'embedding': self.embedding.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        if self.filepath != checkpoint['filepath']:
            logging.warning(f"The saved ModelCheckpoint filepath ({checkpoint['filepath']}) is different from the filepath in config.json ({self.filepath}). Use the saved one.")
            self.filepath = checkpoint['filepath']
        self.monitor = checkpoint['monitor']
        self.best = checkpoint['best']
        self.mode = checkpoint['mode']
        self.all_saved =checkpoint['all_saved']
        
    def on_epoch_end(self, log_train, log_valid, model):
        score = log_valid[self.monitor]
        
        if self.mode == 'min':
            if self.all_saved:
                model.save('{}.{}.pkl'.format(self.filepath, model.epoch))
                logging.info('model.{} saved'.format(model.epoch))
            if score < self.best:
                self.best = score
                model.not_improved_count = 0
                model.save('{}.best.pkl'.format(self.filepath))
                logging.info('model.best saved (min {}: {})'.format(self.monitor, self.best))
            else:
                model.not_improved_count += 1
                logging.info('model.best remained (min {}: {}) (epoch: {})'.format(self.monitor, 
                                                                                   self.best, 
                                                                                   model.epoch - model.not_improved_count))
                
        elif self.mode == 'max':
            if self.all_saved:
                model.save('{}.{}.pkl'.format(self.filepath, model.epoch))
                logging.info('model.{} saved'.format(model.epoch))
            if score > self.best:
                self.best = score
                model.not_improved_count = 0
                model.save('{}.best.pkl'.format(self.filepath))
                logging.info('model.best saved (max {}: {})'.format(self.monitor, self.best))
            else:
                model.not_improved_count += 1
                logging.info('model.best remained (max {}: {}) (epoch: {})'.format(self.monitor, 
                                                                                   self.best, 
                                                                                   model.epoch - model.not_improved_count))