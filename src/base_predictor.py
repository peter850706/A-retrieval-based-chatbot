import torch
from torch.utils.data import DataLoader
import torch.utils.data.dataloader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import logging
import math


class BasePredictor():
    def __init__(self,
                 training=True,
                 valid=None,
                 device=None,
                 metrics={},
                 batch_size=10,
                 max_epochs=10,
                 learning_rate=1e-3,
                 weight_decay=0,
                 early_stopping=0,
                 grad_clipping=0):
        self.training = training
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.valid = valid
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        if device is not None:
            self.device = torch.device('cuda:{}'.format(device) if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.device == torch.device('cpu'):
            logging.warning('Not using GPU.')
            
        if training:
            assert early_stopping >= 0, 'The value of early stopping needs to be greater than zero.'
            if early_stopping == 0:
                logging.warning('Not using early stopping.')
            else:
                logging.info('Using early stopping.')
            self.early_stopping = early_stopping if early_stopping else math.inf

            assert grad_clipping >= 0, 'The value of gradient clipping needs to be greater than zero.'
            if grad_clipping == 0:
                logging.warning('Not using gradient clipping.')
            else:
                logging.info('Using gradient clipping.')
            self.grad_clipping = grad_clipping
            self.epoch = 1
            self.not_improved_count = 0
        
    def fit_dataset(self, data, collate_fn=default_collate, callbacks=[]):
        train_dataloader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
        if self.valid is not None:
            valid_dataloader = DataLoader(dataset=self.valid, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)            
        else:
            log_valid = None
            
        # Start the training loop.
        while self.epoch <= self.max_epochs and self.not_improved_count < self.early_stopping:
            print()
            logging.info('Epoch %i' % self.epoch)
            log_train = self._run_epoch(train_dataloader, True)
            if self.valid is not None:
                log_valid = self._run_epoch(valid_dataloader, False)
            
            # evaluate valid score
            for callback in callbacks:
                callback.on_epoch_end(log_train, log_valid, self)
            self.epoch += 1
            
        if self.not_improved_count == self.early_stopping:
            logging.info('Early stopping')
        
        return callbacks
        
    def predict_dataset(self, data,
                        collate_fn=default_collate,
                        batch_size=None,
                        predict_fn=None):
        if batch_size is None:
            batch_size = self.batch_size
        if predict_fn is None:
            predict_fn = self._predict_batch

        # set model to eval mode
        self.model.eval()

        # make dataloader
        dataloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

        ys_ = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch_y_ = predict_fn(batch)
                ys_.append(batch_y_)

        ys_ = torch.cat(ys_, 0)

        return ys_

    def save(self):
        raise NotImplementedError
        
    def load(self):        
        raise NotImplementedError

    def _run_epoch(self, dataloader, training):
        # set model training/evaluation mode
        self.model.train(training)

        # run batches for train
        loss = 0

        # reset metric accumulators
        for metric in self.metrics:
            metric.reset(training)
        
        # run batches
        iter_in_epoch = len(dataloader)
        trange = tqdm(enumerate(dataloader),
                      total=iter_in_epoch,
                      desc='training' if training else 'evaluating')
        for i, batch in trange:
            if training:
                output, batch_loss = self._run_iter(batch, training)
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    output, batch_loss = self._run_iter(batch, training)

            # accumulate loss and metric scores
            loss += batch_loss.item()
            for metric in self.metrics:
                metric.update(output, batch)
            trange.set_postfix(loss=loss / (i + 1), **{m.name: m.print_score() for m in self.metrics})
            
        # calculate averate loss and metrics
        loss /= iter_in_epoch

        epoch_log = {}
        epoch_log['loss'] = float(loss)
        for metric in self.metrics:
            score = metric.get_score()
            logging.info('{}: {}'.format(metric.name, score))
            epoch_log[metric.name] = score
        logging.info('loss: %f' % loss)
        return epoch_log

    def _run_iter(self):
        """ Run iteration for training.
        """
        raise NotImplementedError

    def _predict_batch(self):
        """ Run iteration for predicting.
        """
        raise NotImplementedError