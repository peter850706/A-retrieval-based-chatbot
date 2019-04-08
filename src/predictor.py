import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
from base_predictor import BasePredictor
import math
import logging
import os


def get_instance(module, config, *args, **kwargs):
    return getattr(module, config['type'])(*args, **config['args'], **kwargs)


class Predictor(BasePredictor):
    """Predictor
    Args:
        arch (dict): the parameters to define model architecture
        embedding (Embedding): the Embedding object defined in /src/embedding.py
        loss (dict): the parameters to define loss
        teacher (dict): the parameters to define teacher model for knowledge distillation
        frozen (bool): whether to frozen the pre-trained word embedding or not
        fine_tuning (bool): if frozen, whether to do fine-tuning or not
    """
    def __init__(self, arch=None, embedding=None, loss=None, teacher=None, frozen=True, fine_tuning=False, **kwargs):
        super(Predictor, self).__init__(**kwargs)        
        assert arch is not None and isinstance(arch, dict)
        assert embedding is not None
        
        module_arch = importlib.import_module("modules." + arch['type'].lower())
        self.model = get_instance(module_arch, arch, dim_embedding=embedding.vectors.size(1))
        self.embedding = nn.Embedding(embedding.vectors.size(0),
                                      embedding.vectors.size(1))
        self.embedding.weight = nn.Parameter(embedding.vectors, requires_grad=False) if frozen else nn.Parameter(embedding.vectors)
                
        if self.training:
            if not frozen and fine_tuning:
                raise ValueError('Need to first frozen the word embedding for fine-tuning.')
            if fine_tuning and self.early_stopping == math.inf:
                raise ValueError('Need early stopping > 0 for fine-tuning.')

            if frozen and fine_tuning:
                logging.info('First frozen the word embedding and then fine-tuning.')
            if frozen and not fine_tuning:
                logging.info('Frozen the word embedding and NO fine-tuning.')
            if not frozen and not fine_tuning:
                logging.info('Train the word embedding and model parameters jointly from scratch.')

            # define loss
            assert loss is not None and isinstance(loss, dict)
            defaulted_losses = [loss for loss in dir(nn) if loss[-4:] == 'Loss']
            if loss['type'] in defaulted_losses:
                self.loss = get_instance(nn, loss)
            else:
                module_loss = importlib.import_module("losses." + loss['type'].lower())
                self.loss = get_instance(module_loss, loss)
                
            # make optimizer
            model_parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
            embedding_parameters = list(filter(lambda p: p.requires_grad, self.embedding.parameters()))
            self.optimizer = torch.optim.Adam([{'params': model_parameters}], lr=self.learning_rate, weight_decay=self.weight_decay) if frozen else torch.optim.Adam([{'params': model_parameters}, {'params': embedding_parameters}], lr=self.learning_rate, weight_decay=self.weight_decay)
            
            # knowledge distillation
            self.teaching = False
            if teacher is not None:
                logging.info('Using teacher model for knowledge distillation.')
                self.teaching = True
                
                module_arch = importlib.import_module("modules." + teacher['type'].lower())
                self.teacher_model = get_instance(module_arch, teacher, dim_embedding=embedding.vectors.size(1))
                self.teacher_embedding = nn.Embedding(embedding.vectors.size(0),
                                                      embedding.vectors.size(1))
                self.teacher_embedding.weight = nn.Parameter(embedding.vectors)
                
                checkpoint = torch.load(teacher['path'])        
                self.teacher_model.load_state_dict(checkpoint['model'])
                self.teacher_embedding.load_state_dict(checkpoint['embedding'])
                self.T = teacher['temperature']
                
                self.teacher_model = self.teacher_model.to(self.device)
                self.teacher_embedding = self.teacher_embedding.to(self.device)
                
            # gradient clipping
            if self.grad_clipping != 0:
                for param_group in self.optimizer.param_groups:
                    for param in param_group['params']:
                        param.register_hook(lambda x: x.clamp(min=-self.grad_clipping, max=self.grad_clipping))
            
            self.fine_tuning = fine_tuning
                        
        # use cuda
        self.model = self.model.to(self.device)
        self.embedding = self.embedding.to(self.device)
        
    def save(self, path):
        torch.save({'epoch': self.epoch + 1,
                    'model': self.model.state_dict(),
                    'embedding': self.embedding.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, path)
        
    def load(self, path):
        checkpoint = torch.load(path)
        if self.training:
            self.epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model'])
            self.embedding.load_state_dict(checkpoint['embedding'])            
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
        else:
            self.model.load_state_dict(checkpoint['model'])
            self.embedding.load_state_dict(checkpoint['embedding'])            
    
    def fit_dataset(self, output_dir, **kwargs):
        callbacks = super(Predictor, self).fit_dataset(**kwargs)
        if self.fine_tuning:
            print()
            logging.info('Loading previous best model for fine-tuning.')
            model_path = os.path.join(output_dir, 'model.best.pkl')
            self.load(model_path)
            self.not_improved_count = 0
            self.embedding.weight.requires_grad = True
            self.optimizer.add_param_group({'params': self.embedding.parameters()})
            
            # use 1/20 learning rate
            for param_group in self.optimizer.param_groups:                
                param_group['lr'] = param_group['lr'] / 20
            
            # use the ground truth labels to finetune if using knowledge distillation
            if self.teaching:
                del self.teacher_model, self.teacher_embedding, self.T
            self.teaching = False            
            
            # gradient clipping
            if self.grad_clipping != 0:
                for param_group in self.optimizer.param_groups:
                    for param in param_group['params']:
                        param.register_hook(lambda x: x.clamp(min=-self.grad_clipping, max=self.grad_clipping))
            
            # 
            kwargs.update({'callbacks': callbacks})
            
            logging.info('Start fine-tuning.')
            _ = super(Predictor, self).fit_dataset(**kwargs)
            
    def _run_iter(self, batch, training):
        """ Run iteration for training.
        Args:
            batch (dict)
            training (bool)
        Returns:
            predicts: Prediction of the batch.
            loss (FloatTensor): Loss of the batch.
        """
        context = self.embedding(batch['context'].to(self.device))
        options = self.embedding(batch['options'].to(self.device))
        logits = self.model.forward(context,
                                    batch['context_lens'].to(self.device),
                                    options,
                                    batch['option_lens'].to(self.device))
        if self.teaching and training:
            with torch.no_grad():
                teacher_logits = self.teacher_model.forward(context,
                                                            batch['context_lens'].to(self.device),
                                                            options,
                                                            batch['option_lens'].to(self.device))
                teacher_logits = F.softmax(teacher_logits / self.T, dim=-1)
            loss = self.loss(logits, teacher_logits)
        else:    
            loss = self.loss(logits, batch['labels'].float().to(self.device))
        return logits, loss

    def _predict_batch(self, batch):
        """ Run iteration for predicting.
        Args:
            batch (dict)
        Returns:
            predicts: Prediction of the batch.
        """
        context = self.embedding(batch['context'].to(self.device))
        options = self.embedding(batch['options'].to(self.device))
        logits = self.model.forward(context,
                                    batch['context_lens'].to(self.device),
                                    options,
                                    batch['option_lens'].to(self.device))
        return logits