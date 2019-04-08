import torch


class Metrics:
    def __init__(self):
        self.name = 'metric name'

    def reset(self):
        raise NotImplementedError

    def update(self, predicts, batch):
        pass NotImplementedError

    def get_score(self):
        pass NotImplementedError


class Recall(Metrics):
    """Recall
    Args:
         at (int): @ to eval.
    """
    def __init__(self, at=10):
        self.at_eval = at
        self.n = 0
        self.n_correct = 0
        self.name = 'recall@{}'.format(at)

    def reset(self, training):
        if training:
            self.at = 1
            self.name = 'recall@{}'.format(1)
        else:
            self.at = self.at_eval
            self.name = 'recall@{}'.format(self.at_eval)
        self.n = 0
        self.n_corrects = 0

    def update(self, predicts, batch):
        """
        Args:
            predicts (FloatTensor): (batch, n_samples).
            batch (dict): batch.
        """
        predicts = predicts.cpu()
        self.n += predicts.size(0)
        predicts = [[index for (index, score) in sorted(enumerate(predict), key=lambda x: x[1], reverse=True)] for predict in predicts]
        labels = batch['labels'].argmax(dim=-1)
        for predict, label in zip(predicts, labels):
            if label in predict[:self.at]:
                self.n_corrects += 1
                
    def get_score(self):
        return self.n_corrects / self.n
        
    def print_score(self):
        return '{:.3f}'.format(self.get_score())