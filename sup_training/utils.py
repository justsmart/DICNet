import numpy as np
import logging
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.vals = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.vals = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.vals.append(val)
        self.sum = np.sum(self.vals)
        self.count = len(self.vals)
        self.avg = np.mean(self.vals)
        self.std = np.std(self.vals)
        self.min = min(self.vals)
        self.min_ind = self.vals.index(self.min)
        self.max = max(self.vals)
        self.max_ind = self.vals.index(self.max)

def setLogger(logfile):
    logger = logging.getLogger()
    
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console = logging.StreamHandler()
    
    
    while logger.handlers:
        logger.handlers.pop()
    if logfile:
        handler = logging.FileHandler(logfile,mode='w') 
        logger.addHandler(handler)
    logger.addHandler(console)
    return logger