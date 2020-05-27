import numpy as np
from sklearn.model_selection import train_test_split

class GenSplit():

    def __init__(self, size, train=0.7, val=0.15, test=0.15, seed=1):
        super().__init__()

        self.train = train
        self.val = val 
        self.test = test
        all_idxs = range(size)

        # split train from validation-test
        train_idxs, test_val_idxs = train_test_split(all_idxs, test_size=(self.val + self.test), random_state=seed)

        # split validation from test
        val_idxs, test_idxs = train_test_split(test_val_idxs, test_size=(self.test / (self.test + self.val)), random_state=seed)     

        self.train_idxs = train_idxs
        self.val_idxs = val_idxs
        self.test_idxs = test_idxs

    def __call__(self):
        
        return self.train_idxs, self.val_idxs, self.test_idxs