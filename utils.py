import numpy as np
import pandas as pd

def get_multihot(targets, n_classes):
    labels = [np.array(list(map(int, str_label.split(' '))))
              for str_label in targets]
    y  = [np.eye(n_classes,dtype=np.float)[label].sum(axis=0)
          for label in labels]
    return y
