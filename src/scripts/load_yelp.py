import numpy as np

from tensorflow.contrib.learn.python.learn.datasets import base
from influence.dataset import DataSet


def load_yelp(train_dir):

  train = np.loadtxt("%s/yelp-ex.train.rating"%train_dir, delimiter='\t')
  valid = np.loadtxt("%s/yelp-ex.valid.rating"%train_dir, delimiter='\t')
  test = np.loadtxt("%s/yelp-ex.test.rating"%train_dir, delimiter='\t')

  train_input = train[:628881,:2].astype(np.int32)
  train_output = train[:628881,2]
  valid_input = valid[:, :2].astype(np.int32)
  valid_output = valid[:, 2]
  test_input = test[:51153, :2].astype(np.int32)
  test_output = test[:51153, 2]

  train = DataSet(train_input, train_output)
  validation = DataSet(valid_input, valid_output)
  test = DataSet(test_input, test_output)

  return base.Datasets(train=train, validation=validation, test=test)


  
