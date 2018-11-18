import pandas as pd
import numpy as np

import utils
from google_drive_api import select_files_from_google_drive

class DataManager:
  def __init__(self):
    self.indexes = []
    self.data = {}

  def download_data(self, configs, dummy=False):
    symbols = configs['input_symbols']
    exchanges = configs['exchanges']
    start_date = configs['start_date']
    end_date = configs['end_date']

    self.batch_index = -1

    # Dummy data initialization
    if dummy:
      self.data = {'train': {'data': np.zeros([10000, configs['input_length'], configs['input_dim']], dtype=np.float),
                             'labels': np.zeros([10000, configs['output_dim']])},
                   'test' : {'data': np.zeros([500, configs['input_length'], configs['input_dim']], dtype=np.float),
                             'labels': np.zeros([500, configs['output_dim']])}}
      return

    # Download from Google Drive
    for cross in symbols:
      for exchange in exchanges:
        df = select_files_from_google_drive(cross, exchange, start_date, end_date)
        df['time'] = pd.to_datetime(df['time'])
        df.head()
        self.data[[cross, exchange]] = df

  def get_new_batch(self, size):
    train_data = self.data['train']['data']
    train_labels = self.data['train']['labels']
    train_data_len = len(train_data)
    if self.batch_index not in range(0, train_data_len):
      self.indexes = np.random.randint(0, train_data_len, train_data_len)
      self.batch_index = 0

    if self.batch_index+size > train_data_len:
      utils.log(type='Info',
                msg='Asked for a {} batch, returning a {} one.'.format(size, train_data_len-self.batch_index),
                during='new batch creation')

    self.batch_index += size

    is_last_batch = self.batch_index >= train_data_len
    data_batch = train_data[self.indexes[self.batch_index:self.batch_index+size]]
    labels_batch = train_labels[self.indexes[self.batch_index:self.batch_index+size]]

    return is_last_batch, data_batch, labels_batch

