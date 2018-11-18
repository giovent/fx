from data_manager import DataManager
from models import SimpleLSTMModel

# Data format parameters
# TODO: Spostare parametri in un file json
configs = dict()
configs['exchanges'] = ['BGN']
configs['input_symbols'] = ['BTCUSD', 'ETHUSD']
configs['input_length'] = 100
configs['input_values'] = ['BID', 'ASK', 'TRADE']
configs['additional_values'] = ['WEEK_DAY', 'HOUR']
configs['input_dim'] = len(configs['exchanges'])*len(configs['input_values'])\
                       *len(configs['input_symbols'])+len(configs['additional_values'])
configs['output_delay'] = 60 # min
configs['output_symbols'] = ['BTCUSD']
configs['output_dim'] = len(configs['output_symbols'])
configs['start_date'] = '2018-09-01'
configs['end_date'] = 'today'

# ML Parameters
# TODO: Spostare parametri in un file json
epochs = 5000
batch_size = 32

# TODO: Download data
database = DataManager()
database.download_data(configs, dummy=True)
# ...

# TODO: Data preprocessing
# ...

# Model initialization
model = SimpleLSTMModel(configs, mode='Train')

# Model training
for epoch in range(epochs):
  # TODO: change get_new_batch structure
  is_last_batch = False
  while not is_last_batch:
    is_last_batch, input_batch, labels_batch = database.get_new_batch(batch_size)
    model.train(input_batch=input_batch, labels_batch=labels_batch)
  print('Epoch {} of {} finished.'.format(epoch+1, epochs))

# Save trained model
