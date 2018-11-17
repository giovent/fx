from models import SimpleLSTMModel
from data_manager import DataManager

# Data format parameters


# ML Parameters
epochs = 5000
batch_size = 32

# TODO: Download data
# ...

# TODO: Data preprocessing
# ...

# Model initialization
model = SimpleLSTMModel(input_length=100, input_dim=100, output_dim=10, mode='Train')

# Model training
for epoch in range(epochs):
  input_batch, labels_batch = DataManager.get_new_batch(batch_size)
  model.train(input_batch=input_batch, labels_batch=labels_batch)

# Save trained model
