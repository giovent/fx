import tensorflow as tf

import ml_utils as ml

class SimpleLSTMModel:
  def __init__(self, input_length=0, input_dim=0, output_dim=0, mode='Infer', saved_model_path=''):
    self.mode = mode
    if mode == 'Infer':
      self.load_trained_model(saved_model_path)
    elif mode == 'Train':
      self.create_model(input_length, input_dim, output_dim)

  def create_model(self, input_length, input_dim, output_dim):
    self.input = tf.placeholder("float", [None, input_length, input_dim])
    self.labels = tf.placeholder("float", [None, 1, output_dim])

    self.rnn_output = ml.LSTM(self.input, input_length)
    self.layer1_output = ml.fully_connected_layer(self.run_output[-1], 512)
    self.output = ml.fully_connected_layer(self.layer1_output)

    self.loss = tf.reduce_mean(tf.square(self.output-Y))
    self.optimizer   = tf.train.AdamOptimizer(0.00001)
    self.train_op    = self.optimizer.minimize(self.loss)
    self.train_sum   = tf.summary.scalar('Training_loss', self.loss)
    self.test_sum    =  tf.summary.scalar('Validation_loss', self.loss)
    self.file_writer = tf.summary.FileWriter('Tensorboard/')
    self.saver = tf.train.Saver()

  def load_trained_model(self, saved_model_path):
    # TODO: load model from file
    pass

  def __call__(self, input):
    return self.sess.run([self.output], feed_dict = {self.input: input})

  def train(self, input_batch, labels_batch):
    self.sess.run(self.train_op, feed_dict={self.input: input_batch, self.labels: labels_batch})


X  = tf.placeholder("float", [None, input_length, input_dimension])
Y  = tf.placeholder("float", [None, 1, output_dimension])
dr = tf.placeholder("float") #dropout parameter