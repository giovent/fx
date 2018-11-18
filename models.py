import tensorflow as tf

import ml_utils as ml

class SimpleLSTMModel:
  def __init__(self, configs, mode, saved_model_path=''):
    self.input_length = configs['input_length']
    self.input_dim = configs['input_dim']
    self.output_dim = configs['output_dim']
    self.mode = mode

    if mode == 'Infer':
      self.load_trained_model(saved_model_path)
      self.warm_up()
    elif mode == 'Train':
      self.create_model()

    self.init_sess(configs)

  def create_model(self):
    self.input = tf.placeholder("float", [None, self.input_length, self.input_dim])
    self.labels = tf.placeholder("float", [None, self.output_dim])

    self.rnn_output = ml.LSTM(self.input, self.input_length)
    self.layer1_output = ml.fully_connected_layer(self.rnn_output[-1], 512)
    self.output = ml.fully_connected_layer(self.layer1_output, self.output_dim)

    self.loss = tf.reduce_mean(tf.square(self.output-self.labels))
    self.optimizer   = tf.train.AdamOptimizer(0.00001)
    self.train_op    = self.optimizer.minimize(self.loss)
    self.train_sum   = tf.summary.scalar('Training_loss', self.loss)
    self.test_sum    =  tf.summary.scalar('Validation_loss', self.loss)
    self.file_writer = tf.summary.FileWriter('Tensorboard/')
    self.saver = tf.train.Saver()

  def init_sess(self, config):
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())

  def warm_up(self):
    pass

  def load_trained_model(self, saved_model_path):
    # TODO: load model from file
    pass

  def __call__(self, input):
    return self.sess.run([self.output], feed_dict = {self.input: input})

  def train(self, input_batch, labels_batch):
    self.sess.run(self.train_op, feed_dict={self.input: input_batch, self.labels: labels_batch})
