import tensorflow as tf
import pandas as pd
import numpy as np

from google_drive_api import select_files_from_google_drive

input_csv = 'fx_XBTUSD+KRKN+Curncy_20180424.csv'
val_type  = {'BID':0, 'ASK':1, 'TRADE':2}

#Data parameters
input_length     = 12
input_dimension  = 3
output_dimension = 3

#Training parameters
rnn_dim    = 256
epochs     = 100
batch_size = 32

### Load the data

# Example 1
data   = [[[0.9,0.99,0.98],[0.99,1,1]]]
labels = [[1,1,1]]

# From file .CSV
df = pd.read_csv(input_csv, index_col=0, parse_dates=[2])

# From Google Drive 
cross = 'XBTUSD'
exchange = 'BGN'
start_date = pd.to_datetime('2000-01-01')
end_date = pd.to_datetime('today')
start_date = pd.to_datetime('2018-09-25')
df = select_files_from_google_drive(cross, exchange, start_date, end_date)
df['time'] = pd.to_datetime(df['time'])

df.head()

df = df.pivot_table(values='value', index='time', columns='type', aggfunc='sum')
df /= 9000

df.head()

def return_raw_data(df, input_length):
    doc = {}
    #for idx, r in df.iterrows():
    #    dt = idx
    #    df_x = df[(df.index-dt)<pd.to_timedelta(1, unit='H')]
    #    df_x = df_x[(df_x.index-dt)>=pd.to_timedelta(0, unit='H')]
    #    df_y = df[df.index>(dt+pd.to_timedelta(2, unit='H'))]
    #    df_y = df_y[df_y.index==df_y.index.min()]
    #    doc[dt] = {'x': df_x, 'y':df_y}
    for i in range(df.shape[0]-input_length):
        dt = df.index[i]
        df_x = df.iloc[i:i+input_length]
        df_y = df[df.index>(dt+pd.to_timedelta(2, unit='H'))]
        df_y = df_y[df_y.index==df_y.index.min()]
        if not df_y.empty:
            doc[dt] = {'x': df_x, 'y':df_y}
    return doc

doc = return_raw_data(df, input_length)
print(doc)

filter_col = ['ASK', 'BID', 'TRADE']
X_raw = [np.array(doc[elem]['x'][filter_col]).tolist() for elem in doc]
Y_raw = [np.array(doc[elem]['y'][filter_col]).tolist() for elem in doc]

### Normalize the data
# (later)

# TODO: Divide the data in training (first 80%) and testing (last 20%)
split_idx = int(len(X_raw)*0.8)
X_train = X_raw[:split_idx]
X_test = X_raw[split_idx:]
Y_train = Y_raw[:split_idx]
Y_test = Y_raw[split_idx:]

#print(X_train[0],'------',Y_train[0])
#pint(split_idx)

def RNN(input, input_length, output_dim=2, rnn_units=128, dropout=1):
    input_ = tf.unstack(input, input_length, 1)
    with tf.name_scope('RNN_Layer'):
        with tf.variable_scope('lstm'):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(rnn_units, forget_bias=1.0)
            outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, input_, dtype=tf.float32)
        rnn_output = outputs[-1] #last output
    with tf.name_scope('Layer'):
        rnn_output = tf.nn.dropout(rnn_output,dropout)
        W = tf.Variable(tf.random_normal([rnn_units,output_dim], 0.0, 0.1),name='W1')
        b = tf.Variable(tf.random_normal([output_dim], 0.0, 0.1), name='Bias')
        return tf.matmul(rnn_output,W)+b

### Define the general input and output
X  = tf.placeholder("float", [None, input_length, input_dimension])
Y  = tf.placeholder("float", [None, 1, output_dimension])
dr = tf.placeholder("float") #dropout parameter

prediction = RNN(X,input_length,output_dim=output_dimension,rnn_units=rnn_dim, dropout=dr)


with tf.name_scope('Training_Stuff'):
    loss = tf.reduce_mean(tf.square(prediction-Y))
    optimizer   = tf.train.AdamOptimizer(0.00001)
    train_op    = optimizer.minimize(loss)
    train_sum   = tf.summary.scalar('Training_loss',loss)
    test_sum    =  tf.summary.scalar('Validation_loss',loss)
    file_writer = tf.summary.FileWriter('Tensorboard/len='+str(input_length)+'_rnn='+str(rnn_dim)+'/')
    saver = tf.train.Saver()

## test
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

indexes = np.random.randint(0,len(X_train),len(X_train))
batch_indexes = indexes[:batch_size]
print(batch_indexes)
print(Y_test.shape)

### Start training
with tf.Session() as sess:
    file_writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs*15):
        #TODO: Create batches from original data
        indexes = np.random.randint(0,len(X_train),len(X_train))
        batch_n = 0
        while batch_n*batch_size < len(X_train):
            batch_indexes = indexes[batch_n*batch_size:(batch_n+1)*batch_size]
            batch_n += 1
            batch_x, batch_y = X_train[batch_indexes], Y_train[batch_indexes]
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, dr: 0.8})

        # Compute the current model loss over training and testing data
        s,lr = sess.run([train_sum,loss],feed_dict={X:X_train,Y:Y_train,dr:1})
        file_writer.add_summary(s,epoch)
        s,le = sess.run([test_sum,loss],feed_dict={X:X_test,Y:Y_test,dr:1})
        file_writer.add_summary(s,epoch)
