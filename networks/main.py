import tensorflow as tf      # Tensorflow 2.1 was used

from utils.data_prep import tokenize, trim_time_sequence, remove_long_sequence, pad_time_sequence
from utils.data_prep import prepare_dataset_v3, oversample_dataset
from utils.model import naive_evaluate, load_embeddings
from utils.model import ResampleLayer
from utils.others import print_train_info_v2, plot_and_save
from utils import plot
import tensorflow as tf
import numpy as np
import pickle
from time import time
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


TS_LIMIT = 120  # samples will be trimmed/picked within this limit (sec)
TS_TRIM = TS_LIMIT
data_fn = r'./data/HDFS/Xy_dataset.pkl'

# Load data
print('\nLoading data...', end='')
start = time()
with open(data_fn,'rb') as f:
  X, y = pickle.load(f)
print('{:.2f}s'.format(time()-start), end='\t')
print('{:d} samples'.format(len(y)))

# separate value & timestamp
print('\nProcessing data...', end='')
start = time()
x_seq =  X[:, 0]
x_ts = X[:, 1].copy()
# y = y[0:5000]
del X

# tokenization
x_tok, tokenizer = tokenize(x_seq)  # 48 tokens
vocab_size = len(tokenizer.word_counts)
# trim/remove data to a certain time length
x_tok, x_ts, y = remove_long_sequence(x_tok, x_ts, y, TS_LIMIT)  # remove sequence that are longer than limit
trim_time_sequence(x_tok, x_ts, TS_TRIM)  # trim sequence to limit
# front pad data to form matrices.
x_tok, x_ts = pad_time_sequence(x_tok, x_ts, 
    maxlen=250, ts_interval=0.1)

print('{:.2f}s'.format(time()-start), end='\t')
print('{:d} samples, {:d} samples with label=1.'.format(len(y), sum(y)))


# Load embeddings
embeddings, embedding_dims = load_embeddings('./log_embeddings_16_sg.txt')
# obtain token - embedding matrix, token zero's embedding is zero.
embedding_matrix = np.zeros((vocab_size+1, embedding_dims))

for word, i in tokenizer.word_index.items():
    embedding_matrix[i] = embeddings[word]

print(embedding_matrix.shape)

# make dataset
test_ratio = 0.2
val_ratio = 0.2

# Dataset split
_train, _val, _test = prepare_dataset_v3(x_tok, x_ts, y, test_ratio=test_ratio, val_ratio=val_ratio)
x_train, ts_train, y_train = _train
x_val, ts_val, y_val = _val
x_test, ts_test, y_test = _test
train_size, val_size, test_size = [len(y_train), len(y_val), len(y_test)]
# del _train, _val, _test

print('Available data samples:', len(y_train)+len(y_val)+len(y_test), end=', \t')
print('train:{}, val:{}, test:{}\n'.format(train_size, val_size, test_size))

# balance training dataset (oversampling)
x_train, ts_train, y_train = oversample_dataset(x_train, ts_train, y_train)
train_size = len(x_train)

# checksum
print('After balancing the training dataset,', end='\t')
print('train:{}, val:{}, test:{}\n'.format(train_size, val_size, test_size))

print('Check the first 20 labels of each dataset are consistent:')
print(y_train[0:20], '\n', y_val[0:20], '\n', y_test[0:20], '\n')

# make tensorflow style dataset for manual training
batch_size = 256
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, ts_train, y_train)).batch(batch_size).prefetch(2)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, ts_val, y_val)).batch(batch_size).prefetch(2)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, ts_test, y_test)).batch(batch_size).prefetch(2)


# models with timestamps
class TimeChangerCnn(tf.keras.Model):
  def __init__(self, vocab_size, embedding_matrix):
    super().__init__()
    
    embedding_dims = 16
    self.embed = tf.keras.layers.Embedding(
      vocab_size+1, embedding_dims,     # +1 because of padding 0
      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
      trainable=False, input_length=250)  # input_length=input_length)
    max_duration = TS_TRIM # seconds
    resolution = 0.1   # seconds
    self.resample = ResampleLayer(max_duration, resolution)
    self.cnn_model = tf.keras.models.Sequential([
      tf.keras.layers.Conv1D(filters=32, kernel_size=8, strides=3),
      tf.keras.layers.GlobalMaxPool1D(),
      tf.keras.layers.Dense(units=32, activation='relu'),
      tf.keras.layers.Dense(units=32, activation='relu'),
      tf.keras.layers.Dense(units=1, activation='sigmoid'),
    ])
    
  #@tf.function
  def call(self, data):
    in_seq, ts_seq = data
    in_seq = self.embed(in_seq)
    output = self.resample((in_seq, ts_seq))
    output = self.cnn_model(output)
    return output

model = TimeChangerCnn(vocab_size, embedding_matrix)
model((x_tok[0:2], x_ts[0:2]))
model.summary()

# define metrics, to display during training
train_loss = tf.keras.metrics.Mean()
train_metrics = {
    'accuracy': tf.keras.metrics.BinaryAccuracy(),
    'precision': tf.keras.metrics.Precision(),
    'recall': tf.keras.metrics.Recall(),
    }
val_metrics = {
    'loss': tf.keras.metrics.Mean(),
    'accuracy': tf.keras.metrics.BinaryAccuracy(),
    }
# training history and curve
history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}

# define other training parameters
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)   #Adam, SGD
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# start training
print('Start training with {} samples, validate with {} samples'.format(train_size, val_size))
epochs = 5   # will run 3 hours worst case.
for epoch in range(epochs):
  start = time()
  
  # reset metrics
  train_loss.reset_states()
  for metric in train_metrics.values():
    metric.reset_states()
  for metric in val_metrics.values():
    metric.reset_states()

  for x_batch, ts_batch, y_batch in train_dataset:
    with tf.GradientTape() as tape:     # watch_accessed_variables=False
      y_pred = model((x_batch, ts_batch))  # ((x_batch, ts_batch))
      y_pred = tf.squeeze(y_pred, 1)   # make it same dimension as y

      # Loss value for this minibatch
      loss = loss_fn(y_batch, y_pred)
      loss += sum(model.losses)  # must have, what does this do?
      
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # update metrics
    train_loss.update_state(loss)
    for metric in train_metrics.values():
        metric.update_state(y_batch, y_pred)
    # print('~', end='')
  
  # update val metrics
  if val_size != 0:
    for x_batch, ts_batch, y_batch in val_dataset:
      y_val_pred = model((x_batch, ts_batch))
      y_val_pred = tf.squeeze(y_val_pred, 1)
      loss = loss_fn(y_batch, y_val_pred)
      val_metrics['loss'].update_state(loss)
      val_metrics['accuracy'].update_state(y_batch, y_val_pred)
  
  # log 
  history['train_loss'].append(train_loss.result().numpy())
  history['train_acc'].append(train_metrics['accuracy'].result().numpy())
  history['val_loss'].append(val_metrics['loss'].result().numpy())
  history['val_acc'].append(val_metrics['accuracy'].result().numpy())

  print_train_info_v2(epoch, time()-start, history)

  # determine early stopping
  # sometimes the loss goes up for a short period, do nothing and wait.
  # stops only when it's absolutely flat.
  
  w = 5     # early stopping moving average window
  n = 5     # early stopping hesitate epochs
  thres = 0.0002        # loss difference threshold
  acc_thres = 0.0   # absolute loss threshold

  if epoch >= n+w+1:
    # loss moving average of the last few epochs
    loss_MA = [history['val_loss'][i-w:i] for i in range(epoch-n, epoch+1)]
    loss_MA = [sum(values) / w for values in loss_MA]
    acc_MA = [history['val_acc'][i-w:i] for i in range(epoch-n, epoch+1)]
    acc_MA = [sum(values) / w for values in acc_MA]
    # for the last n losses, must be going down, and diff < thres
    train_stopping = [abs(a-b)<=thres for (a,b) in zip(loss_MA[0:-1], loss_MA[1:])]
    if sum(train_stopping) == n and acc_MA[-1] >= acc_thres:
      print('Early stopping triggered.')
      print_train_info_v2(epoch, time()-start, history)
      break 
  
  # save weights every epoch
  if epoch % 10 == 0: model.save_weights('./checkpoints/my_checkpoint')

# print evaluation result
print('Evaluation on train dataset:')
(tr_acc, tr_p, tr_r) = naive_evaluate(model, train_dataset)
print('Evaluation on val dataset:')
(tr_acc, tr_p, tr_r) = naive_evaluate(model, val_dataset)
print('Evaluation on test dataset:')
(te_acc, te_p, te_r) = naive_evaluate(model, test_dataset)
