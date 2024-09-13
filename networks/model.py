import tensorflow as tf
import numpy as np


def load_embeddings(filename):
    '''Load word2vec embeddings text file saved by gensim.
    Since the file is in text format, no gensim package needed.
    '''
    embeddings = {}
    vocab_size = 0
    embedding_dims = 0

    with open(filename, 'r') as f:
        # skip the first line
        vocab_size, embedding_dims = [int(v) for v in f.readline().split()]
        print('vocab_size, embedding_dims:', vocab_size, embedding_dims)

        # read every line and store as numpy array
        for line in f:
            values = line.split()
            word = values[0]
            embeddings[word] = np.array([float(v) for v in values[1:]])

    # double check
    if len(embeddings) != vocab_size:
        print('Error loading embedding file. Vocab size does not match:',
            len(embeddings), vocab_size)
    elif len(embeddings[word]) != embedding_dims:
        print('Error loading embedding file. Embedding size does not match:',
            len(embeddings[word]), embedding_dims)
    
    return embeddings, embedding_dims


def tensor_interp(x_new, x, y, method='linear'):
    '''Similar to np.interp, written in tensorflow, can handle multiple dims.
    Given discrete data points (x, y), evaluate at x_new. Tested on 1d, 2d and 3d.
    
    IMPORTANT: When run on some tf CPU version on windows, the output dim is incorrect.
    Parameters:
    -----------
    x: n-d tensor, the last dimension is a sequence of timestamps (discrete
    x values of a function).

    y: n-d tensor, the last dimension is a the function outputs.

    x_new: the new timestamps where the interpolation is evaluated.
    
    Returns:
    --------
    y_new: the evaluated values at x_new timestamps.
    '''
    # Find where in the orignal data, the values to interpolate
    # would be inserted.
    x_new_idx = tf.searchsorted(x, x_new)
    # clip the indices so that the lo won't go to -1
    if method=='zoh':
        x_new_idx = tf.clip_by_value(x_new_idx, 1, x.shape[-1])
    else:
        x_new_idx = tf.clip_by_value(x_new_idx, 1, x.shape[-1]-1)
    # calculate the slope of regions
    lo = x_new_idx - 1
    hi = x_new_idx
    
    # determine batch or single
    batch_dims = -1
    if x.ndim == 1: batch_dims = 0

    # for zero-order hold method
    if method=='zoh':
        return tf.gather(y, lo, batch_dims=batch_dims)

    # gather neighbouring values for every new point.
    x_lo = tf.gather(x, lo, batch_dims=batch_dims)
    x_hi = tf.gather(x, hi, batch_dims=batch_dims)
    y_lo = tf.gather(y, lo, batch_dims=batch_dims)
    y_hi = tf.gather(y, hi, batch_dims=batch_dims)
    
    # There might be a bug on tf CPU version
    warning_msg = 'tf.gather() output dimension error. should be x_new.shape = '
    warning_msg += str(x_new_idx.shape) + ', get tf.gather() shape ' + str(x_lo.shape)
    warning_msg += '. Could be a tensorflow bug on CPU version.'
    assert x_lo.ndim == x.ndim, warning_msg
    
    # the slope btw two data points
    slope = (y_hi - y_lo) / (x_hi - x_lo)
    
    # calculate the actual value for each entry x_new
    y_new = slope*(x_new - x_lo) + y_lo
    
    # not doing bounds yet
    return y_new


def tensor_interp2d(x_new, x, y, extrpl='hold'):
    '''Do interpolation on the last dimension of a 2d matix.
    
    Parameters:
    -----------    
    x_new: see tensor_interp.
    x: see tensor_interp.
    y: see tensor_interp.
    extrpl: choose how to deal with exptrapolation on the right side. Assume no extrapolation on the left side.
    - 'hold': hold the last value in the original data
    - 'linear': linear extrapolation based on the last two points.
    Returns:
    --------
    y_new: the evaluated values at x_new timestamps.
    '''
    # Method to do hold-value extrapolation:
    # create an artificial point at the end of each signal (last dimension)
    if extrpl=='hold':
        y_last = y[:,-1:]       # select the last value of the last dimension
        x_last = x[:,-1:] + 1   # shape=(same, same, 1)
        x = tf.concat([x, x_last], axis=-1)
        y = tf.concat([y, y_last], axis=-1)
    elif extrpl=='linear':
        pass
    else:
        print('Error: parameter extrpl can only be "hold" or "linear". Using linear as default.')

    return tensor_interp(x_new, x, y)


def tensor_interp3d(x_new, x, y, extrpl='hold'):
    '''Do interpolation on the last dimension of a 3d matix.
    
    Parameters:
    -----------    
    x_new: see tensor_interp.
    x: see tensor_interp.
    y: see tensor_interp.
    extrpl: choose how to deal with exptrapolation on the right side. Assume
    no extrapolation on the left side.
    - 'hold': hold the last value in the original data
    - 'linear': linear extrapolation based on the last two points.
    Returns:
    --------
    y_new: the evaluated values at x_new timestamps.
    '''
    # Method to do hold-value extrapolation:
    # create an artificial point at the end of each signal (last dimension)
    if extrpl=='hold':
        y_last = y[:,:,-1:]       # select the last value of the last dimension
        x_last = x[:,:,-1:] + 1   # shape=(same, same, 1)
        x = tf.concat([x, x_last], axis=-1)
        y = tf.concat([y, y_last], axis=-1)
    elif extrpl=='linear':
        pass
    else:
        print('Error: parameter extrpl can only be "hold" or "linear". Using linear as default.')

    return tensor_interp(x_new, x, y)


class ResampleLayer_old(tf.keras.layers.Layer):
    '''Resample the input signal (value, ts) into evenly spaced samples. Throw away timestamps. 
    Then pad value 0.0 at the end of the singal.

    'old' means this one does for loop instead of parallel computing. so it's slow.
    '''
    def __init__(self, duration=100, resolution=0.1):
        super(ResampleLayer, self).__init__()
        self.res = resolution      # resolution, in seconds
        self.duration = duration   # length, in seconds
        self.out_length = int(duration/resolution) + 1   # length of signal, unitless
        
    # def build(self, input_shape):
    
    def call(self, batch_in):
    
        embed_batch, ts_batch = batch_in
        assert(len(embed_batch.shape) == 3)    # (batch_size, seq_length, embed_dims)
        batch_size = embed_batch.shape[0]
        seq_length = embed_batch.shape[1]
        embed_dims = embed_batch.shape[2]
        
        # reference timestamps
        new_ts = tf.linspace(0., self.duration, num=self.out_length)
        
        # iterate through each sample.
        batch_out = []
        for sample, ts in zip(tf.transpose(embed_batch, perm=[0,2,1]), ts_batch):
            # sample: (embed_dims, seq_length)
            sample_out = []
            # iterate through each embedding dimensions.
            # Can be processed as batch?
            for embedding in sample:
                # this function process 1d tensor only
                embedding_out = tensor_interp(new_ts, ts, embedding)
                sample_out.append(embedding_out)
            
            sample_out = tf.stack(sample_out, axis=0)
            batch_out.append(sample_out)
        
        return tf.transpose(tf.stack(batch_out,axis=0), perm=(0,2,1))


class ResampleLayer(tf.keras.layers.Layer):
    '''Resample the input signal (value, ts) into evenly spaced samples. Throw away timestamps.
    
    Parameters
    ----------
    duration: the time duration of resampled signal, in integer seconds.
    resolution: the resolution of resampled signal, in float seconds.

    '''
    def __init__(self, duration=100, resolution=0.1):
        super(ResampleLayer, self).__init__()
        self.res = resolution      # resolution, in seconds
        self.duration = duration   # length, in seconds
        self.out_length = int(duration/resolution) + 1   # length of signal, unitless
        
    # def build(self, input_shape):
    
    def call(self, batch_in):
    
        embed_batch, ts_batch = batch_in

        # embed_batch.shape: (batch_size, seq_length, embed_dims)
        # ts_batch.shape: (batch_size, seq_length)
        assert len(embed_batch.shape) == 3, 'input data should be 3 dims.'
        batch_size = embed_batch.shape[0]
        seq_length = embed_batch.shape[1]
        embed_dims = embed_batch.shape[2]
        
        # reference timestamps
        ts_new = tf.linspace(0., self.duration, num=self.out_length)
        ts_new = tf.expand_dims(ts_new, axis=0)
        ts_new = tf.expand_dims(ts_new, axis=0)
        ts_new = tf.tile(ts_new, tf.TensorShape([batch_size,embed_dims,1])) # (batch_size,embed_dims,new_length)
        
        # adjust the dimensions of inputs
        ts_batch = tf.expand_dims(ts_batch, axis=1)  # (batch_size,1,seq_length)
        ts_batch = tf.tile(ts_batch, [1,embed_dims,1]) # (batch_size,embed_dims,seq_length)
        embed_batch = tf.transpose(embed_batch, perm=[0,2,1]) # (batch_size,embed_dims,seq_length)

        # iterate through each sample.
        batch_out = tensor_interp3d(ts_new, ts_batch, embed_batch)
        # batch_out = tensor_interp(ts_new, ts_batch, embed_batch, method='zoh')
        
        return tf.transpose(tf.stack(batch_out,axis=0), perm=(0,2,1)) # (batch_size, seq_length, embed_dims)


class FFTLayer(tf.keras.layers.Layer):
    
    def __init__(self, clip=(0,10)):
        super(FFTLayer, self).__init__()
        # Add self-defined weights
        self.clip_min = clip[0]
        self.clip_max = clip[1]
        
    def call(self, inputs):
        '''Do FFT calculation.
        A transpose is needed as tf.signal.fft() only works on the inner most dimension. 
        
        Issue: directly call this layer and using batch training give different-sized input.
        Built-in layers does not seem to have this issue.
        '''
        assert inputs.ndim == 3
        inputs = tf.transpose(inputs, perm=[0,2,1])
        spectrum = tf.signal.rfft(inputs)
        spectrum = tf.abs(spectrum)
        spectrum = tf.transpose(spectrum, perm=[0,2,1])
        spectrum = tf.clip_by_value(spectrum, self.clip_min, self.clip_max)
        return spectrum


class TimeChangerLstm(tf.keras.Model):
    def __init__(self, vocab_size):
        super().__init__()
        
        embedding_dims = 16
        self.embed = tf.keras.layers.Embedding(vocab_size+1,    # +1 because of padding 0
                          embedding_dims)  # input_length=input_length
        max_duration = 100 # seconds
        resolution = 0.1   # seconds
        self.resample = ResampleLayer(max_duration, resolution)
        self.conv1d = tf.keras.layers.Conv1D(filters=32, kernel_size=8, strides=3)  # input_shape=(None,21,1)
        self.lstm1 = tf.keras.layers.GRU(units=128, dropout=0.0, return_sequences=True)
        self.lstm2 = tf.keras.layers.GRU(units=128, dropout=0.0, return_sequences=False)
        #self.lstm3 = tf.keras.layers.LSTM(units=32, dropout=0.0)
        # self.pool1d = tf.keras.layers.GlobalMaxPool1D()
        self.fc1 = tf.keras.layers.Dense(units=64, activation='tanh')  # input_shape=input_shape
        #self.fc2 = tf.keras.layers.Dense(units=32, activation='tanh')
        self.final = tf.keras.layers.Dense(units=1, activation='sigmoid')
        
    #@tf.function
    def call(self, data):
        
        in_seq, ts_seq = data
        in_seq = self.embed(in_seq)
        output = self.resample((in_seq, ts_seq))
        output = self.conv1d(output)
        output = self.lstm1(output)
        output = self.lstm2(output)
        #output = self.lstm3(output)
        output = self.fc1(output)
        #output = self.fc2(output)
        output = self.final(output)
        return output


class TimeChanger_FFT(tf.keras.Model):
    '''Add FFT based on TimeChanger.'''
    def __init__(self, vocab_size):
        super().__init__()
        
        embedding_dims = 16
        self.embed = tf.keras.layers.Embedding(vocab_size+1,    # +1 because of padding 0
                          embedding_dims)  # input_length=input_length
        max_duration = 100 # seconds
        resolution = 0.1   # seconds
        self.resample = ResampleLayer(max_duration, resolution)
        self.fft = FFTLayer(clip=(0,20))   # no clip
        self.conv1d = tf.keras.layers.Conv1D(filters=32, kernel_size=8, strides=3)  # input_shape=(None,21,1)
        self.pool1d = tf.keras.layers.GlobalMaxPool1D()
        self.hid1 = tf.keras.layers.Dense(units=32, activation='relu')  # input_shape=input_shape
        self.hid2 = tf.keras.layers.Dense(units=32, activation='relu')
        self.final = tf.keras.layers.Dense(units=1, activation='sigmoid')
        
    #@tf.function
    def call(self, data):
        
        in_seq, ts_seq = data
        in_seq = self.embed(in_seq)
        output = self.resample((in_seq, ts_seq))
        output = self.fft(output)
        output = self.conv1d(output)
        output = self.pool1d(output)
        output = self.hid1(output)
        output = self.hid2(output)
        output = self.final(output)
        return output


class TimeChanger(tf.keras.Model):
    def __init__(self, vocab_size):
        super().__init__()
        
        embedding_dims = 16
        self.embed = tf.keras.layers.Embedding(vocab_size+1,    # +1 because of padding 0
                          embedding_dims)  # input_length=input_length
        max_duration = 100 # seconds
        resolution = 0.1   # seconds
        self.resample = ResampleLayer(max_duration, resolution)
        self.conv1d = tf.keras.layers.Conv1D(filters=32, kernel_size=8, strides=3)  # input_shape=(None,21,1)
        self.pool1d = tf.keras.layers.GlobalMaxPool1D()
        self.hid1 = tf.keras.layers.Dense(units=32, activation='relu')  # input_shape=input_shape
        self.hid2 = tf.keras.layers.Dense(units=32, activation='relu')
        self.final = tf.keras.layers.Dense(units=1, activation='sigmoid')
        
    #@tf.function
    def call(self, data):
        
        in_seq, ts_seq = data
        in_seq = self.embed(in_seq)
        output = self.resample((in_seq, ts_seq))
        output = self.conv1d(output)
        output = self.pool1d(output)
        output = self.hid1(output)
        output = self.hid2(output)
        output = self.final(output)
        return output


class TimeChanger_old(tf.keras.Model):
    ''' _old means this one does for loop instead of parallel computing.'''
    def __init__(self, vocab_size):
        super().__init__()
        
        embedding_dims = 16
        self.embed = tf.keras.layers.Embedding(vocab_size+1,    # +1 because of padding 0
                          embedding_dims)  # input_length=input_length
        max_duration = 100 # seconds
        resolution = 0.1   # seconds
        self.resample = ResampleLayer_old(max_duration, resolution)
        self.conv1d = tf.keras.layers.Conv1D(filters=32, kernel_size=16)  # input_shape=(None,21,1)
        self.pool1d = tf.keras.layers.GlobalMaxPool1D()
        self.hid1 = tf.keras.layers.Dense(units=32, activation='relu')  # input_shape=input_shape
        self.hid2 = tf.keras.layers.Dense(units=32, activation='relu')
        self.final = tf.keras.layers.Dense(units=1, activation='sigmoid')
        
    #@tf.function
    def call(self, data):
        
        in_seq, ts_seq = data
        in_seq = self.embed(in_seq)
        output = self.resample((in_seq, ts_seq))
        output = self.conv1d(output)
        output = self.pool1d(output)
        output = self.hid1(output)
        output = self.hid2(output)
        output = self.final(output)
        return output


def regular_1dcnn(vocab_size):
    '''Simple NN that does embedding and 1d convolution, as a baseline.'''
    embedding_dims = 16
    return tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size+1, embedding_dims),   # +1 because of padding 0
        tf.keras.layers.Conv1D(filters=32, kernel_size=8, strides=3),
        tf.keras.layers.GlobalMaxPool1D(),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid'),
    ])


def regular_1dcnn_lstm(vocab_size):
    '''Simple NN that does embedding and 1d convolution + LSTM, as a baseline.'''
    embedding_dims = 16
    return tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size+1, embedding_dims),   # +1 because of padding 0
        tf.keras.layers.Conv1D(filters=32, kernel_size=8, strides=3),
        tf.keras.layers.LSTM(units=128, return_sequences=True),
        tf.keras.layers.LSTM(units=64, return_sequences=False),
        tf.keras.layers.Dense(units=32, activation='tanh'),
        tf.keras.layers.Dense(units=1, activation='sigmoid'),
    ])


def regular_lstm(vocab_size):
    '''Simple NN that does embedding and LSTM, as a baseline.'''
    embedding_dims = 16
    return tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size+1, embedding_dims),   # +1 because of padding 0
        tf.keras.layers.LSTM(units=64, return_sequences=True),
        tf.keras.layers.LSTM(units=32, return_sequences=False),
        tf.keras.layers.Dense(units=32, activation='tanh'),
        tf.keras.layers.Dense(units=1, activation='sigmoid'),
    ])


def naive_evaluate(model, dataset):
    metrics = {
        'accuracy': tf.keras.metrics.BinaryAccuracy(),
        'precision': tf.keras.metrics.Precision(),
        'recall': tf.keras.metrics.Recall(),
        }
    for x_batch, ts_batch, y_batch in dataset:
        y_pred = model((x_batch, ts_batch))
        y_pred = tf.squeeze(y_pred, 1)
    
        for metric in metrics.values():
            metric.update_state(y_batch, y_pred)

    acc = metrics['accuracy'].result().numpy()
    p = metrics['precision'].result().numpy()
    r = metrics['recall'].result().numpy()
    print('Acc: {:.2%}, P: {:.2%}, R: {:.2%}'.format(acc, p, r))
    
    return (acc, p, r)

