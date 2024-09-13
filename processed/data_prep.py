import pandas as pd
import numpy as np
from collections import OrderedDict
import re
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from sklearn.utils import resample, shuffle
from sklearn.model_selection import train_test_split


def load_HDFS(log_fn, label_fn):
    '''Load HDFS structured log into pandas dataframe. Structured log is produced by the parsing program.
    Original credit to LogPAI.
    (maybe?) This can be written as sklearn pipeline.
    
    Parameters
    ----------
    log_fn: str
    label_fn: str

    Returns
    ----------
    pandas.DataFrame: the data containing block ID, event sequence, timestamp sequence and label.
    '''
    # pandas, try to reduce memory usage
    struct_log = pd.read_csv(log_fn, engine='c', na_filter=False, memory_map=True, 
                         usecols=['LineId','Date','Time','Content','EventId'],
                         header=0, dtype={'Date':str, 'Time':str})
    
    # Convert Date Time col into str
    struct_log['ts'] = '20' + struct_log.Date + ' ' + struct_log.Time  # This piece is recorded in 2008
    struct_log.ts = pd.to_datetime(struct_log.ts, unit='ns')
    
    # Make sequence
    data_dict = OrderedDict()
    for idx, row in struct_log.iterrows():
        # find block ID
        blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
        blkId_set = set(blkId_list)

        # register block ID and date time
        # This loop is not very efficient
        for blk_Id in blkId_set:
            if not blk_Id in data_dict:
                data_dict[blk_Id] = ([],[])  # EventID, ts
            data_dict[blk_Id][0].append(row['EventId'])
            data_dict[blk_Id][1].append(row['ts'])
    data_list = [[k, v[0], v[1]] for (k,v) in data_dict.items()]
    data_df = pd.DataFrame(data_list, columns=['BlockId', 'EventSequence','TimeStamps'])
    
    # process labels
    label_data = pd.read_csv(label_fn, engine='c', na_filter=False, memory_map=True)
    label_data = label_data.set_index('BlockId')
    label_dict = label_data['Label'].to_dict()
    data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)
       
    return data_df


def _list_delta(li:'array-like'):
    '''Calculate the difference btw adjacent list elements. Fill return[0] with zero.'''
    li2 = li.copy()
    li2[1:] = li2[0:-1]
    return [v1 - v2 for v1, v2 in zip(li,li2)]


def _set_zero(array:'list-like'):
    '''Shift the numerical sequence together so that the first element is 0. 
    This function works on Timestamp.'''
    
    # Type of input data
    assert(type(array[0]) is not pd.Timedelta)
    if type(array[0]) is pd.Timestamp:
        zero = pd.Timestamp('1970-01-01 00:00:00') # 0 value in timestamp
    else:
        zero = 0.0

    diff = array[0] - zero
    return [val - diff for val in array]


def _interpolate_ts(array:'list-like'):
    '''Rearrange timestamps evenly within the same second.
    Since timestamp's resolution is only 1 sec, there are multiple ts having the same value.
    E.g. [0, 1, 1, 1, 2, 3, 4, 4] => [0, 1.0, 1.33, 1.67, 2, 3, 4, 4.5]
    '''
    
    tstack = []   # your regular stack
    output = []   # same size as input array
    
    # Doesn't work on Timedelta data
    assert(type(array[0]) is not pd.Timedelta)
    
    # resolution should be 1 sec
    if type(array[0]) is pd.Timestamp:
        res = pd.Timedelta('0 days 00:00:01')
    else:
        res = 1.0
    
    # process ts in the array
    for ts in array:
        # empty stack
        if len(tstack) == 0:
            tstack.append(ts)
            
        # same ts (within this 1 second)
        elif tstack[-1] == ts:
            tstack.append(ts)
            
        # new ts (next second)
        elif tstack[-1] != ts:
            n = len(tstack)
            output += [tstack[j] + j/n*res for j in range(n)]
            tstack = [ts]
            
        # shouldn't be here
        else:
            print('Error tstack value', tstack, ts)
        
    # The last second
    if len(tstack) != 0:
        n = len(tstack)
        output += [tstack[j] + j/n*res for j in range(n)]
    
    return output


class TimeStamper(BaseEstimator, TransformerMixin):
    '''Process timestamp: rearrange ts within the same second, convert to timedelta'''
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, df:'DataFrame') -> 'np.array':
        # Make a copy so don't mess with original df
        X = df.copy()
        # Re-arrange ts evenly within the same second
        X.TimeStamps = X.TimeStamps.apply(_interpolate_ts)
        # Get ts sequence shifted so that the first ts = 0, not sure if needed.
        X.TimeStamps = X.TimeStamps.apply(_set_zero) 
        return X


def tokenize(seq_arr:'Array of list', tokenizer=None):
    '''Convert a list of symbols into tokens.

    Parameters
    ----------
    seq_arr: array of list, each list is consist of symbolic elements (strings)

    tokenizer: use the given Keras tokenizer
    
    Returns
    ----------
    output: converted array of token, numpy array

    tokenizer: the Keras tokenizer used in this process.

    '''
    # convert array of symbols to string.
    seq_txt = _seq_to_text(seq_arr)
    
    if tokenizer is None:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(seq_txt)
        # vocab_size = len(tokenizer.word_counts)
        
    output = tokenizer.texts_to_sequences(seq_txt)

    return output, tokenizer


def _seq_to_text(seq_arr:'Array of list'):
    '''Connect the strings in each list, return a long string. '''
    corpus = []
    
    for seq in seq_arr:
        corpus.append(' '.join(seq))
    
    return corpus


def trim_time_sequence(seq_arr, ts_arr, ts_limit:'sec'):
    '''Trim the timestamped sequence to the given time limit.
    Parameters
    ----------
    seq_arr: numpy array of sequences. sequences are list of integers(tokens).
    ts_arr: numpy array of timestamps. timestamps are list of pd.datetime64.
    ts_limit: numerical value. unit is second.
    
    Returns
    -------
    seq_pad: (in-place update), numpy array of sequences.
    ts_pad: (in-place update), numpy array of timestamps.
    '''
    
    ts_limit = int(ts_limit)
    assert int(ts_limit)>0   # time limit must greater than 0
    
    span = np.datetime64('1970-01-01 00:00:00') + np.timedelta64(ts_limit, 's') 
    for i in range(len(ts_arr)):
        idx = _first_idx(ts_arr[i], span)
        if idx == 0:
            print('index #{}, limit is too small?'.format(i))
            raise Exception('All values are greater than the given limit.')
        ts_arr[i] = ts_arr[i][0:idx]
        seq_arr[i] = seq_arr[i][0:idx]
    # inplace update, no return


def remove_long_sequence(seq_arr, ts_arr, y, ts_limit:'sec'):
    '''remove long sequences that are over the given time limit.
    Parameters
    ----------
    seq_arr: numpy array of sequences. sequences are list of integers(tokens).
    ts_arr: numpy array of timestamps. timestamps are list of pd.datetime64.
    y: numpy array of labels. (0 or 1)
    ts_limit: numerical value. unit is second.
    
    Returns
    -------
    seq_fit, ts_fit, y_fit: 3 array with sequence length within the limit.
    '''
    
    seq_fit = []   # the new sequence data that fits the limit
    ts_fit = []    # the new ts data that fits the limit
    y_fit = []

    ts_limit = np.datetime64('1970-01-01 00:00:00') + np.timedelta64(ts_limit, 's') 

    for seq, ts, label in zip(seq_arr, ts_arr, y):
        # sequence duration too long
        if ts[-1] > ts_limit:
            continue
        # duration fit, keep this sample
        seq_fit.append(seq)
        ts_fit.append(ts)
        y_fit.append(label)
    
    # convert to numpy
    seq_fit = np.array(seq_fit, dtype=object)
    ts_fit = np.array(ts_fit, dtype=object)
    y_fit = np.array(y_fit)

    return (seq_fit, ts_fit, y_fit)


def _first_idx(sorted_list, thres):
    '''Find the first index exceedding threshold in the given sorted list.'''
    for i, val in enumerate(sorted_list):
        if val > thres: return i
    return len(sorted_list)


def pad_time_sequence(seq_arr, ts_arr, maxlen=50, ts_interval=1):
    '''Pad zeros at the beginning of token sequence; pad equal-interval timestamps for ts sequence.
    
    Parameters
    ----------
    seq_arr: numpy array of sequences. sequences are list of integers(tokens).
    ts_arr: numpy array of timestamps. timestamps are list of pd.datetime64.
    
    Returns
    -------
    seq_pad: 2d numpy array, padded sequence(integer)
    ts_pad: 2d numpy array, padded timestamp(np.datetime64)
    '''
    
    
    assert(float(ts_interval*1000).is_integer())    # only accepts ts_interval to be ms / 0.001s
    
    # padding 0 for sequence can be done with keras function
    seq_pad = pad_sequences(seq_arr, maxlen=maxlen)
    
    # padding timestamp    
    # Making a time sequence with equal interval.
    n_samples = len(ts_arr)
    interval = np.timedelta64(int(ts_interval*1000),'ms')
    ts_zero = np.datetime64('1970-01-01 00:00:00.000') # resolution: millisecond
    fill_value = [ts_zero + i*interval for i in range(maxlen)]
    ts_pad = np.full((n_samples, maxlen), fill_value=fill_value)

    #
    for i, ts_seq in enumerate(ts_arr):
        n = len(ts_seq)
        if n >= maxlen:
            # trim sequence
            ts_pad[i] = np.array(ts_seq[n-maxlen:n])
        else:
            # pad sequence
            ts_diff = ts_seq[0] - ts_pad[i, maxlen-n]
            ts_pad[i, maxlen-n:maxlen] = np.array(ts_seq) - ts_diff  # broadcasting
    
    ts_pad = ts_pad.astype('int')/1000    # milliseconds -> seconds
    ts_pad = ts_pad.astype('float32')
    return seq_pad, ts_pad


def prepare_dataset(x, ts, y, test_ratio=0.5, val_ratio=0.0, batch_size=256, seed=42):
    '''DO NOT USE. There is an unknown bug with tf sample_from_dataset.
    make tf.data stream dataset. output dataset is balanced with downsampling.
    
    https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    '''

    # Find total number of positive (abnormal) samples
    bool_labels = y != 0
    pos_size = sum(bool_labels)   # should be 16838
    total_size = pos_size * 2

    # Separate positive and negative samples
    x_pos = x[bool_labels]
    x_neg = x[~bool_labels]
    ts_pos = ts[bool_labels]
    ts_neg = ts[~bool_labels]
    y_pos = y[bool_labels]
    y_neg = y[~bool_labels]

    # Create Dataset instance for pos & neg.
    pos_ds = tf.data.Dataset.from_tensor_slices((x_pos, ts_pos, y_pos))
    neg_ds = tf.data.Dataset.from_tensor_slices((x_neg, ts_neg, y_neg))
    pos_ds = pos_ds.shuffle(buffer_size=total_size+1, seed=seed)
    neg_ds = neg_ds.shuffle(buffer_size=total_size+1, seed=seed)

    # Combine pos & neg dataset. Excessive negative data are dropped.
    balanced_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds],
                        weights=[0.5, 0.5], seed=seed)
    balanced_ds = balanced_ds.take(total_size)  # only the head of dataset is perfectly mixed.

    # train - test split
    if val_ratio == 0:
        train_size = int(total_size * (1 - test_ratio))
        train_dataset = balanced_ds.take(train_size).batch(batch_size).prefetch(2)
        test_dataset = balanced_ds.skip(train_size).batch(batch_size).prefetch(2)

        return train_dataset, test_dataset

    # train - val - test split
    else:
        train_size = int(total_size * (1 - test_ratio - val_ratio))
        val_size = int(total_size * val_ratio)
        train_dataset = balanced_ds.take(train_size+val_size)
        val_dataset = train_dataset.skip(train_size).batch(batch_size).prefetch(2)
        train_dataset = train_dataset.take(train_size).batch(batch_size).prefetch(2)
        test_dataset = balanced_ds.skip(train_size+val_size).batch(batch_size).prefetch(2)

        return train_dataset, val_dataset, test_dataset


def prepare_dataset_v2(x1, x2, y, test_ratio=0.5, val_ratio=0.0, batch_size=256, seed=42):
    '''Make balanced dataset using sklearn. The Tensorflow version is too confusing.

    Parameters
    ----------
    x1, x2: the two features.

    y: labels. Must be binary.

    test_ratio, val_ratio:

    Returns
    -------
    '''

    # Find total number of positive (abnormal) samples
    bool_labels = y != 0
    pos_size = sum(bool_labels)   # should be 16838
    total_size = pos_size * 2

    # Separate positive and negative samples
    x1_pos = x1[bool_labels]
    x1_neg = x1[~bool_labels]
    x2_pos = x2[bool_labels]
    x2_neg = x2[~bool_labels]
    y_pos = y[bool_labels]
    y_neg = y[~bool_labels]

    # resample the negative samples
    x1_neg, x2_neg, y_neg = resample(x1_neg, x2_neg, y_neg,
        n_samples=pos_size, replace=False, random_state=seed,
        ) # make sure replace is false so we don't sample twice.
    
    # combine pos and neg samples
    x1 = np.concatenate([x1_pos, x1_neg])
    x2 = np.concatenate([x2_pos, x2_neg])
    y = np.concatenate([y_pos, y_neg])

    # split and shuffle
    x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
        x1, x2, y, test_size=test_ratio, stratify=y,
        shuffle=True, random_state=seed)
    if val_ratio != 0:
        x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(
            x1_train, x2_train, y_train, test_size=val_ratio, stratify=y_train,
            shuffle=True, random_state=seed)

    if val_ratio != 0: 
        return ((x1_train, x2_train, y_train),
               (x1_val, x2_val, y_val),
               (x1_test, x2_test, y_test))
    else:
        return ((x1_train, x2_train, y_train),
               (x1_test, x2_test, y_test))


def prepare_dataset_v3(x1, x2, y, test_ratio=0.5, val_ratio=0.0, batch_size=256, seed=42):
    '''Make imbalanced dataset. Leave the balancing job to training.

    Parameters
    ----------
    x1, x2: the two features.

    y: labels. Must be binary.

    test_ratio: cannot be zero.
    
    val_ratio: the val size relative to the TRAIN set, 0% - 100%. can be zero.

    batch_size: no longer used.

    Returns
    -------
    '''
    # split and shuffle
    x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
        x1, x2, y, test_size=test_ratio, stratify=y,
        shuffle=True, random_state=seed)
    
    if val_ratio != 0:
        x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(
            x1_train, x2_train, y_train, test_size=val_ratio, stratify=y_train,
            shuffle=True, random_state=seed)

    if val_ratio != 0: 
        return ((x1_train, x2_train, y_train),
               (x1_val, x2_val, y_val),
               (x1_test, x2_test, y_test))
    else:
        return ((x1_train, x2_train, y_train),
               (x1_test, x2_test, y_test))


def oversample_dataset(x1, x2, y, classes=2, seed=42):
    '''
    Oversample an imbalanced dataset [x1, x2, y], to make it balanced (ratio=0.5)

    Assumptions: label 1 is the minority class.

    Parameters
    ----------
    x1, x2, y: the two features and label.

    classes: number of distinct labels (y). 
    Currently only accept two classes: 0,1
    
    '''

    np.random.seed(seed)
    bool_labels = y != 0
    # pos_size = sum(bool_labels)   # should be 16838
    
    x1_pos = x1[bool_labels]
    x1_neg = x1[~bool_labels]
    x2_pos = x2[bool_labels]
    x2_neg = x2[~bool_labels]
    y_pos = y[bool_labels]
    y_neg = y[~bool_labels]

    ids = np.arange(len(y_pos))
    choices = np.random.choice(ids, len(y_neg))

    # resample (oversample) dataset
    res_x1_pos = x1_pos[choices]
    res_x2_pos = x2_pos[choices]
    res_y_pos = y_pos[choices]

    # Concatenate and make dataset
    x1_res = np.concatenate([res_x1_pos, x1_neg], axis=0)
    x2_res = np.concatenate([res_x2_pos, x2_neg], axis=0)
    y_res = np.concatenate([res_y_pos, y_neg], axis=0)

    # shuffle
    order = np.arange(len(y_res))
    np.random.shuffle(order)
    x1_res = x1_res[order]
    x2_res = x2_res[order]
    y_res = y_res[order]

    return x1_res, x2_res, y_res

