import scipy.io as sio
import tensorflow as tf
import numpy as np
import random

class DataSet(object):
    def __init__(self, images1, images2, labels, to_onehot=False, ydim=None, dtype=tf.float32,dset='mnist'):
        """Construct a DataSet.
        `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)
        
        assert images1.shape[0] == labels.shape[0], (
            'images1.shape: %s labels.shape: %s' % (images1.shape,
                                                    labels.shape))
        assert images2.shape[0] == labels.shape[0], (
            'images2.shape: %s labels.shape: %s' % (images2.shape,
                                                    labels.shape))
        self._num_examples = images1.shape[0]
        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        #assert images.shape[3] == 1
        #images = images.reshape(images.shape[0],
        #                        images.shape[1] * images.shape[2])
        if dtype == tf.float32 and images1.dtype != np.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            print("type conversion view 1")
            images1 = images1.astype(np.float32)
        
        if dtype == tf.float32 and images2.dtype != np.float32:
            print("type conversion view 2")
            images2 = images2.astype(np.float32)
        
        if to_onehot:
            #reshape first, so convert to one_hot
            # labels dimension should be [no obserations , ydim]
            if dset == 'mnist':
                # labels in mnist start at 1 and not 0, and are a column vector
                labels = labels.reshape(labels.shape[0],) - 1
            labels = tf.keras.utils.to_categorical(labels, num_classes=ydim)


        self._images1 = images1
        self._images2 = images2
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
    
    @property
    def view1(self):
        return self._images1
    
    @property
    def view2(self):
        return self._images2
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images1 = self._images1[perm]
            self._images2 = self._images2[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        
        end = self._index_in_epoch
        return self._images1[start:end], self._images2[start:end], self._labels[start:end]

def read_dset(dset):
    import numpy as np
    
    if dset=='mnist':
        data=sio.loadmat('../data/MNIST.mat')
    
        train = DataSet(data['X1'],data['X2'],data['trainLabel'],to_onehot=True,ydim=10,dset='mnist')
        val   = DataSet(data['XV1'],data['XV2'],data['tuneLabel'],to_onehot=True,ydim=10,dset='mnist')
        test  = DataSet(data['XTe1'],data['XTe2'],data['testLabel'],to_onehot=True,ydim=10,dset='mnist')
    
    
    return train, val, test

def batch_first(data):
    new_data = np.zeros((data.shape[3],data.shape[0],data.shape[1],data.shape[2]))
    for i in range(data.shape[3]):
        new_data[i,:,:,:] = data[:,:,:,i]
    return new_data

def fixlabels(x):
    same_format = [1+i if i <9 else 0 for i in x]
    return same_format

def fsorted(listval):
    from operator import itemgetter
    indices, L_sorted = zip(*sorted(enumerate(listval), key=itemgetter(1)))
    return list(L_sorted), list(indices)

def match_3m(target_val, values):
    from collections import Counter

    # sort both lists
    t_v, t_i = fsorted(target_val)
    s_v, s_i = fsorted(values)

    # counts. This are the number of obs that we need per digit 
    counts = list(Counter(t_v).values())

    need_idx = []
    for i in np.unique(t_v):
        l_idx = [j for j, e in enumerate(values) if e == i]
        count = counts[i]
        l_idx = l_idx[:count]

        # get values that you need
        need_idx.extend(l_idx)
    # the 1st indices tells where to find the same digit in the 2 list (values)
    # the 2nd indices tells where to marge them to obtain a 3 modal data set
    return need_idx, t_i

def load_mnist_3m(path='../data/',nr_tr=40000,nr_te=8000):
    from scipy.io import loadmat
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit
    import random
    from sklearn import preprocessing
    from sklearn.utils import shuffle

    mnist_tr_idx = np.load('../data/mnist_idx_tr.npy')
    mnist_te_idx = np.load('../data/mnist_idx_te.npy')
    svhn_tr_idx = np.load('../data/svhn_idx_tr.npy')
    svhn_te_idx = np.load('../data/svhn_idx_te.npy')
    

    shvm_path = path+'mnist_shvn/'
    shvm_te = loadmat(shvm_path+'test_32x32.mat')
    x2_te = shvm_te['X']
    x2_te = batch_first(x2_te)
    shvm_tr = loadmat(shvm_path+'train_32x32.mat')
    x2_tr = shvm_tr['X']
    x2_tr = batch_first(x2_tr)

    x1_tr = np.load(path+'mnist_tr.npy')
    x1_te = np.load(path+'mnist_te.npy')
    y_tr = np.load(path+'mnist_y_tr.npy')
    y_te = np.load(path+'mnist_y_te.npy')
    
    # reshape mnist
    x1_te = x1_te.reshape((x1_te.shape[0], x1_te.shape[1]*x1_te.shape[2])) 
    x1_tr = x1_tr.reshape((x1_tr.shape[0], x1_tr.shape[1]*x1_tr.shape[2])) 
    
    #scale 
    x3_tr = x2_tr/255.
    x3_te = x2_te/255.
    x1_tr = x1_tr/255.
    x1_te = x1_te/255.
    #x3_tr = np.where(x3_tr > .5, 1.0, 0.0).astype('float32')
    #x3_te = np.where(x3_te > .5, 1.0, 0.0).astype('float32')
    
    # randomly selected idx
    idx_tr_val = random.sample(range(svhn_tr_idx.shape[0]),nr_tr)
    idx_te_val = random.sample(range(svhn_te_idx.shape[0]),nr_te)

    # get labels
    y_tr_f = y_tr[mnist_tr_idx[idx_tr_val]]
    y_te_f = y_te[mnist_te_idx[idx_te_val]]

    # load mnist VCCA version to use as 3rd modality
    # this is the rotated version of the digit
    data=sio.loadmat('../data/MNIST.mat')
    x2_tr = data['X1']
    y2_tr = data['trainLabel']
    y2_tr = y2_tr-1
    x2_te = data['XTe1']
    y2_te = data['testLabel']
    y2_te = y2_te-1
    y2_tr = fixlabels(y2_tr)
    y2_te = fixlabels(y2_te)
    
    need_idx, t_i = match_3m(y_tr_f, y2_tr)
    # get vales and labels in sorted fashion
    s_v = x2_tr[need_idx]
    # add indexing from y_tr_f
    s_v_o = np.c_[t_i,s_v]
    # sort based on t_i 
    x2_tr = s_v_o[s_v_o[:,0].argsort()][:,1:]

    need_idx, t_i = match_3m(y_te_f, y2_te)
    # get vales and labels in sorted fashion
    s_v = x2_te[need_idx]
    # add indexing from y_tr_f
    s_v_o = np.c_[t_i,s_v]
    # sort based on t_i 
    x2_te = s_v_o[s_v_o[:,0].argsort()][:,1:]

    all_data = (x1_tr[mnist_tr_idx[idx_tr_val],:].astype(np.float32),
                x2_tr.astype(np.float32),
                x3_tr[svhn_tr_idx[idx_tr_val],:].astype(np.float32),
                tf.keras.utils.to_categorical(y_tr[mnist_tr_idx[idx_tr_val]], num_classes=10),
                x1_te[mnist_te_idx[idx_te_val],:].astype(np.float32),
                x2_te.astype(np.float32),
                x3_te[svhn_te_idx[idx_te_val],:].astype(np.float32),
                tf.keras.utils.to_categorical(y_te[mnist_te_idx[idx_te_val]], num_classes=10),
                )

    return all_data

def read_xrmb(nr_tr_ids = 12,nr_te_ids = 2):

    data1=sio.loadmat('data/XRMB1.mat')
    data2=sio.loadmat('data/XRMB2.mat')
   
    id_tr  = data2['trainID']
    id_te  = data2['testID']
    id_val = data2['tuneID']
    
    idx_tr  = random.sample(np.unique(data2['trainID']),nr_tr_ids)
    idx_cls = idx_tr[0:2]
    idx_tr  = idx_tr[2:]
    print ('using ids ', idx_tr, ' for training vcca')
    print ('using ids ', idx_cls, ' for mlp')
    idx_te = random.sample(np.unique(data2['testID']),nr_te_ids)
    print ('using ids ', idx_te, ' for testing')
    idx_val = random.sample(np.unique(data2['tuneID']),nr_te_ids)

    bool_tr  = np.array([any(idx_tr==i) for i in id_tr])
    bool_cls = np.array([any(idx_cls==i) for i in id_tr])
    bool_te  = np.array([any(idx_te==i) for i in id_te])
    bool_val = np.array([any(idx_val==i) for i in id_val])

    train=DataSet(data1['X1'][bool_tr,:], data2['X2'][bool_tr,:], data2['trainLabel'][bool_tr,:])
    
    cls=DataSet(data1['X1'][bool_cls,:],data2['X2'][bool_cls,:],data2['trainLabel'][bool_cls,:])
    
    tune=DataSet(data1['XV1'][bool_val,:],data2['XV2'][bool_val,:],data2['tuneLabel'][bool_val,:])
    
    test=DataSet(data1['XTe1'][bool_te,:],data2['XTe2'][bool_te,:],data2['testLabel'][bool_te,:])
    
    return train, tune, test, cls

if __name__ == "__main__":
    load_mnist_3m()
