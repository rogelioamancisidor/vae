from CVAE import CVAE 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
from utils import read_dset
import json
import os
# supress all tf messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset", default="mnist",help="Name of the data set", type=str)
    parser.add_argument("--outfile", default="cnn_mnist",help="Name of the output folder", type=str)
    parser.add_argument("--epochs", default=10,help="Name of the output folder", type=int)
    parser.add_argument("--no_runs", default=1,help="Name of the output folder", type=int)
    args = parser.parse_args()
    print (args)

    train_size = 60000
    batch_size = 32
    test_size = 10000

    # Iterate over epochs.
    start = time.time()
    
    output_folder = "../output/"+str(args.outfile)
    try:
        os.mkdir(output_folder)
    except OSError:
        print ("Creation of the directory %s failed" % output_folder)
    else:
        print ("Successfully created the directory %s " % output_folder)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss_metric = tf.keras.metrics.Mean()
    
    epochs = args.epochs 

    for r in range(args.no_runs):
        print('run cv %s out of %s' % (r+1, args.no_runs))
        print('building CBMD...')
        cvae = CVAE()
        

        print('loading data')
        # load data
        (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
        train_images = preprocess_images(train_images)
        test_images  = preprocess_images(test_images)
        
        train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                 .shuffle(train_size).batch(batch_size))
        test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                .shuffle(test_size).batch(batch_size))

        for epoch in range(epochs):
            for i, train_x in enumerate(train_dataset):
                costs, obj_fn = cvae.train(train_x,optimizer)

                loss_metric(obj_fn) # this saves the average loss during training

            if epoch % 1 == 0:
                print('epoch {}: loss {}'.format(epoch,loss_metric.result()))
    
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:16, :, :, :]
    z_mean, z_logvar, z = cvae.encoder(test_sample)
    x_recon  = cvae.decoder(z[0],apply_sigmoid=True)
    cvae.generate_images(x_recon, epoch+1)


if __name__ == "__main__":
    train()
