# -*- coding: utf-8 -*-
from VAE import VAE
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

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim_x", default= 784,help="Dimensionality of x", type=int)
    parser.add_argument("--layers_size_enc", default=[2500,2500,2500],nargs='+', help="No of units in the hidden layer", type=int)
    parser.add_argument("--layers_size_dec", default=[1500,1500,1500],nargs='+', help="No of units in the hidden layer", type=int)
    parser.add_argument("--dropout_enc", default= [0.2,0.2,0.2],nargs='+',help="Dropout rate to be used in all hidden layers", type=float)
    parser.add_argument("--dropout_dec", default= [0.2,0.2,0.2],nargs='+',help="Dropout rate to be used in all hidden layers", type=float)
    parser.add_argument("--latent_dim", default= 50,help="Dimensionality of the latent space", type=int)
    parser.add_argument("--n_samples", default= 1,help="No of samples that the encoder draws", type=int)
    parser.add_argument("--no_runs", default= 1,help="No of cross-validations to run", type=int)
    parser.add_argument("--epochs", default= 1001, help="No of samples that the encoder draws", type=int)
    parser.add_argument("--batch_size", default= 100, help="No of samples that the encoder draws", type=int)
    parser.add_argument("--decoder_type", default="Gaussian",help="Name of the distribution assumed in the decoder", type=str)
    parser.add_argument("--dset", default="mnist",help="Name of the data set", type=str)
    parser.add_argument("--outfile", default="mnist",help="Name of the output folder", type=str)
    args = parser.parse_args()
    print (args)

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
        print('building VAE...')
        VAE = VAE(args.dim_x,
                    layers_size_enc=args.layers_size_enc,
                    layers_size_dec=args.layers_size_dec,
                    dropout_enc=args.dropout_enc,
                    dropout_dec=args.dropout_dec,
                    latent_dim=args.latent_dim,
                    n_samples=args.n_samples,
                    decoder_type=args.decoder_type,
                    name='vae',
                    )
        print('loading data')
        # load data
        trainData, tuneData, testData = read_dset(args.dset)
        BUFFER_SIZE = trainData.view1.shape[0]
        tr_data = tf.data.Dataset.from_tensor_slices(trainData.x).shuffle(BUFFER_SIZE).batch(args.batch_size)

        for epoch in range(epochs):
            for i, x in enumerate(tr_data):
                costs = vae.train(x,optimizer)

                loss_metric(costs['vae_loss']) # this saves the average loss during training


            if epoch % 100 == 0:
                print ('epoch %s: loss %0.4f' % (epoch,costs['vae_loss'])) 

        cbmd.save_weights(output_folder+'/checkpoint')

        # clean up
        #tf.keras.backend.clear_session()
        #del cbmd
    
    with open(output_folder +'/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    elapsed_time = time.time() - start
    print ('time elapsed %f' % elapsed_time)
    
if __name__ == "__main__":
    train()

