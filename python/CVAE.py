# -*- coding: utf-8 -*-
from losses import log_diag_mvn, kld_unit_mvn, binary_crossentropy, loss_q_logp, compute_mmd, log_normal_pdf
from Layers import EncoderGaussianCNN, DecoderBernoulliCNN
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

class CVAE(tf.keras.Model):
    def __init__(self,
                latent_dim=2,
                target_shape=(7,7,32),
                n_samples = 1,
                name='cvae',
                **kwargs):
        super(CVAE, self).__init__(name=name, **kwargs)
        
        self.encoder = EncoderGaussianCNN(latent_dim=latent_dim)
        self.decoder = DecoderBernoulliCNN(latent_dim=latent_dim,target_shape=target_shape)

    def call(self, inputs):
        # loss function for classifier
        z_mean, z_logvar, z    = self.encoder(inputs)
        x_logit = self.decoder(z[0])
        
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=inputs)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = -log_normal_pdf(z[0], 0., 0.)
        logqz_x = -log_normal_pdf(z[0], z_mean, z_logvar)

        self.loss =  -tf.reduce_mean(logpx_z + logpz - logqz_x)  

        self.params = self.decoder.trainable_variables + self.encoder.trainable_variables
        
        return {'logpxz':logpx_z, 'logpz':logpz, 'logqz_x':logqz_x} 

    # specifying comp graph to be used during testing
    def generate_images(self, inputs, epoch):
        fig = plt.figure(figsize=(4, 4))

        for i in range(inputs.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(inputs[i, :, :, 0], cmap='gray')
            plt.axis('off')

        # tight_layout minimizes the overlap between 2 sub-plots
        plt.savefig('../output/cnn_mnist/image_epoch_{:04d}.png'.format(epoch))

    @tf.function
    def train(self, x, optimizer):
        with tf.GradientTape() as tape:
            costs = self.call(x)
        gradients = tape.gradient(self.loss, self.params)
        optimizer.apply_gradients(zip(gradients, self.params))

        return costs, self.loss
