# -*- coding: utf-8 -*-
from losses import log_diag_mvn, kld_unit_mvn, binary_crossentropy, loss_q_logp, compute_mmd, log_normal_pdf
from Layers import EncoderGaussian, DecoderBernoulli, CLS, DecoderGaussian
import tensorflow as tf 
import numpy as np

class VAE(tf.keras.Model):
    def __init__(self,
                dim_x,
                layers_size_enc  =[1024,1024,1024],
                layers_size_dec  =[1024,1024,1024],
                dropout_enc=[0.2,0.2,0.2],
                dropout_dec=[0.2,0.2,0.2],
                latent_dim=30,
                n_samples = 1,
                name='vae',
                decoder_type ='Gaussian',
                **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
       
        # q(z|x)
        self.encoder = EncoderGaussian(latent_dim, layers_size=layers_size_enc, dropout_rates=dropout_enc, n_samples=n_samples)
        
        # p(x|z)
        self.decoder_type = decoder_type
        if self.decoder_type == 'Bernoulli':
            self.decoder = DecoderBernoulli(dim_x, layers_size=layers_size_dec, dropout_rates=dropout_dec)
        elif self.decoder_type == 'Gaussian':
            self.decoder = DecoderGaussian(dim_x, layers_size=layers_size_dec, dropout_rates=dropout_dec)

    def call(self, inputs, training=True):
        # loss function for classifier
        z_mean, z_log_var, z    = self.encoder(inputs, training=training)

        kl_loss = kld_unit_mvn(z_mean, z_log_var)
        
        _recon_loss = 0 
        if self.decoder_type == 'Bernoulli':
            for i in range(len(z)):
                input_to_decoder = z[i]
                x2_hat = self.decoder(input_to_decoder, training=training)
                _recon_loss += log_diag_mvn(x2,x2_hat)
        elif self.decoder_type == 'Gaussian': 
            for i in range(len(zpost)):
                input_to_decoder = z[i]
                x_mu, x_log_var, x2_hat = self.decoder(input_to_decoder, training=training)
                _recon_loss += log_diag_mvn(x_mu,tf.math.exp(x_log_var))(x2)
        
        recon_loss = tf.reduce_mean(_recon_loss)
        
        self.vae_loss =  kl_loss + recon_loss  

        self.params = self.decoder.trainable_variables + self.encoder.trainable_variables
        
        
        return {'kl':kl_loss,'recon_loss': recon_loss, 'vae_loss':self.vae_loss}

    # specifying comp graph to be used during testing
    def test(self, inputs, training=False):
        z_mean, z_log_var, z = self.encoder(inputs, training=training)

        if self.decoder_type == 'Bernoulli':
            x2_hat = self.decoder(z[0], training=training)
        elif self.decoder_type == 'Gaussian':
            # use mu as x2_hat
            x2_hat,_,_ = self.decoder(z[0], training=training)
        
        return {'x2_hat': x2_hat, 'z': z}

    @tf.function
    def train(self, x, optimizer):
        with tf.GradientTape() as tape:
            costs = self.call(x)
        gradients = tape.gradient(self.vae_loss, self.params)
        optimizer.apply_gradients(zip(gradients, self.params))

        return costs
