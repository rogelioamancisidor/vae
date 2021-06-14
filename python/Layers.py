# -*- coding: utf-8 -*-
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf 
import numpy as np

class HiddenBlock(layers.Layer):
    def __init__(self, layers_size, dropout_rates, activation = 'softplus'):
        super(HiddenBlock, self).__init__()
        self.activation = activation

        nlayers = len(layers_size)
        _hiddenLayer = [] 
        _hiddenLayer.append(layers.Dense(layers_size[0],activation=self.activation))
        if dropout_rates[0] > 0:
            _hiddenLayer.append(layers.Dropout(dropout_rates[0]))

        for i in range(nlayers-1):
            _hiddenLayer.append(layers.Dense(layers_size[i+1],activation=self.activation))
            if dropout_rates[i+1] > 0:
                _hiddenLayer.append(layers.Dropout(dropout_rates[i+1]))
        self.hidden_layers = Sequential(_hiddenLayer)

    def call(self, inputs, training=True):
        # make the forward pass
        x = self.hidden_layers(inputs, training=training)
        return x

class Sampling(layers.Layer):
    """ Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs, n_samples = 1):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        samples = []

        for i in range(n_samples):
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            sample = z_mean + tf.exp(0.5 * z_log_var) * epsilon
            samples.append(sample)
        
        return samples

class EncoderGaussianCNN(layers.Layer):
    def __init__(self,
                 input_shape = (28,28,1),
                 filters     = 32,
                 kernel_size = 3,
                 strides     = (2,2),
                 activation  = 'relu',
                 latent_dim  = 20,
                 n_samples   = 1,
                 name        = 'encoderCNN',
                 **kwargs):
        super(EncoderGaussianCNN,self).__init__(name=name, **kwargs)
        self.hidden_layers = Sequential(
            [
            layers.InputLayer(input_shape=input_shape),
            layers.Conv2D(
                filters=filters,   kernel_size=kernel_size, strides=strides, activation=activation),
            layers.Conv2D(
                filters=2*filters, kernel_size=kernel_size, strides=strides, activation=activation),
            layers.Flatten(),
            ]
            )
        self.mu       = layers.Dense(latent_dim)
        self.log_var  = layers.Dense(latent_dim)
        self.sampling = Sampling()
        self.n_samples = n_samples

    def call(self, inputs):
        x = self.hidden_layers(inputs)
        z_mean = self.mu(x)
        z_log_var = self.log_var(x)
        z = self.sampling((z_mean, z_log_var), n_samples = self.n_samples)

        return z_mean, z_log_var, z

class DecoderBernoulliCNN(layers.Layer):
    def __init__(self,
                 latent_dim,
                 target_shape,
                 activation='relu',
                 name = 'decoderCNN',
                 **kwargs):
        super(DecoderBernoulliCNN,self).__init__(name=name, **kwargs)
        self.hidden_layers = Sequential(
                [
                layers.InputLayer(input_shape=(latent_dim,)),
                layers.Dense(units=7*7*32, activation=activation),
                layers.Reshape(target_shape=target_shape),
                layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same',
                    activation=activation),
                layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same',
                    activation=activation)
                ]
                )
        self.mu = layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')
    
    def call(self, inputs, apply_sigmoid=False):
        x = self.hidden_layers(inputs)
        x_recon = self.mu(x)
        if apply_sigmoid:
            probs = tf.sigmoid(x_recon)
            return probs
        return x_recon

class DecoderGaussianCNN2(layers.Layer):
    def __init__(self,
                 latent_dim,
                 target_shape,
                 activation='relu',
                 name = 'decoderCNN',
                 **kwargs):
        super(DecoderGaussianCNN2,self).__init__(name=name, **kwargs)
        units = np.prod(target_shape) 
        self.hidden_layers = Sequential(
                [
                layers.InputLayer(input_shape=(latent_dim,)),
                layers.Dense(units=units, activation=activation),
                layers.Reshape(target_shape=target_shape),
                layers.Conv2DTranspose(
                    filters=32*4, kernel_size=4, strides=1, padding='valid',
                    activation=activation),
                layers.Conv2DTranspose(
                    filters=32*2, kernel_size=4, strides=2, padding='same',
                    activation=activation),
                layers.Conv2DTranspose(
                    filters=32, kernel_size=4, strides=2, padding='same',
                    activation=activation)
                ]
                )
        self.mu1 = layers.Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding='same')
        self.mu2 = layers.Conv2DTranspose(filters=1, kernel_size=4, strides=2, padding='same')
    
    def call(self, inputs, apply_sigmoid=False):
        x = self.hidden_layers(inputs)
        
        x1_recon = self.mu1(x)
        x2_recon = self.mu2(x)

        if apply_sigmoid:
            probs1 = tf.sigmoid(x1_recon)
            probs2 = tf.sigmoid(x2_recon)
            return probs1, probs2
        else:
            return x1_recon, x2_recon

class DecoderGaussianCNN(layers.Layer):
    def __init__(self,
                 latent_dim,
                 target_shape,
                 activation='relu',
                 name = 'decoderCNN',
                 **kwargs):
        super(DecoderGaussianCNN,self).__init__(name=name, **kwargs)
        units = np.prod(target_shape) 
        self.hidden_layers = Sequential(
                [
                layers.InputLayer(input_shape=(latent_dim,)),
                layers.Dense(units=units, activation=activation),
                layers.Reshape(target_shape=target_shape),
                layers.Conv2DTranspose(
                    filters=32*4, kernel_size=4, strides=1, padding='valid',
                    activation=activation),
                layers.Conv2DTranspose(
                    filters=32*2, kernel_size=4, strides=2, padding='same',
                    activation=activation),
                layers.Conv2DTranspose(
                    filters=32, kernel_size=4, strides=2, padding='same',
                    activation=activation)
                ]
                )
        self.mu = layers.Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding='same')
        #self.log_var = layers.Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding='same')
    
    def call(self, inputs, apply_sigmoid=False):
        x = self.hidden_layers(inputs)
        
        x_recon = self.mu(x)

        if apply_sigmoid:
            probs = tf.sigmoid(x_recon)
            return probs
        else:
            return x_recon
        '''
        x_mu = self.mu(x)
        x_log_var = self.log_var(x)
        epsilon = tf.keras.backend.random_normal(shape=x_mu.shape)
        sample = x_mu + tf.exp(0.5 * x_log_var) * epsilon
        return x_mu, x_log_var, sample
        '''

class EncoderGaussian(layers.Layer):
    def __init__(self,
                latent_dim,
                layers_size=[64],
                dropout_rates=[0],
                n_samples = 1,
                name='encoder',
                activation = 'softplus',
                **kwargs):
        super(EncoderGaussian, self).__init__(name=name, **kwargs)
        self.hidden_layers = HiddenBlock(layers_size, dropout_rates, activation=activation)
        self.mu = layers.Dense(latent_dim)
        self.log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()
        self.n_samples = n_samples

    def call(self, inputs, training=True):
        x = self.hidden_layers(inputs,training=training)
        z_mean = self.mu(x)
        z_log_var = self.log_var(x)
        z = self.sampling((z_mean, z_log_var), n_samples = self.n_samples)

        return z_mean, z_log_var, z

# Use inheritance to define DecoderGaussian  
class DecoderGaussian(EncoderGaussian):
    def __init__(self,
                latent_dim,
                layers_size=[64],
                dropout_rates=[0],
                n_samples=1,
                name='decoder',
                activation = 'softplus',
                **kwargs):
        EncoderGaussian.__init__(self,latent_dim,name=name, **kwargs)

class DecoderBernoulli(layers.Layer):
    def __init__(self,
                original_dim,
                layers_size=[64],
                dropout_rates=[0],
                name='decoder',
                activation = 'softplus',
                **kwargs):
        super(DecoderBernoulli, self).__init__(name=name, **kwargs)
        self.hidden_layers = HiddenBlock(layers_size,dropout_rates, activation=activation)
        self.mu = layers.Dense(original_dim, activation='sigmoid')

    def call(self, inputs, training=True):
        x = self.hidden_layers(inputs, training=training)
        x_recon = self.mu(x)
        return x_recon

class CLS(layers.Layer):
    def __init__(self,
                y_dim,
                layers_size=[64],
                dropout_rates=[0],
                name='cls',
                activation = 'softplus',
                **kwargs):
        super(CLS, self).__init__(name=name, **kwargs)
        self.hidden_layers = HiddenBlock(layers_size,dropout_rates,activation=activation)
        self.classifier = layers.Dense(y_dim, activation='softmax')

    def call(self, inputs, training=True):
        x = self.hidden_layers(inputs, training=training)
        pi_hat = self.classifier(x)
        return pi_hat
