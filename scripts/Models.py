from glob import glob
import argparse
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img
import tensorflow_addons as tfa
import os


def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
  """Upsamples an input.
  Conv2DTranspose => Batchnorm => Dropout => Relu
  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer
  Returns:
    Upsample Sequential Model
  """
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

  if norm_type.lower() == 'batchnorm':
    result.add(tf.keras.layers.BatchNormalization())
  elif norm_type.lower() == 'instancenorm':
    result.add(InstanceNormalization())

  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))
  result.add(tf.keras.layers.ReLU())

  return result


class unetModel():
    def __init__(self,out=5,shape=(384,640,3),fine_tune=True):
        self.out=out
        self.shape=shape
        base_model = tf.keras.applications.MobileNetV2(input_shape=shape, include_top=False)
        layer_names = ['block_1_expand_relu',   # 192x320
                        'block_3_expand_relu',   # 96x160
                        'block_6_expand_relu',   # 48x80
                        'block_13_expand_relu',  # 24x40
                        'block_16_project',      # 12x20
                        ]
        layers=[base_model.get_layer(layer).output for layer in layer_names]
        self.down_stack=tf.keras.Model(inputs=base_model.input,outputs=layers)
        if fine_tune:
            self.down_stack.trainable=False
        self.up_stack = [
                upsample(512, 3), 
                upsample(256, 3), 
                upsample(128, 3),  
                upsample(64, 3), 
                ]
    
    def get_unet(self):
        inputs = tf.keras.layers.Input(shape=[384, 640, 3])
        x = inputs
        # Downsampling through the model
        skips = self.down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])
        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])
        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(self.out, 3, strides=2,padding='same')  
        x = last(x)
        return tf.keras.Model(inputs=inputs, outputs=x)

def flatten_model(nested_model):
    layers_flat = []
    for layer in nested_model.layers:
      model_layers=[]
      if isinstance(layer,tf.keras.Model):
          for m_l in layer.layers:
            layers_flat.append(m_l)
      try:
          layers_flat.extend(layer.layers)
      except AttributeError:
          layers_flat.append(layer)
    model_flat = tf.keras.models.Sequential(layers_flat)
    return model_flat
    
if __name__=="__main__":
    model=unetModel()
    unet=model.get_unet()
    unet=flatten_model(unet)
    print(unet.summary())
    tf.keras.utils.plot_model(unet, show_shapes=True)   
    model.compile(optimizer=Adam(learning_rate=0.0001), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])      
       




