import tensorflow as tf
import tensorflow.keras as K
import numpy as np
from tensorflow.keras.activations import softmax
from tensorflow.keras.losses import categorical_crossentropy
from PIL import Image


def custom_weighted_crossentropy(weights):
    weights=tf.constant(weights,dtype=tf.float32)
    def loss(y_true,y_pred):
        y_pred=tf.clip_by_value(y_pred,tf.keras.backend.epsilon(),1-tf.keras.backend.epsilon())
        loss = y_true * tf.math.log(y_pred) * weights
        loss=-1*tf.keras.backend.sum(loss, axis=-1)
        return loss
    return loss


@tf.function  # The decorator converts one hot into a `Function`.
def onehot(y_true, n_classes=5):
    s=y_true.shape
    #Squueeze if the tensor is already 4 dimneional
    if (len(s)==4):
        y_true=tf.squeeze(y_true,-1)
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), n_classes)
    y_true = tf.cast(y_true, tf.float32)
    return y_true


def custom_sparse_weighted_crossentropy(weights):
    weights=tf.constant(weights,dtype=tf.float32)
    def loss(y_true,y_pred):
        y_true=onehot(y_true)
        y_pred=softmax(y_pred)
        y_pred=tf.clip_by_value(y_pred,tf.keras.backend.epsilon(),1-tf.keras.backend.epsilon())
        loss = y_true * tf.math.log(y_pred) * weights
        loss=-1*tf.keras.backend.sum(loss, axis=-1)
        loss=tf.keras.backend.mean(loss)
        return loss
    return loss


if __name__=="__main__":
    samples=1
    h=2
    w=2
    c=3
    # prediction tensor
    y_pred_n=np.random.random((samples,h,w,c)).astype(np.float32)
    y_pred= tf.Variable(y_pred_n)
    y_pred=softmax(y_pred)

    # Ground truth tensor
    y_true_n = np.random.random((samples,h,w,c)).astype(np.float32)
    y_true = tf.Variable(y_true_n)
    y_true = softmax(y_true)

    # Weighted loss
    custom_weighted_loss=custom_weighted_crossentropy([1.0,1,1])
    wc_loss=custom_weighted_loss(y_true,y_pred).numpy()
    print(wc_loss)
    #im = Image.open("/home/asad/ld/labels/seq_1_0000.png").resize((512,512))
    #im = tf.keras.utils.to_categorical(im,5)
    #keras loss
    cross_loss=tf.keras.backend.categorical_crossentropy(y_true, y_pred).numpy()
    np.testing.assert_almost_equal(wc_loss,cross_loss,decimal=3)
    print(f"Tested Categorical cross entropy OK ")
    # sparse categorical cross entropy
    y_pred_n=np.random.random((samples,512,512,5)).astype(np.float32)
    y_pred= tf.Variable(y_pred_n)
    #y_pred=softmax(y_pred)
    #y_true = Image.open("/home/asad/ld/labels/seq_1_0000.png").resize((512,512))
    y_true=tf.keras.preprocessing.image.load_img("/home/asad/ld/labels/seq_1_0000.png",color_mode="grayscale",target_size=(512,512))
    y_true=np.array(y_true)
    y_true=y_true[tf.newaxis,...]
    csce=custom_sparse_weighted_crossentropy([1.0,1,1,1,1])
    csce=csce(y_true,y_pred)
    sparse_ce=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    cl=sparse_ce(y_true,y_pred)
    cl=cl.numpy()
    np.testing.assert_almost_equal(csce,cl,decimal=3)
    print(f"Tested custom sparse categorical cross entropy OK ")

    
