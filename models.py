import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def dense_net_imagenet_base_model_non_trainable():
    base_model = keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_tensor=None, input_shape=(1024, 1024, 3), pooling="avg", classes=1000)
    input_tensor = Input((1024, 1024, 3))
    output_tensor = base_model(input_tensor)
    op = Dense(12, activation = "softmax")(output_tensor)
    model = Model(input_tensor, op)
    base_model.trainable = False
    adam = Adam(lr = 1e-5)
    model.compile(loss = "sparse_categorical_crossentropy", optimizer = adam, metrics=['accuracy'])

    return model

def dense_net_imagenet_end2end():
    base_model = keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_tensor=None, input_shape=(1024, 1024, 3), pooling="avg", classes=1000)
    input_tensor = Input((1024, 1024, 3))
    output_tensor = base_model(input_tensor)
    op = Dense(12, activation = "softmax")(output_tensor)
    model = Model(input_tensor, op)
    # model.load_weights("put the model_weights path here")
    adam = Adam(lr = 1e-5)
    model.compile(loss = "sparse_categorical_crossentropy", optimizer = adam, metrics=['accuracy'])

    return model

def dense_net_end2end():
    base_model = keras.applications.densenet.DenseNet121(include_top=False, weights=None, input_tensor=None, input_shape=(1024, 1024, 3), pooling="avg", classes=1000)
    input_tensor = Input((1024, 1024, 3))
    output_tensor = base_model(input_tensor)
    op = Dense(12, activation = "softmax")(output_tensor)
    model = Model(input_tensor, op)
    model.load_weights("put the model_weights path here")
    adam = Adam(lr = 1e-5)
    model.compile(loss = "sparse_categorical_crossentropy", optimizer = adam, metrics=['accuracy'])

    return model