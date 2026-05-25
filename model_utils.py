import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class EfficientNetPreprocessing(tf.keras.layers.Layer):
    def call(self, x):
        return tf.keras.applications.efficientnet.preprocess_input(x)