model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(name='LambdaBlock01'),
    tf.keras.layers.Dense(128, name='Linear02'),
    tf.keras.layers.Activation(tf.keras.activations.relu, name='LambdaBlock03'),
    tf.keras.layers.Dense(64, name='Linear04'),
    tf.keras.layers.Activation(tf.keras.activations.relu, name='LambdaBlock05'),
    tf.keras.layers.Dense(10, name='Linear06')
], name='outputs')

loss = tf.keras.losses.categorical_crossentropy
