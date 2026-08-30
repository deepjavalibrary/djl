inputs = tf.keras.layers.InputLayer(input_shape = (28, 28)).output
LambdaBlock01 = tf.keras.layers.Flatten(name='LambdaBlock01')(inputs)
Linear02 = tf.keras.layers.Dense(128, name='Linear02')(LambdaBlock01)
LambdaBlock03 = tf.keras.layers.Activation(tf.keras.activations.relu, name='LambdaBlock03')(Linear02)
Linear04 = tf.keras.layers.Dense(64, name='Linear04')(LambdaBlock03)
LambdaBlock05 = tf.keras.layers.Activation(tf.keras.activations.relu, name='LambdaBlock05')(Linear04)
Linear06 = tf.keras.layers.Dense(10, name='Linear06')(LambdaBlock05)
outputs = Linear06
model = tf.keras.Model(inputs=inputs, outputs=outputs)

loss = tf.keras.losses.categorical_crossentropy
