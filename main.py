import numpy as np
import tensorflow as tf
from model import MyModel
from loss import BiTemeredLoss
from sklearn.preprocessing import OneHotEncoder
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# add salt
np.random.seed(2021)
shaffle_len = int(0.15*len(y_train))
shaffle_idxs = np.random.choice(range(len(y_train)), size=shaffle_len, replace=False)
y_train[shaffle_idxs] = np.random.randint(0, 10, shaffle_len)

# onehot
onehot = OneHotEncoder()
onehot.fit(np.array([y_train.tolist() + y_test.tolist()]).reshape(-1, 1))
y_train, y_test = onehot.transform(y_train.reshape(-1, 1)).toarray(), onehot.transform(y_test.reshape(-1, 1)).toarray()

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# Create an instance of the model
model = MyModel()
loss_object_bi = BiTemeredLoss(t1=0.2, t2=4.0, multi=True)
loss_object_ce = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

# define training and testing steps 
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    # loss = loss_object_ce(labels, predictions)
    loss = loss_object_bi(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  predictions = model(images, training=False)
  t_loss = loss_object_ce(labels, predictions)
  test_loss(t_loss)
  test_accuracy(labels, predictions)

# start training 
EPOCHS = 5
for epoch in range(EPOCHS):
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'
  )