from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Reshape

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    x = self.d2(x)
    return x

# def get_loss(self,X,Y):
#     boom=self.l1(X)
#     boom1=self.l2(boom)
#     boom2=self.out(boom1)
#     return tf.math.square(boom2-Y)
    
# # get gradients
# def get_grad(self,X,Y):
#     with tf.GradientTape() as tape:
#         tape.watch(self.l1.variables)
#         tape.watch(self.l2.variables)
#         tape.watch(self.out.variables)
#         L = self.get_loss(X,Y)
#         g = tape.gradient(L, [self.l1.variables[0],self.l1.variables[1],self.l2.variables[0],self.l2.variables[1],self.out.variables[0],self.out.variables[1]])
#     return g