import tensorflow as tf
import numpy as np

gpus = tf.config.list_physical_devices('GPU')
class PhysicsInformedNN(tf.keras.Model):
    def __init__(self,x,y,t,u,v,layers,**kwaegs):
        super(PhysicsInformedNN,self).__init__(**kwaegs)
        X = np.concatenate([x,y,t],axis=1)
        self.lb = X.min(0)
        self.ub = X.max(0)
        self.x = X[:,0:1]
        self.y = X[:,1:2]
        self.t = X[:,2:3]
        self.u = u
        self.v = v
        self.layers = layers
        self.weights , self.bias = self.Initialize_NN(layers)




    def Initializer_NN(self,layers):
        layer_size = len(layers)
        layer_list =[]
        for l in range(0,layer_size-1):
            layer = tf.keras.layers.Dense(
                units = layers[l+1],
                input_shape = (layers[l],),
                kernel_initializer = tf.keras.initializers.GlorotNormal(),
                bias_initializer = tf.zeros_initializer()
            )
            layer_list.append(layer)
        return layer_list
    def call(self,X):
        for layer in self.layer_list:
            X = layer(X)
        return X
