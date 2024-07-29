import tensorflow as tf
import numpy as np

gpus = tf.config.list_physical_devices('GPU')
class PhysicsInformedNN(tf.keras.Model):
    def __init__(self,x,y,t,u,v,layers,**kwaegs):
        super().__init__(**kwaegs)
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






def Initialize_NN(self,layers):
    weights = []
    bias = []
    layer_size = len(layers)
    for l in range(0 , layer_size-1):
        w = self.xavier_init(shape=[layers[l],layers[l+1]])
        b = tf.Variable(tf.zeros([1,layers[l+1]],dtype=tf.float32),dtype=tf.float32)
        weights.append(w)
        bias.append(b)
    def xavier_init(self,shape):
        in_dim = shape[0]
        out_dim = shape[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim,out_dim],stddev=xavier_stddev),dtype=tf.float32)
