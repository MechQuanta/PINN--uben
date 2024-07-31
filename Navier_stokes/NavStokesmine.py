import tensorflow as tf
import numpy as np


class PhysicsInformedNN:
    def __init__(self, x, y, t, u, v, layers, **kwargs):
        super(PhysicsInformedNN, self).__init__(**kwargs)
        X = np.concatenate([x, y, t], axis=1)
        self.lb = X.min(axis=0)
        self.X = X
        self.x = X[:, 0:1]
        self.y = X[:, 1:2]
        self.t = X[:, 2:3]
        self.u = u
        self.v = v
        self.layers = layers
        self.model = self.build_model(layers)
        self.x_tf = tf.convert_to_tensor(self.x)
        self.y_tf = tf.convert_to_tensor(self.y)
        self.t_tf = tf.convert_to_tensor(self.t)
        self.u_tf = tf.convert_to_tensor(self.u)
        self.v_tf = tf.convert_to_tensor(self.v)
        self.lambda1 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda2 = tf.Variable([0.0], dtype=tf.float32)
        self.loss = self.compute_loss(self.x_tf, self.y_tf, self.t_tf, self.u_tf, self.v_tf)

    def build_model(self, layers):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
        for layer in layers[1:-1]:
            model.add(tf.keras.layers.Dense(layer, activation="tanh"))
        model.add(tf.keras.layers.Dense(layers[-1]))
        return model

    def net_NS(self, x, y, t):
        lambda_1 = self.lambda1
        lambda_2 = self.lambda2

        X = tf.concat([x, y, t], axis=1)
        psi_and_p = self.model(X)
        psi = psi_and_p[:, 0:1]
        p = psi_and_p[:, 1:2]
        u = tf.gradients(psi, y)[0]
        v = -tf.gradients(psi, x)[0]
        u_t = tf.gradients(u, t)[0]
        v_t = tf.gradients(v, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]
        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]
        f_u = u_t + lambda_1 * (u * u_x + v * u_x) + p_x - lambda_2 * (u_xx + u_yy)
        f_v = v_t + lambda_1 * (u * v_x + v * v_y) + p_y - lambda_2 * (v_xx + v_yy)
        return u, v, p, f_u, f_v

    def compute_loss(self, x, y, t, u, v):
        u_pred, v_pred, p_pred, f_u_pred, f_v_pred = self.net_NS(x, y, t)
        loss = tf.reduce_mean(tf.square(u - u_pred)) + tf.reduce_mean(tf.square(v - v_pred)) + tf.reduce_mean(
            tf.square(f_u_pred)) + tf.reduce_mean(tf.square(f_v_pred))
        return loss
