import tensorflow as tf
import numpy as np

gpus = tf.config.list_physical_devices('GPU')


class PhysicsInformedNN:
    def __init__(self, x, y, t, u, v, layers):
        super().__init__()
        X = np.concatenate([x, y, t], axis=1)
        self.lb = X.min(0)
        self.ub = X.max(0)
        self.x = X[:, 0:1]
        self.y = X[:, 1:2]
        self.t = X[:, 2:3]
        self.u = u
        self.v = v
        self.layers = layers
        self.lambda1 = tf.Variable([0.0],dtype=tf.float32)
        self.lambda2 = tf.Variable([0.0],dtype=tf.float32)
        self.weights, self.bias = self.Initialize_NN(layers)
        self.x_tf = tf.convert_to_tensor(self.x, dtype=tf.float32)
        self.y_tf = tf.convert_to_tensor(self.y, dtype=tf.float32)
        self.t_tf = tf.convert_to_tensor(self.t, dtype=tf.float32)
        self.u_tf = tf.convert_to_tensor(self.u, dtype=tf.float32)
        self.v_tf = tf.convert_to_tensor(self.v, dtype=tf.float32)

        self.u_tf = tf.convert_to_tensor(self.u, dtype=tf.float32)
        self.v_tf = tf.convert_to_tensor(self.v, dtype=tf.float32)
        self.loss = self.compute_loss(self.x_tf, self.y_tf, self.t_tf, self.u_tf, self.v_tf)
        self.adam_optimizer = tf.keras.optimizers.Adam()

    def Initialize_NN(self, layers):
        weights = []
        bias = []
        layer_size = len(layers)
        for l in range(0, layer_size - 1):
            w = self.xavier_init(shape=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(w)
            bias.append(b)
        return weights, bias

    def xavier_init(self, shape):
        in_dim = shape[0]
        out_dim = shape[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, bias):
        layer = len(weights) + 1
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, layer - 1):
            W = weights[l]
            b = bias[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = bias[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H

    def net_NS(self, x, y, t):
        lambda_1 = self.lambda1
        lambda_2 = self.lambda2
        psi_and_p = self.neural_net(tf.concat([x, y, t], axis=1), self.weights, self.bias)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y, t])
            psi = self.neural_net(tf.concat([x, y, t], axis=1), self.weights, self.bias)[:, 0:1]
            u = tape.gradients(psi, y)[0]
            v = - tape.gradients(psi, x)[0]
            p = self.neural_net(tf.concat([x, y, t], axis=1), self.weights, self.bias)[:, 1:2]
            u_t = tape.gradients(u, t)[0]
            v_t = tape.gradients(v, t)[0]
            u_x = tape.gradients(u, x)[0]
            u_y = tape.gradients(u, y)[0]
            u_xx = tape.gradients(u_x, x)[0]
            u_yy = tape.gradients(u_y, y)[0]
            v_x = tape.gradients(v, x)[0]
            v_y = tape.gradients(v, y)[0]
            v_xx = tape.gradients(v_x, x)[0]
            v_yy = tape.gradients(v_y, y)[0]
            p_x = tape.gradients(p, x)[0]
            p_y = tape.gradients(p, y)[0]
        del tape
        f_u = u_t + lambda_1 * (u * u_x + v * u_y) + p_x - lambda_2 * (u_xx + u_yy)
        f_v = v_t + lambda_1 * (u * v_x + v * v_y) + p_y - lambda_2 * (v_xx + v_yy)
        return u, v, p, f_u, f_v

    def compute_loss(self, x, y, t, u, v):
        u_pred, v_pred, p_pred, f_u_pred, f_v_pred = self.net_NS(x, y, t)
        loss = tf.reduce_mean(tf.square(u - u_pred)) + tf.reduce_mean(tf.square(v - v_pred)) + tf.reduce_mean(
            tf.square(f_u_pred)) + tf.reduce_mean(tf.square(f_v_pred))
        return loss

    def train(self,loss):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(self.x_tf,self.y_tf,self.t_tf,self.u_tf,self.v_tf)
            gradient = tape.gradients(loss, self.trainable_variables + [self.lambda1,self.lamdba2])
            self.adam_optimizer.apply_gradients(zip(gradient,self.trainable_variables + [self.lambda1,self.lamdba2]))
        if

