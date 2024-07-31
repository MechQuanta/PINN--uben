import tensorflow as tf
import numpy as np

class PhysicsInformedNN:
    def __init__(self,x,y,t,u,v,layers):
        X = np.concatenate([x,y,t],axis=1)
        self.lb = X.min(axis=0)
        self.ub = X.max(axis=0)
        self.x = X[:,0:1]
        self.y = X[:,1:2]
        self.t = X[:,2:3]

        self.u = u
        self.v = v

        self.lambda1 = tf.Variable([0.0],dtype= tf.float32)
        self.lambda2 = tf.Variable([0.0],dtype= tf.float32)
        self.layers = layers

        self.model = self.build_model()

    def build_model(self):
        inputs = tf.keras.Input(shape=(self.x.shape[1]+self.y.shape[1]+self.t.shape[1],))
        H = 2 * (inputs - self.lb)/(self.ub - self.lb) - 1
        for layer in self.layers[1:-1]:
            H = tf.keras.layers.Dense(units=layer,activation = 'tanh')(H)
        outputs = tf.keras.layers.Dense(units=self.layers[-1],activation = 'tanh')(H)
        model = tf.keras.Model(inputs=inputs,outputs=outputs)
        return model

    def net_NS(self,x,y,t):
        with tf.GradientTape() as tape:
            tape.watch([x,y,t])
            inputs = tf.concat([x,y,t],axis=1)
            psi_and_p = self.model(inputs)
            psi = psi_and_p[:,0:1]
            p = psi_and_p[:,1:2]
            u = tape.gradient(psi,y)
            v = -tape.gradient(psi,x)
            u_t = tape.gradient(u,t)
            u_x = tape.gradient(u,x)
            u_y = tape.gradient(u,y)
            u_xx = tape.gradient(u_x,x)
            u_yy = tape.gradient(u_y,y)
            v_t = tape.gradient(v,t)
            v_x = tape.gradient(v,x)
            v_y = tape.gradient(v,y)
            v_xx = tape.gradient(v_x,x)
            v_yy = tape.gradient(v_y,y)
            p_x = tape.gradient(p,x)
            p_y = tape.gradient(p,y)

        f_u = u_t + self.lambda1*(u*u_x + v*u_y) + p_x - self.lambda2 * (u_xx + u_yy)
        f_v = v_t +self.lambda1*(u*v_x + v*v_y) + p_y - self.lambda2 * (v_xx + v_yy)
        del tape
        return u,v,p,f_u,f_v
    def compute_loss(self,x,y,t,u,v):
        u_pred, v_pred, _ , f_u_pred , f_v_pred = self.net_NS(x,y,t)
        loss = tf.reduce_mean(tf.square(u-u_pred)) + tf.reduce_mean(tf.square(v-v_pred)) + tf.reduce_mean(tf.square(f_u_pred)) + tf.reduce_mean(tf.square(f_v_pred))
        return loss
    def train(self,nIter):
        optimizer = tf.keras.optimizers.Adam()
        x = tf.convert_to_tensor(self.x,dtype=tf.float32)
        y = tf.convert_to_tensor(self.y,dtype=tf.float32)
        t = tf.convert_to_tensor(self.t,dtype=tf.float32)
        u = tf.convert_to_tensor(self.u,dtype=tf.float32)
        v = tf.convert_to_tensor(self.v,dtype=tf.float32)
        for it in range(nIter):
            with tf.GradientTape() as tape:
                loss = self.compute_loss(x,y,t,u,v)
                gradients = tape.gradient(loss,self.model.trainable_variables+[self.lambda1+self.lambda2])
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables+[self.lambda1+self.lambda2]))
            if it%10 == 0 :
                lambda1_value = self.lambda1.numpy()
                lambda2_value = self.lambda2.numpy()
                print(f"Loss : {loss.numpy()} , lambda_1 : {lambda1_value : 0.4f} , lambda2 : {lambda2_value : 0.4f}")

    def predict(self,x_star,y_star,t_star):
        x_star = tf.convert_to_tensor(x_star,dtype=tf.float32)
        y_star = tf.convert_to_tensor(y_star,dtype=tf.float32)
        t_star = tf.convert_to_tensor(t_star,dtype=tf.float32)
        u_star , v_star , p_star , _ , _ = self.net_NS(x_star,y_star,t_star)




