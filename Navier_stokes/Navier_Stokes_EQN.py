import tensorflow as tf
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

physical_gpus = tf.config.list_physical_devices('GPU')
physical_gpus


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
        u_pred, v_pred, p_pred, f_u_pred , f_v_pred = self.net_NS(x,y,t)
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


if __name__ == "__main__":
    N_train = 5000
    layers = [3,20,20,20,20,20,20,20,20,2]
    data = scipy.io.loadmat('/home/sajid/PycharmProjects/NavierStokes/PINN-NavierStokes/data/cylinder_nektar_wake.mat')
    """{'__header__': b'MATLAB 5.0 MAT-file Platform: posix, Created on: Fri Sep 22 23:02:15 2017', '__version__': '1.0', '__globals__': [], 'X_star': array([[ 1.        , -2.        ],
       [ 1.07070707, -2.        ],
       [ 1.14141414, -2.        ],
       ...,
       [ 7.85858586,  2.        ],
       [ 7.92929293,  2.        ],
       [ 8.        ,  2.        ]]), 't': array([[ 0. ],
       [ 0.1],
       [ 0.2],
       [ 0.3],
       [ 0.4],
       [ 0.5],
  
       [ 2.9],
       [ 3. ],
       [ 3.1],
       [ 3.2],
       [ 4.6],
       [ 4.7],
       [ 4.8],
       [ 4.9],
       [ 5. ],
       [ 5.1],
       [ 5.2],
       [ 5.3],
       [ 5.4],
       [ 5.5],
       [ 5.6],
       [ 5.7],
       [ 5.8],
       [ 5.9],
       [ 6. ],
       [ 6.1],
       [ 6.2],
       [ 6.3],
       [ 6.4],
       [ 6.5],
       [ 6.6],
       [ 6.7],
       [ 6.8],
       [ 6.9],
       [ 7. ],
       [ 7.1],
       [ 7.2],
       [ 7.3],
       [ 7.4],
       [ 7.5],
       [ 7.6],
       [ 7.7],
       [ 7.8],
       [ 7.9],
       [ 8. ],
       [ 8.1],
       [ 8.2],
       [ 8.3],
       [ 8.4],
       [ 8.5],
       [ 8.6],
       [ 8.7],
       [ 8.8],
       [ 8.9],
       [ 9. ],
       .
       .
       .
   
       [19.6],
       [19.7],
       [19.8],
       [19.9]]), 'U_star': array([[[ 1.11419216e+00,  1.11755721e+00,  1.11956933e+00, ...,
          1.16105653e+00,  1.16209316e+00,  1.16287446e+00],
        [-4.09649775e-03,  1.09212754e-03,  3.28231641e-03, ...,
         -1.14679740e-02, -1.46226631e-02, -1.78649950e-02]],

       [[ 1.11102707e+00,  1.11424311e+00,  1.11623549e+00, ...,
          1.16119046e+00,  1.16250501e+00,  1.16355997e+00],
        [ 3.93130751e-04,  6.09231658e-03,  8.54240777e-03, ...,
         -3.69331211e-03, -6.90960992e-03, -1.02375054e-02]],

       [[ 1.10748452e+00,  1.11049848e+00,  1.11244362e+00, ...,
          1.16080365e+00,  1.16241577e+00,  1.16375737e+00],
        [ 4.42777717e-03,  1.06585027e-02,  1.33831464e-02, ...,
          4.14245483e-03,  8.86993440e-04, -2.50609725e-03]],

       ...,

       [[ 1.04183122e+00,  1.07052197e+00,  1.08570786e+00, ...,
          1.03561691e+00,  1.01482352e+00,  9.95607444e-01],
        [-1.53233747e-01, -1.52906494e-01, -1.50934775e-01, ...,
          1.70410243e-01,  1.74234015e-01,  1.74978509e-01]],

       [[ 1.03144583e+00,  1.05940483e+00,  1.07424671e+00, ...,
          1.05086925e+00,  1.02921296e+00,  1.00907934e+00],
        [-1.52570327e-01, -1.53932097e-01, -1.52973975e-01, ...,
          1.65974219e-01,  1.72307867e-01,  1.75206081e-01]],

       [[ 1.02128975e+00,  1.04848021e+00,  1.06294889e+00, ...,
          1.06667284e+00,  1.04430400e+00,  1.02333582e+00],
        [-1.51330184e-01, -1.54252780e-01, -1.54219310e-01, ...,
          1.59519620e-01,  1.68613705e-01,  1.73914810e-01]]]), 'p_star': array([[-0.10815457, -0.11115509, -0.1123679 , ..., -0.10035143,
        -0.09845376, -0.09658143],
       [-0.10560371, -0.10881921, -0.1101648 , ..., -0.10137256,
        -0.09959008, -0.09781616],
       [-0.10257633, -0.10599348, -0.10746686, ..., -0.10204929,
        -0.10039933, -0.09874095],
       ...,
       [ 0.0066028 ,  0.00272234,  0.00081532, ...,  0.01152894,
         0.01414437,  0.01651662],
       [ 0.007892  ,  0.00402109,  0.00209731, ...,  0.00927242,
         0.01200894,  0.01451151],
       [ 0.00915729,  0.00530504,  0.00337089, ...,  0.00691424,
         0.00975565,  0.01238328]])}
"""
    U_star = data['U_star']
    p_star = data['p_star']
    X_star = data['X_star']
    t_star = data['t']
    print(U_star.shape,p_star.shape,X_star.shape,t_star.shape)
    N = X_star.shape[0]
    T = t_star.shape[0]
    XX = np.tile(X_star[:,0:1],(1,T))
    YY = np.tile(X_star[:,1:2],(1,T))
    tt = np.tile(t_star[:,0:1],(1,N)).T
    print(XX.shape,YY.shape,tt.shape)

    UU = U_star[:,0,:]
    VV = U_star[:,1,:]
    PP = p_star
    """ plt.plot(X_star[:,0:1],X_star[:,1:2])
    plt.savefig('nseqn.png')"""

    x = XX.flatten()[:,None]
    y = YY.flatten()[:,None]
    t = tt.flatten()[:,None]

    p = PP.flatten()[:,None]
    u = UU.flatten()[:,None]
    v = VV.flatten()[:,None]

    #train data
    indx = np.random.choice(N*T , N_train , replace = False)
    x_star = x[indx,:]
    y_star = y[indx,:]
    t_star = t[indx,:]
    p_star = p[indx,:]
    u_star = u[indx,:]
    v_star = v[indx,:]
    model = PhysicsInformedNN(x_star,y_star,t_star,u_star,v_star,layers)
    model.train(200000)
