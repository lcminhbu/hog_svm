import numpy as np
from numba import cuda
import math

class SVM_Old:
    def __init__(self, C = 1.0):
        # C = error term
        self.C = C
        self.w = 0
        self.b = 0

    # Hinge Loss Function / Calculation
    def hingeloss(self, w, b, x, y):
        # Regularizer term
        reg = 0.5 * (w * w)

        for i in range(x.shape[0]):
            # Optimization term
            opt_term = y[i] * ((np.dot(w, x[i])) + b)

            # calculating loss
            loss = reg + self.C * max(0, 1-opt_term)
        return loss[0]

    def fit(self, X, Y, batch_size=100, learning_rate=0.001, epochs=1000):
        # The number of features in X
        number_of_features = X.shape[1]

        # The number of Samples in X
        number_of_samples = X.shape[0]

        c = self.C

        # Creating ids from 0 to number_of_samples - 1
        ids = np.arange(number_of_samples)

        # Shuffling the samples randomly
        np.random.shuffle(ids)

        # creating an array of zeros
        w = np.zeros(number_of_features)
        b = 0
        losses = []

        # Gradient Descent logic
        for i in range(epochs):
            # Calculating the Hinge Loss
            l = self.hingeloss(w, b, X, Y)

            # Appending all losses 
            losses.append(l)
            
            # Starting from 0 to the number of samples with batch_size as interval
            for batch_initial in range(0, number_of_samples, batch_size):
                gradw = 0
                gradb = 0

                for j in range(batch_initial, batch_initial+ batch_size):
                    if j < number_of_samples:
                        x = ids[j]
                        ti = Y[x] * (np.dot(w, X[x].T) + b)

                        if ti > 1:
                            gradw += 0
                            gradb += 0
                        else:
                            # Calculating the gradients

                            #w.r.t w 
                            gradw += c * Y[x] * X[x]
                            # w.r.t b
                            gradb += c * Y[x]

                # Updating weights and bias
                w = w - learning_rate * w + learning_rate * gradw
                b = b + learning_rate * gradb

        self.w = w
        self.b = b
        return self.w, self.b, losses

    def predict(self, X):
        prediction = np.dot(X, self.w) + self.b # w.x + b
        return np.sign(prediction)

class SVM_New:
    def __init__(self, gamma = -1, kernel = 'rbf', C = 1.0, eps = 1e-3):
        self.C = C
        self.eps = eps
        self.gamma = gamma
        self.tau = 1e-12
        self.kernel = kernel
        self.gamma = gamma
    
    def Kernel(self, x1, x2):
        if self.kernel == 'rbf':
            return self.rbf(x1, x2)
        if self.kernel == 'linear':
            return self.linear(x1, x2)

    def get_Q(self, X, i, j):
        if self.Q[i, j] == None:
            self.Q[i][j] = self.y[i] * self.y[j] * self.get_K(X, i, j)
            self.Q[j][i] = self.Q[i][j]
        return self.Q[i][j]
    
    def get_K(self, X, i, j):
        if self.K[i, j] == None:
            self.K[i][j] = self.Kernel(X[i], X[j])
            self.K[j][i] = self.K[i][j]
        return self.K[i][j]
    
    def linear(self, x1, x2):
        x1_temp = x1.astype(np.float64)
        x2_temp = x2.astype(np.float64)
        return x1_temp.dot(x2_temp)
    
    def rbf(self, x1, x2):
        x1_temp = x1.astype(np.float64)
        x2_temp = x2.astype(np.float64)
        return np.exp(-self.gamma * (x1_temp.dot(x1_temp) + x2_temp.dot(x2_temp) - 2.0 * x1_temp.dot(x2_temp)))
        
    def select_B(self, X):
        i = -1
        G_max = -np.inf
        G_min = np.inf
        for t in range(self.l):
            if (self.y[t] == 1 and self.alphas[t] <self.C) or \
            (self.y[t] == -1 and self.alphas[t] > 0):
                if -self.y[t] * self.G[t] >= G_max:
                    i = t
                    G_max = -self.y[t] * self.G[t]
        j = -1
        obj_min = np.inf
        for t in range(self.l):
            if (self.y[t]==1 and self.alphas[t]>0) or \
                (self.y[t] == -1 and self.alphas[t] < self.C):
                    b = G_max + self.y[t] * self.G[t]
                    if -self.y[t]*self.G[t] <= G_min:
                        G_min = -self.y[t] * self.G[t]
                    if b>0:
                        a = self.get_Q(X, i, i) + self.get_Q(X, t, t) - 2.0*self.y[i]*self.y[t]*self.get_Q(X, i, t)
                        if a<=0:
                            a = self.tau
                        if -(b*b)/a <= obj_min:
                            j = t
                            obj_min = -(b*b)/a
        if G_max - G_min < self.eps:
            return -1, -1
        return i, j

    def predict(self, X):
        pred = []
        for x in X:
            sum = 0.0
            for i in range(self.l):
                sum += self.y[i] * self.alphas[i] * self.Kernel(self.X[i], x)
            sum -= self.b
            pred.append(np.sign(sum))
        return pred

    def get_b(self):
        sum = 0.0
        count = 0
        for i in range(self.l):
            if 0 < self.alphas[i] < self.C:
                count += 1
                sum += self.y[i] * self.G[i]
        if count > 0:
            self.b = sum/count
            return
        max = -np.inf
        min = np.inf
        for i in range(self.l):
            if (self.alphas[i] == 0 and self.y[i] == -1) or \
                (self.alphas[i] == self.C and self.y[i] == 1):
                    if max < self.y[i] * self.G[i]:
                        max = self.y[i] * self.G[i]
            if (self.alphas[i] == 0 and self.y[i] == 1) or \
                (self.alphas[i] == self.C and self.y[i] == -1):
                    if min > self.y[i] *self.G[i]:
                        min = self.y[i] * self.G[i]
        self.b = (min+max) / 2

    def fit(self, X, y):
        self.y = y
        self.X = X
        self.l = len(y)
        if self.gamma == -1:
            self.gamma = 1/(X.shape[1]*X.var())
        self.active_size= self.l
        self.alphas = np.zeros(self.l)
        self.n_iter = 0

        self.K = np.array([[None for _ in range(self.l)] for _ in range(self.l)])
        self.Q = np.array([[None for _ in range(self.l)] for _ in range(self.l)])
        self.G = np.array([-1.0 for _ in range(self.l)])
        while True:
            i, j = self.select_B(X)
            if j == -1:
                break
            self.n_iter += 1
            alphai = self.alphas[i]
            alphaj = self.alphas[j]
            if y[i] != y[j]:
                quad_coef = self.get_Q(X, i, i) + self.get_Q(X, j, j) + 2*self.get_Q(X, i, j)
                if quad_coef <= 0:
                    quad_coef = self.tau
                delta = (-self.G[i] - self.G[j])/quad_coef
                diff = alphai - alphaj
                self.alphas[i] += delta
                self.alphas[j] += delta
                if diff > 0:
                    if self.alphas[j]<0:
                        self.alphas[j] = 0
                        self.alphas[i] = diff
                    if self.alphas[i]>self.C:
                        self.alphas[i] = self.C
                        self.alphas[j] = self.C - diff
                else:
                    if self.alphas[i]<0:
                        self.alphas[i] = 0
                        self.alphas[j] = -diff
                    if self.alphas[j]>self.C:
                        self.alphas[j] = self.C
                        self.alphas[i] = self.C + diff
            else:
                quad_coef = self.get_Q(X, i, i) + self.get_Q(X,j,j) - 2*self.get_Q(X,i,j)
                if quad_coef <=0:
                    quad_coef = self.tau
                delta = (self.G[i]-self.G[j])/quad_coef
                sum = alphai + alphaj
                self.alphas[i] -= delta
                self.alphas[j] += delta
                if sum>self.C:
                    if self.alphas[i]>self.C:
                        self.alphas[i] = self.C
                        self.alphas[j] = sum-self.C
                    if self.alphas[j]>self.C:
                        self.alphas[j] = self.C
                        self.alphas[i] = sum-self.C
                else:
                    if self.alphas[j]<0:
                        self.alphas[j] = 0
                        self.alphas[i] = sum
                    if self.alphas[i]<0:
                        self.alphas[i] = 0
                        self.alphas[j] = sum
            delta_ai = self.alphas[i] - alphai
            delta_aj = self.alphas[j] - alphaj
            
            for t in range(self.l):
                self.G[t] += self.get_Q(X, i, t) * delta_ai + self.get_Q(X, j, t) *delta_aj
        self.get_b()

class SVM_Pa:
    def __init__(self, gamma = -1, kernel = 'rbf', C = 1.0, eps = 1e-3):
        self.C = C
        self.eps = eps
        self.gamma = gamma
        self.tau = 1e-12
        self.kernel = kernel
        self.gamma = gamma
    
    def Kernel(self, x1, x2):
        if self.kernel == 'rbf':
            return self.rbf(x1, x2)
        if self.kernel == 'linear':
            return self.linear(x1, x2)
   
    def linear(self, x1, x2):
        x1_temp = x1.astype(np.float64)
        x2_temp = x2.astype(np.float64)
        return x1_temp.dot(x2_temp)
    
    def rbf(self, x1, x2):
        x1_temp = x1.astype(np.float64)
        x2_temp = x2.astype(np.float64)
        return np.exp(-self.gamma * (x1_temp.dot(x1_temp) + x2_temp.dot(x2_temp) - 2.0 * x1_temp.dot(x2_temp)))
        
    def select_B(self, X):
        i = -1
        G_max = -np.inf
        G_min = np.inf
        for t in range(self.l):
            if (self.y[t] == 1 and self.alphas[t] <self.C) or \
            (self.y[t] == -1 and self.alphas[t] > 0):
                if -self.y[t] * self.G[t] >= G_max:
                    i = t
                    G_max = -self.y[t] * self.G[t]
        j = -1
        obj_min = np.inf
        for t in range(self.l):
            if (self.y[t]==1 and self.alphas[t]>0) or \
                (self.y[t] == -1 and self.alphas[t] < self.C):
                    b = G_max + self.y[t] * self.G[t]
                    if -self.y[t]*self.G[t] <= G_min:
                        G_min = -self.y[t] * self.G[t]
                    if b>0:
                        a = self.Q[i, i] + self.Q[t, t] - 2.0*self.y[i]*self.y[t]*self.Q[i, t]
                        if a<=0:
                            a = self.tau
                        if -(b*b)/a <= obj_min:
                            j = t
                            obj_min = -(b*b)/a
        if G_max - G_min < self.eps:
            return -1, -1
        return i, j

    def predict(self, X):
        return np.sign(self.dual_coef.dot(self.init_K(X)) - self.b)

    @staticmethod
    @cuda.jit 
    def init_K_kernel_rbf(X1, X2, K, n1, n2, m, gamma):
        i, j = cuda.grid(2)
        if i>= n1 or j >= n2:
            return
        sumii = np.float64(0)
        sumij = np.float64(0)
        sumjj = np.float64(0)
        for k in range(m):
            sumii += X1[i][k] * X1[i][k]
            sumij += X1[i][k] * X2[j][k]
            sumjj += X2[j][k] * X2[j][k]
        K[i, j] = math.exp(-gamma * (sumii + sumjj - 2.0 * sumij))
    
    @staticmethod
    @cuda.jit 
    def init_K_kernel_linear(X1, X2, K, n1, n2, m):
        i, j = cuda.grid(2)
        if i>= n1 or j >= n2:
            return
        sumij = np.float64(0)
        for k in range(m):
            sumij += X1[i][k] * X2[j][k]
        K[i, j] = sumij
        
    def init_K(self, x):
        d_x1 = cuda.to_device(self.X.astype(np.float64))
        d_x2 = cuda.to_device(x.astype(np.float64))
        d_K = cuda.device_array((self.l, x.shape[0]), np.float64)
        blocksize = (32, 32)
        gridsize = (math.ceil(self.l/blocksize[0]), math.ceil(x.shape[0]/blocksize[1]))
        if self.kernel == 'rbf':
            self.init_K_kernel_rbf[gridsize, blocksize](d_x1, d_x2, d_K, self.l, x.shape[0], self.n_features, self.gamma)
        elif self.kernel == 'linear':
            self.init_K_kernel_linear[gridsize, blocksize](d_x1, d_x2, d_K, self.l, x.shape[0], self.n_features)
        return np.array(d_K.copy_to_host())

    def get_b(self):
        sum = 0.0
        count = 0
        for i in range(self.l):
            if 0 < self.alphas[i] < self.C:
                count += 1
                sum += self.y[i] * self.G[i]
        if count > 0:
            self.b = sum/count
            return
        max = -np.inf
        min = np.inf
        for i in range(self.l):
            if (self.alphas[i] == 0 and self.y[i] == -1) or \
                (self.alphas[i] == self.C and self.y[i] == 1):
                    if max < self.y[i] * self.G[i]:
                        max = self.y[i] * self.G[i]
            if (self.alphas[i] == 0 and self.y[i] == 1) or \
                (self.alphas[i] == self.C and self.y[i] == -1):
                    if min > self.y[i] *self.G[i]:
                        min = self.y[i] * self.G[i]
        self.b = (min+max) / 2
    
    @staticmethod
    @cuda.jit 
    def init_Q_kernel_rbf(X, y, Q, n, m, gamma):
        i, j = cuda.grid(2)
        if i>= n or j >= n:
            return
        sumii = np.float64(0)
        sumij = np.float64(0)
        sumjj = np.float64(0)
        for k in range(m):
            sumii += X[i][k] * X[i][k]
            sumij += X[i][k] * X[j][k]
            sumjj += X[j][k] * X[j][k]
        Q[i, j] = y[i]*y[j]*math.exp(-gamma * (sumii + sumjj - 2.0 * sumij))
    
    
    @staticmethod
    @cuda.jit 
    def init_Q_kernel_linear(X, y, Q, n, m):
        i, j = cuda.grid(2)
        if i>= n or j >= n:
            return
        sumij = np.float64(0)
        for k in range(m):
            sumij += X[i][k] * X[j][k]
        Q[i, j] = y[i]*y[j]*sumij
        
    def init_Q(self):
        d_x = cuda.to_device(self.X.astype(np.float64))
        d_y = cuda.to_device(self.y.astype(np.float64))
        d_Q = cuda.device_array((self.l, self.l), np.float64)
        blocksize = (32, 32)
        gridsize = (math.ceil(self.l/blocksize[0]), math.ceil(self.l/blocksize[1]))
        if self.kernel == 'rbf':
            self.init_Q_kernel_rbf[gridsize, blocksize](d_x, d_y, d_Q, self.l, self.n_features, self.gamma)
        elif self.kernel == 'linear':
            self.init_Q_kernel_linear[gridsize, blocksize](d_x, d_y, d_Q, self.l, self.n_features)
        self.Q = np.array(d_Q.copy_to_host())
    
    @staticmethod
    @cuda.jit
    def compute_dual_coef_kernel(alpha, y, dual_coef, l):
        i = cuda.grid(1)
        if i > l:
            return
        dual_coef[i] = alpha[i] * y[i]

    def compute_dual_coef(self):
        blocksize = 32
        gridsize = math.ceil(self.l/blocksize)
        d_dualcoef = cuda.device_array(self.l, np.float64)
        d_alphas = cuda.to_device(self.alphas)
        d_y = cuda.to_device(self.y)
        self.compute_dual_coef_kernel[gridsize, blocksize](d_alphas, d_y, d_dualcoef, self.l)
        self.dual_coef = d_dualcoef.copy_to_host()
        

    def fit(self, X, y):
        self.y = y
        self.X = X
        self.l, self.n_features = X.shape
        if self.gamma == -1:
            self.gamma = 1/(self.n_features*X.var())
        self.alphas = np.zeros(self.l)
        self.n_iter = 0
        self.init_Q()
        self.G = np.array([-1.0 for _ in range(self.l)])
        while True:
            i, j = self.select_B(X)
            if j == -1:
                break
            self.n_iter += 1
            alphai = self.alphas[i]
            alphaj = self.alphas[j]
            if y[i] != y[j]:
                quad_coef = self.Q[i, i] + self.Q[j, j] + 2*self.Q[i, j]
                if quad_coef <= 0:
                    quad_coef = self.tau
                delta = (-self.G[i] - self.G[j])/quad_coef
                diff = alphai - alphaj
                self.alphas[i] += delta
                self.alphas[j] += delta
                if diff > 0:
                    if self.alphas[j]<0:
                        self.alphas[j] = 0
                        self.alphas[i] = diff
                    if self.alphas[i]>self.C:
                        self.alphas[i] = self.C
                        self.alphas[j] = self.C - diff
                else:
                    if self.alphas[i]<0:
                        self.alphas[i] = 0
                        self.alphas[j] = -diff
                    if self.alphas[j]>self.C:
                        self.alphas[j] = self.C
                        self.alphas[i] = self.C + diff
            else:
                quad_coef = self.Q[i, i] + self.Q[j,j] - 2*self.Q[i,j]
                if quad_coef <=0:
                    quad_coef = self.tau
                delta = (self.G[i]-self.G[j])/quad_coef
                sum = alphai + alphaj
                self.alphas[i] -= delta
                self.alphas[j] += delta
                if sum>self.C:
                    if self.alphas[i]>self.C:
                        self.alphas[i] = self.C
                        self.alphas[j] = sum-self.C
                    if self.alphas[j]>self.C:
                        self.alphas[j] = self.C
                        self.alphas[i] = sum-self.C
                else:
                    if self.alphas[j]<0:
                        self.alphas[j] = 0
                        self.alphas[i] = sum
                    if self.alphas[i]<0:
                        self.alphas[i] = 0
                        self.alphas[j] = sum
            delta_ai = self.alphas[i] - alphai
            delta_aj = self.alphas[j] - alphaj
            self.G += self.Q[i, :] *delta_ai + self.Q[j, :]*delta_aj
        self.compute_dual_coef()
        self.get_b()