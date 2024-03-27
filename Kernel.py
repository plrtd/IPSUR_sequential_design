from abc import ABCMeta, abstractmethod
import copy
import math
from tabulate import tabulate
from scipy.spatial.distance import pdist, cdist, squareform
import numpy as np

def f_RBF(X1, X2, lengthscales, variance):
    """
    RBF covariance between X1 and X2
    """
    if X2 is None:
        dists = pdist(X1 / lengthscales, metric="sqeuclidean")
        K = np.exp(-0.5*dists)
        K = squareform(K)
        np.fill_diagonal(K, 1)
    else:
        dists = cdist(X1 / lengthscales, X2 / lengthscales, metric="sqeuclidean")
        K = np.exp(-0.5*dists)
    return variance*K

def f_Matern(X1, X2, lengthscales, variance, nu):
    """
    Matern covariance betweeen X1 and X2 for a given choice of nu.
    """
    if X2 is None:
        dists = pdist(X1 / lengthscales, metric="euclidean")
    else:
        dists = cdist(X1 / lengthscales, X2 / lengthscales, metric="euclidean")

    if nu == 0.5:
        K = np.exp(-dists)
    elif nu == 1.5:
        K = dists * math.sqrt(3)
        K = (1.0 + K)*np.exp(-K)
    elif nu == 2.5:
        K = dists * math.sqrt(5)
        K = (1.0 + K + K**2 / 3.0) * np.exp(-K)
    elif nu == np.inf:
        K = np.exp(-0.5 * dists**2)
    else:
        raise ValueError('Matern kernel not available for nu = ' + str(nu))

    if X2 is None:
        K = squareform(K)
        np.fill_diagonal(K, 1)
    return variance*K

class BaseKernel(metaclass=ABCMeta):
    """
    Base class for all kernels
    """

    def clone(self):
        return copy.deepcopy(self)

    @abstractmethod
    def get_params(self):
        """
        Get the parameters
        """

    @abstractmethod
    def set_params(self, theta):
        """
        Set the parameters
        """

    @abstractmethod
    def get_input_dims(self):
        """
        Call the kernel for (X1, X2). If X2=None, call for (X1, X1)
        """

    @abstractmethod
    def get_output_dims(self):
        """
        Call the kernel for (X1, X2). If X2=None, call for (X1, X1)
        """



    @abstractmethod
    def __call__(self, X1, X2=None):
        """
        Call the kernel for (X1, X2). If X2=None, call for (X1, X1)
        """

    @abstractmethod
    def diag(self, X1):
        """
        Get the diagonal of K(X1, X1)
        """

    @property
    @abstractmethod
    def bounds(self):
        """
        Get the bounds for the parmeters
        """
    @property
    @abstractmethod
    def theta(self):
        """
        Get the log transformed hyperparameters
        """
    @theta.setter
    @abstractmethod
    def theta(self, theta):
        """
        Set the log transformed hyperparameters
        """

    def __str__(self):
        params = self.get_params()
        nparams = len(params)
        bounds = self.bounds
        string = tabulate(
            [[item[0], item[1], bounds[i, 0], bounds[i, 1]]
             for i, item in enumerate(params.items())],
            headers=['Name of param', 'Value', 'Bound min', 'Bound max'],
            tablefmt='fancy_grid',
            numalign='center',
            stralign='center'
        ) + '\n' + 'Number of hyperparameters = ' + str(nparams) + '\n'
        return string

    def __add__(self, b):
        if isinstance(b, (int, float)):
            b = ConstantKernel(self.get_input_dims(), cst_value=b)
        return Sum(self, b)

    def __radd__(self, b):
        if isinstance(b, (int, float)):
            b = ConstantKernel(self.get_input_dims(), cst_value=b)
        return Sum(b, self)

    def __mult__(self, b):
        if isinstance(b, (int, float)):
            b = ConstantKernel(self.get_input_dims(), cst_value=b)
        return Product(self, b)

    def __rmult__(self, b):
        if isinstance(b, (int, float)):
            b = ConstantKernel(self.get_input_dims(), cst_value=b)
        return Product(b, self)

class KernelOperator(BaseKernel):
    """
    Base class for all kernel operators
    """

    def __init__(self, k1, k2):
        if k1.get_output_dims() != k2.get_output_dims():
            raise ValueError('Operation of two kernels with different output dimension')
        if k1.get_input_dims() != k2.get_input_dims():
            raise ValueError('Operation of two kernels with different input dimension')
        self.k1 = k1
        self.k2 = k2

    def get_params(self):
        params_k1 = self.k1.get_params()
        items_k1 = params_k1.items()
        params_k2 = self.k2.get_params()
        items_k2 = params_k2.items()
        params = {}
        params.update(("k1_" + k, val) for k, val in items_k1)
        params.update(("k2_" + k, val) for k, val in items_k2)
        return params

    def set_params(self, theta):
        params_k1 = self.k1.get_params()
        nparams_k1 = len(params_k1)
        self.k1.set_params(theta[:nparams_k1])
        self.k2.set_params(theta[nparams_k1:])

    @property
    def theta(self):
        return np.append(self.k1.theta, self.k2.theta)

    @theta.setter
    def theta(self, theta):
        nparams_k1 = len(self.k1.get_params())
        self.k1.theta = theta[:nparams_k1]
        self.k2.theta = theta[nparams_k1:]
        self.set_params(theta)

    @property
    def bounds(self):
        return np.vstack((self.k1.bounds, self.k2.bounds))

    @abstractmethod
    def get_input_dims(self):
        """
        Call the kernel for (X1, X2). If X2=None, call for (X1, X1)
        """

    @abstractmethod
    def get_output_dims(self):
        """
        Call the kernel for (X1, X2). If X2=None, call for (X1, X1)
        """

    @abstractmethod
    def __call__(self, X1, X2=None):
        """
        Call the kernel for (X1, X2). If X2=None, call for (X1, X1)
        """

    @abstractmethod
    def diag(self, X1):
        """
        Get the diagonal of K(X1, X1)
        """

class Sum(KernelOperator):
    """
    Kernel obtained by the sum of two kernels k1 and k2
    Instances of the KernelOperator class
    """

    def __call__(self, X1, X2=None):
        return self.k1(X1, X2) + self.k2(X1, X2)

    def diag(self, X1):
        return self.k1.diag(X1) + self.k2.diag(X1)

    def get_input_dims(self):
        if self.k1.get_input_dims() != self.k2.get_input_dims():
            raise ValueError('Sum of two kernels with different input dimension')
        return self.k1.get_input_dims()

    def get_output_dims(self):
        if self.k1.get_output_dims() != self.k2.get_output_dims():
            raise ValueError('Sum of two kernels with different output dimension')
        return self.k1.get_output_dims()
    
    def set_xtrain(self, Xtrain):
        self.Xtrain = Xtrain
        self.k1.set_xtrain(Xtrain)
        self.k2.set_xtrain(Xtrain)
        

    def k_trans(self, X1):
        return self.k1.k_trans(X1) + self.k2.k_trans(X1)

class Product(KernelOperator):
    """
    Kernel obtained by the product of two kernels k1 and k2
    Instances of the KernelOperator class
    """

    def __call__(self, X1, X2=None):
        return self.k1(X1, X2) * self.k2(X1, X2)

    def diag(self, X1):
        return self.k1.diag(X1) * self.k2.diag(X1)

    def get_input_dims(self):
        if self.k1.get_input_dims() != self.k2.get_input_dims():
            raise ValueError('Sum of two kernels with different input dimension')
        return self.k1.get_input_dims()


    def set_xtrain(self, Xtrain):
        self.Xtrain = Xtrain
        self.k1.set_xtrain(Xtrain)
        self.k2.set_xtrain(Xtrain)

    def get_output_dims(self):
        if self.k1.get_output_dims() != self.k2.get_output_dims():
            raise ValueError('Product of two kernels with different output dimension')
        return self.k1.get_output_dims()

    def k_trans(self, X1):
        return self.k1.k_trans(X1) * self.k2.k_trans(X1)

class ConstantKernel(BaseKernel):
    """
    Constant kernel in single output.
    Defined by a parameter "csv_value" and its bounds.
    """

    def __init__(self, inp_dim=1, cst_value=1.0, cst_value_bounds=(1e-5, 1e5)):
        if isinstance(cst_value, (float, int)):
            self.cst_value = cst_value*np.ones(1)
            self.cst_value_bounds = np.array([cst_value_bounds], dtype=tuple)
        else:
            self.cst_value = cst_value
            self.cst_value_bounds = cst_value_bounds
        if len(self.cst_value) != len(self.cst_value_bounds):
            print('Bounds length was not appropriate',
                  '\nBounds set to (1e-5, 1e5) for all parameters')
            l = [(1e-5, 1e5) for i in range(len(self.cst_value))]
            self.cst_value_bounds = np.array(l)
        self.input_dims = inp_dim
        self.output_dims = len(self.cst_value)
        self.Xtrain = None

    def get_input_dims(self):
        return self.input_dims

    def get_output_dims(self):
        return self.output_dims



    def set_xtrain(self, Xtrain):
        self.Xtrain = Xtrain

    def k_trans(self, X1):
        return self(X1, self.Xtrain)
    
    def __call__(self, X1, X2=None):
        n1 = np.shape(X1)[0]
        if X2 is not None:
            n2 = np.shape(X2)[0]
        else:
            n2 = n1
        K = self.cst_value*np.ones((n1, n2))
        return K

    def diag(self, X1):
        n1 = np.shape(X1)[0]
        K = self.cst_value*np.ones(n1)
        return K

    def get_params(self):
        params = {}
        params.update(('cst_value_' + 'dim_' + str(i+1),
                      self.cst_value[i]) for i in range(len(self.cst_value)))
        return params

    def set_params(self, theta):
        if len(theta) != len(self.cst_value):
            raise ValueError('Theta should have same dimension as nb_params in set_params()')
        self.cst_value = np.exp(theta)

    @property
    def bounds(self):
        return self.cst_value_bounds

    @property
    def theta(self):
        return np.log(self.cst_value)

    @theta.setter
    def theta(self, theta):
        if len(self.theta) != len(theta):
            raise ValueError('New theta should have same dimension')
        self.theta = theta
        self.set_params(theta)

class WhiteKernel(BaseKernel):
    """
    White kernel in single output. used to model noise in training set.
    Defined by a parameter "noise" and its bounds.
    """

    def __init__(self, inp_dim=1, noise=1.0, noise_bounds=(1e-5, 1e5)):
        if isinstance(noise, (float, int)):
            self.noise = noise*np.ones(1)
            self.noise_bounds = np.array([noise_bounds])
        if len(self.noise) != len(self.noise_bounds):
            print('Bounds length was not appropriate.',
                  '\nBounds set to (1e-5, 1e5) for all parameters')
            l = [(1e-5, 1e5) for i in range(len(self.noise))]
            self.noise_bounds = np.array(l)
        self.output_dims = len(self.noise)
        self.input_dims = inp_dim
        self.Xtrain = None

    def get_input_dims(self):
        return self.input_dims

    def get_output_dims(self):
        return self.output_dims



    def set_xtrain(self, Xtrain):
        self.Xtrain = Xtrain

    def k_trans(self, X1):
        return self(X1, self.Xtrain)

    def __call__(self, X1, X2=None):
        if X2 is None and self.get_output_dims() == 1:
            return self.noise*np.eye(np.shape(X1)[0])
        return np.zeros((np.shape(X1)[0], np.shape(X2)[0]))

    def diag(self, X1):
        n1 = np.shape(X1)[0]
        K = self.noise*np.ones(n1)
        return K

    def get_params(self):
        params = {}
        params.update([('noise_dim_' + str(i+1), self.noise[i]) for i in range(len(self.noise))])
        return params

    def set_params(self, theta):
        if len(theta) != len(self.noise):
            raise ValueError('Theta should have same dimension as nb_params in set_params()')
        self.noise = np.exp(theta)

    @property
    def bounds(self):
        return np.array(self.noise_bounds)

    @property
    def theta(self):
        return np.log(self.noise)

    @theta.setter
    def theta(self, theta):
        if len(self.theta) != len(theta):
            raise ValueError('New theta should have same dimension')
        self.set_params(theta)

class RBF(BaseKernel):
    """
    RBF kernel defined by its lengthscales and variance parameters. Single output.
    The kernel is anisotropic. All lengthscales are independent.
    Base Class for the Matern Kernel
    """

    def __init__(self, var=1.0, var_bounds=(1e-5, 1e5), lengthscales=1.0, lengthscale_bounds=(1e-5, 1e5)):
        if isinstance(lengthscales, (float, int)):
            self.lengthscales = lengthscales*np.ones(1)
            self.lengthscale_bounds = np.array([lengthscale_bounds])
        else:
            self.lengthscales = lengthscales
            self.lengthscale_bounds = lengthscale_bounds
        if isinstance(lengthscale_bounds, tuple):
            self.lengthscale_bounds = np.array([lengthscale_bounds])
        if len(self.lengthscales) != len(self.lengthscale_bounds):
            print('Bounds length was not appropriate.',
                  '\nBounds set to (1e-5, 1e5) for all parameters')
            l = [(1e-5, 1e5) for i in range(len(self.lengthscales))]
            self.lengthscale_bounds = np.array(l)
        self.var = var*np.ones(1)
        self.var_bounds = np.array([var_bounds])
        self.output_dims = 1
        self.input_dims = len(self.lengthscales)
        self.Xtrain = None

    def get_input_dims(self):
        return self.input_dims

    def get_output_dims(self):
        return self.output_dims


    def set_xtrain(self, Xtrain):
        self.Xtrain = Xtrain

    def k_trans(self, X1):
        return self(X1, self.Xtrain)
    
    
    def __call__(self, X1, X2=None):
        K = f_RBF(X1, X2, self.lengthscales, self.var)
        return K

    def diag(self, X1):
        n1 = np.shape(X1)[0]
        K = self.var*np.ones(n1)
        return K

    def get_params(self):
        params = {}
        params.update([('variance_dim_' + str(i+1),
                        self.var[i]) for i in range(len(self.var))])
        params.update(('lengthscale_' + 'dim_' + str(i+1),
                       self.lengthscales[i]) for i in range(self.input_dims))
        return params

    def set_params(self, theta):
        if len(theta) != len(self.var)+len(self.lengthscales):
            raise ValueError('Theta should have same dimension as nb_params in set_params()')
        n = len(self.var)
        self.var = np.exp(theta[:n])
        self.lengthscales = np.exp(theta[n:])

    @property
    def bounds(self):
        return np.append(self.var_bounds, self.lengthscale_bounds, axis=0)

    @property
    def theta(self):
        return np.log(np.append(self.var, self.lengthscales))

    @theta.setter
    def theta(self, theta):
        if len(self.theta) != len(theta):
            raise ValueError('New theta should have same dimension')
        self.set_params(theta)

class Matern(RBF):
    """
    Matern Kernel defined by the fixed value "nu".
    Hyperparameters are variance and lengthscales. Single output.
    """

    def __init__(self, var=1.0, var_bounds=(1e-5, 1e5), lengthscales=1.0, lengthscale_bounds=(1e-5, 1e5), nu=2.5):
        super().__init__(var, var_bounds, lengthscales, lengthscale_bounds)
        self.nu = nu

    def __call__(self, X1, X2=None):
        K = f_Matern(X1, X2, self.lengthscales, self.var, self.nu)
        return K

class MultiOutputKernel(BaseKernel, metaclass=ABCMeta):
    """
    Abstract base class for multi output kernels such as ConvolutionKernel.
    Defined by Q independent gaussian processes, each with its own kernel.
    """
    @abstractmethod
    def __call__(self, X1, X2=None):
        """
        Evaluate covariance for multi output kernel
        """

class MultiOutputWhiteKernel(MultiOutputKernel):
    """
    Multioutput kernel for white noise
    """

    def __init__(self, input_dims, output_dims, noise=None, noise_bounds=None):
        self.input_dims = input_dims
        self.output_dims = output_dims
        if noise is None:
            self.noise = np.ones(output_dims)
        else:
            if len(noise) != output_dims:
                raise ValueError('Noise in WhiteOutputkernel should have output dimension')
            self.noise = noise
        if noise_bounds is None:
            self.noise_bounds = [(1e-5, 1e5) for _ in range(output_dims)]
        else:
            if len(noise_bounds) != output_dims:
                raise ValueError('Noise bounds should have output dimension')
            self.noise_bounds = noise_bounds
        self.Xtrain = None
        
        
    def get_input_dims(self):
        return self.input_dims

    def get_output_dims(self):
        return self.output_dims
    
    

    def set_xtrain(self, Xtrain):
        self.Xtrain = Xtrain
        
    def k_trans(self, X1):
        return self(X1, self.Xtrain)

    def __call__(self, X1, X2=None):
        n1 = np.shape(X1)[0]
        if X2 is None:
            return np.kron(np.diag(self.noise), np.eye(n1))
        D = self.get_output_dims()
        n2 = np.shape(X2)[0]
        return np.zeros((D*n1, D*n2))

    def diag(self, X1):
        n1 = np.shape(X1)[0]
        D = self.get_output_dims()
        res = np.zeros(D*n1)
        for d in range(D):
            res[d*n1:(d+1)*n1] = self.noise[d]
        return res

    def get_params(self):
        D = self.get_output_dims()
        params = {}
        params.update([('Multi_noise_d='+str(d+1), str(self.noise[d])) for d in range(D)])
        return params

    def set_params(self, theta):
        D = self.get_output_dims()
        npar_tot = D
        if len(theta) != npar_tot:
            raise ValueError('Theta should have same dimension as nb_params in set_params()')
        self.noise = np.exp(theta)

    @property
    def bounds(self):
        bounds = self.noise_bounds
        return bounds

    @property
    def theta(self):
        theta = np.log(self.noise)
        return theta

    @theta.setter
    def theta(self, theta):
        if len(self.theta) != len(theta):
            raise ValueError('New theta should have same dimension')
        self.set_params(theta)

class LinearCoregionalizationModel(MultiOutputKernel):
    """
    f = Wu with u independent latent GP, with kernels in "latent_kernels".
    W is a DxQ matrix , D=output_dim, Q=num_latent_GPs
    """

    def __init__(self, latent_kernels, output_dims, W=None):
        self.output_dims = output_dims
        self.latent_kernels = latent_kernels
        if W is None:
            self.W = 0.1*np.ones((self.output_dims, len(self.latent_kernels)))
        else:
            self.W = W
        for i in range(min(len(self.latent_kernels), self.output_dims)):
            self.W[i, i] = 1
        if np.shape(self.W)[0] != self.output_dims or np.shape(self.W)[1] != len(self.latent_kernels):
            raise ValueError('Matrix W should have shape DxQ')
        self.Xtrain = None

    def get_input_dims(self):
        inp_dim = 0
        for kern in self.latent_kernels:
            if inp_dim != 0 and inp_dim != kern.get_input_dims():
                raise ValueError('All kernel should have same input dimensions')
            inp_dim = kern.get_input_dims()
        return inp_dim

    def get_output_dims(self):
        return self.output_dims
    

    def set_xtrain(self, Xtrain):
        self.Xtrain = Xtrain
        for kern in self.latent_kernels:
            kern.set_xtrain(Xtrain)
        
    def k_trans(self, X1):
        for kern in self.latent_kernels:
            kern.Xtrain = self.Xtrain
        return self(X1, self.Xtrain)

    def __call__(self, X1, X2=None):
        Q = len(self.latent_kernels)
        D = self.output_dims
        n1 = np.shape(X1)[0]
        if X2 is None:
            n2 = n1
        else:
            n2 = np.shape(X2)[0]
        res = np.zeros((D*n1, D*n2))
        for q in range(Q):
            Aq = np.atleast_2d(self.W[:, q])
            Bq = np.matmul(Aq.T, Aq)
            if X2 is None:
                Kq = self.latent_kernels[q](X1)
            else:
                Kq = self.latent_kernels[q](X1, X2)
            res += np.kron(Bq, Kq)
        return res

    def diag(self, X1):
        n1 = np.shape(X1)[0]
        D = self.output_dims
        Q = len(self.latent_kernels)
        res = np.zeros(D*n1)
        for q in range(Q):
            Bq = self.W[:, q]**2
            Kq = self.latent_kernels[q].diag(X1)
            res += np.kron(Bq, Kq)
        return res

    def get_params(self):
        params = {}
        kern_list = self.latent_kernels
        for d in range(self.output_dims):
            for q in range(len(kern_list)):
                if q != d:
                    params.update([('2+W(d='+str(d)+', q='+str(q)+')', 2+self.W[d, q])])
        for i, kern in enumerate(kern_list):
            dict_kern = kern.get_params()
            params.update([('latent_'+str(i+1)+'_' + key, val) for key, val in dict_kern.items()])
        return params

    def set_params(self, theta):
        npar_tot = 0
        for kern in self.latent_kernels:
            npar_tot += len(kern.theta)
        D = self.output_dims
        Q = len(self.latent_kernels)
        npar_tot += D*Q - Q
        if len(theta) != npar_tot:
            raise ValueError('Theta should have same dimension as nb_params in set_params()')
        W = np.ones((D, Q))
        ind = 0
        for d in range(D):
            for q in range(Q):
                if q != d:
                    W[d, q] = theta[ind]
                    ind += 1
        self.W = W
        index = D*Q - Q
        for kern in self.latent_kernels:
            npar = len(kern.get_params())
            kern.set_params(theta[index:index+npar])
            index += npar

    @property
    def bounds(self):
        D = self.output_dims
        Q = len(self.latent_kernels)
        bounds = np.array([(1, 4) for _ in range(D*Q-Q)])
        for kern in self.latent_kernels:
            bounds = np.append(bounds, kern.bounds, axis=0)
        return bounds

    @property
    def theta(self):
        theta = np.ravel(self.W)
        Q = len(self.latent_kernels)
        theta = np.delete(theta, [q*(Q+1) for q in range(Q)])
        for kern in self.latent_kernels:
            theta = np.append(theta, kern.theta)
        return theta

    @theta.setter
    def theta(self, theta):
        if len(self.theta) != len(theta):
            raise ValueError('New theta should have same dimension')
        self.set_params(theta)

