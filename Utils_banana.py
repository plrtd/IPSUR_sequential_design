import numpy as np
from Kernel import Matern, LinearCoregionalizationModel, MultiOutputWhiteKernel
from Model import GPR
from DataSet import DataSet
import pymc as pm
import arviz as az
from joblib import dump
import scipy
import pytensor.tensor as pt
from scipy.stats import gaussian_kde

class LogLike(pt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """

    itypes = [pt.dvector]  # expects a vector of parameter values when called
    otypes = [pt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, gp, cov_obs_tot, obs, bounds):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.cov_obs_tot = cov_obs_tot
        self.obs = obs
        self.gp = gp
        self.bounds = bounds

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.gp, self.cov_obs_tot, self.obs, self.bounds, transform=False)
        outputs[0][0] = np.array(logl)  # output the log-likelihood

def log_like(inp, gp, cov_obs_tot, obs, bounds, transform=True):
    """

    Parameters
    ----------
    inp : 1D-array of size p, or 2D array of shape (1, p), p is the input dimension.
        Input point for the log-likelihood calculation.
    gp : GPR object.
        Gaussian process surrogate model.
    cov_obs_tot : 2D array of shape (D, D) with D the output dimension.
        Covariance of the noise on the observations.
    obs : array of shape (N, D) with N the number of observations.
        Noisy observations used in the inverse problem.
    transform : boolean 
        Whether or not the input were re-scaled to [0, 1]. 
        If True, the rescaled inputs in [0, 1] are transformed back into the true values.
    Returns
    -------
    res : float
        Log-likelihood value.

    """
    D = 2
    if transform:
        inp = inp * (bounds[1] - bounds[0]) + bounds[0]
    inp = np.atleast_2d(inp)
    N = np.shape(cov_obs_tot)[0] // D
    mean, _, _, cov = gp.predict(inp, full_cov=False, error=None, nugget=False)
    obs = obs.reshape((N, D))
    obs_mean = np.mean(obs, axis=0)
    cov_obs = cov_obs_tot[:D, :D]
    cov_tot = cov[0] + (cov_obs / N)
    
    if np.any(np.isnan(mean)) or np.any(np.isnan(cov)):
        return -np.inf
    res = scipy.stats.multivariate_normal.logpdf(obs_mean, mean=mean[0], cov=cov_tot, allow_singular=True)
    return res

class NUTS_Banana:
    def __init__(self, GP, measures, cov_obs_tot, bounds, input_names=["x1", "x2"]):
        """
        Class for MCMC samplers with PyMC. 
        
        Parameters
        ----------
        GP : GPR Model
            Gaussian process surrogate model.
        measures : array of shape (N, D)
            DESCRIPTION.
        cov_obs_tot : array of shape (D, D)
            Covariance of the noise on the observations.
        bounds : array of shape (2, D)
            Bounds for the MCMC uniform prior.
        input_names : list of str of length p
            The default is ["x1", "x2"].
        prior : str, optional
            Describes the choice of prior. The default is "uniform".
        """
        self.obs = measures
        self.cov_obs_tot = cov_obs_tot
        self.bounds =  bounds
        self.GP = GP
        self.pm_model = pm.Model()
        self.logl = LogLike(log_like, self.GP, self.cov_obs_tot, self.obs, self.bounds)
        self.chain = None
        self.input_names = input_names
    
    def sample(self, n_samples=5000, n_tune=1000, plot=False):
        """
        Parameters
        ----------
        n_samples : int, optional
            Number of HMC-NUTS samples to draw. The default is 5000.
        n_tune : int, optional
            Number of tuning samples. The default is 1000.
        plot : bool, optional
            Whether or not the function should plot autocorrelations and densities.
            The default is True.

        Returns
        -------
        self.chain : array of shape (L, p) 
            All the samples of the posterior.
        idata : inference data
            Inference data for vizualization in arviz for example.

        """
        with self.pm_model:
            x1 = pm.Uniform(self.input_names[0], lower=self.bounds[0, 0], upper=self.bounds[1, 0])
            x2 = pm.Uniform(self.input_names[1], lower=self.bounds[0, 1], upper=self.bounds[1, 1])
            theta = pt.as_tensor_variable([x1, x2])
            pm.Potential("likelihood", self.logl(theta))
            init_dict = {}
            inp = np.random.uniform(low=self.bounds[0], high=self.bounds[1])
            while np.isinf(self.log_like_notransform(inp)):
                inp = np.random.uniform(low=self.bounds[0], high=self.bounds[1])
            for i in range(len(self.input_names)):
                init_dict[self.input_names[i]] = inp[i]
            idata = pm.sample(draws=n_samples//4, tune=n_tune, jitter_max_retries=10, cores=1, chains=4, initvals=init_dict, progressbar=False)
        if plot:      
            az.summary(idata, round_to=2)
            az.plot_trace(idata, combined=True)
            az.plot_autocorr(idata)
        
        keys = idata.posterior.data_vars.keys()
        chain = []
        for i, key in enumerate(keys):
            chain.append(np.ravel(idata.posterior.data_vars[key].to_numpy()))
        chain = np.array(chain).T
        self.chain = chain
        return self.chain, idata

    
    def log_like(self, inp):
        """
        Log likelihood for the input inp
        """
        if np.any(inp) > 1 or np.any(inp) < 0:
            raise ValueError("Normalized input should be between 0 and 1")
        return log_like(inp, self.GP, self.cov_obs_tot, self.obs, self.bounds, transform=True)
    
    
    def log_like_notransform(self, inp):
        """
        Log likelihood for the input inp
        """
        return log_like(inp, self.GP, self.cov_obs_tot, self.obs, self.bounds, transform=False)

    def find_map(self):
        """
        Find the MAP
        """
        with self.pm_model:
            xmap_dict = pm.find_MAP(progressbar=False)
        xmap = np.array([xmap_dict[key] for key in self.input_names])
        self.xmap = xmap
        return xmap

def f_banana(x):
    y = np.zeros((len(x), 2))
    y[:, 0] = x[:, 0]
    y[:, 1] = x[:, 1] + 0.03 * x[:, 0]**2
    return y

def gp_train(N_data, bounds):
    """

    Parameters
    ----------
    N_data : int
        nb of training data.
    bounds : array of shape (2, 2)
        Bounds of the input space.

    Returns
    -------
    gp : GPR object
        Trained GP model with training data obtained via LHS sampling.

    """
    p = 2
    D = 2
    output_names = [r'$y_1$', r'$y_2$']
    lhs = scipy.stats.qmc.LatinHypercube(p)
    inputs = lhs.random(2 * N_data)
    inputs = bounds[0] + (bounds[1] - bounds[0]) * inputs
    outputs = f_banana(inputs)
    
    noise_cov_train = 1 * np.array([[100, 0.0],
                                    [0.0, 1.0]])
                               
    eps = np.random.multivariate_normal(np.zeros(D), noise_cov_train, size=2*N_data)
    outputs = outputs + eps
    
    testfrac = 0.5
    dataset = DataSet(inputs, outputs, output_names, testfrac, boxcox=False, yeo_johnson=False)
    output_bounds_min = np.array([-20, -10])
    output_bounds_max = np.array([ 20, 7])
    dataset.delete_outlier_output(output_bounds_min, output_bounds_max)
    k1 = Matern(lengthscales=np.ones(p), nu=2.5, var=5.0)
    k2 = Matern(lengthscales=np.ones(p), nu=2.5, var=5.0)
    kernel =  LinearCoregionalizationModel([k1, k2], D)  + MultiOutputWhiteKernel(p, D, noise_bounds=[(1e-8, 1e5) for _ in range(D)])
    gp = GPR(dataset, kernel)
    gp.train()
    return gp

def create_observations(N):
    """
    Returns N noisy observations for the inverse problem.
    """
    D = 2
    xth = np.array([[0, 3]])
    measures = f_banana(xth)
    measures = np.tile(measures, (N, 1))
    cov_obs = np.array([[100, 0.0],
                        [0.0, 1.0]])
    
    eps_obs = np.random.multivariate_normal(np.zeros(D), cov_obs, size=N)
    measures = measures + eps_obs
    cov_obs_tot = np.kron(np.eye(N), cov_obs)
    return measures, cov_obs_tot

def sample_mcmc(n_mcmc, gp, observations, cov_obs_tot, bounds_mcmc, n_tune=1000, input_names=[r'$x_1$', r'$x_2$']):
    """
    Sample the posterior according to [1] with HMC-NUTS implemented in PyMC.

    Parameters
    ----------
    n_mcmc : int
        number of mcmc samples.
    gp : GPR 
        Gaussian process surrogate model.
    observations : array of shape (N, D)
        Noisy observations for the inverse problem.
    cov_obs_tot :array of shape (ND, ND)
        Global covariance of the observation noise (see eq (83))
    bounds_mcmc : array of shape (2, D)
        Bounds for uniform prior in MCMC.
    n_tune : int, optional
        Number of tuning samples. The default is 1000.
    input_names : list of str
        list of input names.

    Returns
    -------
    chain : array of shape (K, p)
        All the samples from the posterior distribution.
    sampler : NUTS object
        NUTS object for additional information.
    idata : inference data
        Inference data for data visualization in arviz.

    """
    sampler = NUTS_Banana(gp, observations, cov_obs_tot, bounds_mcmc, input_names=input_names)
    chain, idata = sampler.sample(n_mcmc, n_tune=n_tune)
    return chain, sampler, idata

def get_chain_mean_cov(chain, gp):
    """

    Parameters
    ----------
    chain : array of shape (N, p)
        array of mcmc samples.
    gp : GPR
        GP surrogate model.

    Returns
    -------
    mean : array of shape (N, D)
        array of predictive means.
    var : array of shape (N, D, D)
        array of predictive covariances.

    """
    mean, _, _, var = gp.predict(chain, full_cov=False, nugget=False)
    return mean, var,

def get_chain_new_mean_cov(chain, gp, x, mean_old, var_old):
    """
    

    Parameters
    ----------
    chain : array of shape (N, p)
        array of mcmc samples.
    gp : GPR
        GP surrogate model.
    x : array of shape (1, p)
        new design point.
    mean_old : array of shape (N, D)
        array of predictive means.
    var_old : array of shape (N, D, D)
        array of predictive covariances.

    Returns
    -------
    var_new : array of shape (N, D, D)
        array of updated predictive covariances (eq (81))
    var_x : array of shape (1, D, D)
        predictive covariance at x.
    inv_var_x : array of shape (1, D, D)
        Inverse of var_x.
    corr_x : array of shape (1, D, D)
        array containing k_n(\Tilde{x} , x).
    lambda_n : array of shape (1, D, D)
        array containing lambda_n = (corr_x.T) inv_var_x (corr_x).

    """
    D = 2
    _, _, _, var_x = gp.predict(x, full_cov=True, nugget=False)
    corr_x = np.zeros((len(chain), D, D))
    
    for i in range(len(chain)):
        inp = np.append(np.atleast_2d(chain[i]), np.atleast_2d(x), axis=0)
        _, _, _, cov_x = gp.predict(inp, full_cov=True, nugget=True)
        corr_x[i] = np.reshape(np.array([cov_x[0, 1], cov_x[0, 3],
                                         cov_x[2, 1], cov_x[2, 3]]).T, (D, D))
    new_corr_x = np.swapaxes(corr_x, 1, 2)
    inv_var_x = np.linalg.pinv(var_x) # for numerical stability
    lambda_n = np.matmul(np.matmul(new_corr_x, inv_var_x), corr_x)
    var_new = var_old - lambda_n
    return var_new, var_x, inv_var_x, corr_x, lambda_n

def Sigma_n(cov_obs_tot, var_chain):
    """
    Compute \Sigma_n from equation (83)
    """
    D = 2
    N = len(cov_obs_tot)// D
    res = np.tile(cov_obs_tot[np.newaxis, :, :], (len(var_chain), 1, 1))
    res2 = np.kron(np.ones((N, N)), var_chain)
    res += res2
    return res

def scalar_product(a, b, inv):
    """
    Utilitary function for Mahanalobis scalar product.
    """
    if len(np.shape(a)) != 2:
        a = np.atleast_2d(a)
    if len(np.shape(b)) != 2:
        b = np.atleast_2d(b)
    res = np.matmul(np.matmul(a, inv), b.T)[:, 0, 0]
    return res

def eval_g(Sigma_old, Sigma_new, var_old, var_new):
    """
    Helper function for computing h defined in equation (93).
    """
    det_tot_old = np.linalg.det(Sigma_old)
    det_tot_new = np.linalg.det(Sigma_new)
    res = det_tot_new / det_tot_old
    res[res < 0] = 0
    res = np.sqrt(res)
    res *= np.linalg.det(var_new)
    return res
    
def eval_h(mean, Sigma_old, Sigma_new, inv_Sigma_old, inv_Sigma_new, var_new, var_old, observations):
    """
    Computes h defined in equation (93).
    """
    N = len(observations)
    L = len(mean)
    D = 2
    ext_observations = np.ravel(observations)[np.newaxis, :]
    ext_observations = np.tile(ext_observations, (L, 1))
    ext_observations = ext_observations[:, :, np.newaxis]
    extended_mean = mean[np.newaxis, :, :]
    extended_mean = np.tile(extended_mean, (N, 1, 1))
    extended_mean = np.swapaxes(extended_mean, 0, 1)
    extended_mean = extended_mean.reshape((L, N*D, 1))
    res = eval_g(Sigma_old, Sigma_new, var_old, var_new)
    diff = ext_observations - extended_mean
    diff2 = np.swapaxes(diff, 1, 2)
    dist_old = scalar_product(diff2, diff.T, inv_Sigma_old)
    dist_new = scalar_product(diff2, diff.T, inv_Sigma_new)
    res *= np.exp(- 0.5 * (dist_new - dist_old))
    return res, diff

def eval_I(diff, inv_Sigma_new, var_x, inv_var_x, corr_x):
    """
    Computes I defined in equation (94).
    """
    L = len(diff)
    D = 2
    N = diff.shape[1] // D
    if np.any(np.linalg.det(inv_Sigma_new) < 0): # regularization of determinant
        inv_Sigma_new += 1e-4 * np.eye(D*N)
    det_var_x = np.linalg.det(var_x)
    inv_var_x_ext = np.tile(inv_var_x[np.newaxis, :, :], (L, 1, 1))
    corr_x_ext = np.tile(corr_x, (1, 1, N))
    new_corr_x_ext = np.swapaxes(corr_x_ext, 1, 2)
    S1 = np.matmul(corr_x_ext, np.matmul(inv_Sigma_new, new_corr_x_ext))
    S2 = np.matmul(inv_var_x_ext, np.matmul(S1, inv_var_x_ext))
    B1 = np.matmul(inv_var_x_ext, np.matmul(corr_x_ext, inv_Sigma_new))
    B = np.matmul(B1, diff) # defined in equation (96)
    A = inv_var_x_ext+ S2 # defined in equation (95)
    det_A = np.linalg.det(A)
    res1 = 1 /np.sqrt(det_var_x)
    res2 = 1 /np.sqrt(det_A)
    res3 = np.exp(0.5 * np.matmul(np.matmul(np.swapaxes(B, 1, 2), np.linalg.pinv(A)), B))[:, 0, 0]
    return res1 * res2 * res3

def eval_H(x, gp, observations, chain, mean_old, var_old, Sigma_old, inv_Sigma_old, cov_obs_tot):
    """
    Computes H on the Markov chain (see eq. (87)).
    """
    var_new, cov_x, inv_var_x, corr_x, lambda_n= get_chain_new_mean_cov(chain, gp, x, mean_old, var_old)
    Sigma_new = Sigma_n(cov_obs_tot, var_new)
    inv_Sigma_new = np.linalg.pinv(Sigma_new)
    h_n, diff = eval_h(mean_old, Sigma_old, Sigma_new, inv_Sigma_old, inv_Sigma_new, var_new, var_old, observations)
    I_n = eval_I(diff, inv_Sigma_new, cov_x, inv_var_x, corr_x)
    res = h_n * I_n
    res_mean = np.mean(res, axis=0)
    return res_mean

def acquisition(x, gp, sampler, log_like_bound):
    """
    Acquisition function to maximise for the CSQ method (section 4.1)

    Parameters
    ----------
    x : array of shape (p) or (1, p)
        New design point.
    gp : GPR object
        Gaussian process surrogate model.
    sampler : NUTS_Banana object
        Sampler object used for its method for log-likelihood.
    log_like_bound : float
        Bounds on the difference between MAP and query point log-likelihood..

    Returns
    -------
    log_det : float
        Determinant of predictive covrariance. Set to infinity if outside the query set.
    log_like : float
        Log likelihood of the query point .

    """
    x = np.atleast_2d(x)
    _, _, _, gp_cov = gp.predict(x, full_cov=False, error=None, nugget=False)
    log_like = sampler.log_like_notransform(x)
    log_det = np.log(np.linalg.det(gp_cov))
    if log_like > log_like_bound:
        return log_det, log_like
    return -np.inf, log_like

def find_new_x_csq(sampler, gp, cov_obs_tot, bounds_mcmc, iterations=50, dist_bound_map=3.):
    """

    Parameters
    ----------
    sampler : NUTS_Banana object object
        Sampler object.
    gp : GPR object
        Gaussian process surrogate model.
    cov_obs_tot : array of shape (D, D)
        Covariance for the observation noise.
    iterations : int, optional
        Nb of annealing restarts. The default is 5.
    bounds : array of shape (2, p), optional
        Bound constraints for the optimization. The default is None.
    dist_bound_map : float, optional
        Hyperparameter reflecting the tolerated distance to the MAP (see eq. (21)). The default is 3.

    Returns
    -------
    opt.x : array of shape (p)
        Query point selected to maximize the predictive covariance of the surrogate.

    """
    x0 = sampler.find_map()
    annealing_bound = sampler.log_like_notransform(x0)
    log_like_bound = annealing_bound - dist_bound_map
    def func(x):
        x_trans = x * (bounds_mcmc[1] - bounds_mcmc[0]) + bounds_mcmc[0]
        return -acquisition(x_trans, gp, sampler, log_like_bound)[0]
    bounds_opt = scipy.optimize.Bounds(lb=np.zeros(2), ub=np.ones(2)) # optimization bounds
    x_init = (x0 - bounds_mcmc[0]) / (bounds_mcmc[1] - bounds_mcmc[0])
    opt = scipy.optimize.dual_annealing(func, bounds=bounds_opt, maxiter=iterations, x0=x_init, restart_temp_ratio=1e-3, initial_temp=20000)
    res = opt.x * (bounds_mcmc[1] - bounds_mcmc[0]) + bounds_mcmc[0]
    return res

def find_new_x_sur(chain, gp, observations, cov_obs_tot, bounds_mcmc, iterations=5):
    """

    Parameters
    ----------
    chain : array of shape (L, p)
        MCMC samples.
    gp : GPR object
        GP surrogate model.
    observations : array of shape (N, D)
        Observations of the direct model.
    cov_obs_tot : array of shape (ND, ND)
        Global covariance (see eq. (83)).
    bounds_mcmc : array of shape (2, p)
        Bounds for the uniform prior.
    iterations : int, optional
        Nb of annealing restarts. The default is 5.

    Returns
    -------
    opt.x : array of shape (p)
        New design point of the IP-SUR (see eq. (34))

    """
    mean_old, var_old = get_chain_mean_cov(chain, gp)
    Sigma_old = Sigma_n(cov_obs_tot, var_old)
    inv_Sigma_old = np.linalg.pinv(Sigma_old)
    def func(x):
        x_trans = x * (bounds_mcmc[1] - bounds_mcmc[0]) + bounds_mcmc[0]
        return np.log(eval_H(x_trans, gp, observations, chain, mean_old, var_old, Sigma_old, inv_Sigma_old, cov_obs_tot))
    x0 = chain[np.random.randint(0, len(chain))]
    bounds_opt = scipy.optimize.Bounds(lb=np.zeros(2), ub=np.ones(2)) # optimization bounds
    opt = scipy.optimize.dual_annealing(func, bounds_opt, x0=x0, maxiter=iterations, restart_temp_ratio=1e-2, initial_temp=10000)
    res = opt.x * (bounds_mcmc[1] - bounds_mcmc[0]) + bounds_mcmc[0]
    return res



def int1(mean_old, var_old, obs, cov_obs):
    """
    Helper function for Sinsbeck acquisition function
    """
    cov_tot = cov_obs + var_old
    res = 0
    for k in range(len(mean_old)):
        res += scipy.stats.multivariate_normal(mean=mean_old[k], cov=cov_tot[k]).pdf(obs) ** 2
    return res / len(mean_old)

def int2(mean_old, var_old, var_new, obs, cov_obs):
    """
    Helper function for Sinsbeck acquisition function
    """
    D = 2
    A1 = var_new + cov_obs
    d1 = np.linalg.det(2 * np.pi * A1)
    A2 = 2 * var_old - var_new
    res = np.zeros(len(mean_old))
    for k in range(len(mean_old)):
        res[k] = scipy.stats.multivariate_normal(mean=mean_old[k], cov = 0.5 * A2[k]).pdf(obs)
    res = res / np.sqrt(d1)
    res = res / np.sqrt(2 ** D)
    return np.mean(res)

def sinsbeck_acq(x, gp, samples, obs, cov_obs, bounds_mcmc, denormalize=True):
    """
    Acquisition function to be maximized in Sinsbeck strategy.

    Parameters
    ----------
    x : array of shape (p) or (1, p)
        New design point.
    gp : GPR object
        Gaussian process surrogate model.
    samples : array of shape (L, p)
        MCMC samples.
    obs : array of shape (N, D)
        array of observations.
    cov_obs : array of shape (D, D)
        Observational noise covariance.
    bounds_mcmc : array of shape (2, p)
        Bounds for the uniform prior in MCMC.
    denormalize : bool, optional
        If True, the input are assumed to be previously scaled between 0 and 1. The default is True.

    Returns
    -------
    float
        Returns the value of the acquisition function.

    """
    if denormalize:
        x = bounds_mcmc[0] + x * (bounds_mcmc[1] - bounds_mcmc[0])
    mean_old, _, _, var_old = gp.predict(samples, full_cov=False)
    var_new, var_x, _, _, _ = get_chain_new_mean_cov(samples, gp, x, mean_old, var_old)
    i1 = int1(mean_old, var_old, obs, cov_obs)
    i2 = int2(mean_old, var_old, var_new, obs, cov_obs)
    return i2 - i1

def update_sinsbeck(gp, obs, cov_obs, bounds_mcmc, n=1000, samples=None):
    """
    Find new design point with Sinsbeck strategy..

    Parameters
    ----------
    gp : GPR object
        Gaussian process surrogate model.
    obs : array of shape (N, D)
        array of observations.
    cov_obs : array of shape (D, D)
        Observational noise covariance.
    bounds_mcmc : array of shape (2, p)
        Bounds for the uniform prior in MCMC.
    n : TYPE, optional
        DESCRIPTION. The default is 1000.
    samples : array of shape (n, p), optional
        Samples of the prior distribution. If None, new samples are created with uniform random distribution.The default is None.

    Returns
    -------
    samples : array of shape (n, p)
        Samples of the prior distribution.
    opt_x : array of shape (p)
        New design point.

    """
    if samples is None:
        samples = np.random.uniform(low=bounds_mcmc[0], high=bounds_mcmc[1], size=(n, 2))
    func = lambda x:-sinsbeck_acq(x, gp, samples, obs, cov_obs, bounds_mcmc)
    bounds_opt = scipy.optimize.Bounds(lb=np.zeros(2), ub=np.ones(2))
    opt_x = scipy.optimize.dual_annealing(func, bounds=bounds_opt, maxiter=50, initial_temp=10000, restart_temp_ratio=1e-3).x
    opt_x = bounds_mcmc[0] + opt_x * (bounds_mcmc[1] - bounds_mcmc[0])
    return samples, opt_x

def sequential_design_sur(sampler_sur, gp_sur, observations, cov_obs_tot, iterations,
                          bounds_mcmc, n_mcmc, noise_cov_train, save_path, thinning=10, burnin=500):
    """
    IP-SUR strategy iterated a given number of times. Saves the GP, MCMC chains and new design points in the folder 'save_path'.

    Parameters
    ----------
    sampler_sur : NUTS_Banana object
        MCMC sampler with target distribution p_n.
    gp_sur : GPR object
        GP surrogate model.
    observations : array of shape (N, D)
        array of direct model observations.
    cov_obs_tot : array of shape (ND, ND)
        Global covariance (see eq. (83)).
    iterations : int
        Nb of iteration of the sequential design strategy.
    bounds_mcmc : array of shape (2, p)
        Bounds of the uniform prior.
    n_mcmc : int
        Nb of MCMC samples.
    noise_cov_train : array of shape (D, D)
        Observational noise covariance.
    save_path : str
        Where to save MCMC chains, newly conditioned GPs and list of new design points.
    thinning : int, optional
        Factor by which the chain is thinned to keep only decorrelated samples. The default is 40.
    burnin : int, optional
        Describes how many initial MCMC samples should be discarded (to reach stationary distribution). The default is 1000.


    Returns
    -------
    list_added_sur : list
        list of newly added design points.

    """
    list_added_sur  = []
    for i in range(iterations):
        chain_sur = sampler_sur.chain[burnin::thinning]
        new_x = find_new_x_sur(chain_sur, gp_sur, observations, cov_obs_tot, bounds_mcmc, iterations=20)
        new_x = np.atleast_2d(new_x)
        new_y = f_banana(new_x) + np.random.multivariate_normal(np.zeros(2), noise_cov_train, size=1)
        gp_sur.add_data(new_x, new_y)
        list_added_sur.append(new_x[0])
        chain_sur, sampler_sur, _ = sample_mcmc(n_mcmc, gp_sur, np.ravel(observations), cov_obs_tot, bounds_mcmc)
        dump(gp_sur, f'{save_path}New_GP_SUR_Iteration_' + str(i+1) + '.joblib')
        dump(chain_sur, f'{save_path}New_chain_SUR_Iteration_' + str(i+1) + '.joblib')
        chain_sur = chain_sur[burnin::thinning]
        dump(list_added_sur, f'{save_path}list_added_SUR.joblib')
    return list_added_sur

def sequential_design_csq(sampler_csq, gp_csq, observations, cov_obs_tot, iterations,
                          bounds_mcmc, n_mcmc, noise_cov_train, save_path, thinning=10, burnin=500):
    """
    CSQ design strategy iterated a given number of times. Saves the GP, MCMC chains and new design points in the folder 'save_path'.

    Parameters
    ----------
    sampler_sur : NUTS_Banana object
        MCMC sampler with target distribution p_n.
    gp_sur : GPR object
        GP surrogate model.
    observations : array of shape (N, D)
        array of direct model observations.
    cov_obs_tot : array of shape (ND, ND)
        Global covariance (see eq. (83)).
    iterations : int
        Nb of iteration of the sequential design strategy.
    bounds_mcmc : array of shape (2, p)
        Bounds of the uniform prior.
    n_mcmc : int
        Nb of MCMC samples.
    noise_cov_train : array of shape (D, D)
        Observational noise covariance.
    save_path : str
        Where to save MCMC chains, newly conditioned GPs and list of new design points.
    thinning : int, optional
        Factor by which the chain is thinned to keep only decorrelated samples. The default is 40.
    burnin : int, optional
        Describes how many initial MCMC samples should be discarded (to reach stationary distribution). The default is 1000.


    Returns
    -------
    list_added_sur : list
        list of newly added design points.

    """
    list_added_csq  = []
    for i in range(iterations):
        chain_csq = sampler_csq.chain[burnin::thinning]
        new_x = find_new_x_csq(sampler_csq, gp_csq, cov_obs_tot, bounds_mcmc, iterations=20)
        new_x = np.atleast_2d(new_x)
        new_y = f_banana(new_x) + np.random.multivariate_normal(np.zeros(2), noise_cov_train, size=1)
        gp_csq.add_data(new_x, new_y)
        list_added_csq.append(new_x[0])
        chain_csq, sampler_csq, _ = sample_mcmc(n_mcmc, gp_csq, np.ravel(observations), cov_obs_tot, bounds_mcmc)
        dump(gp_csq, f'{save_path}New_GP_CSQ_Iteration_' + str(i+1) + '.joblib')
        dump(chain_csq, f'{save_path}New_chain_CSQ_Iteration_' + str(i+1) + '.joblib')
        chain_csq = chain_csq[burnin::thinning]
        dump(list_added_csq, f'{save_path}list_added_CSQ.joblib')
    return list_added_csq

def sequential_design_naive(sampler_naive, gp_naive, observations, cov_obs_tot, iterations,
                            bounds_mcmc, n_mcmc, noise_cov_train, save_path, thinning=10, burnin=500):
    """
    Naive design strategy iterated a given number of times. Saves the GP, MCMC chains and new design points in the folder 'save_path'.

    Parameters
    ----------
    sampler_sur : NUTS_Banana object
        MCMC sampler with target distribution p_n.
    gp_sur : GPR object
        GP surrogate model.
    observations : array of shape (N, D)
        array of direct model observations.
    cov_obs_tot : array of shape (ND, ND)
        Global covariance (see eq. (83)).
    iterations : int
        Nb of iteration of the sequential design strategy.
    bounds_mcmc : array of shape (2, p)
        Bounds of the uniform prior.
    n_mcmc : int
        Nb of MCMC samples.
    noise_cov_train : array of shape (D, D)
        Observational noise covariance.
    save_path : str
        Where to save MCMC chains, newly conditioned GPs and list of new design points.
    thinning : int, optional
        Factor by which the chain is thinned to keep only decorrelated samples. The default is 40.
    burnin : int, optional
        Describes how many initial MCMC samples should be discarded (to reach stationary distribution). The default is 1000.


    Returns
    -------
    list_added_sur : list
        list of newly added design points.

    """
    list_added_naive  = []
    for i in range(iterations):
        chain_naive = sampler_naive.chain[burnin::thinning]
        new_x = np.random.uniform(low=bounds_mcmc[0], high=bounds_mcmc[1], size=(1, 2))
        new_y = f_banana(new_x) + np.random.multivariate_normal(np.zeros(2), noise_cov_train, size=1)
        gp_naive.add_data(new_x, new_y)
        list_added_naive.append(new_x[0])
        chain_naive, sampler_naive, _ = sample_mcmc(n_mcmc, gp_naive, np.ravel(observations), cov_obs_tot, bounds_mcmc)
        dump(gp_naive, f'{save_path}New_GP_Naive_Iteration_' + str(i+1) + '.joblib')
        dump(chain_naive, f'{save_path}New_chain_Naive_Iteration_' + str(i+1) + '.joblib')
        chain_naive = chain_naive[burnin::thinning]
        dump(list_added_naive, f'{save_path}list_added_Naive.joblib')
    return list_added_naive


def sequential_design_sinsbeck(sampler_sinsbeck, gp_sinsbeck, observations, cov_obs_tot, iterations,
                               bounds_mcmc, n_mcmc, noise_cov_train, save_path, thinning=10, burnin=500):
    """
    Design strategy from Sinsbeck et al. Saves the GP, MCMC chains and new design points in the folder 'save_path'.

    Parameters
    ----------
    sampler_sinsbeck : NUTS_Banana object
        MCMC sampler with target distribution p_n.
    gp_sinsbeck : GPR object
        GP surrogate model.
    observations : array of shape (N, D)
        array of direct model observations.
    cov_obs_tot : array of shape (ND, ND)
        Global covariance (see eq. (83)).
    iterations : int
        Nb of iteration of the sequential design strategy.
    bounds_mcmc : array of shape (2, p)
        Bounds of the uniform prior.
    n_mcmc : int
        Nb of MCMC samples.
    noise_cov_train : array of shape (D, D)
        Observational noise covariance.
    save_path : str
        Where to save MCMC chains, newly conditioned GPs and list of new design points.
    thinning : int, optional
        Factor by which the chain is thinned to keep only decorrelated samples. The default is 40.
    burnin : int, optional
        Describes how many initial MCMC samples should be discarded (to reach stationary distribution). The default is 1000.


    Returns
    -------
    list_added_sins : list
        list of newly added design points.

    """
    D = 2
    obs = np.mean(observations, axis=0)
    cov_obs = cov_obs_tot[:D, :D] / len(observations)
    list_added_sins = []
    for i in range(iterations):
        chain_sinsbeck = sampler_sinsbeck.chain[burnin::thinning]
        new_x = update_sinsbeck(gp_sinsbeck, obs, cov_obs, bounds_mcmc)[1]
        new_x = np.atleast_2d(new_x)
        new_y = f_banana(new_x) + np.random.multivariate_normal(np.zeros(2), noise_cov_train, size=1)
        gp_sinsbeck.add_data(new_x, new_y)
        list_added_sins.append(new_x[0])
        chain_sinsbeck, sampler_sinsbeck, _ = sample_mcmc(n_mcmc, gp_sinsbeck, np.ravel(observations), cov_obs_tot, bounds_mcmc)
        dump(gp_sinsbeck, f'{save_path}New_GP_sinsbeck_Iteration_' + str(i+1) + '.joblib')
        dump(chain_sinsbeck, f'{save_path}New_chain_sinsbeck_Iteration_' + str(i+1) + '.joblib')
        chain_sinsbeck = chain_sinsbeck[burnin::thinning]
        dump(list_added_sins, f'{save_path}list_added_sinsbeck.joblib')
    return list_added_sins



def kde_mcmc(chain):
    """
    KDE estimation based on the MCMC chain.

    """
    kde = gaussian_kde(chain.T)
    return kde
    
def kl(sampler, sampler_inf, thinning=100, burnin=500, kde=None, kde_inf=None):
    """

    Parameters
    ----------
    sampler : NUTS_Banana object
        MCMC sampler with target distribution p_n.
    sampler_inf : NUTS_Banana object
        MCMC sampler with target distribution p_\infty.
    thinning : int, optional
        Factor by which the chain is thinned to keep only decorrelated samples. The default is 40.
    burnin : int, optional
        Describes how many initial MCMC samples should be discarded (to reach stationary distribution). The default is 1000.
    kde : gaussian_kde object
        Kernel density estimate for the distribution p_n. If None, it is evaluated in the function. Default is None. 
    kde_inf : gaussian_kde object
        Kernel density estimate for the distribution p_\infty. If None, it is evaluated in the function. Default is None. 
    
    Returns
    -------
    kl : float
        Estimated KL(p_n || p_\infty) (equation (58)).

    """
    chain = sampler.chain[burnin::thinning]
    chain_inf = sampler_inf.chain[::thinning]
    chain_test = sampler.chain[thinning//2::thinning]
    if kde is None:
        kde = kde_mcmc(chain)
    if kde_inf is None:
        kde_inf = kde_mcmc(chain_inf)
    else:
        log_density = kde.logpdf(chain_test.T)
        log_density_inf = kde_inf.logpdf(chain_test.T)
        return np.mean(log_density - log_density_inf, axis=0)

def entropy(sampler, thinning=100, burnin=500):
    """

    Parameters
    ----------
    sampler : NUTS_Banana object
        MCMC sampler with target distribution p_n.
    thinning : int, optional
        Factor by which the chain is thinned to keep only decorrelated samples. The default is 40.
    burnin : int, optional
        Describes how many initial MCMC samples should be discarded (to reach stationary distribution). The default is 1000.
    Returns
    -------
    entropy : float
        Estimated entropy of p_n (equation (56)).
    kde : KDE for p_n

    """
    chain = sampler.chain[burnin::thinning]
    chain_test = sampler.chain[thinning//2::thinning]
    kde = kde_mcmc(chain)
    log_density = kde.logpdf(chain_test.T)
    return -np.mean(log_density, axis=0), kde

def Ivar(sampler, thinning=40, burnin=1000):
    """

    Parameters
    ----------
    sampler : NUTS_Banana object
        MCMC sampler with target distribution p_n.
    thinning : int, optional
        Factor by which the chain is thinned to keep only decorrelated samples. The default is 40.
    burnin : int, optional
        Describes how many initial MCMC samples should be discarded (to reach stationary distribution). The default is 1000.

    Returns
    -------
    ivar : float
        Estimated Integrated predictive variance (equation (54)).

    """
    chain = sampler.chain[burnin::thinning]
    _, _, _, cov = sampler.GP.predict(chain, full_cov=False, nugget=False)
    det = np.linalg.det(cov)
    return np.mean(det)

