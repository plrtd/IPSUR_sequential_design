import numpy as np
from joblib import dump, load
import scipy
from Utils_banana import *

## Initialization 
n_mcmc = int(2e5)  # nb of MCMC samples
n_tune = 1000  # nb of tuning samples
iterations = 5 # nb of sequential design iterations
thinning = 10 # MCMC thinning
burnin = 500 # MCMC burnin
n_infinite = 1000 # nb of training points for 'infinite' GP surrogate
bounds_mcmc = np.array([[-20, -10], # MCMC uniform prior bounds
                        [ 20, 10.0]])

bounds = scipy.optimize.Bounds(lb=bounds_mcmc[0], ub=bounds_mcmc[1]) # optimization bounds
noise_cov_train = 0.001 * np.array([[100, 0.0],  # covariance noise for training data
                                    [0.0, 1.0]])

## Load observations, covariance and initial GP
load_path = './Ressources/Banana_case/'
observations = load(f'{load_path}Observations.joblib') # inverse problem noisy observations
cov_obs_tot = load(f'{load_path}Cov_obs_tot.joblib') # covariance of observations in extended form
gp = load(f'{load_path}Initial_GP.joblib') # Initial GP surrogate
gp_infinite = load(f'{load_path}Infinite_GP.joblib')
sampler_inf = NUTS_Banana(gp_infinite, np.ravel(observations), cov_obs_tot, bounds_mcmc)
sampler_inf.chain = load(f'{load_path}Infinite_chain.joblib') # MCMC chain with target distribution p_\infty
kde_inf = kde_mcmc(sampler_inf.chain) # KDE for the distribution p_\infty

## Define save path
save_path = './' # Path where MCMC chains, design points and surrogates should be saved

## Initialize sequential design
gp_sur = load(f'{load_path}Initial_GP.joblib')
gp_csq = load(f'{load_path}Initial_GP.joblib')
gp_naive = load(f'{load_path}Initial_GP.joblib')
gp_sinsbeck = load(f'{load_path}Initial_GP.joblib')
sampler_sur = NUTS_Banana(gp_sur, np.ravel(observations), cov_obs_tot, bounds_mcmc)
sampler_csq = NUTS_Banana(gp_csq, np.ravel(observations), cov_obs_tot, bounds_mcmc)
sampler_naive = NUTS_Banana(gp_naive, np.ravel(observations), cov_obs_tot, bounds_mcmc)
sampler_sinsbeck = NUTS_Banana(gp_sinsbeck, np.ravel(observations), cov_obs_tot, bounds_mcmc)
sampler_sur.sample(n_samples=n_mcmc, n_tune=n_tune)
sampler_csq.sample(n_samples=n_mcmc, n_tune=n_tune)
sampler_naive.sample(n_samples=n_mcmc, n_tune=n_tune)
sampler_sinsbeck.sample(n_samples=n_mcmc, n_tune=n_tune)
dump(sampler_sur.chain, f'{save_path}Initial_chain_SUR.joblib')
dump(sampler_csq.chain, f'{save_path}Initial_chain_CSQ.joblib')
dump(sampler_naive.chain, f'{save_path}Initial_chain_Naive.joblib')
dump(sampler_sinsbeck.chain, f'{save_path}Initial_chain_sinsbeck.joblib')

## Sequential designs 
list_added_sur = sequential_design_sur(sampler_sur, gp_sur, observations, cov_obs_tot, iterations,
                            bounds_mcmc, n_mcmc, noise_cov_train, save_path, thinning=thinning, burnin=burnin)

list_added_csq = sequential_design_csq(sampler_csq, gp_csq, observations, cov_obs_tot, iterations,
                            bounds_mcmc, n_mcmc, noise_cov_train, save_path, thinning=thinning, burnin=burnin)

list_added_naive = sequential_design_naive(sampler_naive, gp_naive, observations, cov_obs_tot, iterations,
                            bounds_mcmc, n_mcmc, noise_cov_train, save_path, thinning=thinning, burnin=burnin)

list_added_sinsbeck = sequential_design_sinsbeck(sampler_sinsbeck, gp_sinsbeck, observations, cov_obs_tot, iterations,
                            bounds_mcmc, n_mcmc, noise_cov_train, save_path, thinning=thinning, burnin=burnin)


## Evaluate metrics
list_kl_sur = []
list_kl_csq = []
list_kl_naive = []
list_kl_sinsbeck = []
list_entropy_sur = []
list_entropy_csq = []
list_entropy_naive = []
list_entropy_sinsbeck = []
list_ivar_sur = []
list_ivar_csq = []
list_ivar_naive = []
list_ivar_sinsbeck = []
for i in range(iterations+1):
    if i == 0:
        chain_sur = load(f'{save_path}Initial_chain_SUR.joblib')
        gp_sur = load(f'{load_path}Initial_GP.joblib')
        chain_csq = load(f'{save_path}Initial_chain_CSQ.joblib')
        gp_csq = load(f'{load_path}Initial_GP.joblib')
        chain_naive = load(f'{save_path}Initial_chain_Naive.joblib')
        gp_naive = load(f'{load_path}Initial_GP.joblib')
        chain_sinsbeck = load(f'{save_path}Initial_chain_sinsbeck.joblib')
        gp_sinsbeck = load(f'{load_path}Initial_GP.joblib')
    else:
        chain_sur = load(f'{save_path}New_chain_SUR_Iteration_' + str(i) + '.joblib')
        gp_sur = load(f'{save_path}New_GP_SUR_Iteration_' + str(i) + '.joblib')
        chain_csq = load(f'{save_path}New_chain_CSQ_Iteration_' + str(i) + '.joblib')
        gp_csq = load(f'{save_path}New_GP_CSQ_Iteration_' + str(i) + '.joblib')
        chain_naive = load(f'{save_path}New_chain_Naive_Iteration_' + str(i) + '.joblib')
        gp_naive = load(f'{save_path}New_GP_Naive_Iteration_' + str(i) + '.joblib')
        chain_sinsbeck = load(f'{save_path}New_chain_sinsbeck_Iteration_' + str(i) + '.joblib')
        gp_sinsbeck = load(f'{save_path}New_GP_sinsbeck_Iteration_' + str(i) + '.joblib')
    sampler_sur = NUTS_Banana(gp_sur, np.ravel(observations), cov_obs_tot, bounds_mcmc)
    sampler_csq = NUTS_Banana(gp_csq, np.ravel(observations), cov_obs_tot, bounds_mcmc)
    sampler_naive = NUTS_Banana(gp_naive, np.ravel(observations), cov_obs_tot, bounds_mcmc)
    sampler_sinsbeck = NUTS_Banana(gp_sinsbeck, np.ravel(observations), cov_obs_tot, bounds_mcmc)
    sampler_sur.chain = chain_sur
    sampler_csq.chain = chain_csq
    sampler_naive.chain = chain_naive
    sampler_sinsbeck.chain = chain_sinsbeck
    ent_sur, kde_sur = entropy(sampler_sur, thinning=thinning, burnin=burnin)
    ent_csq, kde_csq = entropy(sampler_csq, thinning=thinning, burnin=burnin)
    ent_naive, kde_naive = entropy(sampler_naive, thinning=thinning, burnin=burnin)
    ent_sinsbeck, kde_sinsbeck = entropy(sampler_sinsbeck, thinning=thinning, burnin=burnin)
    list_entropy_sur.append(ent_sur)
    list_entropy_csq.append(ent_csq)
    list_entropy_naive.append(ent_naive)
    list_entropy_sinsbeck.append(ent_sinsbeck)
    list_kl_sur.append(kl(sampler_sur, sampler_inf, thinning=thinning, burnin=burnin, kde=kde_sur, kde_inf=kde_inf))
    list_kl_csq.append(kl(sampler_csq, sampler_inf, thinning=thinning, burnin=burnin, kde=kde_csq, kde_inf=kde_inf))
    list_kl_naive.append(kl(sampler_naive, sampler_inf, thinning=thinning, burnin=burnin, kde=kde_naive, kde_inf=kde_inf))
    list_kl_sinsbeck.append(kl(sampler_sinsbeck, sampler_inf, thinning=thinning, burnin=burnin, kde=kde_sinsbeck, kde_inf=kde_inf))
    list_ivar_sur.append(Ivar(sampler_sur, thinning=thinning, burnin=burnin))
    list_ivar_csq.append(Ivar(sampler_csq, thinning=thinning, burnin=burnin))
    list_ivar_naive.append(Ivar(sampler_naive, thinning=thinning, burnin=burnin))
    list_ivar_sinsbeck.append(Ivar(sampler_sinsbeck, thinning=thinning, burnin=burnin))

dump(list_kl_sur, f'{save_path}KL_SUR.joblib')
dump(list_kl_csq, f'{save_path}KL_CSQ.joblib')
dump(list_kl_naive, f'{save_path}KL_Naive.joblib')
dump(list_kl_sinsbeck, f'{save_path}KL_sinsbeck.joblib')
dump(list_entropy_sur, f'{save_path}Entropy_SUR.joblib')
dump(list_entropy_csq, f'{save_path}Entropy_CSQ.joblib')
dump(list_entropy_naive, f'{save_path}Entropy_Naive.joblib')
dump(list_entropy_sinsbeck, f'{save_path}Entropy_sinsbeck.joblib')
dump(list_ivar_sur, f'{save_path}IVAR_SUR.joblib')
dump(list_ivar_csq, f'{save_path}IVAR_CSQ.joblib')
dump(list_ivar_naive, f'{save_path}IVAR_Naive.joblib')
dump(list_ivar_sinsbeck, f'{save_path}IVAR_sinsbeck.joblib')

