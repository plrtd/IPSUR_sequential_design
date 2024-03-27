import numpy as np
import matplotlib.pyplot as plt
import scipy
import random
import operator
import warnings
from tabulate import tabulate

def mean_absolute_error(y_test, y_pred):
    """
    Returns mean absolute error between two sets of data (of same shape)
    """
    mae = np.mean(abs(y_test - y_pred), axis=0)/np.mean(np.abs(y_test), axis=0)
    return mae

def mean_absolute_percentage_error(y_test, y_pred):
    """
    Returns mean absolute percentage error between two sets of data (of same shape)
    """
    mape = np.mean(np.abs((y_test - y_pred)/y_test), axis=0)
    return mape

def mean_squared_error(y_test, y_pred):
    """
    Returns mean squared error between two sets of data (of same shape)
    """
    mse = np.mean((y_test - y_pred)**2, axis=0)
    return mse

class GPR:
    """
    Model Class containing the submodel used (Direct GPR, SVGP, ...),
    the kernel and the data set.
    Contains methods train, error, predict.
    """

    def __init__(self, dataset, kernel):
        """

        Parameters
        ----------
        dataset : DataSet
            Dataset used for model training and testing.
        kernel : BaseKernel
            Kernel for the GP. If output_dims > 1, must be a MultiOutputKernel.

        """
        self.kernel = kernel
        self.dataset = dataset

        self.kernel_ = kernel.clone()
        self.lml_value_ = 0
        self.L_ = 0
        self.alpha_ = 0

    def print_params(self):
        """
        Returns a table containing all hyperparameters, with initial value and bounds.
        """
        print('-----------------------------')
        print('Initial kernel parameters')
        print(self.kernel)

    def print_trained_params(self):
        """
        Returns a table containing all trained hyperparameters, with initial value and bounds.
        """
        print('-----------------------------')
        print('Trained kernel parameters')
        print(self.kernel_)

    def error(self, y_pred, CI, method='MAE', plot=False):
        """
        Returns the error on test set predicitions.
        Prints a table containing error and confidence interval information.

        Parameters
        ----------
        y_pred : 2D array of shape (n_test, output_dims)
            Prediction on the test set.
        CI : 2-tuple of array of shape (n_test, output_dims)
            Lower and Upper bounds onf confidence interval.
        method : str, optional
            Type of error used. 'MAE' is Mean Absolute Error.
            'MSE' is Mean Squared Error. 'MAP' is Mean Absolute Percentage Error.
            If 'all', all errors are evaluated. The default is 'MAE'.

        Returns
        -------
        error : 2D array of shape (n_test, output_dims)
            Error on the test set prediction with the given error measure.

        OR

        mae, mse, mape : tuple of 2D array with shape (n_test, output_dims)
            All types of errors on test set predictions.

        """
        _, y_test = self.dataset.get_test_data()
        y_low = CI[0]
        y_up = CI[1]
        outdim = self.dataset.output_dims
        in_CI = np.zeros(outdim)
        for i in range(len(y_test)):
            for p in range(outdim):
                if y_test[i, p] > y_low[i, p] and y_test[i, p] < y_up[i, p]:
                    in_CI[p] += 1.0
        in_CI = 100.0*in_CI/float(len(y_test))
        print('-------------------------\nConfidence interval information\n' + tabulate(
            [[self.dataset.channel_names[p], in_CI[p]] for p in range(outdim)],
            headers=['Channel name', '% of test set in C.I. '],
            tablefmt='fancy_grid')
            + '\n')
        if method.lower() == 'mae':
            error = mean_absolute_error(y_test, y_pred)
        elif method.lower() == 'mape':
            error = mean_absolute_percentage_error(y_test, y_pred)
        elif method.lower() == 'mse':
            error = mean_squared_error(y_test, y_pred)
        elif method.lower() == 'radius':
            ratio = 100*np.mean((np.abs(CI[1]-CI[0])/np.abs(y_test)), axis=0)
        elif method.lower() == 'all':
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            ratio = 100*np.mean((np.abs(CI[1]-CI[0])/np.abs(y_test)), axis=0)
            print('-------------------------\nError on test set\n' + tabulate(
                [[self.dataset.channel_names[p], mae[p], mse[p], mape[p], ratio[p]]
                    for p in range(self.dataset.output_dims)],
                headers=['Channel name', 'Error (MAE)', 'Error (MSE)',
                         'Error (MAPE)', 'CI width (%)'],
                tablefmt='fancy_grid',
                numalign='center',
                stralign='center'
            ) + '\n')

            if plot:
                mae_all = np.abs(y_test-y_pred)
                mse_all = (y_test-y_pred)**2
                mape_all = np.abs(y_test-y_pred)/np.maximum(y_test, 1e-5*np.ones_like(y_test))
                ratio_all = 100*np.abs(CI[1]-CI[0])/np.abs(y_test)
                errors = [mae_all, mse_all, mape_all, ratio_all]
                labels = ['MAE', 'MSE', 'MAPE', 'CI width (%)']
                colors = ['black', 'black', 'black', 'black']
                for p in range(self.dataset.output_dims):
                    for i in range(len(errors)):
                        plt.figure(figsize=(10, 8))
                        plt.title('Error metrics on the validation set')
                        error = errors[i]
                        label = labels[i]
                        plt.plot(y_test[:, p], error[:, p], linestyle='None',
                                 marker='.', label=label, color=colors[i])
                        plt.xlabel(self.dataset.channel_names[p])
                        plt.ylabel(label)
                        plt.legend(loc='upper left')
                        plt.show()
                        plt.close()
            return mae, mse, mape, in_CI

        print('-------------------------\nError on test set' + tabulate(
            [[self.dataset.channel_names[p], error[p]] for p in range(self.dataset.output_dims)],
            headers=['Channel name', 'Error (' + method + ')'],
            tablefmt='fancy_grid',
            numalign='center',
            stralign='center'
        ) + '\n')
        if plot:
            for p in range(self.dataset.output_dims):
                plt.figure(figsize=(12, 10))
                plt.xlabel('Test set values - Dim ' + str(p+1))
                plt.ylabel('Error')
                plt.plot(y_test, error, linestyle=None, marker='.')
                plt.show()
        return error

    def train(self, method='LBFGS', verbose=False, restart_opt=5):
        """
        Train the model given the dataset and kernel.

        Parameters
        ----------
        method : str, optional
            Method used for the optmization. Only LBFGS is implemented currently.
            The default is 'LBFGS'.
        iters : int, optional
            Nb of iterations in the optimization algorithm.
            The default is 1000.
        verbose : bool, optional
            If True, information on the optimization are printed.
            The default is False.
        restart_opt : int, optional
            Number of restart of the optimizers. The default is 40.

        """
        self.dataset.preprocessing()
        if verbose:
            print('\nStarting optimization with', method, 'method')
        
        if self.kernel_ is None:
            self.kernel_ = self.kernel.clone()
            theta_init_restart = self.kernel_.theta
        else:
            theta_init_restart = None
        
        def loss(theta):
            # print (self.kernel_)
            return -self.log_marginal_likelihood(theta)

        def constrained_opt(loss, initial_theta, bounds):
            opt_res = scipy.optimize.minimize(
                loss,
                initial_theta,
                method="L-BFGS-B",
                jac=False,
                bounds=bounds
            )
            return opt_res.x, opt_res.fun

        bounds = self.kernel_.bounds
        theta_bounds = np.log(bounds)
        optima = [constrained_opt(loss, self.kernel_.theta, theta_bounds)]
        if np.shape(bounds)[0] != len(self.kernel_.theta):
            raise ValueError('Kernel bounds should have same dimension as theta')
        for _ in range(restart_opt):
            if theta_init_restart is None:
                theta_init = [random.uniform(theta_bounds[k][0], theta_bounds[k][1])
                              for k in range(len(theta_bounds))]
            else:
                theta_init = theta_init_restart
            optima.append(constrained_opt(loss, theta_init, theta_bounds))
        lml_values = list(map(operator.itemgetter(1), optima))
        self.kernel_.theta = optima[np.argmin(lml_values)][0]
        self.kernel_.set_params(self.kernel_.theta)
        self.lml_value_ = -np.min(lml_values)
        print('LML value = ', self.lml_value_)

        if verbose:
            print('LML value = ', self.lml_value_)

        x_train, y_train = self.dataset.get_trans_train_data()
        self.kernel_.set_xtrain(x_train)
        K = self.kernel_(x_train)
        try:
            self.L_ = scipy.linalg.cholesky(K, lower=True, check_finite=False)
        except np.linalg.LinAlgError as exc:
            exc.args = (
                "The kernel is not returning a positive "
                "definite matrix.",) + exc.args
            raise
        self.alpha_ = scipy.linalg.cho_solve((self.L_, True), y_train, check_finite=False)
        return self


    def predict(self, x=None, sigma_nb=2.0, full_cov=False, error=None, plot_error=False, untransformed=False, nugget=True):
        """
        Predict on the test set, or a given set. Returns mean and confidence interval bounds.
        Covariance of predictions can be returned for multioutput GPs.

        Parameters
        ----------
        x : array of shape (n, input_dims), optional
            Inputs on which predictions are made. If None, the test set inputs are used.
            The default is None.
        sigma_nb : float, optional
            Confidence interval bounds are provided for any 'sigma_nb'.
            The default is 2.0.
        full_cov : bool, optional
            If True, the prediction covariance is returned as well. The default is False.
        error : str, optional
            Error measure to use to compare with test set. The default is 'MAE'.

        Returns
        -------
        ymean, ylow, yup : tuple of array of shape (n, output_dims).
            Mean, Lower CI bound, and upper CI bound of the predictions.
        ycov if full_cov = True: array of shape (n, output_dims, output_dims).
            Covariance of predicitons for multioutput GPs.
        """
        D = self.dataset.output_dims
        x_train, _ = self.dataset.get_trans_train_data()
        if x is None:
            x_pred = self.dataset.get_trans_test_data()[0]
        else:
            if len(np.shape(x)) < 2:
                if self.dataset.input_dims > 1:
                    x = np.atleast_2d(x)
                else:
                    x = np.atleast_2d(x).T
            x_pred = self.dataset.transform_input(x)
        if full_cov:
            n = len(x_pred)
            if nugget:
                K_trans = self.kernel_.k_trans(x_pred)
            else:
                K_trans = self.kernel_.k1.k_trans(x_pred)
            y_mean = K_trans @ self.alpha_
            V = scipy.linalg.solve_triangular(
                self.L_, K_trans.T, lower=True, check_finite=False
            )
            if nugget:
                y_cov = self.kernel_(x_pred)
            else:
                y_cov = self.kernel_.k1(x_pred)
            y_cov -= np.matmul(V.T, V)
            y_var = np.zeros((n, D, D))
            for p in range(D):
                for q in range(D):
                    y_var[:, p, q] = np.diag(y_cov[p*n:(p+1)*n, q*n:(q+1)*n])
            for k in range(len(y_var)):
                diag = np.diag(y_var[k])
                new_diag = diag * (diag > 0)
                np.fill_diagonal(y_var[k], new_diag)
                y_var[k] += 1e-8 * np.eye(D)
                
            y_low = y_mean - sigma_nb*np.sqrt(np.atleast_2d(np.diag(y_cov)).T)
            y_up = y_mean + sigma_nb*np.sqrt(np.atleast_2d(np.diag(y_cov)).T)

        else:
            if D > 1:
                n = len(x_pred)
                if nugget:
                    K_trans = self.kernel_.k_trans(x_pred)
                else:
                    K_trans = self.kernel_.k1.k_trans(x_pred)
                y_mean = K_trans @ self.alpha_
                V = scipy.linalg.solve_triangular(
                    self.L_, K_trans.T, lower=True, check_finite=False
                )
                if nugget:
                    y_cov = self.kernel_(x_pred)
                else:
                    y_cov = self.kernel_.k1(x_pred)
                y_cov -= np.matmul(V.T, V)
                y_var = np.zeros((n, D, D))
                for p in range(D):
                    for q in range(D):
                        y_var[:, p, q] = np.diag(y_cov[p*n:(p+1)*n, q*n:(q+1)*n])
                for k in range(len(y_var)):
                    diag = np.diag(y_var[k])
                    new_diag = diag * (diag > 0)
                    np.fill_diagonal(y_var[k], new_diag)
                y_low = y_mean - sigma_nb*np.sqrt(np.atleast_2d(np.diag(y_cov)).T)
                y_up = y_mean + sigma_nb*np.sqrt(np.atleast_2d(np.diag(y_cov)).T)
            else:
                n = len(x_pred)
                # K_trans = self.kernel_(x_pred, x_train)
                if nugget:
                    K_trans = self.kernel_.k_trans(x_pred)
                else:
                    K_trans = self.kernel_.k1.k_trans(x_pred)
                y_mean = K_trans @ self.alpha_
                V = scipy.linalg.solve_triangular(
                    self.L_, K_trans.T, lower=True, check_finite=False)
                if nugget:
                    y_var = self.kernel_.diag(x_pred)
                else:
                    y_var = self.kernel_.k1.diag(x_pred)
                y_var -= np.einsum("ij,ji->i", V.T, V)
                y_var = np.reshape(y_var, (len(y_var), 1))
                y_var_neg = y_var < 0
                if np.any(y_var_neg):
                    warnings.warn(
                        "Predicted variances smaller than 0."
                        "Setting these variances to 0."
                    )
                    y_var[y_var_neg] = 0.0
                y_low = y_mean - sigma_nb*np.sqrt(y_var)
                y_up = y_mean + sigma_nb*np.sqrt(y_var)
        if untransformed:
            if self.dataset.flatten and self.dataset.output_dims > 1:
                n = np.shape(y_mean)[0]
                D = self.dataset.output_dims
                y_mean_new = np.zeros((n//D, D))
                y_low_new = np.zeros((n//D, D))
                y_up_new = np.zeros((n//D, D))
                for d in range(D):
                    y_mean_new[:, d] = y_mean[d*n//D:(d+1)*n//D, 0]
                    y_low_new[:, d] = y_low[d*n//D:(d+1)*n//D, 0]
                    y_up_new[:, d] = y_up[d*n//D:(d+1)*n//D, 0]
            if full_cov:
                return y_mean_new, y_low_new, y_up_new, y_cov
            return y_mean_new, y_low_new, y_up_new, y_var
        y_low_new = self.dataset.inv_transform_output(y_low)
        y_up_new = self.dataset.inv_transform_output(y_up)
        y_mean_new = self.dataset.inv_transform_output(y_mean)
        y_std_new = (y_up_new-y_low_new)/(2*sigma_nb)
        y_std_old = np.zeros_like(y_std_new)
        
        if D > 1:
            for p in range(D):
                y_std_old[:, p] = np.copy(np.sqrt(y_var[:, p, p]))
            for p in range(D):
                for q in range(D):
                    y_var[:, p, q] *= y_std_new[:, p]*y_std_new[:, q] / \
                        (y_std_old[:, p]*y_std_old[:, q])
        else:
            y_std_old = (y_up - y_low)/(2*sigma_nb)
            y_var = y_var/(y_std_old**2)
            y_var *= (y_std_new**2)

        if error in ['MAE', 'MSE', 'MAPE', 'all'] and x is None:
            self.error(y_mean_new, (y_low_new, y_up_new), error, plot=plot_error)
        if full_cov:
            if D > 1:
                a = np.atleast_2d(np.ravel(y_std_old.T))
                std_mat_old = np.dot(a.T, a)
                a = np.atleast_2d(np.ravel(y_std_new.T))
                std_mat_new = np.dot(a.T, a)
                y_cov /= std_mat_old
                y_cov *= std_mat_new
                return y_mean_new, y_low_new, y_up_new, y_cov
            else:
                y_cov /= np.dot(y_std_old, y_std_old.T)
                y_cov *= np.dot(y_std_new, y_std_new.T)
            return y_mean_new, y_low_new, y_up_new, y_cov
        return y_mean_new, y_low_new, y_up_new, y_var




    def log_marginal_likelihood(self, theta=None):
        """
        Returns log marginal likelihood for given hyperparameters value.

        Parameters
        ----------
        theta : TYPE, optional
            List of hyperparameters value. If None, current kernel state is used.
            The default is None.

        Returns
        -------
        log_likelihood : float
            Log marginal likelihood value.

        """
        x_train, y_train = self.dataset.get_trans_train_data()
        if theta is None:
            return self.lml_value_
        kern = self.kernel_
        kern.theta = theta
        kern.set_params(theta)
        K = kern(x_train)
        try:
            L = scipy.linalg.cholesky(K, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return -np.inf
        alpha = scipy.linalg.cho_solve((L, True), y_train, check_finite=False)
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        # the log likehood is sum-up across the outputs
        log_likelihood = log_likelihood_dims.sum(axis=-1)
        return log_likelihood


    def add_data(self, x_add, y_add):
        if self.dataset.boxcox and np.any(y_add) < 0:
            y_add[y_add < 0] = 1e-6
        x_train, y_train = self.dataset.get_train_data()
        x_train_tot = np.append(x_train, x_add, axis=0)
        y_train_tot = np.append(y_train, y_add, axis=0)
        self.dataset.set_train_set(x_train_tot, y_train_tot)
        self.dataset.preprocessing()
        
        x_train_tot, y_train_tot = self.dataset.get_trans_train_data()
        
        self.kernel_.set_xtrain(x_train_tot)
        K = self.kernel_(x_train_tot)
        try:
            self.L_ = scipy.linalg.cholesky(K, lower=True, check_finite=False)
        except np.linalg.LinAlgError as exc:
            exc.args = (
                "The kernel is not returning a positive definite matrix.", ) + exc.args
            raise
        self.alpha_ = scipy.linalg.cho_solve((self.L_, True), y_train_tot, check_finite=False)
