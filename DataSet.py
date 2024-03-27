import numpy as np
import matplotlib.pyplot as plt
import scipy


def boxcox_man(x, lmbda):
    """
    Box-Cox transform for a given lambda. Used when x has only one datapoint.
    """
    if lmbda == 0:
        return np.log(x)
    return (np.power(x, lmbda)-1)/lmbda


def boxcox_inv(x, lmbda):
    """
    Inverse Box-Cox transform for a given lambda.
    """
    if lmbda == 0:
        return np.exp(x)
    return np.power(lmbda*x+1, 1/lmbda)


def yj_inv(y, lambdas):
    """
    Inverse Yeo-Johnson transform for a given lambda.
    """
    res = np.zeros_like(y)
    pos = y[:] >= 0
    if abs(lambdas) < np.spacing(1.0):
        res[pos] = np.exp(y[pos]) - 1
    else:
        res[pos] = np.power(y[pos]*lambdas + 1, 1/lambdas) - 1
    if abs(lambdas - 2) < np.spacing(1.0):
        res[~pos] = 1 - np.exp(-y[~pos])
    else:
        res[~pos] = 1 - np.power((lambdas - 2)*y[~pos] + 1, 1/(2 - lambdas))
    return res


def yj_man(x, lambdas):
    """
    Yeo-johnson transform for a given lambda.
    """
    res = np.zeros_like(x)
    pos = x[:] >= 0
    if lambdas == 0:
        res[pos] = np.log(x[pos]+1)
    else:
        res[pos] = (np.power(x[pos] + 1, lambdas) - 1)/lambdas
    if lambdas == 2:
        res[~pos] = - np.log(-x[~pos] + 1)
    else:
        res[~pos] = - (np.power((-x[~pos] + 1), (2 - lambdas)) - 1) / ((2 - lambdas))
    return res


class DataSet:
    """
    DataSet class describes datasets for any given inputs and outputs dimensions.
    To initiate a DataSet, use DataSet(x, y) with x and y np.array of shape (nsamples,input_dims)
    and (nsamples,output_dims)
    """

    def __init__(self, x, y, names, test_frac, standardize=True, boxcox=True,
                 yeo_johnson=False, log=False, flatten=True, no_transform_inputs=False, no_transform_outputs=False,):
        """
        Parameters
        ----------
        x : 2D array of shape (ndata, input_dims)
            Input of dataset.
        y : 2D array of shape (ndata, output_dims)
            Output of dataset.
        names : list of str, of length output_dims
            Names for each output channels.
        test_frac : float between 0 and 1.
            Fraction of the data set that should be used as a test set.
        standardize : boolean.
        Describe whether the data should be standardize or not. Default is True.
        boxcox : boolean.
        Describe whether the Box-Cox transform should be applied to the inputs and outputs or not.
        Default is True.
        yeo_johnson : boolean.
        Describe whether the Yeo-Johnson transform should be applied
        to the inputs and outputs or not. Default is False.
        log : boolean.
        Describe whether the log-data should be considered instead of the data themselves.
        Default is False.
        """
        if len(np.shape(x)) == 1:
            x = np.atleast_2d(x).T
        if len(np.shape(y)) == 1:
            y = np.atleast_2d(y).T
        if len(names) != np.shape(y)[1]:
            raise ValueError('Channel names should be the same length as output dimensions')
        if np.shape(x)[0] != np.shape(y)[0]:
            raise ValueError('x and y should have the same number of sample points')
        if boxcox and yeo_johnson:
            raise ValueError('Cannot apply both Box-Cox and Yeo-Johnson transforms')

        self.all_inp = x
        self.all_out = y
        self.test_frac = test_frac
        self.nsamples = np.shape(x)[0]
        self.input_dims = np.shape(x)[1]
        self.output_dims = np.shape(y)[1]
        self.channel_names = names

        ntest = int(test_frac*self.nsamples)
        ind = np.random.choice(self.nsamples, size=ntest, replace=False)
        self.test_indices = ind
        all_ind = np.arange(len(self.all_inp))
        self.train_indices = np.delete(all_ind, ind)
        self.test_inputs = x[ind, :]
        self.test_outputs = y[ind, :]
        self.train_inputs = np.delete(x, ind, axis=0)
        self.train_outputs = np.delete(y, ind, axis=0)

        self.standardize = standardize
        self.boxcox = boxcox
        self.yeo = yeo_johnson
        self.log = log
        self.flatten = flatten
        self.no_transform_inputs = no_transform_inputs
        self.no_transform_outputs = no_transform_outputs

        self.train_inputs_trans = None
        self.train_outputs_trans = None
        self.test_inputs_trans = None
        self.test_outputs_trans = None
        self.standard_inp = None
        self.standard_out = None
        self.lambdas_out = None
        self.lambdas_inp = None
        self.train_outputs_trans_noflat = None
        self.test_outputs_trans_noflat = None

    def set_test_train_data(self, Xtrain, ytrain, Xtest, ytest):
        """
        Change the training and testing data
        """
        if np.shape(Xtrain)[0] != np.shape(ytrain)[0]:
            raise ValueError('x and y should have the same number of sample points')

        self.all_inp = np.append(Xtrain, Xtest)
        self.all_out = np.append(ytrain, ytest)
        self.nsamples = np.shape(Xtrain)[0] + np.shape(Xtest)[0]
        self.input_dims = np.shape(Xtrain)[1]
        self.output_dims = np.shape(ytrain)[1]

        self.test_frac = float(np.shape(ytest)[0])/(np.shape(ytest)[0]+np.shape(ytrain)[0])
        self.test_inputs = Xtest
        self.test_outputs = ytest
        self.train_inputs = Xtrain
        self.train_outputs = ytrain

    def set_train_set(self, x_train, y_train):
        """
        Set the train set inputs and outputs of the object
        """
        self.train_inputs = x_train
        self.train_outputs = y_train

    def set_test_set(self, x_test, y_test):
        """
        Set the test set inputs and outputs of the object
        """
        self.test_inputs = x_test
        self.test_outputs = y_test

    def get_train_data(self):
        """
        Returns object train set inputs and outputs
        """
        return self.train_inputs, self.train_outputs

    def get_test_data(self):
        """
        Returns object test set inputs and outputs
        """
        return self.test_inputs, self.test_outputs

    def get_noflat_trans_train_data(self):
        """
        Returns object train set inputs and outputs
        """
        return self.train_inputs_trans, self.train_outputs_trans_noflat

    def get_noflat_trans_test_data(self):
        """
        Returns object test set inputs and outputs
        """
        return self.test_inputs_trans, self.test_outputs_trans_noflat

    def get_trans_train_data(self):
        """
        Returns the object train set inputs and outputs after pre-processing.
        """
        return self.train_inputs_trans, self.train_outputs_trans

    def get_trans_test_data(self):
        """
        Returns the object test set inputs and outputs after pre-processing.
        """
        return self.test_inputs_trans, self.test_outputs_trans

    def norm_plot(self):
        """

        Returns
        -------
        None.

        """
        x_train, y_train = self.get_train_data()
        x_train_trans = np.copy(x_train)
        y_train_trans = np.copy(y_train)
        D = self.output_dims
        I = self.input_dims

        if not self.no_transform_outputs:
            if self.boxcox:
                maxlog = np.zeros(D)
                fig = plt.figure(figsize=(9, 10))
                for p in range(D):
                    ax = fig.add_subplot(D, 1, p+1)
                    yp_train_trans = y_train_trans[:, p]
                    _, _ = scipy.stats.boxcox_normplot(yp_train_trans, -20, 20, plot=ax)
                    _, maxlog[p] = scipy.stats.boxcox(yp_train_trans)
                    ax.axvline(maxlog[p], color='r', linestyle='--')
                plt.show()

            if self.yeo:
                maxlog = np.zeros(D)
                fig = plt.figure(figsize=(9, 10))
                for p in range(D):
                    ax = fig.add_subplot(D, 1, p+1)
                    yp_train_trans = y_train_trans[:, p]
                    _, _ = scipy.stats.yeojohnson_normplot(yp_train_trans, -20, 20, plot=ax)
                    _, maxlog[p] = scipy.stats.yeojohnson(yp_train_trans)
                    ax.axvline(maxlog[p], color='r', linestyle='--')
                plt.show()
        if not self.no_transform_inputs:
            if self.boxcox:
                maxlog = np.zeros(I)
                fig = plt.figure(figsize=(9, 20))
                for p in range(I):
                    ax = fig.add_subplot(D, 1, p+1)
                    xp_train_trans = x_train_trans[:, p]
                    _, _ = scipy.stats.boxcox_normplot(xp_train_trans, -20, 20, plot=ax)
                    _, maxlog[p] = scipy.stats.boxcox(xp_train_trans)
                    ax.axvline(maxlog[p], color='r', linestyle='--')
                plt.show()

            if self.yeo:
                maxlog = np.zeros(I)
                fig = plt.figure(figsize=(9, 20))
                for p in range(I):
                    ax = fig.add_subplot(I, 1, p+1)
                    xp_train_trans = x_train_trans[:, p]
                    _, _ = scipy.stats.yeojohnson_normplot(xp_train_trans, -20, 20, plot=ax)
                    _, maxlog[p] = scipy.stats.yeojohnson(xp_train_trans)
                    ax.axvline(maxlog[p], color='r', linestyle='--')
                plt.show()

    def preprocessing(self):
        """
        Apply Yeo-Johnson transform to all outputs and standardization to all inputs.
        YJ is fitted on training set. Returns (y_train, y_test) after transform.
        In Multi-output, transforms the output data in a single column.
        """

        x_train, y_train = self.get_train_data()
        x_test, y_test = self.get_test_data()
        x_train_trans = np.copy(x_train)
        x_test_trans = np.copy(x_test)
        y_train_trans = np.copy(y_train)
        y_test_trans = np.copy(y_test)

        if np.shape(x_train)[0] == 1 or np.shape(x_test)[0] == 1:
            self.train_inputs_trans = x_train_trans
            self.train_outputs_trans = y_train_trans
            self.test_inputs_trans = x_test_trans
            self.test_outputs_trans = y_test_trans
            self.standard_inp = np.zeros((self.output_dims, 2))
            self.standard_inp[:, 1] = np.ones(self.output_dims)
            self.standard_out = np.zeros((self.output_dims, 2))
            self.standard_out[:, 1] = np.ones(self.output_dims)
            self.boxcox = False
            self.yeo = False
            return 1

        if not self.no_transform_outputs:
            if self.log:
                y_train_trans = np.log(y_train_trans)
                y_test_trans = np.log(y_test_trans)
            if self.boxcox:
                lambdas_out = np.zeros(self.output_dims)
                for p in range(self.output_dims):
                    yp_train_trans = y_train_trans[:, p]
                    yp_test_trans = y_test_trans[:, p]
                    yp_train_trans, lambdas_out[p] = scipy.stats.boxcox(yp_train_trans)
                    yp_test_trans = scipy.stats.boxcox(yp_test_trans, lmbda=lambdas_out[p])
                    y_train_trans[:, p] = yp_train_trans
                    y_test_trans[:, p] = yp_test_trans
                self.lambdas_out = lambdas_out
            elif self.yeo:
                lambdas_out = np.zeros(self.output_dims)
                for p in range(self.output_dims):
                    yp_train_trans = y_train_trans[:, p]
                    yp_test_trans = y_test_trans[:, p]
                    yp_train_trans, lambdas_out[p] = scipy.stats.yeojohnson(yp_train_trans)
                    yp_test_trans = scipy.stats.yeojohnson(yp_test_trans, lmbda=lambdas_out[p])
                    y_train_trans[:, p] = yp_train_trans
                    y_test_trans[:, p] = yp_test_trans
                self.lambdas_out = lambdas_out
            if self.standardize:
                standard_out = np.zeros((self.output_dims, 2))
                for p in range(self.output_dims):
                    yp_train_trans = y_train_trans[:, p]
                    yp_test_trans = y_test_trans[:, p]
                    mean = np.mean(yp_train_trans)
                    sigma = np.std(yp_train_trans)
                    y_train_trans[:, p] = (yp_train_trans-mean)/sigma
                    y_test_trans[:, p] = (yp_test_trans-mean)/sigma
                    standard_out[p] = mean, sigma
                    self.standard_out = standard_out

        if not self.no_transform_inputs:
            if self.boxcox:
                lambdas_inp = np.zeros(self.input_dims)
                for p in range(self.input_dims):
                    xp_train_trans = x_train_trans[:, p]
                    xp_test_trans = x_test_trans[:, p]
                    xp_train_trans, lambdas_inp[p] = scipy.stats.boxcox(xp_train_trans+1e-6)
                    xp_test_trans = scipy.stats.boxcox(xp_test_trans+1e-6, lmbda=lambdas_inp[p])
                    x_train_trans[:, p] = xp_train_trans
                    x_test_trans[:, p] = xp_test_trans
                self.lambdas_inp = lambdas_inp
            elif self.yeo:
                lambdas_inp = np.zeros(self.input_dims)
                for p in range(self.input_dims):
                    xp_train_trans = x_train_trans[:, p]
                    xp_test_trans = x_test_trans[:, p]
                    xp_train_trans, lambdas_inp[p] = scipy.stats.yeojohnson(xp_train_trans+1e-6)
                    xp_test_trans = scipy.stats.yeojohnson(xp_test_trans+1e-6, lmbda=lambdas_inp[p])
                    x_train_trans[:, p] = xp_train_trans
                    x_test_trans[:, p] = xp_test_trans
                self.lambdas_inp = lambdas_inp
            if self.standardize:
                standard_inp = np.zeros((self.input_dims, 2))
                for p in range(self.input_dims):
                    xp_train_trans = x_train_trans[:, p]
                    xp_test_trans = x_test_trans[:, p]
                    mean = np.mean(xp_train_trans)
                    sigma = np.std(xp_train_trans)
                    x_train_trans[:, p] = (xp_train_trans-mean)/sigma
                    x_test_trans[:, p] = (xp_test_trans-mean)/sigma
                    standard_inp[p] = mean, sigma
                self.standard_inp = standard_inp

        self.train_outputs_trans_noflat = np.copy(y_train_trans)
        self.test_outputs_trans_noflat = np.copy(y_test_trans)
        if self.output_dims > 1 and self.flatten:
            y_train_trans = np.atleast_2d(np.ravel(y_train_trans.T)).T
            y_test_trans = np.atleast_2d(np.ravel(y_test_trans.T)).T
        self.train_inputs_trans = x_train_trans
        self.train_outputs_trans = y_train_trans
        self.test_inputs_trans = x_test_trans
        self.test_outputs_trans = y_test_trans
        return 1

    def transform_output(self, y, noflatten=False):
        """
        Yeo-Johnson transform applied to y. YJ transform is fit with training set.
        """
        ycop = np.copy(y)
        if not self.no_transform_outputs:
            if self.log:
                ycop = np.log(ycop)
            if self.boxcox:
                if len(np.shape(ycop)) == 1:
                    y = np.atleast_2d(ycop)
                added = False
                if np.shape(ycop)[0] == 1:
                    d = np.shape(ycop)[1]
                    ycop = np.append(ycop, ycop + np.ones(d), axis=0)
                    added = True
                lambdas = self.lambdas_out
                y_new = np.copy(ycop)
                for p in range(self.output_dims):
                    yp = y_new[:, p]
                    yp_new = scipy.stats.boxcox(yp, lmbda=lambdas[p])
                    y_new[:, p] = yp_new
                if added:
                    y_new = y_new[:-1]

            elif self.yeo:
                if len(np.shape(ycop)) == 1:
                    ycop = np.atleast_2d(ycop)
                added = False
                if np.shape(ycop)[0] == 1:
                    d = np.shape(ycop)[1]
                    ycop = np.append(ycop, ycop + np.ones(d), axis=0)
                    added = True
                lambdas = self.lambdas_out
                y_new = np.copy(ycop)
                for p in range(self.output_dims):
                    yp = y_new[:, p]
                    yp_new = scipy.stats.yeojohnson(yp, lmbda=lambdas[p])
                    y_new[:, p] = yp_new
                if added:
                    y_new = y_new[:-1]

            else:
                y_new = ycop
            if self.standardize:
                standard_out = self.standard_out
                for p in range(self.output_dims):
                    yp = np.copy(y_new[:, p])
                    mean = standard_out[p, 0]
                    sigma = standard_out[p, 1]
                    yp = (yp-mean)/sigma
                    y_new[:, p] = yp
        else:
            y_new = np.copy(ycop)
        if noflatten or not self.flatten:
            return y_new
        y_new = np.atleast_2d(np.ravel(y_new.T)).T
        return y_new

    def inv_transform_output(self, y_test, deflatten_data=True):
        """
        Apply Inverse Yeo-Johnson transform to all outputs.
        YJ is fitted on training set.
        """
        if y_test.ndim == 2:
            if deflatten_data and self.flatten:
                n = np.shape(y_test)[0]
                D = self.output_dims
                ynew = np.zeros((n//D, D))
                for d in range(D):
                    ynew[:, d] = y_test[d*n//D:(d+1)*n//D, 0]
            else:
                ynew = np.copy(y_test)
    
            if not self.no_transform_outputs:
                y_back = np.copy(ynew)
                if self.standardize:
                    standard_out = self.standard_out
                    for p in range(self.output_dims):
                        mean, sigma = standard_out[p]
                        yp_back = y_back[:, p]
                        yp_back = yp_back*sigma + mean
                        y_back[:, p] = yp_back
    
                if self.boxcox:
                    for p in range(self.output_dims):
                        y_back[:, p] = boxcox_inv(y_back[:, p], self.lambdas_out[p])
                elif self.yeo:
                    for p in range(self.output_dims):
                        y_back[:, p] = yj_inv(y_back[:, p], self.lambdas_out[p])
                if self.log:
                    y_back = np.exp(y_back)
                return y_back
            return ynew
        elif y_test.ndim == 3:
            K = y_test.shape[0]
            if deflatten_data and self.flatten:
                n = np.shape(y_test)[1]
                D = self.output_dims
                ynew = np.zeros((K, n//D, D))
                for d in range(D):
                    ynew[:, :, d] = y_test[: d*n//D:(d+1)*n//D, 0]
            else:
                ynew = np.copy(y_test)
    
            if not self.no_transform_outputs:
                y_back = np.copy(ynew)
                if self.standardize:
                    standard_out = self.standard_out
                    for p in range(self.output_dims):
                        mean, sigma = standard_out[p]
                        yp_back = y_back[:, :,  p]
                        yp_back = yp_back*sigma + mean
                        y_back[:, :, p] = yp_back
    
                if self.boxcox:
                    for p in range(self.output_dims):
                        y_back[:, :, p] = boxcox_inv(y_back[:, :, p], self.lambdas_out[p])
                elif self.yeo:
                    for p in range(self.output_dims):
                        y_back[:, :, p] = yj_inv(y_back[:, :, p], self.lambdas_out[p])
                if self.log:
                    y_back = np.exp(y_back)
                return y_back
            return ynew
            
            

    def transform_input(self, x):
        """
        Apply standardization to all inputs.
        """
        if len(np.shape(x)) < 2:
            if self.input_dims > 1:
                x = np.atleast_2d(x)
            else:
                x = np.atleast_2d(x).T
        x_new = np.copy(x)
        if not self.no_transform_inputs:
            if self.boxcox:
                for p in range(self.input_dims):
                    xp_new = x_new[:, p]
                    if len(xp_new) == 1:
                        x_new[:, p] = boxcox_man(xp_new, self.lambdas_inp[p])
                    else:
                        x_new[:, p] = scipy.stats.boxcox(
                            xp_new+1e-10*np.linspace(0.1, 1, len(xp_new)), self.lambdas_inp[p])

            if self.yeo:
                for p in range(self.input_dims):
                    xp_new = x_new[:, p]
                    if len(xp_new) == 1:
                        x_new[:, p] = yj_man(xp_new, self.lambdas_inp[p])
                    else:
                        x_new[:, p] = scipy.stats.yeojohnson(
                            xp_new+1e-10*np.linspace(0.1, 1, len(xp_new)), self.lambdas_inp[p])
            if self.standardize:
                standard = self.standard_inp
                for p in range(self.input_dims):
                    xp_new = x_new[:, p]
                    mean, sigma = standard[p]
                    x_new[:, p] = (xp_new-mean)/sigma
        return x_new

    def inv_transform_input(self, x_test):
        """
        Apply Inverse standardization to inputs
        Standardization is fitted on training set.
        """
        x_back = np.copy(x_test)

        if not self.no_transform_inputs:
            if self.standardize:
                standard = self.standard_inp
                for p in range(self.input_dims):
                    mean, sigma = standard[p]
                    xp_test = x_test[:, p]
                    xp_back = xp_test*sigma + mean
                    x_back[:, p] = xp_back
            if self.boxcox:
                for p in range(self.input_dims):
                    xp_back = x_back[:, p]
                    x_back[:, p] = boxcox_inv(xp_back, self.lambdas_inp[p])
            if self.yeo:
                for p in range(self.input_dims):
                    xp_back = x_back[:, p]
                    x_back[:, p] = yj_inv(xp_back, self.lambdas_inp[p])
        return x_back
