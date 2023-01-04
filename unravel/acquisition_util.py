import GPyOpt
import numpy as np


class FUR(GPyOpt.acquisitions.base.AcquisitionBase):
    """
    General template to create a new GPyOPt acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function that provides the evaluation cost and its gradients

    """

    analytical_gradient_prediction = False

    def __init__(
        self, model, space, optimizer, X_init, std, cost_withGradients=None, **kwargs
    ):
        self.optimizer = optimizer
        self.X_init = X_init
        self.iter = 1
        self.std = std
        super(FUR, self).__init__(model, space, optimizer)

    def _compute_acq(self, x):
        if x.shape[0] > 1:
            self.iter += 1
            self.delta = np.random.randn()
        m, s = self.model.predict(x)
        f_acqu = (
            -np.linalg.norm(x - self.X_init - self.std * self.delta / np.log(self.iter))
            + s
        )
        print("x: ",x)
        print("X_init: ", self.X_init)
        print(f_acqu)
        return f_acqu

class IF_FUR(GPyOpt.acquisitions.base.AcquisitionBase):

    """
    General template to create a new GPyOPt acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function that provides the evaluation cost and its gradients

    """

    analytical_gradient_prediction = False

    def __init__(
        self, model, space, optimizer, X_init, std, cost_withGradients=None, **kwargs
    ):
        self.optimizer = optimizer
        self.X_init = X_init
        self.iter = 1
        self.std = std
        super(IF_FUR, self).__init__(model, space, optimizer)
    
    def _compute_IF(self,x):
        import numpy as np
        import argparse
        import time
        import pdb
        import os
        np.random.seed(0)

        from scipy import sparse
        from sklearn.linear_model import LogisticRegression
        from sklearn.linear_model import Ridge
        from sklearn.metrics import log_loss, roc_auc_score
        import sys
        sys.path.insert(0, os.path.dirname(os.getcwd()))

        sys.path.append('..')
        from grad_utils import grad_logloss_theta_lr
        from grad_utils import batch_grad_logloss_lr
        from inverse_hvp import inverse_hvp_lr_newtonCG

        print("IF Calculation")
        #Loading black box utilities
        from evaluation.blackbox_util import BlackBoxSimulator
        dataset_utilities = BlackBoxSimulator().load_breast_cancer_utilities()
        [X_train, y_train, X_test, y_test, features, model, mode, categorical_features, sample_idx] = dataset_utilities.values()
        
        x_va = X_train[-50:]
        y_va = y_train[-50:]
        X_train = X_train[:512-50]
        y_train = y_train[:512-50]
        
        sigmoid_k = 10
        C = 0.1
        sample_ratio = 0.6
        flip_ratio = 0.4
        num_tr_sample = X_train.shape[0]
        obj_sample_size = int(sample_ratio * num_tr_sample)

        clf = LogisticRegression(
                C = C,
                fit_intercept=False,
                tol = 1e-8,
                solver="liblinear",
                multi_class="ovr",
                max_iter=10,
                warm_start=False,
                verbose=0,
                )


        clf.fit(X_train,y_train)
        y_va_pred = clf.predict_proba(x_va)[:,1]
        full_logloss = log_loss(y_va,y_va_pred)
        weight_ar = clf.coef_.flatten()
        y_te_pred = clf.predict_proba(X_test)[:,1]
        full_te_logloss = log_loss(y_test,y_te_pred)
        full_te_auc = roc_auc_score(y_test, y_te_pred)
        y_te_pred = clf.predict(X_test)
        full_te_acc = (y_test == y_te_pred).sum() / y_test.shape[0]

        if_start_time = time.time()
        test_grad_loss_val = grad_logloss_theta_lr(y_va,y_va_pred,x_va,weight_ar,C,False,0.1/(num_tr_sample*C))
        tr_pred = clf.predict_proba(X_train)[:,1]
        batch_size = 100
        M = None
        total_batch = int(np.ceil(num_tr_sample / float(batch_size)))
        for idx in range(total_batch):
            batch_tr_grad = batch_grad_logloss_lr(y_train[idx*batch_size:(idx+1)*batch_size],
                tr_pred[idx*batch_size:(idx+1)*batch_size],
                X_train[idx*batch_size:(idx+1)*batch_size],
                weight_ar,
                C,
                False,
                1.0)

            sum_grad = batch_tr_grad.multiply(X_train[idx*batch_size:(idx+1)*batch_size]).sum(0)
            if M is None:
                M = sum_grad
            else:
                M = M + sum_grad
                
        M = M + 0.1/(num_tr_sample*C) * np.ones(X_train.shape[1])
        M = np.array(M).flatten()

        
        y_x = clf.predict(x)
        y_x0 = clf.predict(self.X_init)

        print("x: ", x)
        print("y_x: ", y_x)
        print("X_init: ", self.X_init)
        print("y_x0: ", y_x0)

        X_train_orig = np.concatenate((X_train, self.X_init))
        y_train_orig = np.concatenate((y_train, y_x0))
        X_train_pert = np.concatenate((X_train, x))
        y_train_pert = np.concatenate((y_train, y_x))
        tr_pred_orig = np.concatenate((tr_pred, y_x0))
        tr_pred_pert = np.concatenate((tr_pred, y_x))

        # iv_hvp_orig = inverse_hvp_lr_newtonCG(X_train_orig,y_train_orig,tr_pred_orig,test_grad_loss_val,C,True,1e-5,True,M,0.1/((num_tr_sample+1)*C))
        iv_hvp_perturbed = inverse_hvp_lr_newtonCG(X_train_pert,y_train_pert,tr_pred_pert,test_grad_loss_val,C,True,1e-5,True,M,0.1/((num_tr_sample+1)*C))
        
        # IF = -1 * (iv_hvp_perturbed - iv_hvp_orig) 
        IF =  -1 * np.linalg.norm(iv_hvp_perturbed)
        print("IF VALUE: ", IF)
        IF = 1
        return IF
    
    def _compute_acq(self, x):
        if x.shape[0] > 1:
            self.iter += 1
            self.delta = np.random.randn()
        m, s = self.model.predict(x)

        IF_val = -1 * 10000 * self._compute_IF(x)

        f_acqu = (
            #-np.linalg.norm(x - self.X_init - self.std * self.delta / np.log(self.iter)) + 
           s + IF_val
        )

        print("f_acqu: ", f_acqu)
        return f_acqu

class UR(GPyOpt.acquisitions.base.AcquisitionBase):

    """
    General template to create a new GPyOPt acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function that provides the evaluation cost and its gradients

    """

    # --- Set this line to true if analytical gradients are available
    analytical_gradient_prediction = False

    def __init__(self, model, space, optimizer, cost_withGradients=None, **kwargs):
        self.optimizer = optimizer
        super(UR, self).__init__(model, space, optimizer)

    def _compute_acq(self, x):
        print("UR")
        m, s = self.model.predict(x)
        f_acqu = s
        return f_acqu


class UCB(GPyOpt.acquisitions.base.AcquisitionBase):

    """
    General template to create a new GPyOPt acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function that provides the evaluation cost and its gradients

    """

    # --- Set this line to true if analytical gradients are available
    analytical_gradient_prediction = False

    def __init__(self, model, space, optimizer, cost_withGradients=None, **kwargs):
        self.optimizer = optimizer
        super(UCB, self).__init__(model, space, optimizer)

    def _compute_acq(self, x):
        print("UCB")
        m, s = self.model.predict(x)
        f_acqu = m + 2 * s
        return f_acqu
