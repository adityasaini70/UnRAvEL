import math
import GPy
import GPyOpt
import numpy as np
from unravel.plot_util import plot_scores
from unravel.kernel_util import Kernel
from unravel.acquisition_util import FUR, UR, UCB, IF_FUR


class UnRAVELTabularExplainer:
    def __init__(
        self,
        bbox_model,
        train_data,
        categorical_features=[],
        mode="regression",
    ):
        """
        Basic constructor

        Args:
            bbox_model (Any Scikit-Learn model): Black box prediction model
            train_data (Numpy Array): Data used to train bbox_model
            categorical_features (List, optional): List containing indices of categorical features
            mode (str, optional): Explanation mode. Can be 'regression', or 'classification'
        """

        self.bbox_model = bbox_model
        self.train_data = train_data
        self.categorical_features = categorical_features
        self.mode = mode
        self.surrogate_data = []
        self.gp_model = None
        self.std = 0

    def generate_domain(self, arr, interval=1, type="std_hypercube"):
        """
        Generates neighborhood around given test sample(arr)

        Args:
            arr (Numpy Array): Test sample for which explanations are required.
            interval (int, optional): Helps define upper and lower limit of exploration neighborhood. Defaults to 1.
            type (str, optional): Type of neighborhood. Defaults to "std_hypercube"(Neighborhood is a hypercube with its limits being defined by the standard deviations of respective features).

        Returns:
            List of dictionaries: Exploration neighborhood/Search space containing the upper and lower limits for each feature.
        """

        exploration_neighborhood = []

        if type == "std_hypercube":

            # Extracting standard deviation from training data
            std = np.std(self.train_data, axis=0)
            self.std = np.mean(std)

            # Confidence interval = 'interval' amounts of sigma
            lower_limit = arr - interval * std
            upper_limit = arr + interval * std

            # Generating domain for each feature
            for idx in range(arr.shape[0]):
                if idx in self.categorical_features:
                    limits = np.arange(
                        math.floor(lower_limit[idx]), math.ceil(upper_limit[idx]) + 1
                    ).tolist()
                    idx_domain = {
                        "name": f"var_{idx}",
                        "type": "discrete",
                        "domain": limits,
                    }
                else:
                    idx_domain = {
                        "name": f"var_{idx}",
                        "type": "continuous",
                        "domain": (lower_limit[idx], upper_limit[idx]),
                    }
                exploration_neighborhood.append(idx_domain)

        return exploration_neighborhood

    def f_p(self, arr):
        """
        Wrapper function for the bbox_model

        Args:
            arr (Numpy Array): Any sample similar to the dimensions of the training set

        Returns:
            Scalar value (float): Prediction for arr
        """
        if self.mode == "regression":
            return np.array([self.bbox_model.predict([arr.ravel()]).tolist()])
        elif self.mode == "classification":
            return np.array([[self.bbox_model.predict_proba([arr.ravel()])[0, 1]]])

    def generate_optimizer(
        self,
        X_init,
        kernel,
        max_iter=50,
        alpha="EI",
        jitter=5,
        interval=1,
        verbosity=False,
        maximize=False,
    ):
        """
        Returns Coefficients of GP(Gaussian Process) model after BO(Bayesian Optimization) routine

        Args:
            X_init (Numpy Array): Starting vector(or point) for the BO routine
            kernel (kernel_util.Kernel object): An instantiated 'Kernel' object
            max_iter (int, optional): Maximum iterations upto which BO should run. Defaults to 50.
            alpha (str, optional): Acquisition function like 'EI', 'LCB', or 'MPI'.  Defaults to 'EI'.
            jitter (float, optional): Exploration-explotation tradeoff for BO. Defaults to 0.05.
            interval (int, optional): Helps define upper and lower limit of exploration neighborhood. Defaults to 1.

        Returns:
            Numpy Array: Coefficients for each feature
        """

        # Generating starting point
        Y_init = self.f_p(X_init)

        # Generating domain for X_init
        bounds = self.generate_domain(X_init[0], interval=interval)

        # Initializing explainer model f_e
        if alpha == "FUR" or alpha == "UR" or alpha == "LCB_custom" or alpha == 'IF_FUR':
            objective = GPyOpt.core.task.SingleObjective(lambda x: self.f_p(x))
            space = GPyOpt.Design_space(space=bounds)
            model = GPyOpt.models.GPModel(kernel=kernel.kernel, verbose=False)
            aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)
            if alpha == "FUR":
                acquisition = FUR(
                    model,
                    space,
                    optimizer=aquisition_optimizer,
                    X_init=X_init,
                    std=self.std,
                )

            elif alpha == "IF_FUR":
                acquisition = IF_FUR(
                    model,
                    space,
                    optimizer=aquisition_optimizer,
                    X_init=X_init,
                    std=self.std,
                )

            elif alpha == "UR":
                acquisition = UR(model, space, optimizer=aquisition_optimizer)

            elif alpha == "UCB":
                acquisition = UCB(
                    model,
                    space,
                    optimizer=aquisition_optimizer,
                )
            evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
            f_optim = GPyOpt.methods.ModularBayesianOptimization(
                model, space, objective, acquisition, evaluator, X_init
            )
        else:
            f_optim = GPyOpt.methods.BayesianOptimization(
                f=lambda x: self.f_p(x),
                domain=bounds,
                model_type="GP",
                kernel=kernel.kernel,
                acquisition_type=alpha,
                X=X_init,
                Y=Y_init,
                acquisition_jitter=jitter,
                exact_feval=False,
                normalize_Y=True,
                maximize=maximize,
            )

        # Running the Bayesian Optimization Routine
        f_optim.run_optimization(max_iter = max_iter, max_time = 0.0001, verbosity=True)
        self.surrogate_data = f_optim.get_evaluations()
        self.gp_model = f_optim.model

        return f_optim

    def generate_scores(self, kernel, f_optim, importance_method="ARD", delta=1):
        """[summary]

        Args:
            kernel ([type]): [description]
            f_e ([type]): [description]
            importance_method (str, optional): [description]. Defaults to "ARD".
            delta (int, optional): [description]. Defaults to 1.

        Returns:
            [type]: [description]
        """
        # Extracting the feature coefficients
        if importance_method == "ARD":
            model_parameters = f_optim.model.get_model_parameters().tolist()[0]
            coefficients = kernel.get_coefficients(model_parameters)
            score = kernel.get_importance_score(coefficients)
        elif importance_method == "KL":
            # Storing the surrogate dataset generated through the BO routine
            X_surrogate = f_optim.get_evaluations()[0]
            # y_surrogate = f_e.get_evaluations()[1]

            # Storing the GP model trained during the BO routine
            gp_model = f_optim.model

            # Code credit: https://github.com/topipa/gp-varsel-kl-var
            n = X_surrogate.shape[0]
            p = X_surrogate.shape[1]

            jitter = 1e-15

            # perturbation
            deltax = np.linspace(-delta, delta, 3)

            # loop through the data points X
            relevances = np.zeros((n, p))

            for j in range(0, n):

                x_n = np.reshape(np.repeat(X_surrogate[j, :], 3), (p, 3))
                # loop through covariates
                for dim in range(0, p):

                    # perturb x_n
                    x_n[dim, :] = x_n[dim, :] + deltax

                    preddeltamean, preddeltavar = gp_model.predict(x_n.T)
                    mean_orig = np.asmatrix(np.repeat(preddeltamean[1], 3)).T
                    var_orig = np.asmatrix(np.repeat(preddeltavar[1] ** 2, 3)).T
                    # compute the relevance estimate at x_n
                    KLsqrt = np.sqrt(
                        0.5
                        * (
                            var_orig / preddeltavar
                            + np.multiply(
                                (preddeltamean.reshape(3, 1) - mean_orig),
                                (preddeltamean.reshape(3, 1) - mean_orig),
                            )
                            / preddeltavar
                            - 1
                        )
                        + np.log(np.sqrt(preddeltavar / var_orig))
                        + jitter
                    )
                    relevances[j, dim] = 0.5 * (KLsqrt[0] + KLsqrt[2]) / delta

                    # remove the perturbation
                    x_n[dim, :] = x_n[dim, :] - deltax

            score = np.mean(relevances, axis=0)

        return score

    def explain(
        self,
        X_init,
        feature_names,
        kernel_type="RBF",
        max_iter=50,
        alpha="EI",
        jitter=5,
        normalize=True,
        plot=True,
        interval=1,
        verbosity=False,
        maximize=False,
        importance_method="ARD",
        delta=1,
    ):
        """
        Returns scores for each feature after BO(Bayesian Optimization) routine


        Args:
            X_init (Numpy Array): Starting vector(or point) for the BO routine
            feature_names (Numpy Array): Array containing names of each feature
            kernel_type (str, optional): Kernel for GP model like 'RBF', 'Matern32', 'Matern52', 'RatQuad', 'Linear', or 'MLP'
            max_iter (int, optional): Maximum iterations upto which BO should run. Defaults to 50.
            alpha (str, optional): Acquisition function like 'FUR', EI', 'LCB', or 'MPI'.  Defaults to 'EI'.
            jitter (float, optional): Exploration-explotation tradeoff for BO. Defaults to 0.05.
            normalize (bool, optional): If False the scores aren't normalized. Defaults to True.
            plot (bool, optional): If False the scores aren't plotted. Defaults to True.
            interval (int, optional): Helps define upper and lower limit of exploration neighborhood. Defaults to 1.
            verbosity (bool, optional): Verbosit y flag for BO
            maximize (bool, optional): BO should maximize or minimize
            importance_method (str, optional): IMportance score method
            delta ()int, optional: Delta for KL method

        Returns:
            [type]: [description]
        """

        # Initializing kernel for f_e
        kernel = Kernel(kernel_type, X_init.shape[1])

        # Extracting the coefficient values
        f_optim = self.generate_optimizer(
            X_init=X_init,
            kernel=kernel,
            max_iter=max_iter,
            alpha=alpha,
            jitter=jitter,
            interval=interval,
            verbosity=verbosity,
            maximize=maximize,
        )

        # Extracting the scores from the coefficients
        scores = self.generate_scores(
            kernel, f_optim, importance_method=importance_method, delta=delta
        )

        # Normalizing the scores for getting relative importance
        if normalize:
            scores = scores / np.max(scores)

        # Plotting the scores
        if plot:
            plot_scores(feature_names, scores)

        # Outputting explanations
        return scores
