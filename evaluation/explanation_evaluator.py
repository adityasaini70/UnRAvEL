import os
import string
import numpy as np
from tqdm import tqdm
from functools import partial
from abc import abstractmethod
from scipy.stats import rankdata
from scipy.spatial import distance
from unravel.kernel_util import Kernel
from multiprocessing import current_process, Pool, Manager
from copy import deepcopy
from itertools import combinations


class ModelEvaluator:
    def __init__(
        self, pred_info={"coefficients": np.array([]), "scores": np.array([])}
    ):
        """[summary]

        Args:
            pred_info (dict, optional): [description]. Defaults to {"coefficients": np.array([]), "scores": np.array([])}.
        """
        self.pred_info = pred_info

    def get_pred_info(self):
        return self.pred_info

    @abstractmethod
    def generate_scores(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def evaluate_model(self, *args, **kwargs):
        raise NotImplementedError

    def kw_util(self, rankings):
        """[summary]

        Args:
            ratings ([type]): [description]

        Returns:
            [type]: [description]
        """
        m = rankings.shape[0]
        n = rankings.shape[1]
        denom = m ** 2 * (n ** 3 - n)
        S = n * np.var(np.sum(rankings, axis=0))
        return 12 * S / denom

    def iod_util(self, rankings, scores):
        """[summary]

        Args:
            rankings ([type]): [description]
            scores ([type]): [description]

        Returns:
            [type]: [description]
        """

        iod = []
        for val in rankings.T:
            iod.append(np.var(val) / np.mean(val))

        mean_scores = np.mean(scores, axis=0)

        return np.dot(iod, mean_scores)

    def evaluate_msd(self, scores):
        """[summary]

        Returns:
            [type]: [description]
        """

        return np.mean(np.std(scores, axis=0))

    def evaluate_mip(self, scores):
        """[summary]

        Returns:
            [type]: [description]
        """

        return np.mean(scores, axis=0)

    def evaluate_kw(self, scores):
        """[summary]

        Returns:
            [type]: [description]
        """

        rankings = rankdata(scores, axis=1)
        print(np.array(rankings).shape)
        return self.kw_util(rankings)

    def evaluate_iod(self, scores):
        """[summary]

        Returns:
            [type]: [description]
        """
        rankings = rankdata(scores, axis=1)

        return self.iod_util(rankings, scores)

    def evaluate_jaccard_distance(self, scores, top_k=5):
        """_summary_

        Args:
            scores (_type_): _description_
            top_k (int, optional): _description_. Defaults to 10.

        Returns:
            _type_: _description_
        """
        jaccard_distances = []
        evaluation_pairs = list(combinations(range(scores.shape[0]), 2))
        for evaluation_pair in evaluation_pairs:
            s1 = scores[evaluation_pair[0]]
            s2 = scores[evaluation_pair[1]]

            # Extracting indices of top-k features in both lists
            s1 = set(np.argpartition(s1, -top_k)[-top_k:])
            s2 = set(np.argpartition(s2, -top_k)[-top_k:])
            #print(s1, s2)

            jaccard_similarity = len(s1.intersection(s2)) / len(s1.union(s2))
            jaccard_distance = 1 - jaccard_similarity
            jaccard_distances.append(jaccard_distance)

        return np.mean(jaccard_distances)


class LimeEvaluator(ModelEvaluator):
    def __init__(
        self,
        evaluation_model="lime",
        pred_info={"coefficients": np.array([]), "scores": np.array([])},
    ):
        """[summary]

        Args:
            pred_info ([type]): [description]
            evalutaion_model (str, optional): Flag for "baylime" or "lime"
        """
        super().__init__(pred_info=pred_info)
        self.evaluation_model = evaluation_model
        self.model_regressor = (
            "non_Bay" if self.evaluation_model == "lime" else "Bay_non_info_prior"
        )

    def evaluate_model(
        self,
        lime_explainer,
        X_init,
        bbox_model,
        iterations,
        num_samples,
        num_features,
        feature_names,
    ):
        """[summary]

        Args:
            lime_explainer ([type]): [description]
            X_init ([type]): [description]
            bbox_model ([type]): [description]
            iterations ([type]): [description]
            num_samples ([type]): [description]
            num_features ([type]): [description]
            feature_names ([type]): [description]
        """

        self.pred_info["scores"] = self.generate_scores(
            lime_explainer,
            X_init,
            bbox_model,
            iterations,
            num_samples,
            num_features,
            feature_names,
        )

        return {
            "kw": self.evaluate_kw(self.pred_info["scores"]),
            "mip": self.evaluate_mip(self.pred_info["scores"]),
            "msd": self.evaluate_msd(self.pred_info["scores"]),
            "iod": self.evaluate_iod(self.pred_info["scores"]),
            "jaccard": self.evaluate_jaccard_distance(self.pred_info["scores"]),
        }

    def generate_scores(
        self,
        lime_explainer,
        X_init,
        bbox_model,
        iterations,
        num_samples,
        num_features,
        feature_names,
    ):
        """[summary]

        Args:
            lime_explainer ([type]): [description]
            X_init ([type]): [description]
            bbox_model ([type]): [description]
            iterations ([type]): [description]
            num_samples ([type]): [description]
            num_features ([type]): [description]
            feature_names ([type]): [description]
        """
        scores_map = {}

        for i in range(iterations):
            if lime_explainer.mode == "classification":
                lime_explanation = lime_explainer.explain_instance(
                    X_init,
                    bbox_model.predict_proba,
                    num_features=num_features,
                    num_samples=num_samples,
                    model_regressor=self.model_regressor,
                )
            else:
                lime_explanation = lime_explainer.explain_instance(
                    X_init,
                    bbox_model.predict,
                    num_features=num_features,
                    num_samples=num_samples,
                    model_regressor=self.model_regressor,
                )
            for feature in lime_explanation.as_list():
                if feature[0] in scores_map:
                    scores_map[feature[0]].append(feature[1])
                else:
                    scores_map[feature[0]] = [feature[1]]

        # Padding LIME to ensure numerical stability; iterations = 100
        for idx in scores_map:
            scores_map[idx] = scores_map[idx] + [
                1e-100 for _ in range(iterations - len(scores_map[idx]))
            ]

        # Updating keys; 5.88 < RM <= 6.21 to RM
        for val in list(scores_map.keys()):
            temp = val
            for c in val:
                if c in string.punctuation or c in string.digits or c == " ":
                    temp = temp.replace(c, "")
            scores_map[temp] = scores_map[val]
            del scores_map[val]

        # Sorting by order mentioned in feature_names
        coefficients = []
        for val in feature_names:
            temp_key = val.replace(" ", "")
            if temp_key in scores_map:
                coefficients.append(scores_map[val.replace(" ", "")])
            else:
                coefficients.append(len(scores_map[list(scores_map.keys())[0]]) * [0])

        coefficients = np.array(coefficients).T

        # Collecting normalized scores from the coefficients
        scores_list = []

        for data in coefficients:

            scores = data

            scores_list.append(scores)

        return np.array(scores_list)


class UnRAVELEvaluator(ModelEvaluator):
    def __init__(
        self,
        kernel_shape,
        kernel_type="RBF",
        pred_info={"coefficients": np.array([]), "scores": np.array([])},
    ):
        """[summary]

        Args:
            pred_info ([type]): [description]
            kernel_shape (int): Number of features inherent in the dataset
            kernel_type (str, optional): Kernel for GP model like 'RBF', 'Matern32', 'Matern52', 'RatQuad', 'Linear', or 'MLP'
        """
        super().__init__(pred_info=pred_info)
        self.kernel = Kernel(kernel_type, kernel_shape)

        # Stores surrogate data generated in each iteration
        try:
            manager = getattr(type(self), "manager")
        except AttributeError:
            manager = type(self).manager = Manager()

        self.surrogate_data = manager.dict()
        self.gp_model = manager.dict()

    def get_gp_model(self):
        return self.gp_model

    def get_surrogate_data(self):
        return self.surrogate_data

    def evaluate_model(
        self,
        iterations,
        unravel_explainer,
        X_init,
        max_iter=50,
        alpha="EI",
        jitter=5,
        interval=1,
    ):
        """[summary]

        Args:
            iterations ([type]): [description]
            unravel_explainer ([typye]): [description]
            X_init (Numpy Array): Starting vector(or point) for the BO routine
            max_iter (Scalar integer): Maximum iterations upto which BO should run. Defaults to 50.
            alpha (str, optional): Acquisition function like 'EI', 'LCB', or 'MPI'.  Defaults to 'EI'.
            jitter (float, optional): Exploration-explotation tradeoff for BO. Defaults to 0.05.
            interval (int, optional): Helps define upper and lower limit of exploration neighborhood. Defaults to 1.
        """

        # Default is ARD scores, rest is KL
        [
            self.pred_info["kl_scores"],
            self.pred_info["ard_scores"],
        ] = self.generate_scores_singlecore(
            iterations, unravel_explainer, X_init, max_iter, alpha, jitter, interval
        )

        return {
            "kl_kw": self.evaluate_kw(self.pred_info["kl_scores"]),
            "kl_mip": self.evaluate_mip(self.pred_info["kl_scores"]),
            "kl_msd": self.evaluate_msd(self.pred_info["kl_scores"]),
            "kl_iod": self.evaluate_iod(self.pred_info["kl_scores"]),
            "kl_jaccard": self.evaluate_jaccard_distance(self.pred_info["kl_scores"]),
            "kw": self.evaluate_kw(self.pred_info["ard_scores"]),
            "mip": self.evaluate_mip(self.pred_info["ard_scores"]),
            "msd": self.evaluate_msd(self.pred_info["ard_scores"]),
            "iod": self.evaluate_iod(self.pred_info["ard_scores"]),
            "jaccard": self.evaluate_jaccard_distance(self.pred_info["ard_scores"]),
        }

    def generate_scores(
        self,
        iterations,
        unravel_explainer,
        X_init,
        max_iter=50,
        alpha="EI",
        jitter=5,
        interval=1,
    ):
        """[summary]

        Args:
            iterations ([type]): [description]
            unravel_explainer ([typye]): [description]
            X_init (Numpy Array): Starting vector(or point) for the BO routine
            max_iter (Scalar integer): Maximum iterations upto which BO should run. Defaults to 50.
            alpha (str, optional): Acquisition function like 'EI', 'LCB', or 'MPI'.  Defaults to 'EI'.
            jitter (float, optional): Exploration-explotation tradeoff for BO. Defaults to 0.05.
            interval (int, optional): Helps define upper and lower limit of exploration neighborhood. Defaults to 1.
        """
        kl_scores = []
        ard_scores = []

        process_id = current_process()._identity[0] % (os.cpu_count())
        for i in tqdm(
            range(iterations),
            total=iterations,
            desc=f"Progress of Core {process_id}",
        ):
            f_optim = unravel_explainer.generate_optimizer(
                X_init=X_init,
                max_iter=max_iter,
                kernel=self.kernel,
                alpha=alpha,
                jitter=jitter,
                interval=interval,
            )

            kl_scores.append(
                unravel_explainer.generate_scores(
                    kernel=None, f_optim=f_optim, importance_method="KL", delta=1
                )
            )

            ard_scores.append(
                unravel_explainer.generate_scores(
                    kernel=self.kernel,
                    f_optim=f_optim,
                    importance_method="ARD",
                    delta=None,
                )
            )

            self.surrogate_data[f"{current_process()._identity[0]}_{i}"] = deepcopy(
                unravel_explainer.surrogate_data
            )

            self.gp_model[f"{current_process()._identity[0]}_{i}"] = deepcopy(
                unravel_explainer.gp_model
            )
        # print(np.array(kl_scores).shape)
        return np.array([np.array(kl_scores), np.array(ard_scores)])

    def generate_scores_multicore(
        self,
        iterations,
        unravel_explainer,
        X_init,
        max_iter=50,
        alpha="EI",
        jitter=0.05,
        interval=1,
    ):
        """[summary]

        Args:
            iterations ([type]): [description]
            unravel_explainer ([typye]): [description]
            X_init (Numpy Array): Starting vector(or point) for the BO routine
            max_iter (Scalar integer): Maximum iterations upto which BO should run. Defaults to 50.
            alpha (str, optional): Acquisition function like 'EI', 'LCB', or 'MPI'.  Defaults to 'EI'.
            jitter (float, optional): Exploration-explotation tradeoff for BO. Defaults to 0.05.
            interval (int, optional): Helps define upper and lower limit of exploration neighborhood. Defaults to 1.
        """

        wrapper_function = partial(
            self.generate_scores,
            unravel_explainer=unravel_explainer,
            X_init=X_init,
            max_iter=max_iter,
            alpha=alpha,
            jitter=jitter,
            interval=interval,
        )

        # Generating iterations list for each core
        num_cores = os.cpu_count()
        iterations_list = [int(iterations / num_cores)] * num_cores + [
            int(iterations % num_cores)
        ]

        # Removing 0's in case iterations < num_cores
        iterations_list = [val for val in iterations_list if val != 0]
        
        
        if num_cores>=iterations:
            iterations_list = [1]*iterations
        print(iterations_list)
        # Initializing multiprocessing loop
        with Pool() as pool:
            results = list(
                tqdm(
                    pool.imap(wrapper_function, iterations_list),
                    total=len(iterations_list),
                    desc="Total progress",
                )
            )

        # Extracting scores
        # kl_scores = results[:, 0]
        # ard_scores = results[:, 1]
        kl_scores = [val[0] for val in results]
        ard_scores = [val[1] for val in results]
        # Converting manager to lists
        self.gp_model = dict(self.gp_model)
        self.surrogate_data = dict(self.surrogate_data)

        return [
            np.vstack(kl_scores).reshape(-1, X_init.shape[1]),
            np.vstack(ard_scores).reshape(-1, X_init.shape[1]),
        ]

    def generate_scores_singlecore(
        self,
        iterations,
        unravel_explainer,
        X_init,
        max_iter=50,
        alpha="EI",
        jitter=0.05,
        interval=1,
    ):
        ard_scores = []
        kl_scores = []

        for i in tqdm(
            range(iterations),
            total=iterations,
        ):
            f_optim = unravel_explainer.generate_optimizer(
                X_init=X_init,
                max_iter=max_iter,
                kernel=self.kernel,
                alpha=alpha,
                jitter=jitter,
                interval=interval,
            )

            kl_scores.append(
                unravel_explainer.generate_scores(
                    kernel=None, f_optim=f_optim, importance_method="KL", delta=1
                )
            )

            ard_scores.append(
                unravel_explainer.generate_scores(
                    kernel=self.kernel,
                    f_optim=f_optim,
                    importance_method="ARD",
                    delta=None,
                )
            )

            self.surrogate_data[f"{max_iter}_{i}"] = deepcopy(
                unravel_explainer.surrogate_data
            )

            self.gp_model[f"{max_iter}_{i}"] = deepcopy(unravel_explainer.gp_model)

        print(np.array(kl_scores).shape)
        return [
            np.vstack(kl_scores).reshape(-1, X_init.shape[1]),
            np.vstack(ard_scores).reshape(-1, X_init.shape[1]),
        ]
