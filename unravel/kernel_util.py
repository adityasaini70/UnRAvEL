import GPy
import numpy as np
import matplotlib.pyplot as plt


class Kernel:
    def __init__(self, kernel_type, input_dim):
        """[summary]

        Args:
            kernel_type ([type]): [description]
            input_dim ([type]): [description]
        """

        self.kernel_type = kernel_type
        self.input_dim = input_dim
        self.kernel = eval(
            f"GPy.kern.{self.kernel_type}(input_dim={self.input_dim},ARD=True)"
        ) + GPy.kern.Bias(input_dim=self.input_dim)

    def get_coefficients(self, model_parameters):
        """[summary]

        Args:
            model_parameters ([type]): [description]

        Returns:
            [type]: [description]
        """

        if "Linear" in self.kernel_type:
            return model_parameters[0 : self.input_dim]
        else:
            return model_parameters[1 : self.input_dim + 1]

    def get_importance_score(self, coefficients):
        """[summary]

        Args:
            coefficients ([type]): [description]

        Returns:
            [type]: [description]
        """
        # plt.stem(self.kernel.input_sensitivity())
        if "Linear" in self.kernel_type or "MLP" in self.kernel_type:
            return coefficients
        else:
            return 1 / np.array(coefficients)
