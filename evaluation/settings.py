import argparse

parser = argparse.ArgumentParser(description="Settings available for experimentation")

parser.add_argument(
    "--evaluation_model",
    type=str,
    default="unravel",
    help='Model to be evaluated. Arguments can include: "unravel", or "lime"',
)

parser.add_argument(
    "--acquisition_jitter",
    type=float,
    default=0.01,
    help="Acquisition jitter for controlling the exploration tradeoff during the BO procedure",
)

parser.add_argument(
    "--acquisition_function",
    type=str,
    default="FUR",
    help='Acquisition function type for the BO process. Arguments can include: "FUR", "UR", "UCB", "EI", "LCB", or "MPI"',
)

parser.add_argument(
    "--kernel_type",
    type=str,
    default="RBF",
    help='Kernel for GP model. Arguments can include: "RBF", "Matern32", "Matern52", "RatQuad", "Linear", or "MLP"',
)

parser.add_argument(
    "--iterations",
    type=int,
    default=50,
    help="Maximum repeatitions for evaluating the stability metrics",
)

parser.add_argument(
    "--dataset",
    type=str,
    default="breast_cancer",
    help='Datasets available for evaluation purposes. Arguments can include: "boston", or "breast_cancer"',
)

args = parser.parse_args()
