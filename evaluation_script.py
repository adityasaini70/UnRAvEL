import pickle
import numpy as np
from datetime import datetime
from copy import deepcopy
from lime.lime_tabular import LimeTabularExplainer
from unravel.tabular import UnRAVELTabularExplainer
from evaluation.settings import args
from evaluation.blackbox_util import BlackBoxSimulator
from evaluation.explanation_evaluator import UnRAVELEvaluator, LimeEvaluator

if __name__ == "__main__":
    # Setting seed for reproducibility
    np.random.seed(50)

    # Loading the dataset utilities and the black-box model(f_p)
    dataset_utilities = eval(f"BlackBoxSimulator().load_{args.dataset}_utilities()")
    [
        X_train,
        y_train,
        X_test,
        y_test,
        features,
        f_p,
        mode,
        categorical_features,
        sample_idx,
    ] = dataset_utilities.values()

    # Sample to be explained - Randomly selected
    # X_init = np.array([X_test[sample_idx]])

    # Initializing empty containers for storing evaluation metrics(Lime or ARD)
    mip = []
    msd = []
    kw = []
    iod = []
    jaccard = []

    if args.evaluation_model == "unravel":
        kl_mip = []
        kl_msd = []
        kl_kw = []
        kl_iod = []
        kl_jaccard = []

    pred_info = []

    # Storing extra information for BO-LIME
    surrogate_data = []
    gp_model = []

    # Initializing the BO loop over the initialized num_samples list
    # num_samples_list = list(range(10, 101, 10))
    # num_samples_list = [100, 150, 200]
    # num_samples_list = [50, 100, 200]
    # num_samples_list = [10, 25, 50, 100, 250, 500, 1000]

    # if args.evaluation_model == "lime":
    #     num_samples_list.append(5000)
    num_samples = 100

    results = {}
    print(sample_idx)
    for idx in range(sample_idx.shape[0]):

        # Initializing starting idx
        X_init = np.array([X_test[idx]])
        print(f"Performing experiment for idx = {idx}")

        # Initializing the explainer module evaluator and generating essential information for analysis
        if args.evaluation_model == "unravel":
            f_e = UnRAVELTabularExplainer(
                bbox_model=f_p,
                train_data=X_train,
                categorical_features=categorical_features,
                mode=mode,
            )

            ModelEvaluator = UnRAVELEvaluator(
                kernel_type=args.kernel_type, kernel_shape=X_init.shape[1]
            )

            evaluations = ModelEvaluator.evaluate_model(
                unravel_explainer=f_e,
                X_init=X_init,
                max_iter=num_samples,
                iterations=args.iterations,
                alpha=args.acquisition_function,
                jitter=args.acquisition_jitter,
            )
        else:
            f_e = LimeTabularExplainer(
                X_train,
                feature_names=features,
                verbose=False,
                mode=mode,
                feature_selection="lasso_path",
                sample_around_instance=True,
            )

            ModelEvaluator = LimeEvaluator(evaluation_model=args.evaluation_model)

            evaluations = ModelEvaluator.evaluate_model(
                lime_explainer=f_e,
                X_init=X_init.ravel(),
                bbox_model=f_p,
                iterations=args.iterations,
                num_samples=num_samples,
                num_features=X_train.shape[1],
                feature_names=features,
            )

        mip.append(evaluations["mip"])
        msd.append(evaluations["msd"])
        kw.append(evaluations["kw"])
        iod.append(evaluations["mip"])
        jaccard.append(evaluations["jaccard"])
        pred_info.append(deepcopy(ModelEvaluator.get_pred_info()))
        print(">>> KW =", kw)
        print(">>> Jaccard =", jaccard)

        # Storing the exploration neighborhood for analysis
        if args.evaluation_model == "unravel":
            surrogate_data.append(deepcopy(ModelEvaluator.get_surrogate_data()))
            gp_model.append(deepcopy(ModelEvaluator.get_gp_model()))
            kl_mip.append(evaluations["kl_mip"])
            kl_msd.append(evaluations["kl_msd"])
            kl_kw.append(evaluations["kl_kw"])
            kl_iod.append(evaluations["kl_iod"])
            kl_jaccard.append(evaluations["jaccard"])
            print(">>> KL KW =", kl_kw)
            print(">>> KL Jaccard =", kl_jaccard)

    # Collecting the obtained metric values
    print("Average Jaccard = ", np.mean(jaccard))
    results = {
        "msd": msd,
        "mis": mip,
        "kw": kw,
        "iod": iod,
        "jaccard": jaccard,
        "mean_jaccard": np.mean(jaccard),
        "pred_info": pred_info,
        "surrogate_data": surrogate_data,
        "gp_model": gp_model,
    }

    if args.evaluation_model == "unravel":
        results["kl_msd"] = kl_msd
        results["kl_kw"] = kl_kw
        results["kl_mip"] = kl_mip
        results["kl_iod"] = kl_iod
        results["kl_jaccard"] = kl_jaccard
        results["kl_mean_jaccard"] = np.mean(kl_jaccard)

    # Caching results for reproducibility and logging their details for further analysis
    if args.evaluation_model == "unravel":
        results_path = (
            f"results/data/{args.evaluation_model}/{args.dataset}_{args.acquisition_function}_{args.kernel_type}_{args.iterations}".replace(
                ".", "dot"
            )
            + ".pickle"
        )
        logging_data = f"{str(datetime.utcnow())},{args.evaluation_model},{args.dataset},{args.acquisition_function},{args.kernel_type},{args.iterations},{results_path}"
    else:
        results_path = f"results/data/{args.evaluation_model}/{args.dataset}_{args.iterations}.pickle"
        logging_data = f"{str(datetime.utcnow())},{args.evaluation_model},{args.dataset},NA,{args.iterations},{results_path}"

    with open(results_path, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f"results/logs.csv", "a") as file:
        file.write("\n")
        file.write(logging_data)

    print("Experiment finished successfully!")
