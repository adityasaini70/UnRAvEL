# BO-LIME

This repository is the official implementation of [LOCALLY INTERPRETABLE MODEL AGNOSTIC EXPLANATIONS USING GAUSSIAN PROCESSES](https://arxiv.org/abs/2108.06907).

![Visualizing Bayesian optimization sampling](https://user-images.githubusercontent.com/49980787/124347202-f694f500-dc00-11eb-8bf5-618db6343d57.png)

## Installation

A local copy of this repository can be made using the following command:

```
git clone https://github.com/adityasaini70/BO-LIME.git
```

After that, the dependencies of the codebase(Python v3.8.5) can be installed using the following command:

```
cd BO-LIME/
pip install -r requirements.txt
```

## Source code

The source code of the API can be found in the [`src`](https://github.com/adityasaini70/BO-LIME/tree/main/src) folder. The description of files are as follows:

- [`bolime.py`](https://github.com/adityasaini70/BO-LIME/blob/main/src/bolime.py): Impelementation of the BO-LIME algorithm
- [`evaluation_metrics.py`](https://github.com/adityasaini70/BO-LIME/blob/main/src/evaluation_metrics.py): Implementations of MSD and MIS evaluation metrics
- [`plot_util.py:`](https://github.com/adityasaini70/BO-LIME/blob/main/src/plot_util.py): Impelementation of plotting utilities

## Tutorials and usage

We're yet to add the documentation of the final API, till then please refer the following tutorial for learning about the implemented calls and methods

- [Extra Trees Regressor on Boston House Pricing dataset](https://github.com/adityasaini70/BO-LIME/blob/main/Notebooks/Introduction.ipynb)

## Evaluation

For evaluating our model against LIME, we used two metrics:

1. **Mean standard deviation(MSD):** For measuring stability of the generated explanations(The lower, the better)
2. **Mean importance score(MIS):** For measuring the quality of the generated explanations

Please refer the following notebook for referring the code used for evaluation:

- [Extra Trees Regressor on Boston House Pricing dataset; MSD & MIS](https://github.com/adityasaini70/BO-LIME/blob/main/Notebooks/Testing_evaluation_metrics.ipynb)

## Results

Our method achieves considerably better [results](https://github.com/adityasaini70/BO-LIME/tree/main/Results) than LIME for both stability and quality metric.

- Quality metric: [Mean importance score](https://github.com/adityasaini70/BO-LIME/blob/main/Results/Importance%20Score.ipynb)

![Importance](https://github.com/adityasaini70/BO-LIME/blob/main/Results/Importance.png)

- Stability metric: [Mean standard deviation](https://github.com/adityasaini70/BO-LIME/blob/main/Results/Inconsistency.ipynb)

![Inconsistency](https://github.com/adityasaini70/BO-LIME/blob/main/Results/Inconsistency.png)

## License

[MIT License](https://github.com/adityasaini70/BO-LIME/blob/main/LICENSE)
