# UnRAvEL

This repository is the official implementation of Select Wisely and Explain: Active Learning and Probabilistic Local Post-hoc Explainability.

![Workflow](https://user-images.githubusercontent.com/49980787/164484714-fdffb1ce-ea73-4d44-8eb1-3ef8a05db27e.png)

## Installation

A local copy of this repository can be made using the following command:

```
git clone https://github.com/adityasaini70/BO-LIME.git
```

After that, the dependencies of the codebase(Python v3.8.5) can be installed using the following command:

```
cd UnRAvEL/
pip install -r requirements.txt
```

## Source code

The source code of the API can be found in the [`unravel`](https://github.com/adityasaini70/UnRAvEL/tree/main/unravel) folder. The description of files are as follows:

- [`tabular.py`](https://github.com/adityasaini70/UnRAvEL/blob/main/unravel/tabular.py): Implementation of the UnRAvEL algorithm for tabular datasets
- [`image.py`](https://github.com/adityasaini70/UnRAvEL/blob/main/unravel/image.py): Implementation of the UnRAvEL algorithm for image datasets
- [`acquisition_util.py`](https://github.com/adityasaini70/UnRAvEL/blob/main/unravel/acquisition_util.py): Implementation of all the used acquisition functions: FUR, UCB and UR
- [`kernel_util.py`](https://github.com/adityasaini70/UnRAvEL/blob/main/unravel/kernel_util.py): Wrapper module for the used GP kernels
- [`plot_util.py`](https://github.com/adityasaini70/UnRAvEL/blob/main/unravel/plot_util.py): Wrapper module for plotting utilities

## Tutorials and usage

We're yet to add the documentation of the final API, till then please refer the following tutorial for learning about the implemented calls and methods

- [Support Vector Classifier on Breast Cancer Classification dataset](https://github.com/adityasaini70/UnRAvEL/blob/main/notebooks/Breast%20Cancer.ipynb)

## Evaluation

The code used in our experiments can be found in the [`evaluation`](https://github.com/adityasaini70/UnRAvEL/tree/main/evaluation) folder. The description of files are as follows:

- [`explanation_evaluator.py`](https://github.com/adityasaini70/UnRAvEL/blob/main/evaluation/explanation_evaluator.py): Implementation of all the stability and fidelity metrics used for evaluating our method against LIME and BayLIME
- [`blackbox_util.py`](https://github.com/adityasaini70/UnRAvEL/blob/main/evaluation/blackbox_util.py): Wrapper module for all the used datasets and corresponding prediction models
- [`settings.py`](https://github.com/adityasaini70/UnRAvEL/blob/main/evaluation/settings.py): Implementation of all the evaluation settings

## Results

Sample Efficiency      |  Stability
:-------------------------:|:-------------------------:
<img src="https://user-images.githubusercontent.com/49980787/164488777-474e2a35-d438-45f8-8eb3-0fadd8c44895.png" width="500" height="400"> | <img src="https://user-images.githubusercontent.com/49980787/164488603-4461b509-c605-48b7-98ac-3ab13830d5e5.png" width="500" height="200">

## License

The API is distributed under the [MIT License](https://github.com/adityasaini70/UnRAvEL/blob/main/LICENSE)
