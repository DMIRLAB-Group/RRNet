# RRNet
Official code for "Generalization bound for estimating causal effects from observational network data"

# dataset
Please find datasets from https://github.com/songjiang0909/Causal-Inference-on-Networked-Data
Thanks for their datasets!

# Cite
```
@inproceedings{10.1145/3583780.3614892,
author = {Cai, Ruichu and Yang, Zeqin and Chen, Weilin and Yan, Yuguang and Hao, Zhifeng},
title = {Generalization Bound for Estimating Causal Effects from Observational Network Data},
year = {2023},
isbn = {9798400701245},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3583780.3614892},
doi = {10.1145/3583780.3614892},
abstract = {Estimating causal effects from observational network data is a significant but challenging problem. Existing works in causal inference for observational network data lack an analysis of the generalization bound, which can theoretically provide support for alleviating the complex confounding bias and practically guide the design of learning objectives in a principled manner. To fill this gap, we derive a generalization bound for causal effect estimation in network scenarios by exploiting 1) the reweighting schema based on joint propensity score and 2) the representation learning schema based on Integral Probability Metric (IPM). We provide two perspectives on the generalization bound in terms of reweighting and representation learning, respectively. Motivated by the analysis of the bound, we propose a weighting regression method based on the joint propensity score augmented with representation learning. Extensive experimental studies on two real-world networks with semi-synthetic data demonstrate the effectiveness of our algorithm.},
booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
pages = {163–172},
numpages = {10},
keywords = {causal inference, generalization bound, observational network data},
location = {Birmingham, United Kingdom},
series = {CIKM '23}
}
```
