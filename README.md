# Incorporating Interpretability into Latent Factor Models via Fast Influence Analysis
This is our Tensorflow implementation for the paper:

>Weiyu Cheng, Yanyan Shen, Linpeng Huang, Yanmin Zhu (2019). [Incorporating Interpretability into Latent Factor Models via Fast Influence Analysis](https://doi.org/10.1145/3292500.3330857). In KDD'19, Anchorage, AK, USA, August 04-08, 2019.

Author: Weiyu Cheng (weiyu_cheng at sjtu.edu.cn)

## Introduction
Fast Influence Analysis (FIA) applies influence functions to latent factor models (LFMs) towards interpretable recommendation. We incorporate interpretability into LFMs by tracing each prediction back to models’ training data, and further provide intuitive neighbor-style explanations for the predictions. FIA significantly reduces the computational cost of influence functions by exploiting the characteristics of LFMs.

## Citation 
If you want to use our codes and datasets in your research, please cite:
```
@inproceedings{DBLP:conf/kdd/ChengSHZ19,
  author    = {Weiyu Cheng and
               Yanyan Shen and
               Linpeng Huang and
               Yanmin Zhu},
  title     = {Incorporating Interpretability into Latent Factor Models via Fast
               Influence Analysis},
  booktitle = {Proceedings of the 25th {ACM} {SIGKDD} International Conference on
               Knowledge Discovery {\&} Data Mining, {KDD} 2019, Anchorage, AK,
               USA, August 4-8, 2019.},
  pages     = {885--893},
  year      = {2019},
  doi       = {10.1145/3292500.3330857}
}
```
## Environment Requirement
The code has been tested running under Python 3.6. The required packages are as follows:
* tensorflow == 1.4.0

## Example to Run the Codes
```
cd src/scripts
sh ./RQ1.sh
sh ./RQ2.sh
```
