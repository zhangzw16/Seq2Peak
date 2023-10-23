# Unlocking the Potential of Deep Learning in Peak-Hour Series Forecasting. (CIKM 2023)

This repo is the official Pytorch implementation of Seq2Peak: "Unlocking the Potential of Deep Learning in Peak-Hour Series Forecasting" 

View on [ACM DL](https://dl.acm.org/doi/10.1145/3583780.3615159)  [Arxiv](https://arxiv.org/abs/2307.01597)

We are still updating the README for this repo. Please bear with us. It will be completed soon.

## Citing

If you find this repository useful for your work, please consider citing it as follows:

```bibtex
@inproceedings{10.1145/3583780.3615159,
author = {Zhang, Zhenwei and Wang, Xin and Xie, Jingyuan and Zhang, Heling and Gu, Yuantao},
title = {Unlocking the Potential of Deep Learning in Peak-Hour Series Forecasting},
year = {2023},
isbn = {9798400701245},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3583780.3615159},
doi = {10.1145/3583780.3615159},
abstract = {Unlocking the potential of deep learning in Peak-Hour Series Forecasting (PHSF) remains a critical yet underexplored task in various domains. While state-of-the-art deep learning models excel in regular Time Series Forecasting (TSF), they struggle to achieve comparable results in PHSF. This can be attributed to the challenges posed by the high degree of non-stationarity in peak-hour series, which makes direct forecasting more difficult than standard TSF. Additionally, manually extracting the maximum value from regular forecasting results leads to suboptimal performance due to models minimizing the mean deficit. To address these issues, this paper presents Seq2Peak, a novel framework designed specifically for PHSF tasks, bridging the performance gap observed in TSF models. Seq2Peak offers two key components: the CyclicNorm pipeline to mitigate the non-stationarity issue and a simple yet effective trainable-parameter-free peak-hour decoder with a hybrid loss function that utilizes both the original series and peak-hour series as supervised signals. Extensive experimentation on publicly available time series datasets demonstrates the effectiveness of the proposed framework, yielding a remarkable average relative improvement of 37.7\% across four real-world datasets for both transformer- and non-transformer-based TSF models.},
booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
pages = {4415–4419},
numpages = {5},
keywords = {peak-hour series, time series forecasting, normalization},
location = {Birmingham, United Kingdom},
series = {CIKM '23}
}
```

Please remember to cite all the datasets and compared methods if you use them in your experiments.

-----

![image](CIKM2023_poster.jpg)
