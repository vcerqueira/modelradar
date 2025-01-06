# Model Radar 🎯

A framework for aspect-based evaluation of time series forecasting models based on Nixtla's ecosystem.

[![PyPi Version](https://img.shields.io/pypi/v/modelradar)](https://pypi.org/project/modelradar/)
[![GitHub](https://img.shields.io/github/stars/vcerqueira/modelradar?style=social)](https://github.com/vcerqueira/modelradar)
[![Downloads](https://static.pepy.tech/badge/modelradar)](https://pepy.tech/project/modelradar)

## Overview

Model Radar introduces a novel aspect-based forecasting evaluation approach that goes beyond traditional aggregate metrics. Our framework enables:
- Fine-grained performance analysis across different forecasting aspects
- Better understanding of model behavior in varying conditions
- More informed model selection based on specific use case requirements

## 🚀 Getting Started

Check the `notebooks` folder for usage examples and tutorials.


### Prerequisites

Required dependencies:
```
utilsforecast==0.2.9
numpy==1.26.4
plotnine==0.14.3
statsmodels==0.14.4
```

### Example outputs

- Spider chart with overall view on several dimensions:

![radar](assets/examples/radar.png)

- Parallel coordinates chart with overall view on several dimensions:

![radar2](assets/examples/parcoords.png)


- Barplot chart controlling for a given variable (in this case, anomaly status):

![radar2](assets/examples/anomaly_status.png)

- Grouped bar plot showing win/draw/loss ratios wrt different models:

![radar2](assets/examples/win_ratios.png)

## 📑 Reference

> Cerqueira, V., Roque, L., & Soares, C. (2024). "Forecasting with Deep Learning: Beyond Average of Average of Average Performance." *arXiv preprint arXiv:2406.16590*

Check DS24 folder to reproduce the experiments published on this paper.
The main repository and package contains an updated framework.

## Contact

Get in touch @ vitorc.research@gmail.com
