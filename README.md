# Predicting Cardiovascular Disease Using Machine Learning

## Overview
Given a medical survey on health/lifestyle, we needed to predict the risk of MICHD by using binart classification, exploring logistic regression optimized by many methods.


## Dataset
The dataset comes from the [CDC BRFSS 2021 survey](https://www.cdc.gov/brfss/). We selected every medical and lifestyle variables(no administrative), handling missing values by adding columns marking missing entries , data split, and balanced the classes.

## Installation & Requirements

```bash
git clone https://github.com/FTognina/ML.P1
cd ML.P1
```
req: Python 3.14, NumPy 1.26, matplotlib 3.8

## Usage
By running this line, you will get the resulting plots in the folder reports/
```bash
python run.py 
```

## Project tructure
```bash
ML.P1/
│
├── project_script/ 
│   ├── reports_grid_loing/ # Resulting plots
│   ├── helper.py
│   └── grid_run.py                 
├── reports/             # Resulting plots
├── REPORT.pdf
├── implementations.py 
├── run.py 
└── README.md
```



