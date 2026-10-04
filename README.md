# NSE Stock Forecasting ML

This project is a Python implementation inspired by the architecture of your design: data preprocessing, feature engineering, train/validation/test partitioning, VAR and NNAR forecast models, a linear benchmark, hybrid validation, multi-horizon forecasting, model comparison, and live close-price validation.

## Architecture implemented

- Historical NSE market data ingestion
- Data cleaning and alignment
- Feature engineering and lag creation
- Chronological dataset split
- VAR forecasting block
- NNAR forecasting block
- LNN benchmark block
- Multi-horizon weighted forecast aggregation
- Performance evaluation and comparison
- Live close-price validation module
- Sector-wise performance analysis extension point

## Project structure

```text
NSE-Stock-Forecasting-ML/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── dataset.py
│   ├── models.py
│   ├── evaluation.py
│   ├── live_validation.py
│   └── pipeline.py
└── notebooks/
    └── .gitkeep
```

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

## Notes

- This starter project uses Yahoo Finance for market data access so it can run immediately.
- To match real NSE production deployment, replace the source with your preferred NSE API or exchange data feed.
- The current implementation is modular and designed to be extended into a richer benchmark dashboard or research notebook.

## Main reference patterns used

This implementation borrows structural ideas from the following public project directions:

- Multi-horizon forecasting and comparison flows
- NSE stock prediction pipelines
- VAR / nonlinear forecasting experimentation
- Benchmark model evaluation workflows
- Live validation and market performance checks
