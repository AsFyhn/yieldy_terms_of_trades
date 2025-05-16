# yieldy_terms_of_trade

## Overview
This repository contains the code and resources for the seminar paper titled *"Seminar: Topics in Sovereign Debt"*, submitted as part of the course *"Seminar: Topics in Sovereign Debt"* in the Master's program in Economics at the University of Copenhagen. The paper, authored by Asbjørn Fyhn and Emil Beckett Kolko, revisits the determinants of sovereign credit spreads and default probabilities in emerging markets, focusing on the role of Terms of Trade (ToT) volatility. The analysis leverages the Emerging Markets Bond Index (EMBI) Diversified to assess credit spreads, while employing logistic regression to estimate default probabilities.

## Project Purpose
The primary objective of this project is to replicate and extend the framework of Hilscher and Nosbusch (2009) by analyzing the impact of ToT volatility and other macroeconomic fundamentals on sovereign credit spreads and default probabilities. The study uses a combination of linear regression models for credit spreads and logistic regression for default probabilities, incorporating country-specific, global, and control variables. Key findings suggest a diminished role of ToT volatility in recent years, attributed to structural improvements in emerging markets and evolving credit rating methodologies.

## Repository Structure
- **data/**: Placeholder directory for datasets used in the analysis (not available due to proprietary restrictions).
  - **output/**: Contains generated outputs from the analysis scripts.

- **src/**: Contains Python scripts for data processing, analysis, and visualization.
  - `analysis_5.ipynb`: Jupyter notebook for Logistic regression models.
  - `analysis.ipynb`: Jupyter notebook for Linear regression models.
  - `data_probability.ipynb`: Notebook for processing data related to default probability estimation.
  - `estimation.py`: Script for estimation tasks (e.g., regression models).
  - `LinearModel.py`: Custom module for linear regression models.
  - `proces_raw.ipynb`: Notebook for processing raw data.
  - `tools.py`: Utility functions for data handling and analysis.

## Dependencies
The project is written in Python and relies on the following libraries:
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical computations.
- `scikit-learn`: For logistic regression and other machine learning tasks.
- `matplotlib` and `seaborn`: For data visualization.
- `statsmodels`: For statistical modeling (e.g., OLS regression).

To install dependencies, you can use:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels
```

## Usage
1. **Data Preparation**: The raw data is proprietary and not included in this repository. The `proces_raw.ipynb` notebook outlines the steps to clean and prepare the data, including calculating ToT volatility and merging datasets.
2. **Analysis**:
   - Run `analysis.ipynb` and `analysis_5.ipynb` to perform the main regression analyses (OLS for spreads, logistic for default probabilities).
   - Use `data_probability.ipynb` to prepare data specifically for default probability estimation.
3. **Estimation and Visualization**:
   - `estimation.py` and `LinearModel.py` contain the core logic for regression models.
   - Outputs such as scatter plots, histograms, and LaTeX tables are saved in the `data/output/` directory.
4. **Utilities**: The `tools.py` script provides helper functions for data processing and visualization.

## Data Availability
Parts of the data used in this project is proprietary and cannot be shared due to licensing restrictions. Data sources include:
- EMBI Diversified consistuents and yields (Bloomberg)
- U.S. Treasury yields (U.S. Department of the Treasury)
- Terms of Trade (World Bank)
- Sovereign default history (Asonuma and Trebesch, 2016)
- VIX index (CBOE)
- GDP, currency reserves, and external debt (World Bank)
- Credit ratings (Moody's and S&P) (retrieved from Bloomberg)

For details on data sources and transformations, refer to Section 7.1 of the seminar paper.

## Notes
- The analysis spans two periods: 1988–2017 for default probability estimation and 2018–2021 for EMBI spread analysis.
- Middle Eastern countries are excluded in some specifications due to their unique oil-driven ToT volatility dynamics.
- The seminar paper is available at the root of the repository (not included here but referenced in the original context).

## Authors
- Asbjørn Fyhn (prn917)
- Emil Beckett Kolko (fsj234)

## Course Information
- **Course**: Sovereign Debt
- **Program**: Master's in Economics, University of Copenhagen
- **Submission Date**: May 2025 (inferred from context)

## License
This project is for academic purposes only. Redistribution of the code is permitted with proper attribution to the authors.