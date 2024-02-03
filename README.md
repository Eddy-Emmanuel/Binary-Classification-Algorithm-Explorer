# Binary Classification Algorithm Explorer

## Overview

This project explores various binary classification algorithms using different datasets. It provides functionalities for Exploratory Data Analysis (EDA) and model training with multiple evaluation options. The supported algorithms include Logistic Regression, K-Nearest Neighbors, Random Forest, XGBoost, CatBoost, and LightGBM.

## Prerequisites

Make sure to install the required Python packages by running:

```bash
pip install numpy pandas scikit-learn seaborn matplotlib xgboost catboost lightgbm streamlit
```

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/binary-classification-explorer.git
   ```

2. Navigate to the project directory:

   ```bash
   cd binary-classification-explorer
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run main.py
   ```

4. Access the application in your web browser at `http://localhost:8501`.

## Features

### Data Selection

- Choose from available datasets, such as the Breast Cancer dataset or the Wine dataset.
- Display dataset information, shape, description, and check for missing values.

### Exploratory Data Analysis (EDA)

- Explore numeric and categorical columns using scatter plots, box plots, violin plots, pair plots, and distribution plots.
- Visualize correlation matrices.

### Model Training

- Select specific features for training.
- Choose classification algorithms like Logistic Regression, K-Nearest Neighbors, Random Forest, XGBoost, CatBoost, or LightGBM.
- Adjust the test size and cross-validation parameters.

### Model Evaluation

- Assess model performance using accuracy scores, confusion matrices, classification reports, ROC AUC curves, and feature importance plots.

## Contributions

Contributions to enhance this binary classification explorer are welcome. Please follow the [contribution guidelines](CONTRIBUTING.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. Feel free to use, modify, and distribute the code for your own purposes.
