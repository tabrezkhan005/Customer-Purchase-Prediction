# Customer Purchase Prediction

This project aims to predict whether an online shopper will complete a purchase based on their browsing behavior and session data. The model is built using the "Online Shoppers Purchasing Intention Dataset" from the UCI Machine Learning Repository.

## Project Overview

The goal is to build a classification model that accurately predicts purchase intent, which can help e-commerce businesses understand customer behavior, optimize user experience, and target potential buyers more effectively.

The project includes data preprocessing, exploratory data analysis (EDA), model training with several algorithms, and model evaluation.

## Dataset

The dataset used is `online_shoppers_intention.csv`. It contains 12,330 sessions, with each session corresponding to a different user in a 1-year period. The dataset consists of 10 numerical and 8 categorical attributes. The 'Revenue' attribute is the class label.

More information about the dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset).

## Project Structure

The project is organized into a modular structure for clarity and scalability.

```
├── customer_purchase_prediction_analysis.ipynb  # Jupyter notebook for EDA and initial modeling
├── online_shoppers_intention.csv              # Raw dataset
├── README.md                                    # This file
└── src/                                         # Source code directory
    ├── pipeline.py                            # Main pipeline to run the project
    ├── analysis/                              # Scripts for analysis and insights
    │   ├── business_insights.py
    │   ├── feature_importance.py
    │   └── visualization.py
    ├── config/                                # Configuration files
    │   └── settings.py
    ├── data/                                  # Data loading and preprocessing modules
    │   ├── loader.py
    │   └── preprocessing.py
    ├── evaluation/                            # Model evaluation and validation
    │   ├── cross_validator.py
    │   └── evaluator.py
    └── models/                                # Machine learning model implementations
        ├── gradient_boosting.py
        ├── logistic_regression.py
        ├── random_forest.py
        └── trainer.py
```

## Getting Started

### Prerequisites

- Python 3.9+
- A virtual environment (recommended)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Customer_Purchase_Prediction
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    A `requirements.txt` file is not yet present. Based on the project structure, you will likely need the following libraries. You can install them manually:
    ```bash
    pip install pandas scikit-learn matplotlib seaborn jupyterlab
    ```
    *(It is recommended to create a `requirements.txt` file for easier dependency management.)*

## Usage

There are two main ways to run this project:

1.  **Run the full pipeline:**
    Execute the main pipeline script to preprocess the data, train the models, and evaluate them.
    ```bash
    python src/pipeline.py
    ```

2.  **Exploratory Analysis via Jupyter Notebook:**
    For a detailed, step-by-step walkthrough of the data analysis, feature engineering, and model building process, you can use the Jupyter Notebook.
    ```bash
    jupyter lab customer_purchase_prediction_analysis.ipynb
    ```

## Models Implemented

The following classification models are implemented in the `src/models/` directory:
-   **Logistic Regression**
-   **Random Forest**
-   **Gradient Boosting**

The models are trained and evaluated within the `trainer.py` and `evaluator.py` modules, respectively.
