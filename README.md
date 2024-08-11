# Titanic Survival Prediction

This project involves building a machine learning model to predict the survival of passengers on the Titanic, based on various features such as age, gender, class, and other attributes. The model is developed using the Titanic dataset, a well-known benchmark in data science and machine learning.

## Overview

The Titanic dataset provides information on the fate of passengers aboard the RMS Titanic, which sank after colliding with an iceberg. The goal of this project is to develop a predictive model that can accurately determine whether a passenger survived or perished based on the available data.

## Key Features

- **Data Preprocessing**: 
  - Handling missing data, outliers, and incorrect values.
  - Encoding categorical features into numerical values for model compatibility.
  - Feature scaling and normalization to improve model performance.

- **Exploratory Data Analysis (EDA)**:
  - Visualizing the distribution of key features.
  - Analyzing correlations between features and survival outcomes.
  - Identifying patterns and trends within the data.

- **Feature Engineering**:
  - Creating new features such as family size, title extraction from names, and fare per person.
  - Selecting the most significant features to enhance model accuracy.

- **Model Development**:
  - Implementing multiple machine learning algorithms including Logistic Regression, Decision Trees, Random Forests, and Support Vector Machines (SVM).
  - Tuning hyperparameters to optimize model performance.
  - Utilizing cross-validation techniques to ensure the model generalizes well to unseen data.

- **Model Evaluation**:
  - Measuring model performance using accuracy, precision, recall, F1 score, and AUC-ROC curve.
  - Generating and interpreting confusion matrices to assess classification performance.

- **Prediction**:
  - Applying the trained model to make predictions on test data.
  - Providing insights into feature importance and model interpretability.

## Technologies Used

- **Python**: Core programming language for development.
- **Pandas**: Data manipulation and preprocessing.
- **NumPy**: Numerical computations.
- **Scikit-learn**: Machine learning model implementation and evaluation.
- **Matplotlib & Seaborn**: Data visualization tools for EDA.

## Getting Started

### Prerequisites

Ensure you have Python installed along with the necessary libraries. You can install the required dependencies using the provided `requirements.txt` file.

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/youssefa7med/Titanic-survived-prediction.git
   ```
2. **Navigate to the Project Directory**:
   ```bash
   cd Titanic-survived-prediction
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook Titanic_survival_prediction.ipynb
   ```

## Usage

1. **Data Exploration**: 
   - Begin with the exploratory data analysis to understand the dataset and its features.
2. **Model Training**: 
   - Execute the notebook cells to preprocess the data, train the models, and evaluate their performance.
3. **Making Predictions**: 
   - Use the trained model to predict the survival of passengers based on input features.

## Contributing

We welcome contributions from the community! If you have suggestions for improvements or new features, feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file.

## Acknowledgments

This project is inspired by the famous Titanic dataset provided by Kaggle. Special thanks to the open-source community for their contributions to the tools and libraries used in this project.
