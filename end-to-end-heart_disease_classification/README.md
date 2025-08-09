# End-to-End Heart Disease Prediction

A comprehensive machine learning project that predicts heart disease using multiple classification algorithms including Random Forest, Logistic Regression, and K-Nearest Neighbors (KNN).

## 🎯 Project Overview

This project implements an end-to-end machine learning pipeline for heart disease prediction using the Cleveland Heart Disease dataset. The system compares multiple classification algorithms to identify the best performing model for predicting the presence of heart disease.

## 📊 Dataset Information

- **Source**: Cleveland Heart Disease Dataset
- **File**: `heart-disease.csv`
- **Total Records**: 303 patients
- **Features**: 13 medical attributes
- **Target**: Heart disease diagnosis (0 = No disease, 1 = Disease)

### Features Used:
1. Age
2. Sex
3. Chest pain type (cp)
4. Resting blood pressure (trestbps)
5. Serum cholesterol (chol)
6. Fasting blood sugar (fbs)
7. Resting ECG results (restecg)
8. Maximum heart rate achieved (thalach)
9. Exercise induced angina (exang)
10. ST depression (oldpeak)
11. Slope of peak exercise ST segment
12. Number of major vessels (ca)
13. Thalassemia (thal)

## 🛠️ Technologies Used

- **Python**: 3.10+
- **Libraries**:
  - NumPy: Numerical computing
  - Pandas: Data manipulation
  - Scikit-learn: Machine learning algorithms
  - Matplotlib & Seaborn: Data visualization
  - MLflow: Experiment tracking and model management

## 🚀 Model Comparison

The project evaluates three different classification algorithms:

| Model | Accuracy | Description |
|-------|----------|-------------|
| **Random Forest** | ~87% | Ensemble method using multiple decision trees |
| **Logistic Regression** | ~85% | Linear model for binary classification |
| **KNN** | ~83% | Instance-based learning algorithm |

### Hyperparameter Tuning
- **RandomizedSearchCV**: Initial parameter exploration
- **GridSearchCV**: Fine-tuning optimal parameters
- **Cross-validation**: 5-fold CV for robust evaluation

## 📈 Model Performance Metrics

After hyperparameter tuning, the models achieve:
- **Precision**: Measures the accuracy of positive predictions
- **Recall**: Measures the ability to find all positive cases
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

## 🔧 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/IamAbhinav01/end-to-end-heart-disease-prediction-with-randomclassifier-logistic-knn.git
cd end-to-end-heart-disease-prediction-with-randomclassifier-logistic-knn
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the project**
```bash
python prediction.ipynb
```

## 📁 Project Structure

```
end-to-end-heart-disease-classification/
├── heart-disease.csv          # Dataset file
├── prediction.ipynb          # Main analysis notebook
├── main.py                   # Python script version
├── README.md                 # Project documentation
├── pyproject.toml           # Project dependencies
├── requirements.txt          # Python packages
└── mlruns/                 # MLflow experiment tracking
```

## 🎯 Usage

### Running the Analysis
1. Open `prediction.ipynb` in Jupyter Notebook
2. Run all cells to execute the complete pipeline
3. View model comparison results and visualizations

### Key Outputs:
- Model accuracy comparison chart
- KNN performance visualization
- ROC curves for model evaluation
- Classification reports with detailed metrics

## 🔍 Key Findings

1. **Random Forest** emerged as the best performing model with ~87% accuracy
2. **Feature importance** analysis reveals that chest pain type and maximum heart rate are strong predictors
3. **Hyperparameter tuning** significantly improved model performance across all algorithms
4. **Cross-validation** ensured robust model evaluation and prevented overfitting

## 🚀 Future Improvements

- [ ] Add more sophisticated feature engineering
- [ ] Implement ensemble methods combining multiple models
- [ ] Add model explainability using SHAP values
- [ ] Deploy the model as a REST API
- [ ] Create a web interface for predictions
- [ ] Add continuous integration/deployment (CI/CD)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Contact

For questions or suggestions, please open an issue or contact the repository maintainer.

---

**Note**: This project is for educational purposes. The predictions should not be used for actual medical diagnosis without proper validation and consultation with healthcare professionals.
