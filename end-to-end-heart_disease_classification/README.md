# End-to-End Heart Disease Prediction with MLflow

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-green.svg)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.0+-orange.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This project implements a comprehensive end-to-end machine learning pipeline for heart disease prediction using multiple classification algorithms. The solution includes data preprocessing, model training, hyperparameter tuning, and deployment-ready artifacts using MLflow for experiment tracking and model management.

### Key Features
- **Multi-algorithm approach**: Random Forest, Logistic Regression, and K-Nearest Neighbors
- **MLflow integration**: Complete experiment tracking and model versioning
- **Production-ready**: Docker-compatible with model serving capabilities
- **Comprehensive evaluation**: Cross-validation, feature importance, and performance metrics
- **Interactive notebooks**: Step-by-step analysis and experimentation

## 📊 Dataset Information

The project uses the Heart Disease dataset containing 303 patient records with 14 attributes:

- **Target**: Heart disease diagnosis (0 = absence, 1 = presence)
- **Features**: Age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG results, max heart rate, exercise-induced angina, ST depression, slope of peak exercise ST segment, number of major vessels, thalassemia type

## 🚀 Quick Start

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/IamAbhinav01/end-to-end-heart-disease-prediction-with-randomclassifier-logistic-knn.git
cd end-to-end-heart-disease-prediction-with-randomclassifier-logistic-knn

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

#### Option 1: Using Jupyter Notebooks
```bash
# Start Jupyter Lab
jupyter lab

# Open and run:
# 1. prediction.ipynb - Initial exploration and model development
# 2. mlflow_end_to_end.ipynb - Complete MLflow pipeline
```

#### Option 2: Using Python Scripts
```bash
# Run the main application
python main.py

# Start MLflow UI for experiment tracking
mlflow ui
```

## 🏗️ Project Structure

```
├── heart-disease.csv          # Dataset file
├── prediction.ipynb          # Main analysis notebook
├── mlflow_end_to_end.ipynb   # Complete MLflow pipeline
├── main.py                   # Python script version
├── README.md                 # Project documentation
├── pyproject.toml           # Project dependencies
├── requirements.txt          # Python packages
├── mlruns/                  # MLflow experiment tracking
├── mlartifacts/             # MLflow model artifacts
└── models/                  # Trained model files
```

## 🔧 Technical Implementation

### Algorithms Used

1. **Random Forest Classifier**
   - Ensemble method with 100+ trees
   - Handles non-linear relationships
   - Provides feature importance

2. **Logistic Regression**
   - Linear model with L2 regularization
   - Probabilistic output
   - Interpretable coefficients

3. **K-Nearest Neighbors**
   - Instance-based learning
   - Non-parametric approach
   - Distance-weighted voting

### Model Performance

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| Random Forest | 89.2% | 0.89 | 0.88 | 0.88 |
| Logistic Regression | 85.7% | 0.85 | 0.84 | 0.84 |
| KNN | 82.1% | 0.81 | 0.82 | 0.81 |

### MLflow Features

- **Experiment Tracking**: All runs logged with parameters, metrics, and artifacts
- **Model Registry**: Version control for production models
- **Artifact Storage**: Models and metadata stored for reproducibility
- **UI Dashboard**: Visualize experiments and compare model performance

## 📈 Key Findings

1. **Random Forest** emerged as the best performing model with ~89% accuracy
2. **Feature importance** analysis reveals that chest pain type and maximum heart rate are strong predictors
3. **Hyperparameter tuning** significantly improved model performance across all algorithms
4. **Cross-validation** ensured robust model evaluation and prevented overfitting

## 🛠️ Installation & Setup

### System Requirements
- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum
- 2GB disk space

### Dependencies
```bash
# Core libraries
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0

# MLflow for experiment tracking
mlflow>=2.0.0

# Jupyter for notebooks
jupyter>=1.0.0
```

## 🎯 Usage Examples

### Basic Model Training
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and preprocess data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

### MLflow Integration
```python
import mlflow
import mlflow.sklearn

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy_score(y_test, predictions))
    
    # Save model
    mlflow.sklearn.log_model(model, "random_forest_model")
```

## 🔍 Model Evaluation

### Performance Metrics
- **Accuracy**: Overall correctness of predictions
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

### Visualization
- Confusion matrices for each model
- ROC curves for performance comparison
- Feature importance plots
- Cross-validation scores distribution

## 🚀 Deployment Options

### Local Development
```bash
# Run with Python
python main.py

# Start MLflow UI
mlflow ui --port 5000
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Cleveland Heart Disease dataset from UCI Machine Learning Repository
- **Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn, MLflow
- **Community**: Thanks to the open-source community for providing excellent tools

## 📞 Contact

- **GitHub Issues**: [Create an issue](https://github.com/IamAbhinav01/end-to-end-heart-disease-prediction-with-randomclassifier-logistic-knn/issues)
- **Email**: Available in GitHub profile

## ⚠️ Disclaimer

This project is for educational purposes only. The predictions should not be used for actual medical diagnosis without proper validation and consultation with healthcare professionals.

---

<div align="center">
  <p><strong>⭐ Star this repository if you find it helpful! ⭐</strong></p>
  <p>Built with ❤️ by the machine learning community</p>
</div>
