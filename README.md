# Machine Learning Projects Repository

A comprehensive collection of machine learning projects covering data preprocessing, classification, clustering, regression, and end-to-end ML workflows with experiment tracking.

## 📁 Repository Structure

### 🔧 Data Preprocessing
- **DataPreprocessing/**: Essential data preprocessing techniques and templates
  - Data cleaning and transformation scripts
  - Titanic dataset processing
  - Data preprocessing templates for ML pipelines

### 🎯 Classification Models
- **Classification/**: Various classification algorithms and implementations
  - Naive Bayes (`Bayes.ipynb`)
  - Decision Trees (`DescisonTree.ipynb`)
  - Support Vector Machines (`svm.ipynb`, `kernel_svm.ipynb`)
  - K-Nearest Neighbors (`knn_neighbours.ipynb`)
  - Logistic Regression (`logistic_regression.ipynb`)
  - Social Network Ads dataset for classification tasks

### 🎯 Clustering
- **Clustering/**: Unsupervised learning techniques
  - K-Means clustering implementations
  - Mall Customers dataset for customer segmentation

### 📈 Regression Models
- **Regression Model/**: Comprehensive regression analysis
  - Simple and Multiple Linear Regression
  - Polynomial Regression
  - Support Vector Regression
  - Decision Tree Regression
  - Various datasets: Boston Housing, Energy Efficiency, Student Performance, etc.

### 🚀 End-to-End ML Projects
- **end-to-end-heart_disease_classification/**: Complete ML pipeline with MLflow
  - Heart disease prediction model
  - MLflow experiment tracking and model versioning
  - Production-ready deployment setup
  - Comprehensive requirements and project configuration

- **house_price_mlflow/**: House price prediction with MLflow
  - Advanced regression modeling
  - Model performance tracking
  - Artifact management

### 📚 Machine Learning A-Z Course Materials
- **S13 & S23 - Machine Learning A-Z (Model Selection)/**: Course implementations
  - Classification algorithms with both Jupyter notebooks and Python scripts
  - Regression models and techniques
  - Model selection and evaluation methods

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Jupyter Notebooks**: Interactive development and analysis
- **Scikit-learn**: Machine learning algorithms and tools
- **MLflow**: Experiment tracking and model management
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **UV**: Modern Python package management

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Machine_Learning
   ```

2. **Install dependencies**
   
   For the end-to-end heart disease project:
   ```bash
   cd end-to-end-heart_disease_classification
   pip install -r requirements.txt
   ```
   
   Or using UV (recommended):
   ```bash
   cd end-to-end-heart_disease_classification
   uv sync
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

### Running Projects

1. **Data Preprocessing**: Start with `DataPreprocessing/data_preprocessing_template.py` for a comprehensive preprocessing pipeline
2. **Classification**: Explore various algorithms in the `Classification/` folder
3. **Regression**: Run regression models from the `Regression Model/` directory
4. **End-to-End Projects**: 
   - Heart Disease: `cd end-to-end-heart_disease_classification && python main.py`
   - House Price: Open `house_price_mlflow/house_price.ipynb`

## 📊 Key Features

- **Comprehensive ML Algorithms**: From basic to advanced machine learning techniques
- **Real-world Datasets**: Practical datasets for hands-on learning
- **Experiment Tracking**: MLflow integration for model versioning and performance monitoring
- **Production Ready**: End-to-end projects with deployment considerations
- **Educational Content**: Course materials and step-by-step implementations

## 🎯 Project Highlights

### Heart Disease Classification
- Complete ML pipeline with data preprocessing, feature engineering, and model evaluation
- MLflow integration for experiment tracking
- Multiple algorithms comparison
- Model deployment ready

### House Price Prediction
- Advanced regression techniques
- Feature importance analysis
- Model performance optimization
- MLflow artifact management

## 📈 Learning Path

1. **Start with Data Preprocessing** - Learn essential data cleaning and preparation techniques
2. **Explore Classification** - Understand various classification algorithms
3. **Study Regression** - Master regression techniques and evaluation
4. **Practice Clustering** - Learn unsupervised learning methods
5. **Build End-to-End Projects** - Apply knowledge in complete ML workflows

## 🤝 Contributing

This repository serves as a learning resource and portfolio of machine learning projects. Feel free to:
- Report issues or bugs
- Suggest improvements
- Add new algorithms or datasets
- Enhance documentation

## 📝 License

This project is for educational and portfolio purposes.

---

**Created by iamAbhinav01** | *Machine Learning Enthusiast & Data Scientist*
