## Project Summary

This project is an end-to-end machine learning pipeline for predicting student performance based on demographic and educational factors. The workflow covers data exploration, preprocessing, model training, evaluation, and deployment.

### Tools & Libraries Used

- **Python**: Core programming language
- **Pandas, NumPy**: Data manipulation and analysis
- **Matplotlib, Seaborn**: Data visualization
- **scikit-learn**: Preprocessing, model selection, and evaluation
- **CatBoost**: Advanced machine learning modeling
- **Jupyter Notebook**: Exploratory data analysis and prototyping
- **Flask**: Web application for model inference (see [`application.py`](application.py))
- **AWS Elastic Beanstalk**: Deployment platform

### Project Workflow

1. **Exploratory Data Analysis (EDA)**
   - Performed in [`notebook/1 . EDA STUDENT PERFORMANCE .ipynb`](notebook/1%20.%20EDA%20STUDENT%20PERFORMANCE%20.ipynb)
   - Visualizes distributions, checks for missing values, and explores relationships between features and target.

2. **Data Preprocessing & Feature Engineering**
   - Handles categorical encoding, scaling, and feature selection.
   - Uses `ColumnTransformer`, `OneHotEncoder`, and `StandardScaler` from scikit-learn.

3. **Model Training & Evaluation**
   - Implemented in [`notebook/2. MODEL TRAINING.ipynb`](notebook/2.%20MODEL%20TRAINING.ipynb)
   - Trains multiple models (e.g., Linear Regression, Decision Tree, CatBoost).
   - Evaluates models using metrics like R² score and visualizes predictions.

4. **Model Serialization**
   - Trained model and preprocessor are saved as `.pkl` files in the `artifacts/` directory for deployment.

5. **Web Application**
   - [`application.py`](application.py) provides a Flask-based interface for users to input data and receive predictions.
   - Uses the serialized model and preprocessor for inference.

6. **Deployment**
   - The application is containerized and deployed to AWS Elastic Beanstalk using a `Procfile` and `.ebextensions/` for configuration.

### Component Functionality

- **notebook/1 . EDA STUDENT PERFORMANCE .ipynb**: Data understanding, visualization, and insights.
- **notebook/2. MODEL TRAINING.ipynb**: Data preprocessing, model training, evaluation, and saving artifacts.
- **artifacts/**: Stores trained model (`model.pkl`), preprocessor (`preprocessor.pkl`), and datasets.
- **application.py**: Flask app for serving predictions.
- **templates/**: HTML templates for the web interface.
- **logs/**: Stores logs for model training and inference (will be created when code is run).
- **catboost_info/**: Stores CatBoost training logs and metrics (will be created when code is run).

### Python Files in `components/` and `pipeline/`

- **components/data_ingestion.py**: Handles loading raw data, splitting into train/test sets, and saving them for further processing.
- **components/data_transformation.py**: Performs data cleaning, encoding, scaling, and prepares features for modeling.
- **components/model_trainer.py**: Trains machine learning models using the processed data and evaluates their performance.
- **pipeline/training_pipeline.py**: Orchestrates the full training workflow by sequentially calling data ingestion, transformation, and model training components.
- **pipeline/predict_pipeline.py**: Loads the trained model and preprocessor to make predictions on new input data.

This modular structure ensures reproducibility, scalability, and ease of deployment for real-world machine learning

### More Info

- **actual_req.txt**: This is the exhaustive list of libraries required for training, testing and deploying
- **requirements.txt**: This contains only the libraries necessary for deploying in AWS BeanStalk to save resources by installing only the required libraries at the time of deployment