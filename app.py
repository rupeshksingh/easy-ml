import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Metrics
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score
)

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# XGBoost (if available)
try:
    from xgboost import XGBRegressor, XGBClassifier
except ImportError:
    XGBRegressor = None
    XGBClassifier = None

class MachineLearningApp:
    def __init__(self):
        # Initialize session state
        self.initialize_session_state()
        
        # Regression Metrics
        self.REGRESSION_METRICS = {
            'Mean Squared Error': mean_squared_error,
            'Mean Absolute Error': mean_absolute_error,
            'R-squared': r2_score,
            'Root Mean Squared Error': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
        }
        
        # Classification Metrics
        self.CLASSIFICATION_METRICS = {
            'Accuracy': accuracy_score,
            'Precision (Weighted)': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
            'Recall (Weighted)': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'),
            'F1 Score (Weighted)': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
            'ROC AUC Score': roc_auc_score
        }
        
    def initialize_session_state(self):
        """Initialize or reset session state variables"""
        session_keys = [
            'data', 'preprocessed_data', 'problem_type', 'target_column', 
            'feature_columns', 'model', 'train_test_split', 'evaluation_metrics',
            'X', 'y', 'preprocessor', 'selected_model'
        ]
        for key in session_keys:
            if key not in st.session_state:
                st.session_state[key] = None
    
    def load_data(self):
        """Data Upload and Initial Analysis"""
        st.header("Data Upload")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read the CSV
                df = pd.read_csv(uploaded_file)
                
                # Store in session state
                st.session_state.data = df
                
                # Display basic information
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Rows", df.shape[0])
                    st.metric("Total Columns", df.shape[1])
                
                with col2:
                    st.subheader("Column Types")
                    st.dataframe(df.dtypes)
                
                # Missing values visualization
                st.subheader("Missing Values")
                missing_data = df.isnull().sum()
                if len((missing_data[missing_data > 0])) == 0:
                    st.write("No Missing Values")
                else:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    missing_data[missing_data > 0].plot(kind='bar', ax=ax)
                    st.pyplot(fig)
                
                # Correlation Heatmap
                st.subheader("Correlation Heatmap")
                numeric_df = df.select_dtypes(include=['float64', 'int64'])
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
                
                return df
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
                return None
    
    def problem_type_selection(self, df):
        """Problem Type and Target Column Selection"""
        st.header("🎯 Problem Type Selection")
        
        # Problem Type Selection
        problem_type = st.selectbox(
            "Select Problem Type", 
            ["Classification", "Regression"]
        )
        st.session_state.problem_type = problem_type
        
        # Target Column Selection
        target_column = st.selectbox(
            "Select Target Column", 
            df.columns.tolist()
        )
        st.session_state.target_column = target_column
        
        # Feature Columns Selection
        feature_columns = st.multiselect(
            "Select Feature Columns", 
            [col for col in df.columns if col != target_column]
        )
        st.session_state.feature_columns = feature_columns
        
        return problem_type, target_column, feature_columns
    
    def preprocess_data(self, df, target_column, feature_columns, problem_type):
        """Advanced Data Preprocessing"""
        st.header("Data Preprocessing")
        
        # Separate features and target
        X = df[feature_columns]
        y = df[target_column]
        
        # Store X and y in session state
        st.session_state.X = X
        st.session_state.y = y
        
        # Identify column types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        # Preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_features)
            ])
        
        # Store preprocessor in session state
        st.session_state.preprocessor = preprocessor
        
        # Handle target column encoding
        if problem_type == 'Classification':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        
        # Display preprocessing details
        st.write("Numeric Features:", list(numeric_features))
        st.write("Categorical Features:", list(categorical_features))
        
        return preprocessor, X, y
    
    def train_test_split_configuration(self, X, y):
        """Configure Train-Test Split"""
        st.header("Train-Test Split Configuration")
        
        # Test Size Selection
        test_size = st.slider(
            "Test Set Percentage", 
            min_value=10, 
            max_value=50, 
            value=20, 
            step=5
        )
        
        # Random State Selection
        random_state = st.number_input(
            "Random State (for reproducibility)", 
            min_value=0, 
            max_value=1000, 
            value=42
        )
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=random_state
        )
        
        # Store split data in session state
        st.session_state.train_test_split = {
            'X_train': X_train, 
            'X_test': X_test, 
            'y_train': y_train, 
            'y_test': y_test,
            'test_size': test_size
        }
        
        return X_train, X_test, y_train, y_test
    
    def model_selection(self, problem_type):
        """Model Selection Based on Problem Type"""
        st.header("Model Selection")
        
        if problem_type == 'Regression':
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "Support Vector Regression": SVR()
            }
            if XGBRegressor:
                models["XGBoost Regressor"] = XGBRegressor()
        
        else:  # Classification
            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest Classifier": RandomForestClassifier(),
                "Gradient Boosting Classifier": GradientBoostingClassifier(),
                "Support Vector Machine": SVC(),
                "Naive Bayes": GaussianNB(),
                "K-Nearest Neighbors": KNeighborsClassifier()
            }
            if XGBClassifier:
                models["XGBoost Classifier"] = XGBClassifier()
        
        # Model Selection
        selected_model_name = st.selectbox("Choose a Model", list(models.keys()))
        selected_model = models[selected_model_name]
        
        # Store model in session state
        st.session_state.selected_model = selected_model
        
        return selected_model, selected_model_name
    
    def metrics_selection(self, problem_type):
        """Metrics Selection"""
        st.header("Metrics Selection")
        
        # Select metrics based on problem type
        if problem_type == 'Regression':
            available_metrics = self.REGRESSION_METRICS
        else:
            available_metrics = self.CLASSIFICATION_METRICS
        
        # Multiselect metrics
        selected_metrics = st.multiselect(
            "Choose Metrics to Display", 
            list(available_metrics.keys()),
            default=list(available_metrics.keys())
        )
        
        return {metric: available_metrics[metric] for metric in selected_metrics}
    
    def train_and_evaluate(self, preprocessor, X_train, X_test, y_train, y_test, model, problem_type, selected_metrics):
        """Model Training and Evaluation"""
        st.header("Model Training & Evaluation")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        
        # Compute selected metrics
        metrics_results = {}
        for metric_name, metric_func in selected_metrics.items():
            try:
                # Special handling for ROC AUC (only for binary classification)
                if metric_name == 'ROC AUC Score' and problem_type == 'Classification':
                    # Check if binary classification
                    if len(np.unique(y_test)) == 2:
                        metrics_results[metric_name] = metric_func(y_test, y_pred)
                    else:
                        st.warning("ROC AUC Score is only applicable for binary classification")
                else:
                    metrics_results[metric_name] = metric_func(y_test, y_pred)
            except Exception as e:
                st.error(f"Error computing {metric_name}: {e}")
        
        # Display Metrics
        for metric, value in metrics_results.items():
            st.metric(metric, f"{value:.4f}")
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        if problem_type == 'Regression':
            ax.scatter(y_test, y_pred)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Regression Prediction Comparison")
        else:
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            ax.set_title("Confusion Matrix")
        
        st.pyplot(fig)
        
        return metrics_results, pipeline
    
    def run(self):
        """Main Application Runner"""
        st.title("Universal Machine Learning Prediction App")
        
        # Data Loading
        df = self.load_data()
        
        if df is not None:
            # Problem Type Selection
            problem_type, target_column, feature_columns = self.problem_type_selection(df)
            
            # Preprocessing
            preprocessor, X, y = self.preprocess_data(df, target_column, feature_columns, problem_type)
            
            # Train-Test Split Configuration
            X_train, X_test, y_train, y_test = self.train_test_split_configuration(X, y)
            
            # Model Selection
            model, model_name = self.model_selection(problem_type)
            
            # Metrics Selection
            selected_metrics = self.metrics_selection(problem_type)
            
            # Training and Evaluation
            if st.button("Train Model"):
                metrics, trained_pipeline = self.train_and_evaluate(
                    preprocessor, X_train, X_test, y_train, y_test, 
                    model, problem_type, selected_metrics
                )

def main():
    app = MachineLearningApp()
    app.run()

if __name__ == "__main__":
    main()