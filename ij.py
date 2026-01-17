import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Optional imports with fallbacks
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

# Deep Learning Imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Set page configuration
st.set_page_config(page_title="Advanced Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Getting the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(working_dir, 'saved_models')

# Create models directory if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

class AdvancedDiseasePredictor:
    def __init__(self):
        self.ml_models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Naive Bayes': GaussianNB()
        }
        
        # Add models that can handle single class with proper configuration
        self.ml_models['Logistic Regression'] = LogisticRegression(random_state=42)
        
        if XGB_AVAILABLE:
            self.ml_models['XGBoost'] = XGBClassifier(random_state=42, verbosity=0)
        
        if LGBM_AVAILABLE:
            self.ml_models['LightGBM'] = LGBMClassifier(random_state=42)
        
    def create_dl_model(self, input_dim, num_classes=2):
        """Create a deep learning model"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available")
            
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            Dropout(0.3),
            
            Dense(16, activation='relu'),
            Dropout(0.2),
            
            Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train_models(self, X, y, disease_type):
        """Train all models and save them"""
        results = {}
        
        # Check if we have at least 2 classes
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            st.warning(f"⚠️ Dataset has only one class ({unique_classes[0]}). Adding synthetic data for second class.")
            # Add synthetic samples for the missing class
            n_synthetic = min(50, len(X) // 10)  # Add 5-10% synthetic samples
            X_synthetic = X[:n_synthetic] + np.random.normal(0, 0.1, (n_synthetic, X.shape[1]))
            y_synthetic = np.array([1 - unique_classes[0]] * n_synthetic)  # Opposite class
            X = np.vstack([X, X_synthetic])
            y = np.hstack([y, y_synthetic])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler
        scaler_path = os.path.join(models_dir, f'{disease_type}_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Train ML models
        ml_results = {}
        for name, model in self.ml_models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                ml_results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'predictions': y_pred
                }
                
                # Save model
                model_path = os.path.join(models_dir, f'{disease_type}_{name.replace(" ", "_").lower()}.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                    
                st.success(f"✅ {name} trained successfully - Accuracy: {accuracy:.3f}")
                    
            except Exception as e:
                st.warning(f"⚠️ Error training {name}: {str(e)}")
        
        # Train DL model if available
        if TF_AVAILABLE:
            try:
                dl_model = self.create_dl_model(X_train_scaled.shape[1], len(np.unique(y)))
                
                early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
                reduce_lr = ReduceLROnPlateau(factor=0.5, patience=5)
                
                history = dl_model.fit(
                    X_train_scaled, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_test_scaled, y_test),
                    callbacks=[early_stopping, reduce_lr],
                    verbose=0
                )
                
                dl_accuracy = dl_model.evaluate(X_test_scaled, y_test, verbose=0)[1]
                
                # Save DL model
                dl_model_path = os.path.join(models_dir, f'{disease_type}_dl_model.h5')
                dl_model.save(dl_model_path)
                
                ml_results['Deep Learning'] = {
                    'model': dl_model,
                    'accuracy': dl_accuracy,
                    'history': history
                }
                
                st.success(f"✅ Deep Learning trained successfully - Accuracy: {dl_accuracy:.3f}")
                
            except Exception as e:
                st.warning(f"⚠️ Error training Deep Learning model: {str(e)}")
        
        return ml_results, X_test_scaled, y_test

# Improved sample data generation with proper class distribution
def load_sample_data(disease_type):
    """Load or generate sample data for each disease type"""
    np.random.seed(42)
    
    if disease_type == 'diabetes':
        n_samples = 1000
        n_features = 8
        X = np.random.randn(n_samples, n_features)
        # Create realistic correlations for diabetes
        X[:, 1] = X[:, 1] * 0.5 + X[:, 0] * 0.3 + np.random.randn(n_samples) * 0.2
        X[:, 5] = X[:, 5] * 0.6 + X[:, 1] * 0.2 + np.random.randn(n_samples) * 0.2
        # Ensure we have both classes
        y = (X[:, 1] * 0.3 + X[:, 5] * 0.2 + X[:, 7] * 0.1 + np.random.randn(n_samples) * 0.5 > 0).astype(int)
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
    elif disease_type == 'heart':
        n_samples = 1000
        n_features = 13
        X = np.random.randn(n_samples, n_features)
        # Create realistic correlations for heart disease
        X[:, 0] = np.random.normal(55, 10, n_samples)
        X[:, 3] = np.random.normal(130, 20, n_samples)
        X[:, 4] = np.random.normal(250, 50, n_samples)
        # Ensure we have both classes with proper distribution
        risk_score = X[:, 0] * 0.1 + X[:, 3] * 0.2 + X[:, 4] * 0.15 + np.random.randn(n_samples) * 0.5
        # Create balanced classes
        threshold = np.percentile(risk_score, 60)  # 40% positive, 60% negative
        y = (risk_score > threshold).astype(int)
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
    elif disease_type == 'parkinsons':
        n_samples = 1000
        n_features = 22
        X = np.random.randn(n_samples, n_features)
        # Create realistic correlations for Parkinson's
        X[:, 0] = np.random.normal(150, 30, n_samples)
        X[:, 1] = X[:, 0] + np.random.normal(20, 5, n_samples)
        X[:, 2] = X[:, 0] - np.random.normal(20, 5, n_samples)
        # Ensure we have both classes
        risk_score = X[:, 16] * 0.3 + X[:, 17] * 0.2 + X[:, 21] * 0.4 + np.random.randn(n_samples) * 0.3
        threshold = np.percentile(risk_score, 70)  # 30% positive, 70% negative
        y = (risk_score > threshold).astype(int)
        feature_names = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
                        'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
                        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
                        'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
                        'spread1', 'spread2', 'D2', 'PPE']
    
    st.info(f"📊 Generated dataset with {np.sum(y==1)} positive and {np.sum(y==0)} negative samples")
    return X, y, feature_names

# Load models function with improved error handling
def load_saved_models(disease_type):
    """Load all saved models for a specific disease"""
    models = {}
    
    # ML models to load
    ml_model_names = [
        'random_forest', 'k_nearest_neighbors', 'decision_tree', 
        'naive_bayes', 'logistic_regression'
    ]
    
    if XGB_AVAILABLE:
        ml_model_names.append('xgboost')
    if LGBM_AVAILABLE:
        ml_model_names.append('lightgbm')
    
    for model_name in ml_model_names:
        model_path = os.path.join(models_dir, f'{disease_type}_{model_name}.pkl')
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
            except Exception as e:
                st.warning(f"Error loading {model_name}: {str(e)}")
    
    # Load DL model if available
    if TF_AVAILABLE:
        dl_model_path = os.path.join(models_dir, f'{disease_type}_dl_model.h5')
        if os.path.exists(dl_model_path):
            try:
                models['deep_learning'] = load_model(dl_model_path)
            except Exception as e:
                st.warning(f"Error loading Deep Learning model: {str(e)}")
    
    # Load scaler
    scaler_path = os.path.join(models_dir, f'{disease_type}_scaler.pkl')
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                models['scaler'] = pickle.load(f)
        except Exception as e:
            st.warning(f"Error loading scaler: {str(e)}")
    
    return models

# Get available algorithms
def get_available_algorithms():
    """Get list of available algorithms based on installed packages"""
    algorithms = [
        'Random Forest', 'K-Nearest Neighbors', 'Decision Tree', 
        'Naive Bayes', 'Logistic Regression'
    ]
    
    if XGB_AVAILABLE:
        algorithms.append('XGBoost')
    if LGBM_AVAILABLE:
        algorithms.append('LightGBM')
    if TF_AVAILABLE:
        algorithms.append('Deep Learning')
    
    return algorithms

# Safe prediction function
def safe_predict(model, input_data, model_type='ml'):
    """Safely make predictions and handle different model types"""
    try:
        if model_type == 'dl':
            prediction = model.predict(input_data)
            if prediction.shape[1] == 2:
                probability = prediction[0][1]
                result = 1 if probability > 0.5 else 0
            else:
                probability = prediction[0][0]
                result = 1 if probability > 0.5 else 0
        else:
            # For ML models
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_data)
                if probabilities.shape[1] == 2:
                    probability = probabilities[0][1]
                else:
                    probability = probabilities[0][0]
                result = model.predict(input_data)[0]
            else:
                result = model.predict(input_data)[0]
                probability = 0.8 if result == 1 else 0.2
        
        return result, probability
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return 0, 0.5

# Initialize the predictor
predictor = AdvancedDiseasePredictor()

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Advanced Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Model Training',
                            'Model Comparison'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person', 'cpu', 'graph-up'],
                           default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('🧬 Diabetes Prediction')
    
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=1)
        Glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=120)
        BloodPressure = st.number_input('Blood Pressure value', min_value=0, max_value=150, value=70)

    with col2:
        SkinThickness = st.number_input('Skin Thickness value', min_value=0, max_value=100, value=20)
        Insulin = st.number_input('Insulin Level', min_value=0, max_value=900, value=80)
        BMI = st.number_input('BMI value', min_value=0.0, max_value=70.0, value=25.0)

    with col3:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', 
                                                 min_value=0.0, max_value=3.0, value=0.5)
        Age = st.number_input('Age of the Person', min_value=0, max_value=120, value=30)
    
    available_algorithms = get_available_algorithms()
    algorithm = st.selectbox('Select Algorithm', available_algorithms)

    if st.button('🔍 Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]
        
        models = load_saved_models('diabetes')
        
        if not models:
            st.error("No trained models found. Please train models first.")
        else:
            scaler = models.get('scaler')
            if scaler:
                user_input_scaled = scaler.transform([user_input])
                
                if algorithm == 'Deep Learning':
                    if 'deep_learning' in models:
                        result, probability = safe_predict(models['deep_learning'], user_input_scaled, 'dl')
                    else:
                        st.error("Deep Learning model not found")
                        result, probability = 0, 0.5
                else:
                    model_key = algorithm.lower().replace(' ', '_')
                    if model_key in models:
                        result, probability = safe_predict(models[model_key], user_input_scaled, 'ml')
                    else:
                        st.error(f"{algorithm} model not found")
                        result, probability = 0, 0.5
                
                if result == 1:
                    st.error('🩺 The person is likely diabetic')
                else:
                    st.success('✅ The person is not diabetic')
                
                confidence = probability if result == 1 else 1 - probability
                st.info(f'🔬 Confidence: {confidence:.2%}')

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction')
    
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, value=50)
        sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
        cp = st.number_input('Chest Pain types', min_value=0, max_value=3, value=1)
        trestbps = st.number_input('Resting Blood Pressure', min_value=0, max_value=200, value=120)

    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl', min_value=0, max_value=600, value=200)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1], 
                          format_func=lambda x: 'No' if x == 0 else 'Yes')
        restecg = st.number_input('Resting Electrocardiographic results', min_value=0, max_value=2, value=0)
        thalach = st.number_input('Maximum Heart Rate achieved', min_value=0, max_value=220, value=150)

    with col3:
        exang = st.selectbox('Exercise Induced Angina', [0, 1], 
                           format_func=lambda x: 'No' if x == 0 else 'Yes')
        oldpeak = st.number_input('ST depression induced by exercise', min_value=0.0, max_value=10.0, value=1.0)
        slope = st.number_input('Slope of the peak exercise ST segment', min_value=0, max_value=2, value=1)
        ca = st.number_input('Major vessels colored by flourosopy', min_value=0, max_value=4, value=0)
        thal = st.number_input('thal', min_value=0, max_value=3, value=1)
    
    available_algorithms = get_available_algorithms()
    algorithm = st.selectbox('Select Algorithm', available_algorithms, key='heart_algo')

    if st.button('🔍 Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        
        models = load_saved_models('heart')
        
        if not models:
            st.error("No trained models found. Please train models first.")
        else:
            scaler = models.get('scaler')
            if scaler:
                user_input_scaled = scaler.transform([user_input])
                
                if algorithm == 'Deep Learning':
                    if 'deep_learning' in models:
                        result, probability = safe_predict(models['deep_learning'], user_input_scaled, 'dl')
                    else:
                        st.error("Deep Learning model not found")
                        result, probability = 0, 0.5
                else:
                    model_key = algorithm.lower().replace(' ', '_')
                    if model_key in models:
                        result, probability = safe_predict(models[model_key], user_input_scaled, 'ml')
                    else:
                        st.error(f"{algorithm} model not found")
                        result, probability = 0, 0.5
                
                if result == 1:
                    st.error('🩺 The person is likely to have heart disease')
                else:
                    st.success('✅ The person is not likely to have heart disease')
                
                confidence = probability if result == 1 else 1 - probability
                st.info(f'🔬 Confidence: {confidence:.2%}')

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    st.title("🧠 Parkinson's Disease Prediction")
    
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.number_input('MDVP:Fo(Hz)', min_value=0.0, max_value=300.0, value=150.0)
        fhi = st.number_input('MDVP:Fhi(Hz)', min_value=0.0, max_value=300.0, value=170.0)
        flo = st.number_input('MDVP:Flo(Hz)', min_value=0.0, max_value=300.0, value=130.0)
        Jitter_percent = st.number_input('MDVP:Jitter(%)', min_value=0.0, max_value=1.0, value=0.005)

    with col2:
        Jitter_Abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0, max_value=1.0, value=0.00003)
        RAP = st.number_input('MDVP:RAP', min_value=0.0, max_value=1.0, value=0.003)
        PPQ = st.number_input('MDVP:PPQ', min_value=0.0, max_value=1.0, value=0.003)
        DDP = st.number_input('Jitter:DDP', min_value=0.0, max_value=1.0, value=0.01)

    with col3:
        Shimmer = st.number_input('MDVP:Shimmer', min_value=0.0, max_value=1.0, value=0.03)
        Shimmer_dB = st.number_input('MDVP:Shimmer(dB)', min_value=0.0, max_value=1.0, value=0.3)
        APQ3 = st.number_input('Shimmer:APQ3', min_value=0.0, max_value=1.0, value=0.02)
        APQ5 = st.number_input('Shimmer:APQ5', min_value=0.0, max_value=1.0, value=0.03)

    with col4:
        APQ = st.number_input('MDVP:APQ', min_value=0.0, max_value=1.0, value=0.04)
        DDA = st.number_input('Shimmer:DDA', min_value=0.0, max_value=1.0, value=0.06)
        NHR = st.number_input('NHR', min_value=0.0, max_value=1.0, value=0.02)
        HNR = st.number_input('HNR', min_value=0.0, max_value=40.0, value=20.0)

    with col5:
        RPDE = st.number_input('RPDE', min_value=0.0, max_value=1.0, value=0.5)
        DFA = st.number_input('DFA', min_value=0.0, max_value=1.0, value=0.7)
        spread1 = st.number_input('spread1', min_value=-10.0, max_value=10.0, value=-5.0)
        spread2 = st.number_input('spread2', min_value=0.0, max_value=1.0, value=0.2)
        D2 = st.number_input('D2', min_value=0.0, max_value=10.0, value=2.0)
        PPE = st.number_input('PPE', min_value=0.0, max_value=1.0, value=0.2)

    available_algorithms = get_available_algorithms()
    algorithm = st.selectbox('Select Algorithm', available_algorithms, key='parkinsons_algo')

    if st.button("🔍 Parkinson's Test Result"):
        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                     Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, 
                     RPDE, DFA, spread1, spread2, D2, PPE]
        
        models = load_saved_models('parkinsons')
        
        if not models:
            st.error("No trained models found. Please train models first.")
        else:
            scaler = models.get('scaler')
            if scaler:
                user_input_scaled = scaler.transform([user_input])
                
                if algorithm == 'Deep Learning':
                    if 'deep_learning' in models:
                        result, probability = safe_predict(models['deep_learning'], user_input_scaled, 'dl')
                    else:
                        st.error("Deep Learning model not found")
                        result, probability = 0, 0.5
                else:
                    model_key = algorithm.lower().replace(' ', '_')
                    if model_key in models:
                        result, probability = safe_predict(models[model_key], user_input_scaled, 'ml')
                    else:
                        st.error(f"{algorithm} model not found")
                        result, probability = 0, 0.5
                
                if result == 1:
                    st.error("🩺 The person is likely to have Parkinson's disease")
                else:
                    st.success("✅ The person is not likely to have Parkinson's disease")
                
                confidence = probability if result == 1 else 1 - probability
                st.info(f'🔬 Confidence: {confidence:.2%}')

# Model Training Page
if selected == 'Model Training':
    st.title('🤖 Train Models')
    
    disease_type = st.selectbox('Select Disease Type', ['diabetes', 'heart', 'parkinsons'])
    
    if st.button('🚀 Start Model Training'):
        with st.spinner('Training models...'):
            X, y, feature_names = load_sample_data(disease_type)
            results, X_test, y_test = predictor.train_models(X, y, disease_type)
            
            st.success('✅ Model training completed!')
            
            # Show accuracy comparison
            st.subheader('📊 Model Performance')
            
            accuracy_data = []
            for model_name, result in results.items():
                if 'accuracy' in result:
                    accuracy_data.append({
                        'Model': model_name,
                        'Accuracy': f"{result['accuracy']:.3f}"
                    })
            
            if accuracy_data:
                accuracy_df = pd.DataFrame(accuracy_data)
                st.dataframe(accuracy_df, use_container_width=True)

# Model Comparison Page
if selected == 'Model Comparison':
    st.title('📈 Model Comparison')
    
    disease_type = st.selectbox('Select Disease Type', ['diabetes', 'heart', 'parkinsons'], key='compare_disease')
    
    results_path = os.path.join(models_dir, f'{disease_type}_training_results.pkl')
    
    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        
        st.subheader(f'Model Accuracies for {disease_type.title()}')
        
        accuracy_data = []
        for model_name, result in results.items():
            if 'accuracy' in result:
                accuracy_data.append({
                    'Model': model_name,
                    'Accuracy': result['accuracy']
                })
        
        if accuracy_data:
            accuracy_df = pd.DataFrame(accuracy_data)
            accuracy_df = accuracy_df.sort_values('Accuracy', ascending=False)
            
            st.dataframe(accuracy_df, use_container_width=True)
            
            st.subheader('Accuracy Chart')
            chart_data = accuracy_df.set_index('Model')['Accuracy']
            st.bar_chart(chart_data)
    else:
        st.warning("No training results found. Please train models first.")

# Footer
st.markdown("---")
st.markdown("### 🏥 Health Prediction System")