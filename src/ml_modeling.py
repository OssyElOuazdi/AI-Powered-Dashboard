import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.pipeline import make_pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


def get_supervised_models(is_classification: bool):
    classification_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Support Vector Machine (SVM)": SVC(),
        "Random Forest": RandomForestClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree (CART)": DecisionTreeClassifier(),
        "Neural Network": MLPClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Bernoulli Naive Bayes": BernoulliNB(),
    }
    
    regression_models = {
        "Linear Regression": LinearRegression(),
        "Polynomial Regression": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
        "Support Vector Machine (SVM)": SVR(),
        "Random Forest": RandomForestRegressor(),
        "Decision Tree (CART)": DecisionTreeRegressor(),
        "K-Nearest Neighbors": KNeighborsRegressor(),
        "Neural Network": MLPRegressor(),
    }
    
    return classification_models if is_classification else regression_models

def get_unsupervised_models():
    return {
        "K-Means": KMeans(n_init=10),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5)
    }

def get_param_space(model_name: str, problem_type: str, is_classification: bool = None) -> dict:
    supervised_classification_spaces = {
        "Logistic Regression": {
            'C': Real(0.1, 100, prior='log-uniform'),
            'max_iter': Integer(100, 1000)
        },
        "Support Vector Machine (SVM)": {
            'C': Real(0.1, 100, prior='log-uniform'),
            'kernel': Categorical(['rbf', 'linear']),
            'gamma': Real(1e-4, 1.0, prior='log-uniform')
        },
        "Random Forest": {
            'n_estimators': Integer(10, 200),
            'max_depth': Integer(3, 20),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10)
        },
        "K-Nearest Neighbors": {
            'n_neighbors': Integer(1, 20),
            'weights': Categorical(['uniform', 'distance']),
            'p': Integer(1, 2)
        },
        "Decision Tree (CART)": {
            'max_depth': Integer(3, 20),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10)
        },
        "Neural Network": {
            'hidden_layer_sizes': Categorical([(50,), (100,), (50, 50), (100, 50)]),
            'alpha': Real(1e-5, 1.0, prior='log-uniform'),
            'learning_rate_init': Real(1e-4, 0.1, prior='log-uniform')
        },
        "Gaussian Naive Bayes": {
            'var_smoothing': Real(1e-10, 1e-8, prior='log-uniform')
        }
    }
    
    supervised_regression_spaces = {
        "Linear Regression": {},
        "Polynomial Regression": {
            'polynomialfeatures__degree': Integer(2, 4)
        },
        "Support Vector Machine (SVM)": {
            'C': Real(0.1, 100, prior='log-uniform'),
            'kernel': Categorical(['rbf', 'linear']),
            'gamma': Real(1e-4, 1.0, prior='log-uniform')
        },
        "Random Forest": {
            'n_estimators': Integer(10, 200),
            'max_depth': Integer(3, 20),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10)
        },
        "Decision Tree (CART)": {
            'max_depth': Integer(3, 20),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10)
        },
        "K-Nearest Neighbors": {
            'n_neighbors': Integer(1, 20),
            'weights': Categorical(['uniform', 'distance']),
            'p': Integer(1, 2)
        },
        "Neural Network": {
            'hidden_layer_sizes': Categorical([(50,), (100,), (50, 50), (100, 50)]),
            'alpha': Real(1e-5, 1.0, prior='log-uniform'),
            'learning_rate_init': Real(1e-4, 0.1, prior='log-uniform')
        }
    }
    
    unsupervised_spaces = {
        "K-Means": {
            'n_clusters': Integer(2, 10),
            'max_iter': Integer(100, 500),
        },
        "DBSCAN": {
            'eps': Real(0.1, 2.0),
            'min_samples': Integer(2, 10)
        }
    }
    
    if problem_type == "Supervised":
        return supervised_classification_spaces.get(model_name, {}) if is_classification else supervised_regression_spaces.get(model_name, {})
    else:
        return unsupervised_spaces.get(model_name, {})

def get_param_descriptions(model_name: str) -> Dict[str, str]:
    descriptions = {
        "Logistic Regression": {
            'C': "Inverse regularization strength (higher = less regularization)",
            'max_iter': "Maximum number of iterations for solver convergence"
        },
        "Support Vector Machine (SVM)": {
            'C': "Model complexity control (higher = more complex patterns)",
            'kernel': "Type of decision boundary (rbf = curved, linear = straight)",
            'gamma': "Influence of single training points (higher = more influence)"
        },
        "Random Forest": {
            'n_estimators': "Number of trees in the forest (higher = more complex model)",
            'max_depth': "Maximum depth of each tree (higher = more complex patterns)",
            'min_samples_split': "Minimum samples required to split a node (higher = simpler model)",
            'min_samples_leaf': "Minimum samples required in leaf nodes (higher = simpler model)"
        },
        "K-Nearest Neighbors": {
            'n_neighbors': "Number of neighbors to consider (higher = smoother decision boundary)",
            'weights': "How to weight neighbor votes (uniform = equal, distance = closer more important)",
            'p': "Distance calculation method (1 = manhattan, 2 = euclidean)"
        },
        "Decision Tree (CART)": {
            'max_depth': "Maximum depth of the tree (higher = more complex patterns)",
            'min_samples_split': "Minimum samples required to split a node (higher = simpler model)",
            'min_samples_leaf': "Minimum samples required in leaf nodes (higher = simpler model)"
        },
        "Neural Network": {
            'hidden_layer_sizes': "Network architecture (larger = more complex model)",
            'alpha': "Regularization strength (higher = simpler model)",
            'learning_rate_init': "Initial learning speed (higher = faster learning but less stable)"
        },
        "Gaussian Naive Bayes": {
            'var_smoothing': "Portion of the largest variance added to variances for calculation stability"
        },
        "Polynomial Regression": {
            'polynomialfeatures__degree': "Degree of polynomial features (higher = more complex model)"
        }
    }
    return descriptions.get(model_name, {})
def get_clustering_settings(df: pd.DataFrame, model_name: str):
    """Get appropriate settings for clustering models"""
    n_samples = len(df)
    
    if model_name == "K-Means":
        # Step 1: Compute the Elbow Method
        st.markdown(" Elbow Method to Determine Optimal Clusters :")
        max_k = 20  # Define the maximum number of clusters to evaluate
        inertia_values = []
        for k in range(2, max_k + 1):
           kmeans = KMeans(n_clusters=k, random_state=42)
           kmeans.fit(df)  # X is the dataset
           inertia_values.append(kmeans.inertia_)

        # Plot the Elbow Method
        fig, ax = plt.subplots()
        ax.plot(range(2, max_k + 1), inertia_values, 'bx-')
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Method to Determine Optimal Clusters")
        st.pyplot(fig)   

        # Suggest the optimal number of clusters based on the "elbow"
    
        suggested_clusters = st.number_input("Suggested Number of Clusters", min_value=2, max_value=max_k, value=4, step=1)
        st.info(f"Suggested number of clusters: {suggested_clusters}")

        return {
                'n_clusters': suggested_clusters,
                'max_iter': st.slider("Maximum iterations", 100, 1000, 300),
                'n_init': st.slider("Number of initializations", 1, 20, 10)
           
                  }
    elif model_name == "DBSCAN":
        # Calculate suggested eps using nearest neighbors
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
        distances = np.sort(np.linalg.norm(X_scaled[:, None] - X_scaled, axis=2), axis=1)
        suggested_eps = np.median(distances[:, 1])  # Use median of nearest neighbor distances
        
        return {
            'eps': st.slider("Epsilon (neighborhood radius)", 0.1, 2.0, float(suggested_eps), 0.1),
            'min_samples': st.slider("Minimum samples per cluster", 2, 20, 5)
        }
    
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import streamlit as st

from tensorflow.keras import layers


def create_lenet5_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        layers.Conv2D(6, (5, 5), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        layers.Conv2D(16, (5, 5), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def plot_training_history(history):
    """Plot training metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'])
    
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'])
    
    return fig

def ml_modeling_page():
    st.markdown("""
        <style>
            .navigation-bar {
                text-align: center;
                background-color: #272727;
                color: #E07B39;
                border-radius: 5px;
            }
            .navigation-bar h2 {
                margin: 0;
                color: #E07B39;
            }
        </style>
    """, unsafe_allow_html=True)
    # Custom CSS
    st.markdown(
        r"""
        <style>
            /* General Page Styling */
           body {
                background-color: #f4f4f9;
                font-family: 'Arial', sans-serif;
            }
            .navigation-bar {
            text-align: center;
            background-color: #2C3E50;
            color: #E07B39;
            border-radius: 5px;
        }
            .navigation-bar h2 {
                margin: 0;
                color: #E07B39; /* Header text in beige */
            }
            .stTabs [role="tablist"] {
            display: flex;
            justify-content: center;
            border-bottom: 1px solid #B6A28E; /* Subtle line under all tabs */
        }
            .stTabs [role="tab"] {
                flex-grow: 1;
                text-align: center;
                padding: 10px 20px;
                border: none; /* Remove borders around individual tabs */
                border-bottom: 2px solid transparent; /* Default: no underline */
                background-color: transparent; /* No background for inactive tabs */
                color: #B6A28E; /* Subtle text color for inactive tabs */
                cursor: pointer;
            }
            .stTabs [role="tab"][aria-selected="true"] {
                border-bottom: 2px solid #E07B39; /* Active tab underline */
                color: #E07B39; /* Active tab text color */
            }
            .stTabs [role="tab"]:hover {
                color: #E07B39; /* Darker hover color */
            }
            .data-preview {
                display: flex;
                justify-content: space-between;
                margin-top: 20px;
            }
            .data-section {
                width: 48%;
                padding: 10px;
                background-color: #ffffff;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            .data-section h4 {
                margin-bottom: 10px;
                color: #004080;
            }

            /* Header Section */
            .header {
                text-align: center;
                background-color: #272727;
                color: #FFFFFF;
                padding: 30px;
                border-radius: 10px;
            }
            .header h1 {
                font-size: 3rem;
                margin: 0;
                color: #E07B39;
            }
            .header p {
                font-size: 1.3rem;
                margin: 10px 0 0 0;
                color: #B6A28E;
            }

            /* Features Section */
            .features {
                display: flex;
                justify-content: center;
                gap: 30px;
                margin: 40px 0;
            }
            .feature-card {
                text-align: center;
                background-color: #F5F5DC;
                border-radius: 10px;
                padding: 20px;
                width: 22%;
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                transition: transform 0.3s ease;
            }
            .feature-card:hover {
                transform: translateY(-5px);
                background-color: #E07B39;
                color: #FFFFFF;
            }
            .feature-icon {
                display: inline-block;
                width: 60px;
                height: 60px;
                margin-bottom: 15px;
                background-size: contain;
                background-repeat: no-repeat;
                margin: 0 auto;
            }
            .feature-title {
                font-size: 1.2rem;
                font-weight: bold;
                color: #272727;
            }
            .feature-description {
                font-size: 1rem;
                color: #555555;
            }

            /* Sidebar Styling */
            [data-testid="stSidebar"] {
                background-color: #F0F4F8 !important;
                padding: 15px;
            }
            .sidebar-button {
                width: 100%;
                text-align: left;
                padding: 8px 10px;
                margin: 5px 0;
                background-color: transparent;
                border: none;
                cursor: pointer;
                color: #272727;
                font-size: 1rem;
                border-radius: 5px;
                transition: background-color 0.3s;
            }
            .sidebar-button:hover {
                background-color: #E07B39;
                color: #FFFFFF;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="navigation-bar"><h2>🧠 Data Modeling</h2></div>', unsafe_allow_html=True)

    # Initialize problem_type at the start
    problem_type = None
    selected_model = None
    is_classification = None
    target_column = None
    models = {}

    # Handle MNIST data case
    if 'X_processed' in st.session_state and 'y_processed' in st.session_state:
        st.subheader(" MNIST Model Configuration")
        
        X = st.session_state['X_processed']
        y = st.session_state['y_processed']

        with st.expander("LeNet-5 Architecture", expanded=True):
            st.write("""
            LeNet-5 CNN Architecture:
            1. Input Layer (28x28x1)
            2. Conv2D (6 filters, 5x5 kernel)
            3. MaxPooling2D (2x2)
            4. Conv2D (16 filters, 5x5 kernel)
            5. MaxPooling2D (2x2)
            6. Dense Layer (120 units)
            7. Dense Layer (84 units)
            8. Output Layer (10 units)
            """)
        
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.slider("Number of Epochs", 5, 50, 10)
            batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
        with col2:
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.1, 0.01, 0.001, 0.0001],
                value=0.001,
                format_func=lambda x: f"{x:.4f}"
            )
            validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2)
        
        if st.button("Train Model"):
            with st.spinner("Training LeNet-5 model..."):
                try:
                    model = create_lenet5_model(
                        input_shape=X.shape[1:],
                        num_classes=y.shape[1]
                    )
                    
                    model.compile(
                        optimizer=Adam(learning_rate=learning_rate),
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    callbacks = [
                        EarlyStopping(
                            monitor='val_loss',
                            patience=5,
                            restore_best_weights=True
                        ),
                        ModelCheckpoint(
                            'best_model.keras',
                            monitor='val_accuracy',
                            save_best_only=True
                        )
                    ]
                    
                    history = model.fit(
                        X, y,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        callbacks=callbacks,
                        verbose=1
                    )
                    
                    st.session_state['trained_model'] = model
                    st.session_state['training_history'] = history
                    st.session_state['problem_type'] = "CNN"  # Storing the problem type for CNN
                    st.success(" Model training completed!")
                    
                    #st.subheader(" Training History")
                    #history_fig = plot_training_history(history)
                    #st.pyplot(history_fig)
                    
                except Exception as e:
                    st.error(f"An error occurred during training: {str(e)}")

    # Handle standard ML data case
    elif 'processed_data' in st.session_state:
        df = st.session_state['processed_data']
        
        if 'problem_type' not in st.session_state:
            st.session_state['problem_type'] = None
        

        st.markdown("###  ")


        # Card for Problem Type Selection
        st.markdown("###  Select the Type of Problem :")
        # Problem type selection
        problem_type = st.radio(
            "",
            ["Supervised", "Unsupervised"],
            key="problem_type_selection"
        )

        st.session_state['problem_type'] = problem_type

        if problem_type == "Supervised":
            st.header("Select Target Variable :")
            target_column = st.selectbox(
                "Which variable do you want to predict?", 
                options=list(df.columns),
                index=len(df.columns) - 1
            )
            
            is_classification = df[target_column].nunique() < 10
            task_type = "Classification" if is_classification else "Regression"
            st.info(f"Detected Problem Type: {task_type}")
            models = get_supervised_models(is_classification)
        else:
            models = get_unsupervised_models()

        st.header("Choose Your Model :")
        selected_model = st.selectbox("Select model", list(models.keys()))
        st.session_state['current_model_name'] = selected_model

        # Only show these sections if problem_type is defined and model is selected
        if problem_type and selected_model:
            st.header("Configure Training Settings :")
            col1, col2 = st.columns(2)

            with col1:
                if problem_type == "Supervised":
                    test_size = st.slider("Test Data Size (%)", 10, 40, 20)
                    st.caption("Higher % = more data for testing, lower % = more data for training")
                else:
                    if selected_model in ["K-Means", "DBSCAN"]:
                        clustering_params = get_clustering_settings(df, selected_model)

            with col2:
                if problem_type == "Supervised":
                    cv_folds = st.slider("Cross-validation Folds", 3, 10, 5)
                    st.caption("Higher = more robust evaluation but slower training")

            st.header("Model Optimization :")
            perform_tuning = st.checkbox("Optimize model parameters automatically")

            if perform_tuning:
                n_iter = st.slider("Number of optimization trials", 10, 100, 20)
                st.caption("More trials = better optimization but longer training time")

                param_space = get_param_space(selected_model, problem_type, is_classification)
                param_descriptions = get_param_descriptions(selected_model)

                if param_descriptions:
                    st.expander("Parameter Descriptions").write(param_descriptions)

            if st.button("Train Model", type="primary"):
                st.session_state['training_in_progress'] = True

                with st.spinner("Training your model..."):
                    st.session_state['model_name'] = selected_model

                    if problem_type == "Supervised":
                        X = df.drop(columns=[target_column])
                        y = df[target_column]

                        if is_classification and y.dtype == 'object':
                            le = LabelEncoder()
                            y = le.fit_transform(y)
                    else:
                        X = df

                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    if problem_type == "Supervised":
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_scaled, y, test_size=test_size/100, random_state=42
                        )
                        if perform_tuning and param_space:
                            opt = BayesSearchCV(
                                models[selected_model],
                                param_space,
                                n_iter=n_iter,
                                cv=cv_folds,
                                n_jobs=-1,
                                random_state=42
                            )
                            opt.fit(X_train, y_train)
                            model = opt.best_estimator_

                            st.success("Model Optimization Results")
                            for param, value in opt.best_params_.items():
                                desc = param_descriptions.get(param, param)
                                st.write(f"- {desc}: {value}")
                        else:
                            model = models[selected_model]
                            model.fit(X_train, y_train)
                    else:
                        # Clustering models
                        if selected_model == "K-Means":
                            model = KMeans(**clustering_params)
                        elif selected_model == "DBSCAN":
                            model = DBSCAN(**clustering_params)
                        
                        model.fit(X_scaled)
                        
                        """# Evaluate clustering performance
                        if selected_model == "K-Means":
                            inertia = model.inertia_
                            st.write(f"Inertia (within-cluster sum of squares): {inertia:.2f}")
                            
                            silhouette_avg = silhouette_score(X_scaled, model.labels_)
                            st.write(f"Silhouette Score: {silhouette_avg:.2f}")
                        
                        elif selected_model == "DBSCAN":
                            n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
                            n_noise = list(model.labels_).count(-1)
                            st.write(f"Number of clusters found: {n_clusters}")
                            st.write(f"Number of noise points: {n_noise}")
                            
                            if n_clusters > 1:
                                mask = model.labels_ != -1
                                if mask.sum() > 1:
                                    silhouette_avg = silhouette_score(
                                        X_scaled[mask], 
                                        model.labels_[mask]
                                    )
                                    st.write(f"Silhouette Score (excluding noise): {silhouette_avg:.2f}")"""

                    # Update session state
                    st.session_state.update({
                        'trained_model': model,
                        'scaler': scaler,
                        'X_train': X_train if problem_type == "Supervised" else X_scaled,
                        'y_train': y_train if problem_type == "Supervised" else None,
                        'X_test': X_test if problem_type == "Supervised" else None,
                        'y_test': y_test if problem_type == "Supervised" else None,
                        'feature_names': X.columns.tolist(),
                        'target_name': target_column,
                        'problem_type': problem_type,
                        'is_classification': is_classification if problem_type == "Supervised" else None,
                        'current_model_name': selected_model
                    })

                    if problem_type == "Unsupervised":
                        st.session_state['cluster_labels'] = model.labels_

                    st.success(" Model trained successfully! Check the evaluation page for results.")
                    st.session_state['training_in_progress'] = False
    else:
        st.warning("Please process and save data first!")

if __name__ == "__main__":
    ml_modeling_page()