import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error,silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from sklearn.model_selection import learning_curve, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
#import shap
import zipfile
import os
import io
import json
from datetime import datetime
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from tensorflow.keras.preprocessing import image
from streamlit_drawable_canvas import st_canvas
from PIL import Image  # For handling image files
import numpy as np     # For numerical operations
from tensorflow.keras.preprocessing.image import img_to_array  # For image array conversion



def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return fig

def plot_clustering_2d(X, labels, centers=None, title="Clustering Results"):
    """Create a 2D visualization of clustering results"""
    # Apply PCA if data has more than 2 dimensions
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
    else:
        X_2d = X

    fig = plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis')
    
    if centers is not None:
        centers_2d = pca.transform(centers) if X.shape[1] > 2 else centers
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='x', s=200, linewidths=3)
    
    plt.title(title)
    plt.colorbar(scatter)
    return fig

def plot_elbow_curve(X, k_range):
    """Create elbow curve for K-means"""
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    return fig

def plot_silhouette(X, labels):
    """Create silhouette plot"""
    from sklearn.metrics import silhouette_samples
    
    n_clusters = len(np.unique(labels[labels != -1]))
    if n_clusters <= 1:
        return None
        
    silhouette_vals = silhouette_samples(X, labels)
    
    fig = plt.figure(figsize=(10, 6))
    y_lower = 10
    
    for i in range(n_clusters):
        ith_cluster_silhouette_values = silhouette_vals[labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         alpha=0.7)
        
        y_lower = y_upper + 10
    
    plt.title("Silhouette Plot")
    plt.xlabel("Silhouette Coefficient Values")
    plt.ylabel("Cluster Label")
    return fig
def add_prediction_section():
    st.header("Make Predictions :")
    
    # Get feature names and model from session state
    feature_names = st.session_state['feature_names']
    model = st.session_state['trained_model']
    
    # Create input method selection
    input_method = st.radio("Select input method:", ["Single Prediction", "Batch Prediction"])
    
    if input_method == "Single Prediction":
        # Create columns for feature inputs
        cols = st.columns(3)
        feature_values = {}
        
        # Create input fields for each feature
        for i, feature in enumerate(feature_names):
            with cols[i % 3]:
                feature_values[feature] = st.number_input(f"{feature}", value=0.0)
        
        if st.button("Predict"):
            # Create input array
            input_data = np.array([[feature_values[feature] for feature in feature_names]])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display result
            st.success(f"Prediction: {prediction}")
            
    else:  # Batch Prediction
        st.write("Upload a CSV file with the same features as your training data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                input_df = pd.read_csv(uploaded_file)
                
                # Verify columns match
                if set(feature_names) != set(input_df.columns):
                    st.error("Columns in uploaded file don't match the model features!")
                    return
                
                # Make predictions
                predictions = model.predict(input_df[feature_names])
                
                # Add predictions to dataframe
                input_df['Prediction'] = predictions
                
                # Display results
                st.write("Predictions:")
                st.dataframe(input_df)
                
                # Download results
                csv = input_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

def add_clustering_prediction_section():
    st.header("Predict Clusters")
    
    model = st.session_state['trained_model']
    feature_names = st.session_state['feature_names']
    scaler = st.session_state.get('scaler')  # Get scaler if available
    
    input_method = st.radio("Select input method:", ["Single Sample", "Batch Samples"])
    
    if input_method == "Single Sample":
        cols = st.columns(3)
        feature_values = {}
        
        for i, feature in enumerate(feature_names):
            with cols[i % 3]:
                feature_values[feature] = st.number_input(f"{feature}", value=0.0)
        
        if st.button("Predict Cluster"):
            input_data = np.array([[feature_values[feature] for feature in feature_names]])
            if scaler:
                input_data = scaler.transform(input_data)
            cluster = model.predict(input_data)[0]
            st.success(f"Predicted Cluster: {cluster}")
            
    else:
        st.write("Upload a CSV file with features")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file:
            try:
                input_df = pd.read_csv(uploaded_file)
                
                if set(feature_names) != set(input_df.columns):
                    st.error("Column mismatch with model features!")
                    return
                
                data = input_df[feature_names].values
                if scaler:
                    data = scaler.transform(data)
                    
                clusters = model.predict(data)
                input_df['Predicted_Cluster'] = clusters
                
                st.write("Predictions:")
                st.dataframe(input_df)
                
                csv = input_df.to_csv(index=False)
                st.download_button(
                    "Download Predictions",
                    data=csv,
                    file_name="cluster_predictions.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

def save_model_results(model, metrics, model_name):
    """
    Save model and its evaluation metrics to session state
    """
    if 'saved_models' not in st.session_state:
        st.session_state['saved_models'] = {}
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_key = f"{model_name}_{timestamp}"
    
    st.session_state['saved_models'][model_key] = {
        'model': model,
        'metrics': metrics,
        'name': model_name,
        'timestamp': timestamp,
        'is_classification': st.session_state['is_classification'],
        'feature_names': st.session_state['feature_names'],
        'target_name': st.session_state['target_name']
    }
    
    return model_key

def plot_models_comparison():
    """
    Create comparison plots for all saved models
    """
    saved_models = st.session_state.get('saved_models', {})
    if not saved_models:
        return None
    
    # Extract metrics for all models
    model_metrics = []
    for key, data in saved_models.items():
        metrics = data['metrics']
        metrics['Model'] = f"{data['name']}\n({data['timestamp']})"
        model_metrics.append(metrics)
    
    df_metrics = pd.DataFrame(model_metrics)
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    metrics_to_plot = [col for col in df_metrics.columns if col != 'Model']
    
    df_plot = df_metrics.melt(id_vars=['Model'], value_vars=metrics_to_plot, 
                             var_name='Metric', value_name='Value')
    
    sns.barplot(data=df_plot, x='Model', y='Value', hue='Metric')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return plt.gcf()

# Helper function to save plots as PNG
def save_plot_as_png(plot, filename):
    """
    Save a Matplotlib plot as a PNG file.
    """
    buf = io.BytesIO()
    plot.savefig(buf, format="png")
    buf.seek(0)
    with open(filename, "wb") as f:
        f.write(buf.read())
    buf.close()

'''def plot_feature_importance_lime(model, feature_names, X_train):
    """
    Generate feature importance plot using LIME for model-agnostic explanation.
    """
    try:
        # Create a LIME explainer for tabular data
        explainer = LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=feature_names,
            class_names=['class_0', 'class_1'],  # or adapt based on the model's output classes
            mode='classification'  # Use 'regression' for regression models
        )
        
        # Select a random instance from the training data to explain
        instance_idx = np.random.randint(0, X_train.shape[0])
        explanation = explainer.explain_instance(X_train[instance_idx], model.predict_proba)
        
        # Get the feature importance from the LIME explanation
        importance = explanation.as_list()
        importance_df = pd.DataFrame(importance, columns=['feature', 'importance'])
        
        # Sort and plot the top 10 features
        importance_df = importance_df.sort_values('importance', ascending=False).head(10)

        plt.figure(figsize=(6, 4))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('Top 10 Feature Importance (LIME)')
        return plt.gcf()
    
    except Exception as e:
        print(f"Error: {e}")
    return None'''

'''
# Plot Feature Importance
def plot_feature_importance(model, feature_names):
    """
    Generate feature importance plot using SHAP or model-specific methods.
    """
    try:
        # Use SHAP for feature importance
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(np.array(feature_names))
        shap.summary_plot(shap_values, feature_names, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance")
        return plt.gcf()
    except Exception:
        # Fallback for models with feature_importances_ or coef_
        if hasattr(model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(6, 4))
            sns.barplot(data=importances.head(10), x='importance', y='feature')
            plt.title('Top 10 Feature Importance')
            return plt.gcf()
        elif hasattr(model, 'coef_'):
            coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            importances = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(coef)
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(6, 4))
            sns.barplot(data=importances.head(10), x='importance', y='feature')
            plt.title('Top 10 Feature Importance')
            return plt.gcf()
    return None
'''

def plot_training_history(history):
    """Plot training metrics"""
    import matplotlib.pyplot as plt
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
# Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt.gcf()


# Plot Regression Scatter Plot
def plot_regression_scatter(y_true, y_pred):
    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predicted vs Actual Values')
    return plt.gcf()


# Plot Learning Curve
def plot_learning_curve(model, X, y, cv=5):
    """
    Generate the learning curve for a model.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )

    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), label='Cross-validation score')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    return plt.gcf()

def plot_models_comparison():
    """
    Create enhanced comparison plots for all saved models
    """
    saved_models = st.session_state.get('saved_models', {})
    if not saved_models:
        return None
    
    # Extract metrics for all models
    model_metrics = []
    for key, data in saved_models.items():
        metrics = data['metrics'].copy()
        # Simplify model name by removing timestamp
        metrics['Model'] = data['name']
        model_metrics.append(metrics)
    
    df_metrics = pd.DataFrame(model_metrics)
    
    # Create multiple comparison plots
    plots = []
    
    # 1. Overall metrics comparison
    plt.figure(figsize=(12, 6))
    metrics_to_plot = [col for col in df_metrics.columns if col != 'Model']
    df_plot = df_metrics.melt(id_vars=['Model'], value_vars=metrics_to_plot, 
                             var_name='Metric', value_name='Value')
    sns.barplot(data=df_plot, x='Model', y='Value', hue='Metric')
    plt.xticks(rotation=45)
    plt.title('Model Performance Comparison - All Metrics')
    plt.tight_layout()
    plots.append(plt.gcf())
    plt.close()
    
    
    
    return plots, df_metrics


# Main Evaluation Page
def evaluation_page():
    # Custom CSS
    st.markdown("""
        <style>
            /* General Page Styling */
            body {
                background-color: #DCE4C9;
                font-family: 'Arial', sans-serif;
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
        </style>
    """, unsafe_allow_html=True)
    #st.title("Model Evaluation Dashboard")
    st.markdown('<div class="navigation-bar"><h2>🧭 Model Evaluation</h2></div>', unsafe_allow_html=True)

    if 'trained_model' not in st.session_state:
        st.warning("No trained model found. Please train a model first.")
        return

    # Ensure 'problem_type' is defined in session state
    if 'problem_type' not in st.session_state:
        st.error("The 'problem_type' variable is not defined. Please define the problem type before proceeding.")
        return

    # Load session state variables
    model = st.session_state['trained_model']
    problem_type = st.session_state['problem_type']

    # Create directory for results
    result_dir = "evaluation_results"
    os.makedirs(result_dir, exist_ok=True)

    # Call appropriate evaluation function
    if problem_type == "Supervised":
        evaluate_supervised_model()
    elif problem_type == "CNN":
        evaluate_cnn_model()
    else:
        evaluate_clustering_model()

    
import os
import zipfile
import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from keras.preprocessing import image

def evaluate_cnn_model():
    st.header("CNN Evaluation")
    
    # Retrieve the trained model and history from session_state
    model = st.session_state.get('trained_model')
    history = st.session_state.get('training_history')
    
    if model is None or history is None:
        st.error("No trained model found. Please train a model first.")
        return
    
    # Display training history
    st.subheader("Training History")
    history_fig = plot_training_history(history)
    st.pyplot(history_fig)
    
    # Prediction Interface
    #st.subheader("🔮 Make a Prediction")
    import numpy as np
    from tensorflow.keras.utils import load_img, img_to_array
    import matplotlib.pyplot as plt

    # Function to preprocess uploaded image
    def preprocess_image(uploaded_file, target_size=(28, 28)):
        # Load and preprocess the image
        img = load_img(uploaded_file, target_size=target_size, color_mode="grayscale")
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        img_array = img_array.reshape(1, *target_size, 1)  # Reshape to match model input
        return img_array

    #st.title("Upload an Image for Prediction")

    # File uploader widget
    """uploaded_file = st.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        # Preprocess the image
        preprocessed_image = preprocess_image(uploaded_file)
        
        # Make prediction using the trained model
        if 'trained_model' in st.session_state:
            model = st.session_state['trained_model']
            prediction = model.predict(preprocessed_image)
            predicted_class = np.argmax(prediction, axis=1)[0]
            
            st.success(f"Predicted Class: {predicted_class}")
        else:
            st.error("No trained model found. Please train the model first!")"""

    
    # Upload image for prediction
    #uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    """if uploaded_file is not None:
        try:
            # Display the uploaded image
            uploaded_image = Image.open(uploaded_file)
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
            
            # Preprocess the image
            processed_image = preprocess_uploaded_image(uploaded_image)
            
            # Display preprocessed image (remove batch and channel dimensions for display)
            st.image(processed_image.squeeze(), caption="Preprocessed Image (28x28)", use_container_width=True)
            
            # Add debug information
            st.write(f"Processed image shape: {processed_image.shape}")
            
            # Predict the class
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = prediction[0][predicted_class] * 100
            
            # Display the prediction result
            st.success(f"Predicted Digit: {predicted_class}")
            st.write(f"Confidence: {confidence:.2f}%")
            
            # Display probability distribution
            st.write("Probability Distribution:")
            probabilities = prediction[0] * 100
            prob_df = pd.DataFrame({
                'Digit': range(10),
                'Probability (%)': probabilities
            })
            st.bar_chart(prob_df.set_index('Digit'))
        
        except Exception as e:
            st.error(f"Error processing the image: {str(e)}")
            st.write("Full error message:", str(e))"""
    

    """# Model Evaluation Metrics
    st.subheader("📊 Model Evaluation Metrics")
    
    if 'test_data' not in st.session_state:
        st.error("Test data not available in session_state.")
        return
    
    X_test, y_test = st.session_state.get('test_data', (None, None))
    
    if X_test is None or y_test is None:
        st.error("Test data is not properly loaded.")
        return
    
    try:
        # Ensure X_test has the correct shape (n_samples, 28, 28, 1)
        if len(X_test.shape) == 3:
            X_test = np.expand_dims(X_test, axis=-1)
        
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate and display evaluation metrics
        accuracy = np.mean(y_pred_classes == y_test) * 100
        st.write(f"Model Accuracy: {accuracy:.2f}%")
        
        # Add confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        st.pyplot(plt.gcf())
        
    except Exception as e:
        st.error(f"Error evaluating the model: {str(e)}")
        st.write("Full error message:", str(e))"""

def preprocess_uploaded_image(image):
    
    # Convert to grayscale if the image is not already
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to 28x28 pixels
    image = image.resize((28, 28))
    
    # Convert to numpy array
    img_array = np.array(image, dtype='float32')
    
    # Normalize pixel values to [0, 1]
    img_array = img_array / 255.0
    
    # Reshape to (1, 28, 28, 1) - adding batch and channel dimensions
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array


def evaluate_clustering_model():
    st.header("Clustering Evaluation")
    
    # Get data and model from session state
    model = st.session_state['trained_model']
    X_scaled = st.session_state['X_train']  # Using scaled data
    labels = st.session_state['cluster_labels']
    model_name = st.session_state['current_model_name']
    
    # Display basic clustering information
    n_clusters = len(np.unique(labels[labels != -1]))
    st.write(f"Number of clusters found: {n_clusters}")
    
    if model_name == "DBSCAN":
        n_noise = np.sum(labels == -1)
        st.write(f"Number of noise points: {n_noise} ({n_noise/len(labels)*100:.2f}%)")
    
    # Clustering metrics
    st.subheader("Clustering Metrics")
    
    metrics_col1, metrics_col2 = st.columns(2)
    
    # Define the metrics dictionary
    metrics = {}

    with metrics_col1:
        if n_clusters > 1:  # Silhouette score requires at least 2 clusters
            try:
                sil_score = silhouette_score(X_scaled, labels)
                metrics['Silhouette Score'] = sil_score
                st.metric("Silhouette Score", f"{sil_score:.3f}")
            except Exception as e:
                st.write("Could not calculate Silhouette Score")
            
            try:
                ch_score = calinski_harabasz_score(X_scaled, labels)
                metrics['Calinski-Harabasz Score'] = ch_score
                st.metric("Calinski-Harabasz Score", f"{ch_score:.3f}")
            except Exception as e:
                st.write("Could not calculate Calinski-Harabasz Score")
    
    with metrics_col2:
        if n_clusters > 1:
            try:
                db_score = davies_bouldin_score(X_scaled, labels)
                metrics['Davies-Bouldin Score'] = db_score
                st.metric("Davies-Bouldin Score", f"{db_score:.3f}")
            except Exception as e:
                st.write("Could not calculate Davies-Bouldin Score")
    
    # Visualization section
    st.subheader("Clustering Visualization")
    
    # 2D visualization of clusters
    st.write("2D Visualization (using PCA if dimensions > 2)")
    centers = getattr(model, 'cluster_centers_', None) if model_name == "K-Means" else None
    cluster_viz = plot_clustering_2d(X_scaled, labels, centers)
    st.pyplot(cluster_viz)
    plt.close()
    
    # Model-specific visualizations
    if model_name == "K-Means":
        st.subheader("K-Means Specific Analysis")
        
    """  # Elbow curve
        if st.checkbox("Show Elbow Curve"):
            k_range = range(1, min(11, len(X_scaled)))
            elbow_fig = plot_elbow_curve(X_scaled, k_range)
            st.pyplot(elbow_fig)
            plt.close()"""
    
    # Silhouette analysis
    if st.checkbox("Show Silhouette Analysis") and n_clusters > 1:
        silhouette_fig = plot_silhouette(X_scaled, labels)
        if silhouette_fig:
            st.pyplot(silhouette_fig)
            plt.close()
    
    # Cluster distribution
    st.subheader("Cluster Distribution")
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    fig = plt.figure(figsize=(10, 6))
    cluster_counts.plot(kind='bar')
    plt.title("Number of Samples per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Samples")
    st.pyplot(fig)
    plt.close()
    
    # Feature analysis per cluster
    st.subheader("Feature Analysis per Cluster")
    feature_names = st.session_state['feature_names']
    df_features = pd.DataFrame(X_scaled, columns=feature_names)
    df_features['Cluster'] = labels
    
    # Allow user to select features for analysis
    selected_features = st.multiselect(
        "Select features to analyze",
        feature_names,
        default=feature_names[:2] if len(feature_names) > 1 else feature_names
    )
    
    if selected_features:
        fig = plt.figure(figsize=(12, 6))
        sns.boxplot(data=df_features, x='Cluster', y=selected_features[0])
        plt.title(f"{selected_features[0]} Distribution by Cluster")
        st.pyplot(fig)
        plt.close()
    
    # Define result directory
    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)
    
    # Save current model and results
    if st.button("Save Model Results"):
        save_model_results(model, metrics, model_name)  # Call the function without using the returned model_key
        st.success(f"Model results saved successfully for model: {model_name}")
    
    # Navigation options
    st.header("Next Steps")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'saved_models' in st.session_state and len(st.session_state['saved_models']) > 1:
            
            if st.button("Compare All Models"):
                st.header("Model Comparison")
                
                comparison_container = st.container()
                with comparison_container:
                    # Get comparison plots and metrics
                    plots, comparison_df = plot_models_comparison()
                    
                    # Display metrics table with centered alignment
                    st.subheader("Metrics Comparison")
                    styled_df = comparison_df.round(4).style.set_properties(**{
                        'text-align': 'center'
                    })
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Display all comparison plots
                    st.subheader("Visual Comparisons")
                    for i, plot in enumerate(plots):
                        st.pyplot(plot)
                        plt.close()

                   # Identify the best model for clustering
                if 'Silhouette Score' in comparison_df.columns:
                    primary_metric = 'Silhouette Score'
                    best_model_idx = comparison_df[primary_metric].idxmax()
                    best_model_name = comparison_df.loc[best_model_idx, 'Model']
                    st.success(f"🏆 Best performing clustering model: {best_model_name}")
                else:
                    st.warning("Silhouette Score not available for comparison.")
        else:
            st.info("Train at least one more model to enable comparison")
        
    with col2:
       if st.button("Train another model", key="model", use_container_width=True):
        st.session_state.current_page = "ML Modeling"
        st.rerun()
    
    add_clustering_prediction_section()
    
    # Create ZIP file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for root, _, files in os.walk(result_dir):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=file)

    # Provide ZIP file for download
    st.header("Download Results")
    st.download_button(
        label="Download Results as ZIP",
        data=zip_buffer.getvalue(),
        file_name="evaluation_results.zip",
        mime="application/zip"
    )

def evaluate_supervised_model():
    if 'trained_model' not in st.session_state:
        st.warning("No trained model found. Please train a model first.")
        return

    # Load session state variables
    model = st.session_state['trained_model']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']
    is_classification = st.session_state['is_classification']
    feature_names = st.session_state['feature_names']
    
    # Get model name from session state
    model_name = st.session_state.get('current_model_name', 'Unknown Model')

    # Make predictions
    y_pred = model.predict(X_test)

    # Prepare directory to store results
    result_dir = "evaluation_results"
    os.makedirs(result_dir, exist_ok=True)

    # Model Performance Metrics
    st.header("Model Performance :")
    metrics = {}

    if is_classification:
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'F1 Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }

        col1, col2 = st.columns(2)
        with col1:
            for metric, value in metrics.items():
                st.metric(metric, f"{value:.4f}")

        with col2:
            conf_matrix = plot_confusion_matrix(y_test, y_pred)
            st.pyplot(conf_matrix)
            save_plot_as_png(conf_matrix, os.path.join(result_dir, "confusion_matrix.png"))
            plt.close()
    else:  # Regression
        metrics = {
            'R² Score': r2_score(y_test, y_pred),
            'Mean Squared Error': mean_squared_error(y_test, y_pred),
            'Mean Absolute Error': mean_absolute_error(y_test, y_pred),
            'Root Mean Squared Error': np.sqrt(mean_squared_error(y_test, y_pred))
        }

        col1, col2 = st.columns(2)
        with col1:
            for metric, value in metrics.items():
                st.metric(metric, f"{value:.4f}")

        with col2:
            scatter_plot = plot_regression_scatter(y_test, y_pred)
            st.pyplot(scatter_plot)
            save_plot_as_png(scatter_plot, os.path.join(result_dir, "regression_scatter.png"))
            plt.close()

    # Learning Curve
    st.header("Learning Curve Analysis :")
    if st.checkbox("Show Learning Curve"):
        with st.spinner("Generating learning curve..."):
            X_train = st.session_state.get('X_train', None)
            y_train = st.session_state.get('y_train', None)

            if X_train is None or y_train is None:
                X_train, _, y_train, _ = train_test_split(
                    X_test, y_test, test_size=0.2, random_state=42
                )

            learning_curve_plot = plot_learning_curve(model, X_train, y_train)
            st.pyplot(learning_curve_plot)
            save_plot_as_png(learning_curve_plot, os.path.join(result_dir, "learning_curve.png"))
            plt.close()

    # Feature Importance
    '''st.header("Feature Analysis")
    importance_plot = plot_feature_importance_lime(model, feature_names, X_train)
    if importance_plot:
        st.pyplot(importance_plot)
        save_plot_as_png(importance_plot, os.path.join(result_dir, "feature_importance.png"))
        plt.close()
    else:
        st.info("Feature importance visualization not available for this model type")'''

    # Save metrics to a text file
    metrics_file = os.path.join(result_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
            
    
    # Save current model and results
    if st.button("Save Model Results"):
        save_model_results(model, metrics, model_name)  # Call the function without using the returned model_key
        st.success(f"Model results saved successfully for model: {model_name}")
    # Navigation options
    st.header("Next Steps :")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'saved_models' in st.session_state and len(st.session_state['saved_models']) > 1:
            
            if st.button("Compare All Models"):
                st.header("Model Comparison :")
                
                
# Center the comparison section
                comparison_container = st.container()
                with comparison_container:
                    # Get comparison plots and metrics
                    plots, comparison_df = plot_models_comparison()
                    
                    # Display metrics table with centered alignment
                    st.subheader("Metrics Comparison")
                    styled_df = comparison_df.round(4).style.set_properties(**{
                        'text-align': 'center'
                    })
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Display all comparison plots
                    st.subheader("Visual Comparisons")
                    for i, plot in enumerate(plots):
                        st.pyplot(plot)
                        plt.close()
                    
                    # Identify best model
                    primary_metric = 'Accuracy' if is_classification else 'R² Score'
                    best_model_idx = comparison_df[primary_metric].idxmax()
                    best_model_name = comparison_df.loc[best_model_idx, 'Model']
                    
                    st.success(f"🏆 Best performing model: {best_model_name}")
        else:
            st.info("Train at least one more model to enable comparison")
            
    with col2:
        if st.button("Train another model", key="model", use_container_width=True):
            st.session_state.current_page = "ML Modeling"
            st.rerun()


    add_prediction_section()

    # Create ZIP file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for root, _, files in os.walk(result_dir):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=file)

    # Provide ZIP file for download
    st.header("Download Results :")
    st.download_button(
        label="Download Results as ZIP",
        data=zip_buffer.getvalue(),
        file_name="evaluation_results.zip",
        mime="application/zip"
    )

    # Assuming model is your trained machine learning model
    st.header("Save Model")

    # Define the local path where the model will be saved
    local_path = "model.pkl"

    # Create a button
    if st.button("Save Model"):
        # Save the model when the button is pressed
        joblib.dump(model, local_path)
        st.success(f"Model saved to {local_path}")
    
# Run the Streamlit App
if __name__ == "__main__":
    evaluation_page()
