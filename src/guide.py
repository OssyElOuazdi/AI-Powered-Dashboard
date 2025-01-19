import streamlit as st

def guide_page():
    
    """
    Guide page for the App.
    Provides a tutorial and user guide.
    """
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
            .sidebar-button:hover {
                background-color: #E07B39;
                color: #FFFFFF;
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="navigation-bar"><h2>ⓘ Guide Utilisateur</h2></div>', unsafe_allow_html=True)

    """
    Guide page for the App.
    Provides a tutorial and user guide.
    """
    st.header("Watch the video tutorials :")

    # Local file paths for the videos
    supervised_video_path = "uploads\supervised.mp4"
    unsupervised_video_path = "uploads\supervised.mp4"
    deep_learning_video_path = "uploads\supervised.mp4"

    # Thumbnails for the videos (ensure to replace the image paths with the actual paths)
    supervised_thumbnail = "uploads\supervised.png"
    unsupervised_thumbnail = r"uploads\unsupervised.png"
    deep_learning_thumbnail = "uploads\deep.png"

    # Display video tutorials with thumbnails
    st.subheader("1. Supervised Learning - Classification Problem:")
    st.image(supervised_thumbnail, caption="Supervised Learning - Classification", use_container_width=True)
    if st.button("Watch Video Tutorial 1"):
        try:
            st.video(supervised_video_path, format="mp4")
        except Exception as e:
            st.error(f"Error while loading the video: {str(e)}")

    st.subheader("2. Supervised Learning - Regression Problem:")
    st.image(unsupervised_thumbnail, caption="Supervised Learning - Regression", use_container_width=True)
    if st.button("Watch Video Tutorial 2"):
        try:
            st.video(unsupervised_video_path, format="mp4")
        except Exception as e:
            st.error(f"Error while loading the video: {str(e)}")

    st.subheader("3. Unsupervised Learning :")
    st.image(deep_learning_thumbnail, caption="Unsupervised Learning", use_container_width=True)
    if st.button("Watch Video Tutorial 3"):
        try:
            st.video(deep_learning_video_path, format="mp4",)
        except Exception as e:
            st.error(f"Error while loading the video: {str(e)}")

    st.write("""
    Once you have watched the video tutorials, you can start exploring the other features of the application.
    """)


    
    st.write("""
    

### Overview
This application provides an end-to-end solution for data analysis and machine learning modeling. It includes functionalities for:

1. Importing and visualizing datasets.
2. Preprocessing data with advanced techniques.
3. Training and optimizing machine learning models.
4. Evaluating and comparing model performance.

### Navigation Instructions
The application is divided into the following sections, accessible through the sidebar:

1. **Data Handling**
2. **Data Visualization**
3. **Data Preparation**
4. **Machine Learning Modeling**
5. **Model Evaluation**

### Detailed Page Descriptions

#### 1. **Data Handling**
This section allows users to import, create, and manage datasets.

**Features:**
- **Import Methods**: Choose from Local File, Example Dataset, or Manual Creation.
  - Supported formats: CSV, Excel, JSON.
- **Reset Dataset**: Clears the currently loaded dataset.
- **Display Dataset**: View the uploaded data and its basic statistics.

**How to Use:**
1. Select an import method.
2. If importing locally, upload the file.
3. For example datasets, select a dataset like Iris or Titanic.
4. Use manual creation to specify the number of rows and columns.



#### 2. **Data Visualization**
Explore and understand the dataset visually.

**Features:**
- **Target Column Selection**: Choose the dependent variable.
- **Problem Type Detection**: Automatically detects whether it is a regression or classification problem.
- **Regression Visualizations**: Scatter plots, correlation heatmaps, and boxplots.
- **Classification Visualizations**: Class distribution, pair plots, and violin plots.

**How to Use:**
1. Upload a dataset in the Data Handling section.
2. Select a target column.
3. Use the visualizations provided based on the problem type.


#### 3. **Data Preparation**
Prepare your dataset for modeling with various preprocessing tools.

**Features:**
- **Basic Operations**: Remove duplicates, drop unused columns.
- **Handle Missing Values**: Impute missing data for numerical and categorical columns.
- **Feature Engineering**: Encoding, normalization, and outlier handling.
- **Advanced Operations**: Apply SMOTE for balancing, PCA for dimensionality reduction, and feature selection methods.

**How to Use:**
1. Use tabs to navigate through operations (Basic, Missing Values, Feature Engineering, Advanced).
2. Perform actions like imputation, encoding, and outlier handling.
3. Save processed data for the modeling step.


#### 4. **Machine Learning Modeling**
Train machine learning models with customizable settings.

**Features:**
- **Model Selection**: Choose from models like Logistic Regression, SVM, Random Forest, etc.
- **Configure Training Settings**: Set test data size and cross-validation folds.
- **Optimization**: Automatically tune hyperparameters with Bayesian optimization.

**How to Use:**
1. Select the target variable.
2. Choose a model and configure training settings.
3. Enable optimization for hyperparameter tuning if desired.
4. Train the model and proceed to evaluation.



#### 5. **Model Evaluation**
Analyze the performance of trained models.

**Features:**
- **Metrics Display**: View metrics like accuracy, precision, recall, R² score, etc.
- **Visualizations**: Confusion matrix for classification, scatter plot for regression.
- **Learning Curve Analysis**: Evaluate model training behavior.
- **Comparison**: Compare metrics of multiple models.
- **Save and Download**: Save results and download evaluation reports.

**How to Use:**
1. Train a model in the Machine Learning Modeling section.
2. View performance metrics and visualizations.
3. Save results for comparison or download them as a ZIP file.



### Examples and Tips

**Example Workflow:**
1. **Data Handling**: Import the Titanic dataset.
2. **Visualization**: Identify that "Survived" is the target column and a classification problem.
3. **Preparation**: Encode categorical variables and handle missing values.
4. **Modeling**: Train a Random Forest classifier.
5. **Evaluation**: Analyze accuracy and compare with other models like SVM.

**Tips:**
- Ensure data is well-prepared before training models to avoid poor performance.
- Use feature selection to improve model interpretability.
- Save your processed data and models for reproducibility.

This guide covers the core functionality of the application and should help you navigate and utilize all its features effectively. If you encounter any problems or issues, feel free to contact us at mlacademyapp@support.ma for assistance.

    """)
