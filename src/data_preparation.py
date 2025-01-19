import pandas as pd
import streamlit as st
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, mutual_info_regression

from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
#from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import matplotlib.pyplot as plt

import statsmodels.api as sm
import numpy as np 
from sklearn.feature_selection import VarianceThreshold

def prepare_mnist_data(df):
    """
    Prepare MNIST dataset specifically for deep learning tasks
    """
    st.markdown("### MNIST Data Preparation :")
    
    # Separate features and target
    pixel_columns = [col for col in df.columns if col.startswith('pixel_')]
    X = df[pixel_columns].values
    y = df['target'].values
    
    # Display original shape
    st.write("Original Data Shape:", X.shape)
    
    # Preprocessing options
    preprocessing_options = st.multiselect(
        "Select preprocessing steps:",
        ["Reshape Images", "Normalize Pixels", "One-Hot Encode Labels"],
        default=["Reshape Images", "Normalize Pixels", "One-Hot Encode Labels"]
    )
    
    # Initialize processed data
    X_processed = X.copy()
    y_processed = y.copy()
    
    if "Reshape Images" in preprocessing_options:
        # Get image dimensions
        img_height = df['image_height'].iloc[0]
        img_width = df['image_width'].iloc[0]
        
        # Reshape to 4D: (samples, height, width, channels)
        X_processed = X_processed.reshape(-1, img_height, img_width, 1)
        st.success(f"✅ Images reshaped to: {X_processed.shape}")
        
        # Visualize some examples
        st.write("### Sample Reshaped Images")
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            if idx < 5:
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(X_processed[idx, :, :, 0], cmap='gray')
                ax.axis('off')
                ax.set_title(f'Digit: {y[idx]}')
                col.pyplot(fig)
                plt.close(fig)
    
    if "Normalize Pixels" in preprocessing_options:
        # Normalize to [0, 1] range
        X_processed = X_processed.astype('float32') / 255.0
        st.success("✅ Pixel values normalized to [0, 1] range")
        
        # Show statistics
        st.write("### Normalized Data Statistics")
        st.write(f"Min value: {X_processed.min():.3f}")
        st.write(f"Max value: {X_processed.max():.3f}")
        st.write(f"Mean value: {X_processed.mean():.3f}")
    
    if "One-Hot Encode Labels" in preprocessing_options:
        # One-hot encode the labels
        n_classes = len(np.unique(y))
        y_processed = pd.get_dummies(y).values
        st.success(f"✅ Labels one-hot encoded to {n_classes} classes")
        
        # Show example of one-hot encoding
        st.write("### One-Hot Encoding Example")
        example_df = pd.DataFrame({
            'Original Label': y[:5],
            'One-Hot Encoded': [str(vec) for vec in y_processed[:5]]
        })
        st.table(example_df)
    
    # Save processed data
    if st.button("Save Processed MNIST Data"):
        st.session_state['X_processed'] = X_processed
        st.session_state['y_processed'] = y_processed
        st.session_state['is_mnist_processed'] = True
        
        # Save preprocessing info
        st.session_state['mnist_preprocessing_info'] = {
            'image_shape': X_processed.shape[1:],
            'n_classes': y_processed.shape[1] if "One-Hot Encode Labels" in preprocessing_options else 1,
            'preprocessing_steps': preprocessing_options
        }
        
        st.success("Processed MNIST data saved successfully!")
        
        # Display final shapes
        st.write("### Final Data Shapes")
        st.write(f"Features shape: {X_processed.shape}")
        st.write(f"Labels shape: {y_processed.shape}")


def data_preparation_page():
    """
    Data Preparation Page for the ML Exploration App.
    Uses the previously loaded dataset from session state.
    """
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

    # Check if data exists in session state
    if 'uploaded_data' not in st.session_state:
        st.warning("Please upload a dataset in the Import Data section first.")
        return
    

    # Load data from session state
    df = st.session_state['uploaded_data']
    
    # Page Title and Navigation Bar
    st.markdown('<div class="navigation-bar"><h2>⚙️ Prepare Data</h2></div>', unsafe_allow_html=True)
    #st.write("### Dataset Preview")
    #st.dataframe(df.head())
    # Check if it's MNIST data
    is_mnist = ('image_width' in df.columns and 'image_height' in df.columns and 
                'target' in df.columns and 'label_name' in df.columns)
    
    if is_mnist:
        prepare_mnist_data(df)
    else:
       

        st.markdown("###  ")


        # Card for Problem Type Selection
        st.markdown("###  Select the Type of Problem :")
        # Problem type selection
        problem_type = st.radio(
            "",
            ["Supervised", "Unsupervised"],
            key="problem_type_selection"
        )
        # Dynamically determine numeric and categorical columns
        numeric_columns = df.select_dtypes(include=['number']).columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns

        if problem_type == "Supervised":
            prepare_supervised_data(df, numeric_columns, categorical_columns)
        else:
            prepare_unsupervised_data(df, numeric_columns, categorical_columns)

def prepare_supervised_data(df, numeric_columns, categorical_columns):
    """Handle supervised learning visualizations"""

    # Select target column
    #target_column = st.selectbox("Select the target column", options=list(df.columns), index=len(df.columns) - 1)
    if 'target_column' not in st.session_state or st.session_state['target_column'] is None:
        st.warning("Target column not set. Please select the target column in the visualization page.")
    else:
        target_column = st.session_state['target_column']
        #st.write(f"Target column: {target_column}")
        st.markdown(
        f"""
        <div style="margin-top: 20px; font-size: 16px; color: #ffffff;margin-bottom: 20px;">
            Target column: <strong>{target_column}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
    processed_data = df.copy()

    # Tabs for preprocessing steps
    tabs = st.tabs([
        "Basic Operations",
        "Missing Values",
        "Manage Columns Values",
        "Feature Engineering",
        "Advanced Operations"
    ])
    

    # Basic Operations Tab
    with tabs[0]:
        st.markdown("### Basic Data Operations :")
        # Check if problem_type is set
        if 'problem_type' not in st.session_state or st.session_state['problem_type'] is None:
            st.error("Problem type not detected. Please set it in the visualization page.")
            st.stop()
        col1, col2 = st.columns(2)

        with col1:
            
            if st.checkbox("Remove Duplicates"):
                initial_rows = processed_data.shape[0]
                processed_data.drop_duplicates(inplace=True)
                st.info(f"Removed {initial_rows - processed_data.shape[0]} duplicate rows.")
                #st.write("### Dataset After Basic Operations")
            if st.checkbox("Drop Unused Columns"):
                unused_cols = st.multiselect("Select columns to drop", df.columns)
                processed_data.drop(columns=unused_cols, inplace=True)
        with col2:
            if st.checkbox("Drop Rows With Missing Values"):
                missing_rows = processed_data.isnull().sum().sum()
                if missing_rows > 0:
                    initial_rows = processed_data.shape[0]
                    processed_data.dropna(inplace=True)
                    st.info(f"Dropped {initial_rows - processed_data.shape[0]} rows with missing values.")
                else:
                    st.info("No missing values found in the dataset.")

    # Missing Values Tab
    
    with tabs[1]:
        st.markdown("### Handle Missing Values :")
        if st.checkbox("Handle Missing Values"):
            numerical_cols = processed_data.select_dtypes(include=['float64', 'int64']).columns
            categorical_cols = processed_data.select_dtypes(include=['object', 'category', 'bool']).columns

            col1, col2 = st.columns(2)

            with col1:
                st.write("Numerical Columns")
                strategy = st.selectbox("Strategy for numerical columns", ["Mean", "Median", "Mode", "Constant"])
                if strategy == "Constant":
                    constant = st.number_input("Enter constant value", value=0)
                    processed_data[numerical_cols] = processed_data[numerical_cols].fillna(constant)
                else:
                    for col in numerical_cols:
                        if strategy == "Mean":
                            processed_data[col] = processed_data[col].fillna(processed_data[col].mean())
                        elif strategy == "Median":
                            processed_data[col] = processed_data[col].fillna(processed_data[col].median())
                        elif strategy == "Mode":
                            processed_data[col] = processed_data[col].fillna(processed_data[col].mode().iloc[0])
            

            with col2:
                st.write("Categorical Columns")
                strategy = st.selectbox("Strategy for categorical columns", ["Mode", "Constant"])
                if strategy == "Constant":
                    constant = st.text_input("Enter constant value", "Unknown")
                    processed_data[categorical_cols] = processed_data[categorical_cols].fillna(constant)
                else:
                    for col in categorical_cols:
                        processed_data[col] = processed_data[col].fillna(processed_data[col].mode().iloc[0])
            st.write("### Dataset After Basic Operations")
            #st.dataframe(processed_data)
            st.markdown('<div class="data-preview">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="data-section">', unsafe_allow_html=True)
                st.write("### Dataset Preview")
                st.dataframe(df.head())
                st.markdown('</div>', unsafe_allow_html=True)

            
            with col2:
                st.markdown('<div class="data-section">', unsafe_allow_html=True)
                st.write("### Final Processed Data")
                st.dataframe(processed_data.head())
                #st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)  # End of data preview section


# Feature Engineering Tab

    with tabs[2]:
        st.markdown("### Manage Columns Values :")
        if st.checkbox("Replace Multiple Values with Multiple Replacements"):
            num_operations = st.number_input("How many columns do you want to modify?", min_value=1, step=1)

            replacement_operations = []  # List to store all operations

            for i in range(num_operations):
                st.write(f"### Replacement Operation {i + 1}")
                column_to_replace = st.selectbox(f"Select column for operation {i + 1}", processed_data.columns, key=f"col_{i}")
                
                mapping_input = st.text_area(f"Enter values and replacements (format: old1:new1, old2:new2) for operation {i + 1}", key=f"map_{i}")
                
                if column_to_replace and mapping_input:
                    # Parse the mapping input into a dictionary
                    mapping = {pair.split(":")[0].strip(): pair.split(":")[1].strip() for pair in mapping_input.split(",")}
                    replacement_operations.append((column_to_replace, mapping))

            if st.button("Apply Replacements"):
                if replacement_operations:
                    for col, mapping in replacement_operations:
                        processed_data[col] = processed_data[col].replace(mapping, regex=True)
                        st.info(f"Applied replacements in column '{col}': {mapping}")
                else:
                    st.warning("Please define at least one replacement operation before applying.")


            st.markdown('<div class="data-preview">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="data-section">', unsafe_allow_html=True)
                st.write("### Dataset Preview")
                st.dataframe(df.head())
                st.markdown('</div>', unsafe_allow_html=True)

            
            with col2:
                st.markdown('<div class="data-section">', unsafe_allow_html=True)
                st.write("### Final Processed Data")
                st.dataframe(processed_data.head())
                #st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)  # End of data preview section

    with tabs[3]:
        st.markdown("### Feature Engineering :")
        cool1, cool2 = st.columns(2)

        with cool1:
            if st.checkbox("Encode Categorical Variables"):
                categorical_cols = processed_data.select_dtypes(include=['object', 'category', 'bool']).columns
                method = st.selectbox("Encoding method", ["Label Encoding", "One-Hot Encoding"])
                if method == "Label Encoding":
                    st.info("Categorical variables encoded using Label Encoding.")
                    for col in categorical_cols:
                        le = LabelEncoder()
                        processed_data[col] = le.fit_transform(processed_data[col])
                        st.write(f"### Encoding for {col}:")
                        encoding_mapping = pd.DataFrame({
                            "Original Value": le.classes_,
                            "Encoded Value": le.transform(le.classes_)
                        })
                        st.table(encoding_mapping)
                else:
                    processed_data = pd.get_dummies(processed_data, drop_first=True)
                    st.info("Categorical variables encoded using One-Hot Encoding.")

            if st.checkbox("Normalize Columns"):
                numerical_cols = [col for col in processed_data.select_dtypes(include=['float64', 'int64']).columns if col != target_column]
                if numerical_cols:
                    scaler = MinMaxScaler()
                    choice = st.radio("Choose normalization type", ["All Columns", "Selected Columns"])
                    if choice == "All Columns":
                        processed_data[numerical_cols] = scaler.fit_transform(processed_data[numerical_cols])
                        st.info("All numerical columns (except target) normalized.")
                    elif choice == "Selected Columns":
                        selected_cols = st.multiselect("Select columns to normalize", numerical_cols)
                        if selected_cols:
                            processed_data[selected_cols] = scaler.fit_transform(processed_data[selected_cols])
                            st.info(f"Normalized columns: {', '.join(selected_cols)}")


            if st.checkbox("Determine The Type"):
                st.markdown("### Column Types Inferred :")

                # Identify initial types
                numeric_cols = processed_data.select_dtypes(include=['number']).columns.tolist()
                categorical_cols = processed_data.select_dtypes(include=['object', 'category']).columns.tolist()
                datetime_cols = processed_data.select_dtypes(include=['datetime']).columns.tolist()

                # Display the inferred types
                st.write(f"*Numeric Columns:* {', '.join(numeric_cols) if numeric_cols else 'None'}")
                st.write(f"*Categorical Columns:* {', '.join(categorical_cols) if categorical_cols else 'None'}")
                st.write(f"*Datetime Columns:* {', '.join(datetime_cols) if datetime_cols else 'None'}")

                # Option to change data type
                st.markdown("### Change Column Data Types :")
                column_to_change = st.selectbox("Select Column to Change Type", processed_data.columns)
                new_type = st.radio("Select New Data Type", ["Numeric", "Categorical", "Datetime"])

                if st.button("Change Type"):
                    try:
                        if new_type == "Numeric":
                            processed_data[column_to_change] = pd.to_numeric(processed_data[column_to_change], errors='coerce')
                        elif new_type == "Categorical":
                            processed_data[column_to_change] = processed_data[column_to_change].astype('category')
                        elif new_type == "Datetime":
                            processed_data[column_to_change] = pd.to_datetime(
                                processed_data[column_to_change], errors='coerce')
                            
                        st.success(f"Successfully changed type of '{column_to_change}' to {new_type}.")
                    except Exception as e:
                        st.error(f"Error changing type: {e}")
                if st.checkbox("Convert Dates to Sequential Numbers"):
                    # Detect potential date columns
                    date_cols = [col for col in processed_data.columns if "date" in col.lower() or "month" in col.lower()]
                    if date_cols:
                        selected_date_col = st.selectbox("Select the date column to convert", date_cols)
                        if selected_date_col:
                            # Convert to datetime with error handling
                            processed_data[selected_date_col] = pd.to_datetime(
                                processed_data[selected_date_col],
                                format='%y-%b',  # Update format based on the input
                                errors='coerce'
                            )
                            invalid_dates = processed_data[selected_date_col].isna().sum()
                            if invalid_dates > 0:
                                st.warning(f"{invalid_dates} invalid date entries found and replaced with NaT.")
                                fix_option = st.radio(
                                    "How would you like to handle invalid dates?",
                                    ["Remove rows with NaT", "Replace NaT with the earliest valid date"]
                                )
                                if fix_option == "Remove rows with NaT":
                                    processed_data.dropna(subset=[selected_date_col], inplace=True)
                                    st.info("Rows with NaT have been removed.")
                                elif fix_option == "Replace NaT with the earliest valid date":
                                    earliest_date = processed_data[selected_date_col].min()
                                    processed_data[selected_date_col].fillna(earliest_date, inplace=True)
                                    st.info(f"NaT values replaced with the earliest valid date: {earliest_date}.")

                            # Convert valid dates to sequential numbers
                            processed_data['Time_Step'] = processed_data[selected_date_col].rank().astype(int)

                            # Drop the original date column
                            processed_data.drop(columns=[selected_date_col], inplace=True)
                            st.info(f"Original date column {selected_date_col} has been removed and replaced with Time_Step.")

                            
                    else:
                        st.warning("No potential date columns detected in the dataset.")

        with cool2:
            if st.checkbox("Handle Outliers"):
                numerical_cols = processed_data.select_dtypes(include=['float64', 'int64']).columns
                z_scores = (processed_data[numerical_cols] - processed_data[numerical_cols].mean()) / processed_data[numerical_cols].std()
                processed_data = processed_data[(z_scores < 3).all(axis=1)]
                st.info("Outliers removed.")

            if st.checkbox("Standardize Columns"):
                if 'scaler' in locals():
                    st.warning("The dataset is already normalized. No need to standardize.")
                else:
                    choice = st.radio("Choose standardization type", ["All Columns", "Selected Columns"])
                    numerical_cols = [col for col in processed_data.select_dtypes(include=['float64', 'int64']).columns if col != target_column]

                    robust_scaler = RobustScaler()
                    if choice == "All Columns":
                        processed_data[numerical_cols] = robust_scaler.fit_transform(processed_data[numerical_cols])
                        st.info("All numerical columns standardized.")
                    elif choice == "Selected Columns":
                        selected_cols = st.multiselect("Select columns to standardize", numerical_cols)
                        if selected_cols:
                            processed_data[selected_cols] = robust_scaler.fit_transform(processed_data[selected_cols])
                            st.info(f"Standardized columns: {', '.join(selected_cols)}")

        st.markdown('<div class="data-preview">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="data-section">', unsafe_allow_html=True)
            st.write("### Dataset Preview :")
            st.dataframe(df.head())
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="data-section">', unsafe_allow_html=True)
            st.write("### Final Processed Data :")
            st.dataframe(processed_data.head())

        st.markdown('</div>', unsafe_allow_html=True)  # End of data preview section

# Advanced Operations Tab
    with tabs[4]:
        st.markdown("### Advanced Operations :")
        
        # Split processed_data into features and target
        X_features = processed_data.drop(columns=[target_column])  # Default: Exclude target column
        y_target = processed_data[target_column]  # Target column

        # Problem type detection
        problem_type = st.session_state.get("problem_type", None)
        if not problem_type:
            st.warning("Problem type not detected. Please set it in the visualization or preprocessing step.")
            return

        # Columns for layout
        ao_col1, ao_col2 = st.columns(2)

        with ao_col1:
            #SMOTE Checkbox (only for classification problems)
            if problem_type == "Classification":
                if st.checkbox("Balance Dataset (SMOTE)"):
                    smote = SMOTE()
                    X_res, y_res = smote.fit_resample(X_features, y_target)

                    # Update features and target with resampled data
                    X_features = pd.DataFrame(X_res, columns=X_features.columns)
                    y_target = pd.Series(y_res, name=target_column)

                    st.write(f"Original Data shape: {processed_data.shape}")
                    st.write(f"Resampled Data shape: {X_features.shape}")
                    st.info("Dataset balanced using SMOTE.")
            


            # PCA Option
            if st.checkbox("Apply PCA"):
                pca = PCA()
                pca.fit(X_features)

                explained_variance_ratio = pca.explained_variance_ratio_.cumsum()
                n_components = next(i for i, var in enumerate(explained_variance_ratio) if var >= 0.95) + 1
                st.write(f"Number of components selected with 95% variance explained: {n_components}")

                # Transform data with selected components
                pca = PCA(n_components=n_components)
                X_features = pd.DataFrame(pca.fit_transform(X_features), columns=[f"PC{i+1}" for i in range(n_components)])

                st.info("PCA applied to the dataset.")

            # Feature selection
            if st.checkbox("Apply Feature Selection"):
                st.markdown("### Feature Selection :")

                # Select a single feature selection method
                methods = []
                if problem_type == "Classification":
                    methods = ["SelectKBest", "RFE", "Mutual Information", "Random Forest"]
                elif problem_type == "Regression":
                    methods = ["SelectKBest", "LassoCV", "Pearson Method", "Backward Elimination", "Forward Selection"]

                selected_method = st.selectbox("Select a Feature Selection Method", methods)

                # Perform feature selection using the selected method
                selected_features = set()
                X = processed_data.drop(columns=[target_column])
                y = processed_data[target_column]

                if selected_method == "SelectKBest":
                    score_func = f_classif if problem_type == "Classification" else f_regression
                    selector = SelectKBest(score_func=score_func, k=5)
                    selector.fit(X, y)
                    selected_features = set(X.columns[selector.get_support()])

                elif selected_method == "RFE":
                    model = RandomForestClassifier() if problem_type == "Classification" else RandomForestRegressor()
                    rfe = RFE(model, n_features_to_select=5)
                    rfe.fit(X, y)
                    selected_features = set(X.columns[rfe.support_])

                elif selected_method == "Random Forest":
                    model = RandomForestClassifier() if problem_type == "Classification" else RandomForestRegressor()
                    model.fit(X, y)
                    selected_features = set(X.columns[model.feature_importances_.argsort()[-5:]])

                elif selected_method == "Mutual Information":
                    mi_func = mutual_info_classif if problem_type == "Classification" else mutual_info_regression
                    mi = mi_func(X, y)
                    selected_features = set(X.columns[mi.argsort()[-5:]])

                elif selected_method == "LassoCV" and problem_type == "Regression":
                    lasso = LassoCV().fit(X, y)
                    selected_features = set(X.columns[lasso.coef_ != 0])

                elif selected_method == "Backward Elimination" and problem_type == "Regression":
                    X_with_const = sm.add_constant(X)
                    model = sm.OLS(y, X_with_const).fit()
                    p_values = model.pvalues.sort_values()
                    selected_features = set(p_values[p_values < 0.05].index)
                    if "const" in selected_features:
                        selected_features.remove("const")

                elif selected_method == "Forward Selection":
                    model = RandomForestClassifier() if problem_type == "Classification" else RandomForestRegressor()
                    sfs = SFS(model, k_features=5, forward=True, scoring="accuracy" if problem_type == "Classification" else "r2")
                    sfs.fit(X, y)
                    selected_features = set(X.columns[list(sfs.k_feature_idx_)])

                # Display selected features for the single method
                st.markdown("### Selected Features :")
                st.write(f"**Selected Features ({selected_method}):** {', '.join(selected_features)}")

                # Option to combine methods using union or intersection
                apply_combination = st.checkbox("Do you want to combine multiple selection methods?")

                if apply_combination:
                    combination_methods = st.multiselect(
                        "Select Methods to Combine",
                        [m for m in methods if m != selected_method]
                    )
                    combination_strategy = st.radio("Combination Strategy", ["Union", "Intersection"])

                    selected_features_union = set()
                    selected_features_intersection = None

                    for method in combination_methods:
                        current_method_features = set()

                        if method == "SelectKBest":
                            score_func = f_classif if problem_type == "Classification" else f_regression
                            selector = SelectKBest(score_func=score_func, k=5)
                            selector.fit(X, y)
                            current_method_features = set(X.columns[selector.get_support()])

                        elif method == "RFE":
                            model = RandomForestClassifier() if problem_type == "Classification" else RandomForestRegressor()
                            rfe = RFE(model, n_features_to_select=5)
                            rfe.fit(X, y)
                            current_method_features = set(X.columns[rfe.support_])

                        elif method == "Random Forest":
                            model = RandomForestClassifier() if problem_type == "Classification" else RandomForestRegressor()
                            model.fit(X, y)
                            current_method_features = set(X.columns[model.feature_importances_.argsort()[-5:]])

                        elif method == "Mutual Information":
                            mi_func = mutual_info_classif if problem_type == "Classification" else mutual_info_regression
                            mi = mi_func(X, y)
                            current_method_features = set(X.columns[mi.argsort()[-5:]])

                        elif method == "LassoCV" and problem_type == "Regression":
                            lasso = LassoCV().fit(X, y)
                            current_method_features = set(X.columns[lasso.coef_ != 0])

                        elif method == "Backward Elimination" and problem_type == "Regression":
                            X_with_const = sm.add_constant(X)
                            model = sm.OLS(y, X_with_const).fit()
                            p_values = model.pvalues.sort_values()
                            current_method_features = set(p_values[p_values < 0.05].index)
                            if "const" in current_method_features:
                                current_method_features.remove("const")

                        elif method == "Forward Selection":
                            model = RandomForestClassifier() if problem_type == "Classification" else RandomForestRegressor()
                            sfs = SFS(model, k_features=5, forward=True, scoring="accuracy" if problem_type == "Classification" else "r2")
                            sfs.fit(X, y)
                            current_method_features = set(X.columns[list(sfs.k_feature_idx_)])

                        # Update union and intersection
                        selected_features_union.update(current_method_features)
                        if selected_features_intersection is None:
                            selected_features_intersection = current_method_features
                        else:
                            selected_features_intersection.intersection_update(current_method_features)

                    # Apply selected combination strategy
                    if combination_strategy == "Union":
                        combined_features = list(selected_features_union)
                    else:  # Intersection
                        combined_features = list(selected_features_intersection or [])

                    # Display combined results
                    st.markdown("### Combined Feature Selection Results :")
                    st.write("**Combined Features:**")
                    st.write(", ".join(combined_features) if combined_features else "No features selected.")

                    # Update processed data with combined features
                    if combined_features:
                        processed_data = processed_data[combined_features + [target_column]]


                    

        # Final Display of Processed Data
        st.markdown('<div class="data-preview :">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="data-section">', unsafe_allow_html=True)
            st.write("### Dataset Preview")
            st.dataframe(df.head())
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="data-section">', unsafe_allow_html=True)
            processed_data = pd.concat([X_features, y_target], axis=1)
            st.write("### Final Processed Data")
            st.dataframe(processed_data.head())
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # End of data preview section

        # Save processed data to session state
        if st.button("Save Processed Data"):
            st.session_state['processed_data'] = processed_data
            st.success("Processed data saved to session state!")

            # Offer download option
            csv = processed_data.to_csv(index=False)
            st.download_button(
                label="Download Processed Data",
                data=csv,
                file_name="processed_data.csv",
                mime="text/csv"
            )


        # Final Dataset
        #st.markdown("### Final Processed Data")
        #processed_data = pd.concat([X_features, y_target], axis=1)
        #st.dataframe(processed_data.head())

def prepare_unsupervised_data(df, numeric_columns, categorical_columns):
    """Handle supervised learning visualizations"""

    
    processed_data = df.copy()

    # Tabs for preprocessing steps
    tabs = st.tabs([
        "Basic Operations",
        "Missing Values",
        "Feature Engineering",
        "Advanced Operations"
    ])
    

    # Basic Operations Tab
    with tabs[0]:
        st.markdown("### Basic Data Operations :")
        col1, col2 = st.columns(2)

        with col1:
            
            if st.checkbox("Remove Duplicates"):
                initial_rows = processed_data.shape[0]
                processed_data.drop_duplicates(inplace=True)
                st.info(f"Removed {initial_rows - processed_data.shape[0]} duplicate rows.")
                #st.write("### Dataset After Basic Operations")
            if st.checkbox("Drop Unused Columns"):
                unused_cols = st.multiselect("Select columns to drop", df.columns)
                processed_data.drop(columns=unused_cols, inplace=True)
              

    # Missing Values Tab
    
    with tabs[1]:
        st.markdown("### Handle Missing Values :")
        if st.checkbox("Handle Missing Values"):
            numerical_cols = processed_data.select_dtypes(include=['float64', 'int64']).columns
            categorical_cols = processed_data.select_dtypes(include=['object', 'category', 'bool']).columns

            col1, col2 = st.columns(2)

            with col1:
                st.write("Numerical Columns")
                strategy = st.selectbox("Strategy for numerical columns", ["Mean", "Median", "Mode", "Constant"])
                if strategy == "Constant":
                    constant = st.number_input("Enter constant value", value=0)
                    processed_data[numerical_cols] = processed_data[numerical_cols].fillna(constant)
                else:
                    for col in numerical_cols:
                        if strategy == "Mean":
                            processed_data[col] = processed_data[col].fillna(processed_data[col].mean())
                        elif strategy == "Median":
                            processed_data[col] = processed_data[col].fillna(processed_data[col].median())
                        elif strategy == "Mode":
                            processed_data[col] = processed_data[col].fillna(processed_data[col].mode().iloc[0])
            

            with col2:
                st.write("Categorical Columns")
                strategy = st.selectbox("Strategy for categorical columns", ["Mode", "Constant"])
                if strategy == "Constant":
                    constant = st.text_input("Enter constant value", "Unknown")
                    processed_data[categorical_cols] = processed_data[categorical_cols].fillna(constant)
                else:
                    for col in categorical_cols:
                        processed_data[col] = processed_data[col].fillna(processed_data[col].mode().iloc[0])
            st.write("### Dataset After Basic Operations")
            #st.dataframe(processed_data)
            st.markdown('<div class="data-preview">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="data-section">', unsafe_allow_html=True)
                st.write("### Dataset Preview")
                st.dataframe(df.head())
                st.markdown('</div>', unsafe_allow_html=True)

            
            with col2:
                st.markdown('<div class="data-section">', unsafe_allow_html=True)
                st.write("### Final Processed Data")
                st.dataframe(processed_data.head())
                #st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)  # End of data preview section


# Feature Engineering Tab
    with tabs[2]:
        st.markdown("### Feature Engineering :")
        cool1, cool2 = st.columns(2)
        

        with cool1:
            if st.checkbox("Encode Categorical Variables"):
                categorical_cols = processed_data.select_dtypes(include=['object', 'category', 'bool']).columns
                method = st.selectbox("Encoding method", ["Label Encoding", "One-Hot Encoding"])
                if method == "Label Encoding":
                    le = LabelEncoder()
                    for col in categorical_cols:
                        processed_data[col] = le.fit_transform(processed_data[col])
                    st.info("Categorical variables encoded using Label Encoding.")
                else:
                    processed_data = pd.get_dummies(processed_data, drop_first=True)
                    st.info("Categorical variables encoded using One-Hot Encoding.")

            if st.checkbox("Normalize Columns"):
                numerical_cols = processed_data.select_dtypes(include=['float64', 'int64']).columns
                if numerical_cols.any():
                    scaler = MinMaxScaler()
                    choice = st.radio("Choose normalization type", ["All Columns", "Selected Columns"])
                    if choice == "All Columns":
                        processed_data[numerical_cols] = scaler.fit_transform(processed_data[numerical_cols])
                        st.info("All numerical columns normalized.")
                    elif choice == "Selected Columns":
                        selected_cols = st.multiselect("Select columns to normalize", numerical_cols)
                        if selected_cols:
                            processed_data[selected_cols] = scaler.fit_transform(processed_data[selected_cols])
                            st.info(f"Normalized columns: {', '.join(selected_cols)}")
        with cool2:
            if st.checkbox("Handle Outliers"):
                numerical_cols = processed_data.select_dtypes(include=['float64', 'int64']).columns
                z_scores = (processed_data[numerical_cols] - processed_data[numerical_cols].mean()) / processed_data[numerical_cols].std()
                processed_data = processed_data[(z_scores < 3).all(axis=1)]
                st.info("Outliers removed.")

            if st.checkbox("Standardize Columns"):
                choice = st.radio("Choose standardization type", ["All Columns", "Selected Columns"])
                numerical_cols = processed_data.select_dtypes(include=['float64', 'int64']).columns

                robust_scaler = RobustScaler()
                if choice == "All Columns":
                    processed_data[numerical_cols] = robust_scaler.fit_transform(processed_data[numerical_cols])
                    st.info("All numerical columns standardized.")
                elif choice == "Selected Columns":
                    selected_cols = st.multiselect("Select columns to standardize", numerical_cols)
                    if selected_cols:
                        processed_data[selected_cols] = robust_scaler.fit_transform(processed_data[selected_cols])
                        st.info(f"Standardized columns: {', '.join(selected_cols)}")


        #st.write("### Dataset After Basic Operations")
        #st.dataframe(processed_data)
        st.markdown('<div class="data-preview">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="data-section">', unsafe_allow_html=True)
            st.write("### Dataset Preview")
            st.dataframe(df.head())
            st.markdown('</div>', unsafe_allow_html=True)

        
        with col2:
            st.markdown('<div class="data-section">', unsafe_allow_html=True)
            st.write("### Final Processed Data")
            st.dataframe(processed_data.head())
            #st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # End of data preview section

    # Advanced Operations Tab
    # Advanced Operations Tab
    with tabs[3]:
        st.markdown("### Advanced Operations :")

        # Split processed_data into features (no target column for clustering)
        X_features = processed_data.copy()

        # Columns for layout
        ao_col1, ao_col2 = st.columns(2)

        with ao_col1:
            # PCA Option
            if st.checkbox("Apply PCA"):
                # Store original column names
                original_columns = X_features.columns.tolist()
                
                # Apply PCA
                pca = PCA()
                pca.fit(X_features)

                explained_variance_ratio = pca.explained_variance_ratio_.cumsum()
                n_components = next(i for i, var in enumerate(explained_variance_ratio) if var >= 0.95) + 1
                st.write(f"Number of components selected with 95% variance explained: {n_components}")

                # Transform data with selected components
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X_features)
                
                # Create new dataframe with PCA components
                X_features = pd.DataFrame(
                    X_pca,
                    columns=[f"PC{i+1}" for i in range(n_components)]
                )

                st.info("PCA applied to the dataset.")
                
                # Update processed_data with PCA components
                processed_data = X_features.copy()

            # Feature selection (focused on clustering relevance)
            if st.checkbox("Apply Feature Selection"):
                st.markdown("### Feature Selection :")

                methods = ["Variance Threshold", "Correlation-based Selection"]
                selected_method = st.selectbox("Select a Feature Selection Method", methods)

                # Perform feature selection using the selected method
                selected_features = set()

                if selected_method == "Variance Threshold":
                    threshold = st.slider("Select Variance Threshold", 0.0, 1.0, 0.1)
                    selector = VarianceThreshold(threshold=threshold)
                    selector.fit(X_features)
                    selected_features = set(X_features.columns[selector.get_support()])

                elif selected_method == "Correlation-based Selection":
                    corr_threshold = st.slider("Select Correlation Threshold", 0.0, 1.0, 0.8)
                    correlation_matrix = X_features.corr().abs()

                    # Select features with low correlation
                    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
                    selected_features = set(
                        X_features.columns[~(upper_triangle > corr_threshold).any(axis=0)]
                    )

                # Display selected features
                st.markdown("### Selected Features :")
                st.write(f"**Selected Features ({selected_method}):** {', '.join(selected_features)}")

                # Update processed data with selected features
                if selected_features:
                    X_features = X_features[list(selected_features)]
                    processed_data = X_features.copy()

        # Display data preview
        st.markdown("### Dataset Preview :")
        col1, col2 = st.columns(2)

        with col1:
            st.write("### Original Data :")
            st.dataframe(df.head())

        with col2:
            st.write("### Processed Features")
            st.dataframe(X_features.head())

        # Save processed data to session state
        if st.button("Save Processed Data"):
            if 'y_target' in locals():
                st.session_state['processed_data'] = processed_data
                st.success("Processed data saved to session state!")
                
                # Offer download option
                csv = processed_data.to_csv(index=False)
                st.download_button(
                    label="Download Processed Data",
                    data=csv,
                    file_name="processed_data.csv",
                    mime="text/csv"
                )
            else:
                st.session_state['processed_data'] = X_features
                st.success("Processed features saved to session state!")
                
                # Offer download option
                csv = X_features.to_csv(index=False)
                st.download_button(
                    label="Download Processed Features",
                    data=csv,
                    file_name="processed_features.csv",
                    mime="text/csv"
                )