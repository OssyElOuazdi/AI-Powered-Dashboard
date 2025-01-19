import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_digits
import seaborn as sns
from pathlib import Path
import requests


def data_import_page():
    """
    Page for importing and creating datasets with session persistence
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

    
    st.markdown('<div class="navigation-bar"><h2>🗂️ Import Data</h2></div>', unsafe_allow_html=True)

    # Reset dataset button in the sidebar
    if st.sidebar.button("Reset Dataset", key="reset_dataset_btn"):
        # List of session state keys to clear
        keys_to_clear = [
            'uploaded_data', 'data_source', 'processed_data', 'X_train', 'X_test', 
            'y_train', 'y_test', 'trained_model', 'current_model_name', 
            'problem_type', 'is_classification', 'cluster_labels', 
            'feature_names', 'target_name'
        ]
        
        # Remove each key from session state
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Rerun the app to refresh the state
        st.rerun()

    # If a dataset is already loaded, display it with statistics
    if 'uploaded_data' in st.session_state:
        st.success("A dataset is already loaded. Explore it below.")
        show_dataset_and_statistics(st.session_state['uploaded_data'])
        return


    # Custom CSS for larger text
    st.markdown(
        """
        <style>
        .custom-radio-label {
            font-size: 24px; /* Adjust the size as needed */
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Display the text with larger size
    st.markdown('<div class="custom-radio-label">Select your import method :</div>', unsafe_allow_html=True)

    # Import method selection
    import_method = st.radio(
        "",
        ["Local File", "Example Dataset", "Manual Creation", "API Import", "Direct Link Import"],
        key="import_method_radio"
    )


    if import_method == "Local File":
        local_file_import()
    elif import_method == "Example Dataset":
        example_dataset_import()
    elif import_method == "Manual Creation":
        manual_data_creation()
    elif import_method == "API Import":
        api_import()
    elif import_method == "Direct Link Import":
        direct_link_import()

def show_dataset_and_statistics(df):
    """
    Display the dataset and descriptive statistics
    """
    # Display the dataset and its shape
    st.write("### Dataset :")
    st.caption(f"Number of Rows: {df.shape[0]}")
    st.caption(f"Number of Columns: {df.shape[1]}")
    st.dataframe(df, height=300)

    # If it's the MNIST dataset, show some example images
    """if 'image_width' in df.columns:
        st.write("### 🖼️ Example Images")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            if i < len(df):
                img_data = df.iloc[i].drop(['target', 'image_width', 'image_height', 'label_name'])
                img_array = np.array(img_data).reshape(8, 8)
                col.image(img_array, caption=f"Digit: {df.iloc[i]['target']}", 
                         use_container_width=True)"""

    # Descriptive Statistics for Numerical Values
    st.write("### 📊 Descriptive Statistics: Numerical")
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    if numerical_columns.empty:
        st.info("No numerical columns found in the dataset.")
    else:
        stats = df.describe().transpose()
        st.dataframe(stats, height=300)

    # Descriptive Statistics for Categorical Values
    st.write("### 📊 Descriptive Statistics: Categorical")
    categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns
    if categorical_columns.empty:
        st.info("No categorical columns found in the dataset.")
    else:
        for col in categorical_columns:
            st.write(f"#### {col} - Category Distribution")
            counts = df[col].value_counts()
            percentage = (counts / counts.sum()) * 100
            summary = pd.DataFrame({
                'Count': counts,
                'Percentage': percentage
            })
            st.dataframe(
                summary.style.format({
                    'Percentage': '{:.2f}%'
                }).highlight_max(axis=0, color='lightblue')
            )

def local_file_import():
    """
    Import files from the local computer
    """
    uploaded_file = st.file_uploader(
        "",
        type=['csv', 'xlsx', 'json'],
        key="file_uploader"
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)

            st.session_state['uploaded_data'] = df
            st.session_state['data_source'] = 'local file'
            st.success("File successfully imported!")
            show_dataset_and_statistics(df)
        except Exception as e:
            st.error(f"Error during import: {e}")

def example_dataset_import():
    """
    Import example datasets
    """
    dataset_choices = {
        "Iris": load_iris(as_frame=True).frame,
        "MNIST Digits": "MNIST Digits (Small subset for classification)",
        "Titanic": sns.load_dataset('titanic')
    }

    selected_dataset = st.selectbox(
        "Choose an example dataset :",
        list(dataset_choices.keys()),
        key="example_dataset_select"
    )

    if st.button("Load Dataset", key="load_dataset_btn"):
        if selected_dataset == "MNIST Digits":
            digits = load_digits()
            n_samples = 1000
            df = pd.DataFrame(digits.data[:n_samples], 
                            columns=[f'pixel_{i}' for i in range(64)])
            df['target'] = digits.target[:n_samples]
            df['image_width'] = 8
            df['image_height'] = 8
            df['label_name'] = df['target'].map(lambda x: f"Digit {x}")

            st.info("""
            Dataset Info:
            - Contains 1000 handwritten digit images (0-9)
            - Each image is 8x8 pixels (64 features)
            - 'target' column contains the actual digit (0-9)
            - 'label_name' provides a readable version of the target
            - Suitable for CNN classification tasks
            """)
        else:
            df = dataset_choices[selected_dataset]

        st.session_state['uploaded_data'] = df
        st.session_state['data_source'] = f'example dataset {selected_dataset}'
        st.success(f"Dataset {selected_dataset} loaded successfully!")
        show_dataset_and_statistics(df)

def api_import():
    """
    Import data via an API with just the URL.
    """
    st.write("### Import Data via API :")
    api_url = st.text_input("API Endpoint", placeholder="Enter the API endpoint URL")

    if st.button("Fetch Data"):
        try:
            # Make the API request
            response = requests.get(api_url)
            response.raise_for_status()  # Raise an error for bad responses

            # Detect JSON or CSV format
            if api_url.endswith('.csv'):
                df = pd.read_csv(api_url)  # If CSV, load into DataFrame
            else:
                data = response.json()  # If JSON, parse it
                df = pd.DataFrame(data)

            # Save to session state
            st.session_state['uploaded_data'] = df
            st.session_state['data_source'] = 'API Import'

            st.success("Data successfully imported from API!")
            st.dataframe(df.head())  # Display the first few rows
        except Exception as e:
            st.error(f"Error fetching data: {e}")




def direct_link_import():
    """
    Import data via a direct download link.
    """
    st.write("### Import Data via Direct Link :")
    file_url = st.text_input("File URL", placeholder="Enter the direct file URL (CSV, JSON, Excel)")

    if st.button("Download and Import"):
        try:
            # Download the file
            response = requests.get(file_url, stream=True)
            response.raise_for_status()

            # Save the file temporarily
            temp_file = Path("temp_downloaded_file")
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Load the file into a DataFrame
            if file_url.endswith('.csv'):
                df = pd.read_csv(temp_file)
            elif file_url.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(temp_file)
            elif file_url.endswith('.json'):
                df = pd.read_json(temp_file)
            else:
                raise ValueError("Unsupported file type. Please provide a CSV, Excel, or JSON file.")

            # Clean up temporary file
            temp_file.unlink()

            # Save to session state
            st.session_state['uploaded_data'] = df
            st.session_state['data_source'] = 'Direct Link Import'

            st.success("File successfully imported from the provided link!")
            show_dataset_and_statistics(df)
        except Exception as e:
            st.error(f"Error downloading or importing the file: {e}")

def manual_data_creation():
    """
    Manually create a dataset
    """
    st.write("Create your own dataset :")

    num_rows = st.number_input("Number of rows", min_value=1, max_value=100, value=10, key="num_rows_input")
    num_columns = st.number_input("Number of columns", min_value=1, max_value=10, value=3, key="num_cols_input")

    columns = []
    data_types = {}

    for i in range(num_columns):
        col_name = st.text_input(f"Column name {i+1}", key=f"col_name_{i}")
        col_type = st.selectbox(
            f"Data type for {col_name if col_name else f'column {i+1}'}", 
            ['Numeric', 'Categorical'],
            key=f"col_type_{i}"
        )

        if col_name:
            columns.append(col_name)
            data_types[col_name] = col_type

    if st.button("Generate Dataset", key="generate_dataset_btn"):
        if len(columns) == num_columns:
            df_data = {}
            for col, col_type in data_types.items():
                if col_type == 'Numeric':
                    df_data[col] = np.random.rand(num_rows) * 100
                else:
                    df_data[col] = np.random.choice(['A', 'B', 'C'], num_rows)

            df = pd.DataFrame(df_data)
            st.session_state['uploaded_data'] = df
            st.session_state['data_source'] = 'manual creation'
            st.success("Dataset successfully created!")
            show_dataset_and_statistics(df)
        else:
            st.warning("Please name all columns")

if __name__ == "__main__":
    data_import_page()