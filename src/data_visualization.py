import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np 
import matplotlib.pyplot as plt
import numpy as np
def data_visualization_page():
    # Custom CSS
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
    st.markdown('<div class="navigation-bar"><h2>📊 Dashboard</h2></div>', unsafe_allow_html=True)




    # Check if a dataset is uploaded
    if 'uploaded_data' not in st.session_state:
        st.warning("Please upload a dataset first to use the visualization features.")
        return

    # Use the uploaded dataset
    df = st.session_state['uploaded_data']
    data_source = st.session_state.get('data_source', '')

    # Check if it's MNIST data by looking for specific characteristics
    is_mnist = ('image_width' in df.columns and 'image_height' in df.columns and 
                'target' in df.columns and 'label_name' in df.columns)

    if is_mnist:
        visualize_mnist_data(df)
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
            visualize_supervised_data(df, numeric_columns, categorical_columns)
        else:
            visualize_unsupervised_data(df, numeric_columns, categorical_columns)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def visualize_mnist_data(df):
    """Specialized visualization for MNIST digits dataset"""
    st.markdown("###  MNIST Digits Dataset Visualization")

    # Convert all image-related columns to numeric explicitly
    image_columns = df.columns.difference(['target', 'image_width', 'image_height', 'label_name'])

    # First ensure the data is numeric and handle any conversion issues
    for col in image_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows with NaN values that resulted from failed conversions
    df = df.dropna()

    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Sample Images", "Class Distribution", "Image Grid"])

    with tab1:
        st.markdown("#### Sample Images from Dataset")

        # Allow user to select number of images to display
        num_images = st.slider("Number of images to display", min_value=1, max_value=10, value=5)

        # Create columns for images
        cols = st.columns(num_images)

        # Display random sample images
        random_indices = np.random.choice(len(df), num_images, replace=False)

        for idx, col in enumerate(cols):
            try:
                # Convert to float explicitly and reshape
                img_data = df.iloc[random_indices[idx]][image_columns].astype(float).to_numpy()
                img_array = img_data.reshape(8, 8)  # Reshape to 8x8 for digits dataset
                digit = df.iloc[random_indices[idx]]['target']

                # Create figure with dark background
                fig, ax = plt.subplots(figsize=(3, 3))
                fig.patch.set_facecolor('#1E1E1E')
                ax.set_facecolor('#1E1E1E')

                # Display image
                ax.imshow(img_array, cmap='gray')
                ax.axis('off')
                ax.set_title(f'Digit: {digit}', color='white', pad=10)

                col.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                col.error(f"Error displaying image: {str(e)}")

    with tab2:
        st.markdown("#### Class Distribution")

        # Plot the distribution of digit classes
        class_counts = df['target'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis", ax=ax)
        ax.set_title("Class Distribution of MNIST Digits", fontsize=16)
        ax.set_xlabel("Digit", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        ax.set_xticks(range(10))
        ax.set_xticklabels(range(10))
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        st.markdown("#### Image Grid")

        # Allow user to select the grid size
        grid_size = st.slider("Grid size (NxN)", min_value=2, max_value=10, value=5)

        # Select random samples to display
        num_images = grid_size ** 2
        random_indices = np.random.choice(len(df), num_images, replace=False)

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        fig.patch.set_facecolor('#1E1E1E')

        for i, ax in enumerate(axes.flat):
            try:
                img_data = df.iloc[random_indices[i]][image_columns].astype(float).to_numpy()
                img_array = img_data.reshape(8, 8)  # Reshape to 8x8 for digits dataset
                digit = df.iloc[random_indices[i]]['target']

                ax.imshow(img_array, cmap='gray')
                ax.axis('off')
                ax.set_title(f'{digit}', color='white', fontsize=10)
            except Exception as e:
                ax.axis('off')

        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        st.pyplot(fig)
        plt.close(fig)



def visualize_supervised_data(df, numeric_columns, categorical_columns):
    """Handle supervised learning visualizations"""
   
    # Select Target Column
    options = [""] + list(df.columns)
    target_col = st.selectbox("Select Target Column", options=options, index=0)
    st.session_state['target_column'] = target_col

    # Determine Problem Type
    if target_col in numeric_columns:
        unique_values = df[target_col].nunique()
        problem_type = "Classification" if unique_values <= 20 else "Regression"
    elif target_col in categorical_columns:
        problem_type = "Classification"
    else:
        problem_type = None

    # Store target column and problem type in session state
    st.session_state['target_col'] = target_col
    st.session_state['problem_type'] = problem_type
    # Display inferred problem type
    if problem_type:
        st.info(f"Inferred Problem Type: {problem_type}")
    else:
        st.warning("Could not determine problem type. Please choose the target column.")

    # Regression Visualizations
    if problem_type == "Regression":
        st.markdown("###  Regression Visualizations :")
        
        # Target Variable Distribution
        st.markdown("#### Target Variable Line Plot :")
        fig = px.line(df, x=df.index, y=target_col, title="Line Plot of Target Variable")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#F0F0F0", font=dict(color="#ffffff"))
        st.plotly_chart(fig, use_container_width=True)

        # Scatter Plot
        if len(numeric_columns) > 1:
            st.markdown("#### Scatter Plot :")
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("Select X-axis:", numeric_columns)
            with col2:
                y_axis = st.selectbox("Select Y-axis:", numeric_columns)
            fig = px.scatter(df, x=x_axis, y=y_axis, trendline="ols", title="Scatter Plot with Trendline")
            fig.update_layout(template="plotly_dark", paper_bgcolor="#F0F0F0", font=dict(color="#ffffff"))
            st.plotly_chart(fig, use_container_width=True)

        # Correlation Heatmap
        st.markdown("#### Correlation Heatmap :")
        corr_matrix = df[numeric_columns].corr()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
        st.pyplot(fig)

        # Boxplot for Outliers
        st.markdown("#### Boxplot of Target Variable :")
        fig = px.box(df, y=target_col, title="Boxplot of Target Variable")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#F0F0F0", font=dict(color="#ffffff"))
        st.plotly_chart(fig, use_container_width=True)

    # Classification Visualizations
    elif problem_type == "Classification":
        st.markdown("###  Classification Visualizations :")

        # Class Distribution
        st.markdown("#### Class Distribution :")
        class_counts = df[target_col].value_counts()
        fig = px.bar(class_counts, x=class_counts.index, y=class_counts.values,
                     labels={'x': target_col, 'y': 'Count'}, title="Class Distribution")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#F0F0F0", font=dict(color="#ffffff"))
        st.plotly_chart(fig, use_container_width=True)

        # Pair Plot
        if len(numeric_columns) > 1:
            st.markdown("#### Pair Plot (Colored by Target) :")
            fig = px.scatter_matrix(df, dimensions=numeric_columns, color=target_col)
            st.plotly_chart(fig, use_container_width=True)

        # Feature Correlation Heatmap
        st.markdown("#### Feature Correlation Heatmap :")
        corr_matrix = df[numeric_columns].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
        st.pyplot(fig)

        # Grouped Bar Chart for Categorical Features
        if len(categorical_columns) > 0:
            st.markdown("#### Grouped Bar Chart for Categorical Features :")
            category = st.selectbox("Select Categorical Feature for Grouped Bar Chart:", categorical_columns)
            grouped = df.groupby([category, target_col]).size().reset_index(name="count")
            fig = px.bar(grouped, x=category, y="count", color=target_col, barmode="group",
                         title=f"Grouped Bar Chart of {category} by {target_col}",
                         labels={category: "Feature", "count": "Count"})
            fig.update_layout(template="plotly_dark", paper_bgcolor="#F0F0F0", font=dict(color="#ffffff"))
            st.plotly_chart(fig, use_container_width=True)

        # Violin Plot by Class
        st.markdown("#### Violin Plot by Class :")
        feature = st.selectbox("Select Feature for Violin Plot:", numeric_columns)
        fig = px.violin(df, y=feature, x=target_col, box=True, points="all", color=target_col,
                        title=f"Violin Plot of {feature} by {target_col}")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#F0F0F0", font=dict(color="#ffffff"))
        st.plotly_chart(fig, use_container_width=True)

    # General Visualizations (Fallback)
    else:
        pass
        """st.markdown("### General Visualizations :")
        if len(numeric_columns) > 0:
            st.markdown("#### Feature Histogram :")
            column = st.selectbox("Select a Numeric Column for Histogram:", numeric_columns)
            fig = px.histogram(df, x=column)
            fig.update_layout(template="plotly_dark", paper_bgcolor="#F0F0F0", font=dict(color="#ffffff"))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Boxplot :")
            fig = px.box(df, y=column)
            fig.update_layout(template="plotly_dark", paper_bgcolor="#F0F0F0", font=dict(color="#ffffff"))
            st.plotly_chart(fig, use_container_width=True)"""

def visualize_unsupervised_data(df, numeric_columns, categorical_columns):
    """
    Handle unsupervised learning visualizations focusing on raw data exploration
    """
    st.markdown("###  Unsupervised Learning Analysis :")

    # Feature Selection
    selected_features = st.multiselect(
        "Select features for analysis",
        numeric_columns,
        default=list(numeric_columns[:min(len(numeric_columns), 5)])
    )

    if not selected_features:
        st.warning("Please select at least one feature for analysis")
        return

    # Analysis Type Selection
    analysis_type = st.radio(
        "Select Analysis Type",
        ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"]
    )

    if analysis_type == "Univariate Analysis":
        st.markdown("###  Univariate Analysis :")
        
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Histograms", "Box Plots", "Kernel Density Plots", "All"]
        )

        if viz_type in ["Histograms", "All"]:
            st.markdown("####  Histograms :")
            cols = st.columns(2)
            for idx, feature in enumerate(selected_features):
                with cols[idx % 2]:
                    fig = px.histogram(
                        df, 
                        x=feature,
                        title=f"Histogram of {feature}",
                        nbins=30,
                        marginal="box"
                    )
                    fig.update_layout(template="plotly_dark", paper_bgcolor="#F0F0F0")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    

        if viz_type in ["Box Plots", "All"]:
            st.markdown("####  Box Plots :")
            fig = px.box(
                df,
                y=selected_features,
                title="Box Plots of Selected Features"
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor="#F0F0F0")
            st.plotly_chart(fig, use_container_width=True)

        if viz_type in ["Kernel Density Plots", "All"]:
            st.markdown("####  Kernel Density Plots :")
            fig = go.Figure()
            for feature in selected_features:
                fig.add_trace(go.Violin(
                    y=df[feature],
                    name=feature,
                    box_visible=True,
                    meanline_visible=True
                ))
            fig.update_layout(
                title="Kernel Density Estimation with Violin Plots",
                template="plotly_dark",
                paper_bgcolor="#F0F0F0"
            )
            st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Bivariate Analysis":
        st.markdown("###  Bivariate Analysis :")
        
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Scatter Plots", "Hexbin Plots"]
        )

        if viz_type == "Scatter Plots":
            col1, col2 = st.columns(2)
            with col1:
                x_feature = st.selectbox("Select X-axis feature", selected_features, key='x_feat')
            with col2:
                y_feature = st.selectbox("Select Y-axis feature", selected_features, key='y_feat')

            fig = px.scatter(
                df,
                x=x_feature,
                y=y_feature,
                title=f"Scatter Plot: {x_feature} vs {y_feature}",
                trendline="ols"
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor="#F0F0F0")
            st.plotly_chart(fig, use_container_width=True)

            # Add correlation coefficient
            correlation = df[x_feature].corr(df[y_feature])
            st.write(f"Correlation coefficient: {correlation:.3f}")

        else: 
            col1, col2 = st.columns(2)
            with col1:
                x_feature = st.selectbox("Select X-axis feature", selected_features, key='x_hex')
            with col2:
                y_feature = st.selectbox("Select Y-axis feature", selected_features, key='y_hex')

            fig = px.density_heatmap(
                df,
                x=x_feature,
                y=y_feature,
                title=f"Hexbin Plot: {x_feature} vs {y_feature}",
                marginal_x="histogram",
                marginal_y="histogram"
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor="#F0F0F0")
            st.plotly_chart(fig, use_container_width=True)

        """else:  # Correlation Analysis
            st.markdown("####  Correlation Heatmap")
            corr_matrix = df[selected_features].corr()
            
            fig = px.imshow(
                corr_matrix,
                title="Feature Correlation Heatmap",
                labels=dict(color="Correlation"),
                text=np.round(corr_matrix, 2),
                aspect="auto"
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor="#F0F0F0")
            st.plotly_chart(fig, use_container_width=True)

            # Identify highly correlated features
            threshold = st.slider("Correlation Threshold", 0.0, 1.0, 0.8)
            high_corr = np.where(np.abs(corr_matrix) > threshold)
            high_corr = [(corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y])
                        for x, y in zip(*high_corr) if x != y]
            
            if high_corr:
                st.write("Highly correlated feature pairs:")
                for feat1, feat2, corr in high_corr:
                    st.write(f"{feat1} - {feat2}: {corr:.3f}")"""

    else:  # Multivariate Analysis
        st.markdown("###  Multivariate Analysis :")
        
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Pair Plot", "3D Scatter Plot"]
        )

        if viz_type == "Pair Plot":
            # Limit to 5 features for performance
            if len(selected_features) > 5:
                st.warning("Limiting to first 5 features for performance")
                selected_features = selected_features[:5]
            
            fig = px.scatter_matrix(
                df[selected_features],
                dimensions=selected_features,
                title="Pair Plot of Selected Features"
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor="#F0F0F0")
            st.plotly_chart(fig, use_container_width=True)

        else:  # 3D Scatter Plot
            if len(selected_features) >= 3:
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_feature = st.selectbox("X-axis feature", selected_features, key='x_3d')
                with col2:
                    y_feature = st.selectbox("Y-axis feature", selected_features, key='y_3d')
                with col3:
                    z_feature = st.selectbox("Z-axis feature", selected_features, key='z_3d')

                fig = px.scatter_3d(
                    df,
                    x=x_feature,
                    y=y_feature,
                    z=z_feature,
                    title="3D Scatter Plot"
                )
                fig.update_layout(template="plotly_dark", paper_bgcolor="#F0F0F0")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least 3 features for 3D visualization")
