import streamlit as st
from src.data_import import data_import_page
from src.data_visualization import data_visualization_page
from src.data_preparation import data_preparation_page
from src.ml_modeling import ml_modeling_page
from src.evaluation import evaluation_page
from src.guide import guide_page



def home_page():
    """
    Enhanced Home Page for the App Presenting Its Features
    """
    # Custom CSS
    st.markdown("""
        <style>
            /* General Page Styling */
            body {
                background-color: #F0F4F8; /* Changed from #DCE4C9 */
                font-family: 'Arial', sans-serif;
            }

            /* Header Section */
            .header {
                text-align: center;
                background-color: #2C3E50; /* Changed from #272727 */
                color: #FFFFFF;
                padding: 30px;
                border-radius: 10px;
            }
            .header h1 {
                font-size: 3rem;
                margin: 0;
                color: #E67E22; /* Changed from #E07B39 */
            }
            .header p {
                font-size: 1.3rem;
                margin: 10px 0 0 0;
                color: #BDC3C7; /* Changed from #B6A28E */
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
                background-color: #FFFFFF; /* Changed from #F5F5DC */
                border-radius: 10px;
                padding: 20px;
                width: 22%;
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                transition: transform 0.3s ease;
            }
            .feature-card:hover {
                transform: translateY(-5px);
                background-color: #f0f0f0; /* Changed from #E07B39 */
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
                color: #2C3E50; /* Changed from #272727 */
            }
            .feature-description {
                font-size: 1rem;
                color: #7F8C8D; /* Changed from #555555 */
            }

            /* Sidebar Styling */
            [data-testid="stSidebar"] {
                background-color: #F0F4F8 !important; /* Changed from #DCE4C9 */
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
                color: #2C3E50; /* Changed from #272727 */
                font-size: 1rem;
                border-radius: 5px;
                transition: background-color 0.3s;
            }
            .sidebar-button:hover {
                background-color: #E67E22; /* Changed from #FFFFFF */
                color: #FFFFFF;
            }
            .header {
                display: flex;
                align-items: center;
                justify-content: center;
                background-color: #2C3E50; /* Changed from #272727 */
                color: #FFFFFF;
                padding: 30px;
                border-radius: 10px;
            }
            .header h1 {
                font-size: 3rem;
                margin: 0 15px 0 0; /* Add space between logo and title */
                color: #E67E22; /* Changed from #E07B39 */
            }
            .header p {
                font-size: 1.3rem;
                margin: 10px 0 0 0;
                color: #BDC3C7; /* Changed from #B6A28E */
            }
            .header img {
                height: 80px; /* Adjust the size of the logo */
                margin-right: 15px; /* Add space between logo and text */
            }
        </style>
    """, unsafe_allow_html=True)

    # # Header Section
    # st.markdown("""
    #     <div class="header">
    #         <h1>ML Academy</h1>
    #         <p>Your intuitive guide to a complete machine learning workflow.</p>
    #     </div>
    # """, unsafe_allow_html=True)
    st.markdown(f"""
   
    <div class="header">
        <div>
            <h1>AI Powered Dashboard</h1>
        <p>Bring Your Data..Get Insights..Make Decisions.</p>
        </div>
    </div>
""", unsafe_allow_html=True)


    # Features Section
    st.markdown("""
        <div class="features">
            <div class="feature-card">
                <div class="feature-icon" style="background-image: url('https://img.icons8.com/ios/50/000000/upload-to-cloud.png');"></div>
                <div class="feature-title">Data Import</div>
                <div class="feature-description">Upload datasets with ease.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon" style="background-image: url('https://img.icons8.com/ios/50/000000/combo-chart.png');"></div>
                <div class="feature-title">Dashboard</div>
                <div class="feature-description">Analyze your data with clear visualizations.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon" style="background-image: url('https://img.icons8.com/ios/50/000000/broom.png');"></div>
                <div class="feature-title">Data Preparation</div>
                <div class="feature-description">Clean and structure your data.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon" style="background-image: url('https://img.icons8.com/ios/50/000000/artificial-intelligence.png');"></div>
                <div class="feature-title">AI Modeling</div>
                <div class="feature-description">Train and test AI models.</div>
            </div>
                <div class="feature-card">
                <div class="feature-icon" style="background-image: url('https://img.icons8.com/ios/50/000000/rating.png');"></div>
                <div class="feature-title">Models Evaluation</div>
                <div class="feature-description">Assess model performance with metrics.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def app():
    """
    Main Application Entry Point
    """
    st.set_page_config(
        page_title="AI Powered Dashboard",
        page_icon="uploads/logo.ico",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    


    # Initialize session state for navigation if not exists
    if 'current_page' not in st.session_state:
       st.session_state.current_page = "Home"

    # Sidebar navigation using buttons
    with st.sidebar:
        st.markdown("<h1 style='text-align: center; font-size: 2.5rem;'>Navigation</h1>", unsafe_allow_html=True)
        if st.button(" Home", key="home", use_container_width=True):
            st.session_state.current_page = "Home"
        if st.button(" Import Data", key="data_import", use_container_width=True):
            st.session_state.current_page = "Data Import"
        if st.button(" Dashboard", key="visualization", use_container_width=True):
            st.session_state.current_page = "Dashboard"
        if st.button(" Prepare Data", key="preparation", use_container_width=True):
            st.session_state.current_page = "Data Preparation"
        if st.button(" Train AI Models", key="modeling", use_container_width=True):
            st.session_state.current_page = "AI Models"
        if st.button(" Evaluation", key="evaluation", use_container_width=True):
            st.session_state.current_page = "Evaluation"
        if st.button(" Guide", key="guide", use_container_width=True):
            st.session_state.current_page = "Guide"
        

    # Page Routing based on session state
    if st.session_state.current_page == "Home":
        home_page()
    elif st.session_state.current_page == "Data Import":
        data_import_page()
    elif st.session_state.current_page == "Dashboard":
        data_visualization_page()
    elif st.session_state.current_page == "Data Preparation":
        data_preparation_page()
    elif st.session_state.current_page == "AI Models":
        ml_modeling_page()
    elif st.session_state.current_page == "Evaluation":
        evaluation_page()
    elif st.session_state.current_page == "Guide":
        guide_page()
    
    
if __name__ == "__main__":
    app()