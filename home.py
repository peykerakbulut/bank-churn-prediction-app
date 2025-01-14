import streamlit as st
import os
from streamlit_option_menu import option_menu


# Streamlit'in otomatik menüsünü gizle
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="images/data_scientist_badge.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)


# Custom CSS for sidebar menu
st.markdown("""
    <style>
    [data-testid=stSidebar] {
        background-color: white;
    }
    
    section[data-testid="stSidebar"] > div {
        background-color: #eff2f6;
        padding: 1rem;
    }
    
    .logo-container {
        padding: 5px 0;
        margin-top: -35px;
        text-align: center;
    }

    /* Varsayılan menüyü gizle */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none;}
    
    /* Pages sekmesini gizle */
    section[data-testid="stSidebar"] > div.css-1d391kg {display: none;}
    
    </style>
""", unsafe_allow_html=True)

# Sidebar içeriği
with st.sidebar:
    # Logo bölümü
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    st.image("images/data_scientist_logo.png", width=250)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation menüsü
    selected = option_menu(
        menu_title=None,
        options=[
            "Homepage", 
            "About DataSet", 
            "Project Charts",
            "Evaluation",
            "Single Prediction",
            "Batch Prediction"
        ],
        icons=[
            'house-fill',  # Homepage
            'database-fill',  # About DataSet
            'graph-up',  # Project Charts 
            'clipboard-data',  # Evaluation
            'bullseye',  # Single Prediction
            'cloud-upload',  # Batch Prediction
        ],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#eff2f6"},
            "icon": {"color": "#31333F", "font-size": "16px"}, 
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "padding": "15px 20px",
                "--hover-color": "#F0F2F6",
                "background-color": "white",
                "margin-bottom": "5px",
                "border-radius": "10px",
            },
            "nav-link-selected": {
                "background-color": "#ffbd59",
                "color": "#000000",
            },
        }
    )

    # Initialize session state for section if not exists
    if 'section' not in st.session_state:
        st.session_state.section = "Homepage"
    
    # Update session state based on selection
    st.session_state.section = selected

# Sayfa yönlendirmesi
try:
    if st.session_state.section == "Homepage":
        import modules.project as project
        project.home_page()
    elif st.session_state.section == "About DataSet":
        import modules.dataset as dataset
        dataset.dataset_page()
    elif st.session_state.section == "Project Charts":
        import modules.charts as charts
        charts.charts_page()
    elif st.session_state.section == "Evaluation":
        import modules.evaluation as evaluation
        evaluation.evaluation_page()
    elif st.session_state.section == "Single Prediction":
        import modules.predict as predict
        predict.predict_page()
    elif st.session_state.section == "Batch Prediction":
        import modules.batch as batch
        batch.batch_page()
except Exception as e:
    st.error(f"Hata: {e}")
