import streamlit as st

# Tüm sayfalardaki config bilgilerini birleştirdik
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="images/data_scientist_badge.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={

        "About": "For More Information\n" + "https://data-scientist.ai/",
        "Report a bug": "mailto:peykererkol@gmail.com",
        "Get help": "https://data-scientist.ai/"
    }
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
    
    .main-menu {
        font-size: 24px;
        margin: 20px 0;
        padding: 10px;
        color: #31333F;
        font-weight: 500;
    }

    /* Buton stilini özelleştir */
    .stButton > button {
        width: 100%;
        border-radius: 10px !important;
        padding: 15px 20px !important;
        border: none !important;
        background-color: white !important;
        color: #31333F !important;
        font-weight: normal !important;
        transition: all 0.3s ease;
    }

    /* Buton içeriğini sola yasla */
    .stButton > button p {
        text-align: left !important;
        display: flex !important;
        align-items: left !important;
        justify-content: flex-start !important;
        margin: 0 !important;
    }

    /* Aktif menü */
    .stButton > button:active, .stButton > button:focus {
        background-color: #ffbd59 !important;
        color: #000000 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    }

    /* Hover efekti */
    .stButton > button:hover {
        background-color: #F0F2F6 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    }
    
    .logo-container {
        padding: 5px 0;
        margin-top: -35px;
        text-align: center;
    }

    /* İkon ve metin arasındaki boşluğu ayarla */
    .stButton > button div {
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar içeriği
with st.sidebar:
    # Logo bölümü
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    st.image("images/data_scientist_logo.png", width=250)
    st.markdown('</div>', unsafe_allow_html=True)
    
    
    
    # Navigation menüsü
    menu_items = {
        "Project Description": "📝",
        "About DataSet": "📊",
        "Project Charts": "📈",
        "Evaluation": "📋",
        "Prediction": "🎯"
    }
    
    # Initialize session state for section if not exists
    if 'section' not in st.session_state:
        st.session_state.section = "Project Description"
    
    # Navigation buttons
    for page, icon in menu_items.items():
        if st.sidebar.button(f"{icon} {page}", key=page, use_container_width=True):
            st.session_state.section = page


# Sayfa yönlendirmesi
try:
    if st.session_state.section == "Project Description":
        import pages._project as project
    elif st.session_state.section == "About DataSet":
        import pages._dataset as dataset
    elif st.session_state.section == "Project Charts":
        import pages._charts as charts
    elif st.session_state.section == "Evaluation":
        import pages._evaluation as evaluation
    elif st.session_state.section == "Prediction":
        import pages._predict as predict
except Exception as e:
    st.error(f"Hata: {e}")
