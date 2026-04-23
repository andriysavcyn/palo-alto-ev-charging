import pandas as pd
import streamlit as st
import joblib

st.set_page_config("EV Charging Station Usage of California City", layout='wide', page_icon='🚙')

@st.cache_resource
def load_models():
    try:
        pipeline = joblib.load("models/ev_scaler.pkl")
        model = joblib.load("models/kmeans_ev_model.pkl")
        return pipeline, model
    except Exception as e:
        st.error(f":Error loading models: {e}")
        return None, None

pipeline, model = load_models()

st.sidebar.header("Session Parameters")
with st.sidebar:
    st.markdown("Adjust the values to simulate a charging session:")
    duration = st.slider("Time at station (minutes)", min_value=1, max_value=2000, value=120)
    energy = st.slider("Energy consumed (kWh)", min_value=0.0, max_value=100.0, value=15.0)

    current_power = (energy / (duration / 60)) if duration > 0 else 0
   
    
    MAX_POWER_LIMIT = 350 
    is_physically_possible = current_power <= MAX_POWER_LIMIT

    if not is_physically_possible:
        st.sidebar.error(f"⚠️ Power exceeds {MAX_POWER_LIMIT}kW! This session is technically impossible.")

    st.divider()
    st.info("The model uses K-Means clustering to identify driver profiles.")

col_head1, col_head2 = st.columns([1, 10])
with col_head1:
    st.title("🚙")
with col_head2:
    st.title("EV Charging Behavior Analysis")

st.markdown("""
    This application utilizes Machine Learning to classify Electric Vehicle charging sessions in California. 
    By analyzing duration and energy consumption, we can optimize station turnover and identify power users.
""")

st.divider()

st.subheader("📊 Current Session Data")
m_col1, m_col2 = st.columns(2)
m_col1.metric("Selected Duration", f"{duration} min", delta_color="normal")
m_col2.metric("Selected Energy", f"{energy} kWh")

st.write("") 

if st.button("Analyze a behaviour of driver", type="primary", use_container_width=True):
    if not is_physically_possible:
        st.error(f"""
            ### ❌ Invalid Data Detected
            You entered **{energy} kWh** for **{duration} minutes**. 
            This would require a charging power of **{current_power:.1f} kW**. 
            
            Even the most advanced DC Fast Chargers (Ultra-fast) are limited to **350 kW**. 
            Please increase the time or decrease the energy amount.
        """)
    elif pipeline is not None and model is not None:
        with st.spinner('Running ML interface...'):
            user_data = {
                'duration_min': duration, 
                'Energy (kWh)': energy
            }
            df = pd.DataFrame([user_data])

            try:
                processed_data = pipeline.transform(df)
                prediction = model.predict(processed_data)[0]

                st.subheader("Analysis Result:")
                if prediction == 0:
                    st.info("🔋 **Cluster 0: Heavy Users**\n\nLasts about 4 hours and uses a lot of power. This is a good value client with a large battery.")
                if prediction == 1:
                    st.info("🚩 **Cluster 1: Cheeky parking attendants (Overstayers)**\n\nAttention! Abnormally long time at the station. This driver is taking up space as free parking. It is recommended to apply the Idle Fee tariff.")
                if prediction == 2:
                    st.info("☕ **Cluster 2: Coffee break (Quick snack)**\n\nShort session (about 1 hour), minimal energy. Driver just dropped in for a quick recharge.")
                if prediction == 3:
                    st.info("🛒 **Cluster 3: Standard shopping**\n\nStandard use (about 2.5 hours). Adequate balance of time and energy consumption.")

            except Exception as e:
                st.error(f"Error during prediction: {e}")
    else:
        st.error("Model files not found. Please check path: `../models/`")

st.divider()
st.markdown(f"""<div style="text-align: center; color: grey; font-size: 0.8rem;">
        Developed by Andriy Savchyn | Lviv | 2026<br>
        <a href="https://github.com/andriysavcyn" target="_blank">GitHub</a> • 
        <a href="https://www.linkedin.com/in/andriy-savchyn" target="_blank">LinkedIn</a>
    </div>""", unsafe_allow_html=True)