import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Intelligent EV Analytics Pro", layout="wide", page_icon="⚡")

# Custom CSS for Professional UI
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stMetric {background-color: #ffffff; padding: 15px; border-radius: 12px; border-left: 5px solid #4CAF50; box-shadow: 0px 4px 6px rgba(0,0,0,0.05);}
    h1 {color: #111; font-weight: 700;}
    h2, h3 {color: #2c3e50;}
    </style>
    """, unsafe_allow_html=True)

st.title("⚡ Intelligent EV Analytics: Range Prediction & Market Trends")
st.markdown("### 🚀 End-to-End Pipeline: Raw Data -> Cleaning -> Intelligence")

# --- 2. SMART DATA PIPELINE ---
@st.cache_data
def load_and_process_data():
    try:
        # STEP A: LOAD RAW DATA FROM ZIP
        # We use zipfile to open the zip and find the first CSV inside, regardless of its name
        with zipfile.ZipFile("raw_data.zip", "r") as z:
            csv_file = [f for f in z.namelist() if f.endswith('.csv')][0]
            df_raw = pd.read_csv(z.open(csv_file))
        
        raw_count = len(df_raw)
        
        # STEP B: DATA CLEANING (The "Mining" Logic)
        df_clean = df_raw.copy()
        
        # 1. Standardize Columns
        if 'Country' in df_clean.columns and 'County' not in df_clean.columns:
            df_clean.rename(columns={'Country': 'County'}, inplace=True)
            
        # 2. Remove administrative/irrelevant columns
        drop_cols = ['Unnamed: 3', 'Unnamed: 14', 'VIN (1-10)', 'DOL Vehicle ID', 'Legislative District', '2020 Census Tract']
        df_clean = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns], errors='ignore')

        # 3. CRITICAL: Handle Range = 0 (Data Quality Fix)
        # Many datasets have '0' for range. We must filter these out for accurate training.
        if 'Electric Range' in df_clean.columns:
            df_clean = df_clean[df_clean['Electric Range'] > 0]
        
        # 4. Fill Missing Numeric Data (Imputation)
        num_cols = df_clean.select_dtypes(include=[int, float]).columns
        df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())
        
        # 5. Fill Missing Categorical Data
        str_cols = df_clean.select_dtypes(include=["object"]).columns
        for col in str_cols:
            df_clean[col] = df_clean[col].fillna("Unknown")

        # 6. MSRP Cleaning
        if 'Base MSRP' in df_clean.columns:
             # If MSRP is 0, we can't really fix it easily without external data, 
             # but we will keep it for general trends, just maybe not for heavy price correlation if it's all 0.
             pass 

        clean_count = len(df_clean)
        
        return df_raw, df_clean, raw_count, clean_count
    except Exception as e:
        st.error(f"Data Pipeline Error: {e}")
        return None, None, 0, 0

# EXECUTE PIPELINE
df_raw_view, df, count_raw, count_clean = load_and_process_data()

if df is not None:
    # --- SIDEBAR: PIPELINE VISUALIZATION ---
    st.sidebar.header("⚙️ Data Pipeline Status")
    st.sidebar.success("✅ Raw Data Ingested (raw_data.zip)")
    st.sidebar.success("✅ ETL Processing Complete")
    
    # Show the "Before vs After" stats
    st.sidebar.markdown("---")
    st.sidebar.metric("Raw Rows", f"{count_raw:,}")
    st.sidebar.metric("Cleaned Rows", f"{count_clean:,}")
    st.sidebar.markdown(f"**Filtered Out:** {count_raw - count_clean:,} rows ({(count_raw - count_clean)/count_raw:.1%} junk data)")
    
    # Toggle to show raw data
    show_raw = st.sidebar.checkbox("Show Raw vs Clean Data Inspection")

    # --- MAIN APP LOGIC ---
    if show_raw:
        st.subheader("🔍 Data Quality Inspection (EDA)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original Raw Data (Sample)**")
            st.dataframe(df_raw_view.head(50), height=300)
        with c2:
            st.markdown("**Processed Clean Data (Sample)**")
            st.dataframe(df.head(50), height=300)
        st.markdown("---")

    options = st.sidebar.radio("Select Module:", 
                               ["Market Intelligence (EDA)", "Model Training Lab", "Real-Time Prediction"])

    # Global Filters
    min_year = int(df['Model Year'].min())
    max_year = int(df['Model Year'].max())
    selected_years = st.sidebar.slider("Filter Model Years", min_year, max_year, (2015, max_year))
    df_filtered = df[(df['Model Year'] >= selected_years[0]) & (df['Model Year'] <= selected_years[1])]

    # --- MODULE A: EDA (VISUALIZATIONS) ---
    if options == "Market Intelligence (EDA)":
        st.header("📈 Market Trends & Visualizations")
        
        # KPI Row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Vehicles", len(df_filtered))
        k2.metric("Avg Range", f"{df_filtered['Electric Range'].mean():.0f} mi")
        k3.metric("Avg Price", f"${df_filtered['Base MSRP'].mean():,.0f}")
        k4.metric("Top Brand", df_filtered['Make'].mode()[0])

        st.markdown("### 📊 Project Visualizations")
        
        c1, c2 = st.columns(2)
        with c1:
             # 1. Growth Trend
            year_counts = df_filtered['Model Year'].value_counts().reset_index()
            year_counts.columns = ['Year', 'Count']
            year_counts = year_counts.sort_values('Year')
            fig1 = px.area(year_counts, x='Year', y='Count', title="1. EV Market Growth Trend", color_discrete_sequence=['#4CAF50'])
            st.plotly_chart(fig1, use_container_width=True)
        
        with c2:
            # 2. Top Manufacturers
            top_makes = df_filtered['Make'].value_counts().nlargest(10).reset_index()
            top_makes.columns = ['Make', 'Count']
            fig2 = px.bar(top_makes, x='Count', y='Make', orientation='h', title="2. Market Share by Manufacturer", color='Count')
            st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            # 3. Range Distribution
            fig3 = px.histogram(df_filtered, x='Electric Range', nbins=30, title="3. Distribution of Electric Range", color_discrete_sequence=['#FF4B4B'])
            st.plotly_chart(fig3, use_container_width=True)
        
        with c4:
             # 4. Price vs Range Correlation
            fig4 = px.scatter(df_filtered, x='Base MSRP', y='Electric Range', color='Make', title="4. Range vs Price Correlation")
            st.plotly_chart(fig4, use_container_width=True)

    # --- MODULE B: TRAINING LAB ---
    elif options == "Model Training Lab":
        st.header("🧪 Model Training Lab")
        st.markdown("Train a Random Forest Regressor to predict EV Range based on Make, Year, and Price.")
        
        col1, col2 = st.columns(2)
        with col1:
            split_size = st.slider("Train/Test Split Ratio", 0.1, 0.5, 0.2)
        with col2:
            n_trees = st.slider("Number of Trees (Estimators)", 10, 200, 100)

        if st.button("Start Training Sequence", type="primary"):
            with st.spinner("Training Model..."):
                # Prepare Data
                le_make = LabelEncoder()
                train_df = df_filtered[['Make', 'Model Year', 'Base MSRP', 'Electric Range']].copy()
                train_df['Make_Code'] = le_make.fit_transform(train_df['Make'])
                
                # Features & Target
                X = train_df[['Make_Code', 'Model Year', 'Base MSRP']]
                y = train_df['Electric Range']
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=42)
                
                # Train
                model = RandomForestRegressor(n_estimators=n_trees, random_state=42)
                model.fit(X_train, y_train)
                
                # Evaluate
                preds = model.predict(X_test)
                mae = mean_absolute_error(y_test, preds)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                r2 = r2_score(y_test, preds)
                
                # Save for Prediction Tab
                st.session_state['trained_model'] = {
                    'model': model, 'le': le_make, 'r2': r2, 'mae': mae
                }
                
                st.success(f"Training Complete! R² Score: {r2:.4f}")
                
                # Metrics Display
                st.markdown("#### 📉 Model Evaluation Matrices")
                m1, m2, m3 = st.columns(3)
                m1.metric("R² Score (Accuracy)", f"{r2:.2%}", delta="Higher is better")
                m2.metric("MAE (Avg Error)", f"{mae:.2f} mi", delta="Lower is better", delta_color="inverse")
                m3.metric("RMSE", f"{rmse:.2f}", delta="Lower is better", delta_color="inverse")
                
                # Residual Plot
                residuals = y_test - preds
                fig_res = px.scatter(x=preds, y=residuals, labels={'x': 'Predicted Range', 'y': 'Residuals'}, 
                                     title="Residual Plot (Prediction Errors)")
                fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_res, use_container_width=True)

    # --- MODULE C: PREDICTION ---
    elif options == "Real-Time Prediction":
        st.header("🤖 Real-Time Prediction Dashboard")
        
        if 'trained_model' not in st.session_state:
            st.warning("⚠️ No active model found. Please go to the 'Model Training Lab' tab and train a model first.")
        else:
            data = st.session_state['trained_model']
            model = data['model']
            le = data['le']
            
            st.success(f"Model Active (Accuracy: {data['r2']:.2%})")
            
            with st.form("predict_form"):
                st.subheader("Input Vehicle Specs")
                c1, c2, c3 = st.columns(3)
                with c1:
                    make = st.selectbox("Manufacturer", le.classes_)
                with c2:
                    year = st.number_input("Model Year", 2010, 2030, 2025)
                with c3:
                    price = st.number_input("Base MSRP ($)", 0, 500000, 55000)
                
                submit = st.form_submit_button("Predict Range")
                
            if submit:
                # Encode and Predict
                make_code = le.transform([make])[0]
                prediction = model.predict([[make_code, year, price]])[0]
                
                st.markdown("---")
                st.metric("Estimated Electric Range", f"{prediction:.1f} Miles")
                st.info(f"Note: Based on historical data, this prediction has an average margin of error of ±{data['mae']:.1f} miles.")
