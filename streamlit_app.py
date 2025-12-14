import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# --- 1. PAGE CONFIGURATION & EXPERT STYLING ---
st.set_page_config(page_title="Intelligent EV Analytics Pro", layout="wide", page_icon="⚡")

# Custom CSS for Industrial Look
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stMetric {background-color: #ffffff; padding: 15px; border-radius: 12px; border-left: 5px solid #4CAF50; box-shadow: 0px 4px 6px rgba(0,0,0,0.05);}
    h1 {color: #111; font-weight: 700;}
    h2, h3 {color: #2c3e50;}
    .stButton>button {width: 100%; border-radius: 8px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;}
    </style>
    """, unsafe_allow_html=True)

st.title("⚡ Rajpoot & Sheikh EV Analytics: Industrial Edition")
st.markdown("### 🚀 Advanced Range Prediction & Market Intelligence System")
st.markdown("---")

# --- 2. ROBUST DATA PIPELINE ---
@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_csv("cleaned_Electric_Vehicle_Population_Data.csv")
        
        # Standardization
        if 'Country' in df.columns and 'County' not in df.columns:
            df.rename(columns={'Country': 'County'}, inplace=True)
            
        drop_cols = [c for c in ['Unnamed: 3', 'Unnamed: 14', 'VIN (1-10)', 'DOL Vehicle ID'] if c in df.columns]
        df = df.drop(columns=drop_cols, errors='ignore')

        # CRITICAL FIX: Filter out rows where Range is 0 (useless for training)
        if 'Electric Range' in df.columns:
            df = df[df['Electric Range'] > 0]
        
        # Imputation
        numeric_cols = df.select_dtypes(include=[int, float]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        str_cols = df.select_dtypes(include=["object"]).columns
        for col in str_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
            
        # Clean Logic
        if 'Base MSRP' in df.columns:
            df = df[df['Base MSRP'] >= 0]
        
        return df
    except Exception as e:
        st.error(f"Critical Error: {e}")
        return None

df_raw = load_and_clean_data()

if df_raw is not None:
    # --- 3. ADVANCED SIDEBAR CONTROLS ---
    st.sidebar.header("🕹️ Control Center")
    options = st.sidebar.radio("System Module:", 
                               ["Market Intelligence (EDA)", "Model Training Lab", "Real-Time Prediction"])

    # Global Filters (Applies to EDA)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌍 Global Data Filters")
    
    # Year Filter
    min_year = int(df_raw['Model Year'].min())
    max_year = int(df_raw['Model Year'].max())
    selected_years = st.sidebar.slider("Select Model Years", min_year, max_year, (2015, max_year))
    
    # Filter Data
    df = df_raw[(df_raw['Model Year'] >= selected_years[0]) & (df_raw['Model Year'] <= selected_years[1])]
    
    st.sidebar.info(f"Active Data Points: {len(df):,}")

    # --- MODULE A: MARKET INTELLIGENCE ---
    if options == "Market Intelligence (EDA)":
        st.header("📈 Market Intelligence & Strategic Insights")
        
        if len(df) == 0:
            st.warning("No data available for the selected time range. Please adjust filters.")
        else:
            # KPI Row
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Active Fleet", f"{len(df):,}", delta="Global Filter Applied")
            col2.metric("Avg. Range", f"{df['Electric Range'].mean():.0f} mi")
            col3.metric("Top Maker", df['Make'].mode()[0])
            bev_count = len(df[df['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)'])
            col4.metric("BEV Share", f"{(bev_count / len(df) * 100):.1f}%")

            st.markdown("### 📊 Interactive Visualizations")
            
            # Row 1: Forecasting & Hierarchy
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("1. AI Growth Forecast")
                # Forecasting Logic
                year_counts = df['Model Year'].value_counts().reset_index()
                year_counts.columns = ['Year', 'Count']
                year_counts = year_counts.sort_values('Year')
                
                if len(year_counts) > 3:
                    z = np.polyfit(year_counts['Year'], year_counts['Count'], 2)
                    p = np.poly1d(z)
                    future_years = np.arange(selected_years[0], selected_years[1] + 5)
                    future_counts = p(future_years)
                    
                    fig_trend = px.scatter(year_counts, x='Year', y='Count', title="Growth Trajectory")
                    fig_trend.add_traces(px.line(x=future_years, y=future_counts, color_discrete_sequence=['#FF4B4B']).data[0])
                    fig_trend.data[1].name = "Polynomial Projection"
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.info("Insufficient data for forecasting.")

            with c2:
                st.subheader("2. Market Hierarchy (Treemap)")
                top_makes = df['Make'].value_counts().nlargest(15).index
                tree_df = df[df['Make'].isin(top_makes)]
                top_models = tree_df.groupby(['Make', 'Model']).size().reset_index(name='Count')
                fig_tree = px.treemap(top_models, path=['Make', 'Model'], values='Count', title='Manufacturer > Model')
                st.plotly_chart(fig_tree, use_container_width=True)

            # Row 2: Geographic & Technical Evolution
            c3, c4 = st.columns(2)
            with c3:
                st.subheader("3. Geographic Hotspots")
                if 'City' in df.columns:
                    city_counts = df['City'].value_counts().nlargest(10).reset_index()
                    city_counts.columns = ['City', 'Count']
                    fig_city = px.bar(city_counts, x='Count', y='City', orientation='h', color='Count', title="Top 10 Cities")
                    st.plotly_chart(fig_city, use_container_width=True)
                else:
                    st.warning("City data not available.")

            with c4:
                st.subheader("4. Technological Evolution")
                avg_range = df[df['Electric Range'] > 0].groupby('Model Year')['Electric Range'].mean().reset_index()
                fig_line = px.line(avg_range, x='Model Year', y='Electric Range', markers=True, 
                                   title='Avg Electric Range (Yearly)', color_discrete_sequence=['#2ecc71'])
                st.plotly_chart(fig_line, use_container_width=True)

            # Row 3: Range Box Plot & CAFV Pie
            c5, c6 = st.columns(2)
            with c5:
                st.subheader("5. Range Efficiency (Box Plot)")
                top_10 = df['Make'].value_counts().nlargest(10).index
                fig_box = px.box(df[df['Make'].isin(top_10)], x='Make', y='Electric Range', color='Make', title="Range Consistency")
                st.plotly_chart(fig_box, use_container_width=True)

            with c6:
                st.subheader("6. Clean Energy Eligibility")
                cafv_col = 'Clean Alternative Fuel Vehicle (CAFV) Eligibility'
                if cafv_col in df.columns:
                    fig_pie = px.pie(df, names=cafv_col, title='Incentive Eligibility', hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("CAFV Eligibility data missing.")

            # Row 4: Correlation
            st.subheader("7. Variable Correlations (Heatmap)")
            corr = df[['Model Year', 'Electric Range', 'Base MSRP']].corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)

    # --- MODULE B: ADVANCED MODEL TRAINING LAB ---
    elif options == "Model Training Lab":
        st.header("🧪 Advanced Machine Learning Lab")
        st.markdown("Customize, Train, and Export your predictive models.")

        # Layout
        col_conf, col_params = st.columns([1, 2])
        
        with col_conf:
            st.subheader("1. Target Configuration")
            target_type = st.radio("Goal:", ["Classify Vehicle Type", "Predict Range (Regression)"])
            test_size = st.slider("Test Set Split", 0.1, 0.4, 0.2, help="Percentage of data reserved for validation")

        with col_params:
            st.subheader("2. Hyperparameter Tuning (Optimization)")
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                n_estimators = st.slider("Number of Trees", 50, 500, 100, step=50)
                max_depth = st.slider("Max Tree Depth", 5, 50, 20)
            with c_p2:
                min_samples = st.slider("Min Samples Split", 2, 10, 2)
                bootstrap = st.checkbox("Bootstrap Sampling", value=True)

        # Data Prep
        le_make = LabelEncoder()
        le_type = LabelEncoder()
        model_df = df[['Make', 'Model Year', 'Electric Range', 'Base MSRP', 'Electric Vehicle Type']].copy().dropna()
        
        model_df['Make_Encoded'] = le_make.fit_transform(model_df['Make'])
        model_df['Type_Encoded'] = le_type.fit_transform(model_df['Electric Vehicle Type'])

        if st.button("🚀 Train Optimized Model", type="primary"):
            with st.spinner("Optimizing weights and training model..."):
                if "Classify" in target_type:
                    X = model_df[['Make_Encoded', 'Model Year', 'Electric Range', 'Base MSRP']]
                    y = model_df['Type_Encoded']
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                                   min_samples_split=min_samples, bootstrap=bootstrap, random_state=42)
                    metric = "Accuracy"
                else:
                    X = model_df[['Make_Encoded', 'Model Year', 'Type_Encoded', 'Base MSRP']]
                    y = model_df['Electric Range']
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                                  min_samples_split=min_samples, bootstrap=bootstrap, random_state=42)
                    metric = "R² Score"

                # Train
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                model.fit(X_train, y_train)
                
                # Evaluate
                score = model.score(X_test, y_test)
                
                # Visuals
                c_res1, c_res2 = st.columns(2)
                with c_res1:
                    st.success(f"Model Performance: {metric} = {score:.4f}")
                    feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=True)
                    fig_feat = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', title="Feature Importance")
                    st.plotly_chart(fig_feat, use_container_width=True)
                
                with c_res2:
                    if "Classify" in target_type:
                        cm = confusion_matrix(y_test, model.predict(X_test))
                        fig_cm = px.imshow(cm, text_auto=True, x=le_type.classes_, y=le_type.classes_, title="Confusion Matrix")
                        st.plotly_chart(fig_cm, use_container_width=True)
                    else:
                        # NEW: Regression Metrics
                        preds = model.predict(X_test)
                        mae = mean_absolute_error(y_test, preds)
                        rmse = np.sqrt(mean_squared_error(y_test, preds))
                        
                        st.markdown("#### 📉 Error Metrics")
                        m1, m2 = st.columns(2)
                        m1.metric("Mean Absolute Error (MAE)", f"{mae:.2f} mi")
                        m2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")

                        # Residual Plot
                        residuals = y_test - preds
                        fig_res = px.scatter(x=preds, y=residuals, labels={'x': 'Predicted Range', 'y': 'Residuals'}, 
                                             title="Residual Plot (Errors)")
                        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig_res, use_container_width=True)

                # --- MLOps: DOWNLOAD MODEL ---
                st.subheader("💾 MLOps: Export Artifacts")
                model_data = {
                    "model": model,
                    "le_make": le_make,
                    "le_type": le_type,
                    "target": target_type
                }
                serialized_model = pickle.dumps(model_data)
                st.download_button(
                    label="Download Trained Model (.pkl)",
                    data=serialized_model,
                    file_name=f"ev_model_{metric}_{score:.2f}.pkl",
                    mime="application/octet-stream"
                )
                
                # Session State
                st.session_state['model_pack'] = model_data

    # --- MODULE C: REAL-TIME PREDICTION ---
    elif options == "Real-Time Prediction":
        st.header("🤖 Prediction Interface")
        
        if 'model_pack' not in st.session_state:
            st.warning("⚠️ No active model found. Please train a model in the 'Lab' tab.")
        else:
            pack = st.session_state['model_pack']
            model = pack['model']
            le_make = pack['le_make']
            le_type = pack['le_type']
            target = pack['target']
            
            st.success(f"Model Loaded: **{target}**")
            
            with st.form("pred_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    make = st.selectbox("Make", le_make.classes_)
                with c2:
                    year = st.number_input("Year", 1995, 2030, 2024)
                with c3:
                    msrp = st.number_input("MSRP ($)", 0, 200000, 45000)
                
                val_range = 0
                val_type = ""
                if "Classify" in target:
                    val_range = st.number_input("Range (mi)", 0, 600, 250)
                else:
                    val_type = st.selectbox("Type", le_type.classes_)
                
                submitted = st.form_submit_button("Run Prediction")
                
            if submitted:
                make_enc = le_make.transform([make])[0]
                if "Classify" in target:
                    vec = np.array([[make_enc, year, val_range, msrp]])
                    res = le_type.inverse_transform([model.predict(vec)[0]])[0]
                    st.metric("Predicted Class", res)
                else:
                    type_enc = le_type.transform([val_type])[0]
                    vec = np.array([[make_enc, year, type_enc, msrp]])
                    res = model.predict(vec)[0]
                    st.metric("Predicted Range", f"{res:.1f} Miles")
