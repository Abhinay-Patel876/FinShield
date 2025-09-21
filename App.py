# import streamlit as st
# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
# import joblib
# import plotly.express as px
# import plotly.graph_objects as go

# # ---------------------- Load & Preprocess Data ----------------------
# def load_data(file):
#     data = pd.read_excel(file, sheet_name='bankruptcy-prevention', engine='openpyxl')
#     if data.iloc[:, 0].dtype != object:
#         data.iloc[:, 0] = data.iloc[:, 0].astype(str)
#     data_split = data.iloc[:, 0].str.split(';', expand=True)
#     data_split.columns = ["industrial_risk", "management_risk", "financial_flexibility",
#                           "credibility", "competitiveness", "operating_risk", "class"]
#     for col in data_split.columns[:-1]:
#         data_split[col] = data_split[col].astype(float)
#     data_split['class'] = data_split['class'].map({'non-bankruptcy': 0, 'bankruptcy': 1})
#     return data_split

# # ---------------------- Train Model ----------------------
# def train_and_save_model(data):
#     X = data.drop('class', axis=1)
#     y = data['class']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = xgb.XGBClassifier(random_state=42)
#     model.fit(X_train, y_train)
#     joblib.dump(model, 'xgboost_model.pkl')
#     y_pred = model.predict(X_test)
#     y_prob = model.predict_proba(X_test)[:, 1]
#     return model, X_test, y_test, y_pred, y_prob

# def predict_bankruptcy(model, features):
#     df = pd.DataFrame([features], columns=[
#         'industrial_risk', 'management_risk', 'financial_flexibility',
#         'credibility', 'competitiveness', 'operating_risk'
#     ])
#     prediction = model.predict(df)[0]
#     probability = model.predict_proba(df)[0][1]
#     return prediction, probability

# # ---------------------- Feature Importance ----------------------
# def plot_feature_importance(model):
#     importance = model.feature_importances_
#     features = ['industrial_risk', 'management_risk', 'financial_flexibility',
#                 'credibility', 'competitiveness', 'operating_risk']
#     fig = px.bar(x=importance, y=features, orientation="h",
#                  labels={'x': 'Importance', 'y': 'Features'},
#                  title="Feature Importance", color=importance,
#                  color_continuous_scale='Blues')
#     st.plotly_chart(fig, use_container_width=True)

# # ---------------------- Main App ----------------------
# def main():
#     st.set_page_config(page_title="Bankruptcy Predictor", layout="wide")

#     # Custom CSS
#     st.markdown("""
#         <style>
#             body {background-color: #f9fafc;}
#             .prediction-card {
#                 padding: 20px; border-radius: 10px; text-align: center; font-size: 20px;
#                 font-weight: bold; color: white;
#             }
#             .success {background-color: #4CAF50;}
#             .danger {background-color: #f44336;}
#         </style>
#     """, unsafe_allow_html=True)

#     st.title("📉 Bankruptcy Prediction Dashboard")
#     st.write("Upload your dataset or use the default one, adjust financial risk indicators, and get instant predictions.")

#     # Sidebar Inputs
#     st.sidebar.header("🔍 Input Features")
#     industrial_risk = st.sidebar.slider("Industrial Risk", 0.0, 1.0, 0.5, step=0.1, help="Risk from industry sector")
#     management_risk = st.sidebar.slider("Management Risk", 0.0, 1.0, 0.5, step=0.1, help="Risk due to management decisions")
#     financial_flexibility = st.sidebar.slider("Financial Flexibility", 0.0, 1.0, 0.5, step=0.1)
#     credibility = st.sidebar.slider("Credibility", 0.0, 1.0, 0.5, step=0.1)
#     competitiveness = st.sidebar.slider("Competitiveness", 0.0, 1.0, 0.5, step=0.1)
#     operating_risk = st.sidebar.slider("Operating Risk", 0.0, 1.0, 0.5, step=0.1)

#     st.sidebar.markdown("📁 Upload your Excel file")
#     uploaded_file = st.sidebar.file_uploader("Choose a file", type=["xlsx"])
#     if uploaded_file:
#         data = load_data(uploaded_file)
#     else:
#         data = load_data(r"C:\Users\HP\OneDrive\Documents\bankruptcy-prevention (1).xlsx")

#     model, X_test, y_test, y_pred, y_prob = train_and_save_model(data)

#     # Tabs
#     tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Metrics", "📌 Feature Importance"])

#     with tab1:
#         st.subheader("Prediction Result")
#         if st.button("Predict Bankruptcy"):
#             features = [industrial_risk, management_risk, financial_flexibility,
#                         credibility, competitiveness, operating_risk]
#             prediction, probability = predict_bankruptcy(model, features)

#             # Colored card
#             if prediction == 1:
#                 st.markdown(f"<div class='prediction-card danger'>🚨 Bankruptcy Predicted<br>Confidence: {probability:.2%}</div>", unsafe_allow_html=True)
#             else:
#                 st.markdown(f"<div class='prediction-card success'>✅ Non-Bankruptcy Predicted<br>Confidence: {probability:.2%}</div>", unsafe_allow_html=True)

#             # Gauge Chart
#             fig = go.Figure(go.Indicator(
#                 mode="gauge+number",
#                 value=probability * 100,
#                 title={'text': "Bankruptcy Probability (%)"},
#                 gauge={'axis': {'range': [0, 100]},
#                        'bar': {'color': "darkred" if prediction == 1 else "green"}}
#             ))
#             st.plotly_chart(fig, use_container_width=True)

#     with tab2:
#         st.subheader("Model Evaluation")
#         col1, col2 = st.columns(2)
#         with col1:
#             st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
#             st.metric("ROC AUC", f"{roc_auc_score(y_test, y_prob):.2f}")
#         with col2:
#             with st.expander("📌 Confusion Matrix"):
#                 st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred),
#                                           columns=["Predicted Non-Bankruptcy", "Predicted Bankruptcy"],
#                                           index=["Actual Non-Bankruptcy", "Actual Bankruptcy"]))
#         with st.expander("📄 Preview Data"):
#             st.dataframe(data.head())

#     with tab3:
#         st.subheader("Feature Importance")
#         plot_feature_importance(model)

#     # Feedback
#     st.markdown("---")
#     with st.expander("💬 Feedback"):
#         st.text_area("Share your thoughts or suggestions:")

#     st.markdown("<p style='text-align: center; color: gray;'>Made with ❤️ by Abhinay</p>", unsafe_allow_html=True)


# if __name__ == "__main__":
#     main()



import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta

# ---------------------- Load & Preprocess Data ----------------------
@st.cache_data
def load_data(file):
    if isinstance(file, str):
        data = pd.read_excel(file, sheet_name='bankruptcy-prevention', engine='openpyxl')
    else:
        data = pd.read_excel(file, sheet_name='bankruptcy-prevention', engine='openpyxl')
    
    if data.iloc[:, 0].dtype != object:
        data.iloc[:, 0] = data.iloc[:, 0].astype(str)
    data_split = data.iloc[:, 0].str.split(';', expand=True)
    data_split.columns = ["industrial_risk", "management_risk", "financial_flexibility",
                          "credibility", "competitiveness", "operating_risk", "class"]
    for col in data_split.columns[:-1]:
        data_split[col] = data_split[col].astype(float)
    data_split['class'] = data_split['class'].map({'non-bankruptcy': 0, 'bankruptcy': 1})
    return data_split

# ---------------------- Train Multiple Models ----------------------
@st.cache_resource
def train_models(data):
    X = data.drop('class', axis=1)
    y = data['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'XGBoost': xgb.XGBClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    trained_models = {}
    model_scores = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        trained_models[name] = model
        model_scores[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'cv_scores': cross_val_score(model, X_train, y_train, cv=5)
        }
    
    return trained_models, model_scores, X_test, y_test

# ---------------------- Prediction Functions ----------------------
def predict_bankruptcy(model, features):
    df = pd.DataFrame([features], columns=[
        'industrial_risk', 'management_risk', 'financial_flexibility',
        'credibility', 'competitiveness', 'operating_risk'
    ])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]
    return prediction, probability

def risk_assessment(features):
    """Provide risk assessment based on feature values"""
    risks = []
    feature_names = ['Industrial Risk', 'Management Risk', 'Financial Flexibility',
                    'Credibility', 'Competitiveness', 'Operating Risk']
    
    for i, (name, value) in enumerate(zip(feature_names, features)):
        if i == 2:  # Financial flexibility is inverse (higher = better)
            if value < 0.3:
                risks.append(f"⚠️ Low {name}: {value:.1f}")
            elif value < 0.6:
                risks.append(f"⚡ Moderate {name}: {value:.1f}")
        else:
            if value > 0.7:
                risks.append(f"🚨 High {name}: {value:.1f}")
            elif value > 0.5:
                risks.append(f"⚠️ Moderate {name}: {value:.1f}")
    
    return risks

# ---------------------- Interactive Visualizations ----------------------
def create_radar_chart(features):
    categories = ['Industrial Risk', 'Management Risk', 'Financial Flexibility',
                 'Credibility', 'Competitiveness', 'Operating Risk']
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=features,
        theta=categories,
        fill='toself',
        name='Current Company',
        line_color='rgb(1,90,200)'
    ))
    
    # Add industry average (simulated)
    avg_features = [0.4, 0.3, 0.7, 0.8, 0.6, 0.4]
    fig.add_trace(go.Scatterpolar(
        r=avg_features,
        theta=categories,
        fill='toself',
        name='Industry Average',
        line_color='rgb(255,140,0)',
        opacity=0.6
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title="Risk Profile Comparison"
    )
    return fig

def create_time_series_simulation(probability, days=30):
    """Simulate probability changes over time"""
    dates = pd.date_range(start=datetime.now(), periods=days, freq='D')
    # Add some random variation
    probabilities = [probability + np.random.normal(0, 0.02) for _ in range(days)]
    probabilities = [max(0, min(1, p)) for p in probabilities]  # Keep within bounds
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=probabilities,
        mode='lines+markers',
        name='Bankruptcy Probability',
        line=dict(color='red' if probability > 0.5 else 'green')
    ))
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange", 
                  annotation_text="Risk Threshold")
    
    fig.update_layout(
        title="Bankruptcy Risk Trend (Simulated)",
        xaxis_title="Date",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1])
    )
    return fig

def plot_model_comparison(model_scores):
    models = list(model_scores.keys())
    accuracies = [model_scores[model]['accuracy'] for model in models]
    roc_aucs = [model_scores[model]['roc_auc'] for model in models]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Accuracy', 'ROC AUC'])
    
    fig.add_trace(go.Bar(x=models, y=accuracies, name='Accuracy', marker_color='lightblue'), row=1, col=1)
    fig.add_trace(go.Bar(x=models, y=roc_aucs, name='ROC AUC', marker_color='lightcoral'), row=1, col=2)
    
    fig.update_layout(showlegend=False, title="Model Performance Comparison")
    return fig

# ---------------------- Scenario Analysis ----------------------
def scenario_analysis(model, base_features):
    scenarios = {
        'Best Case': [max(0, f - 0.2) if i != 2 else min(1, f + 0.2) for i, f in enumerate(base_features)],
        'Worst Case': [min(1, f + 0.2) if i != 2 else max(0, f - 0.2) for i, f in enumerate(base_features)],
        'Current': base_features
    }
    
    results = {}
    for scenario, features in scenarios.items():
        pred, prob = predict_bankruptcy(model, features)
        results[scenario] = {'prediction': pred, 'probability': prob[1]}
    
    return results

# ---------------------- Main App ----------------------
def main():
    st.set_page_config(
        page_title="AI Bankruptcy Predictor Pro", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Enhanced CSS
    st.markdown("""
        <style>
            .main-header {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                padding: 2rem;
                border-radius: 10px;
                margin-bottom: 2rem;
                text-align: center;
                color: white;
            }
            .prediction-card {
                padding: 30px; 
                border-radius: 15px; 
                text-align: center; 
                font-size: 24px;
                font-weight: bold; 
                color: white;
                margin: 20px 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .success {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .danger {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            }
            .metric-card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin: 10px 0;
            }
            .risk-item {
                padding: 10px;
                margin: 5px 0;
                border-radius: 5px;
                background: #f8f9fa;
                border-left: 4px solid #dc3545;
            }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🤖 AI Bankruptcy Predictor </h1>
            <p>Advanced ML-powered financial risk assessment with real-time analytics</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar with Enhanced Controls
    st.sidebar.markdown("## 🎛️ Control Panel")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "🧠 Choose AI Model",
        ["XGBoost", "Random Forest", "Logistic Regression"],
        help="Different algorithms for prediction"
    )
    
    # Interactive mode
    interactive_mode = st.sidebar.toggle("🔄 Real-time Mode", value=True, help="Update predictions automatically")
    
    st.sidebar.markdown("### 📊 Financial Risk Indicators")
    
    # Enhanced sliders with better descriptions
    industrial_risk = st.sidebar.slider(
        "🏭 Industrial Risk", 0.0, 1.0, 0.5, step=0.1,
        help="Risk from industry volatility and market conditions"
    )
    management_risk = st.sidebar.slider(
        "👥 Management Risk", 0.0, 1.0, 0.5, step=0.1,
        help="Risk from management decisions and leadership quality"
    )
    financial_flexibility = st.sidebar.slider(
        "💰 Financial Flexibility", 0.0, 1.0, 0.5, step=0.1,
        help="Company's ability to adapt to financial challenges"
    )
    credibility = st.sidebar.slider(
        "🏆 Credibility", 0.0, 1.0, 0.5, step=0.1,
        help="Market trust and reputation score"
    )
    competitiveness = st.sidebar.slider(
        "⚡ Competitiveness", 0.0, 1.0, 0.5, step=0.1,
        help="Ability to compete in the market"
    )
    operating_risk = st.sidebar.slider(
        "⚙️ Operating Risk", 0.0, 1.0, 0.5, step=0.1,
        help="Risk from operational processes and efficiency"
    )
    
    # Quick preset buttons
    st.sidebar.markdown("### 🎯 Quick Presets")
    col1, col2 = st.sidebar.columns(2)
    
    if col1.button("🟢 Healthy Co."):
        industrial_risk, management_risk, operating_risk = 0.2, 0.1, 0.2
        financial_flexibility, credibility, competitiveness = 0.8, 0.9, 0.8
    
    if col2.button("🔴 At-Risk Co."):
        industrial_risk, management_risk, operating_risk = 0.8, 0.7, 0.8
        financial_flexibility, credibility, competitiveness = 0.2, 0.3, 0.2

    # File upload
    st.sidebar.markdown("### 📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"])
    
    # Load data
    try:
        if uploaded_file:
            data = load_data(uploaded_file)
            st.sidebar.success("✅ Custom data loaded!")
        else:
            # Use default file path - you may need to adjust this
            data = load_data("bankruptcy-prevention.xlsx")  # Adjust path as needed
            st.sidebar.info("📋 Using sample data")
    except Exception as e:
        st.sidebar.error("❌ Could not load data. Using synthetic data.")
        # Create synthetic data if file not found
        np.random.seed(42)
        data = pd.DataFrame({
            'industrial_risk': np.random.uniform(0, 1, 250),
            'management_risk': np.random.uniform(0, 1, 250),
            'financial_flexibility': np.random.uniform(0, 1, 250),
            'credibility': np.random.uniform(0, 1, 250),
            'competitiveness': np.random.uniform(0, 1, 250),
            'operating_risk': np.random.uniform(0, 1, 250),
            'class': np.random.choice([0, 1], 250, p=[0.7, 0.3])
        })

    # Train models
    with st.spinner("🔄 Training AI models..."):
        trained_models, model_scores, X_test, y_test = train_models(data)

    # Current features
    features = [industrial_risk, management_risk, financial_flexibility,
                credibility, competitiveness, operating_risk]
    
    # Real-time prediction
    selected_model = trained_models[model_choice]
    prediction, probability = predict_bankruptcy(selected_model, features)

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔮 Live Prediction", 
        "📊 Advanced Analytics", 
        "🎯 Scenario Analysis",
        "📈 Model Performance",
        "🔍 Risk Assessment",
        "📋 Data Explorer"
    ])

    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Dynamic prediction display
            if interactive_mode:
                prediction_placeholder = st.empty()
                gauge_placeholder = st.empty()
                
                with prediction_placeholder.container():
                    if prediction == 1:
                        st.markdown(f"""
                            <div class='prediction-card danger'>
                                🚨 BANKRUPTCY RISK DETECTED<br>
                                <span style='font-size:18px'>Probability: {probability[1]:.2%}</span><br>
                                <span style='font-size:14px'>Model: {model_choice}</span>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class='prediction-card success'>
                                ✅ FINANCIALLY STABLE<br>
                                <span style='font-size:18px'>Safety Score: {(1-probability[1]):.2%}</span><br>
                                <span style='font-size:14px'>Model: {model_choice}</span>
                            </div>
                        """, unsafe_allow_html=True)
                
                with gauge_placeholder.container():
                    # Enhanced gauge chart
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = probability[1] * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Bankruptcy Risk Level (%)"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkred" if prediction == 1 else "darkgreen"},
                            'steps': [
                                {'range': [0, 25], 'color': "lightgreen"},
                                {'range': [25, 50], 'color': "yellow"},
                                {'range': [50, 75], 'color': "orange"},
                                {'range': [75, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Risk factors analysis
            risks = risk_assessment(features)
            if risks:
                st.subheader("⚠️ Risk Factors Identified")
                for risk in risks:
                    st.markdown(f"<div class='risk-item'>{risk}</div>", unsafe_allow_html=True)
        
        with col2:
            # Real-time metrics
            st.subheader("📊 Live Metrics")
            
            st.metric(
                label="Risk Score",
                value=f"{probability[1]:.2%}",
                delta=f"{(probability[1] - 0.5):.2%}" if probability[1] != 0.5 else "0.00%"
            )
            
            st.metric(
                label="Safety Buffer",
                value=f"{max(0, 0.5 - probability[1]):.2%}",
                delta="Healthy" if probability[1] < 0.5 else "At Risk"
            )
            
            st.metric(
                label="Model Confidence",
                value=f"{max(probability):.2%}",
                delta="High" if max(probability) > 0.7 else "Moderate"
            )
            
            # Radar chart
            st.subheader("🎯 Risk Profile")
            radar_fig = create_radar_chart(features)
            st.plotly_chart(radar_fig, use_container_width=True, height=300)

    with tab2:
        st.subheader("📊 Advanced Analytics Dashboard")
        
        # Time series simulation
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Risk Trend Analysis")
            time_fig = create_time_series_simulation(probability[1])
            st.plotly_chart(time_fig, use_container_width=True)
        
        with col2:
            st.subheader("🔍 Feature Impact Analysis")
            importance = selected_model.feature_importances_ if hasattr(selected_model, 'feature_importances_') else [1/6]*6
            features_names = ['Industrial Risk', 'Management Risk', 'Financial Flexibility',
                             'Credibility', 'Competitiveness', 'Operating Risk']
            
            fig = px.bar(
                x=importance, 
                y=features_names, 
                orientation="h",
                title="Feature Importance",
                color=importance,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution analysis
        st.subheader("📊 Data Distribution")
        feature_to_analyze = st.selectbox("Select feature to analyze:", features_names)
        feature_idx = features_names.index(feature_to_analyze)
        feature_col = data.columns[feature_idx]
        
        fig = px.histogram(
            data, 
            x=feature_col, 
            color='class', 
            title=f"Distribution of {feature_to_analyze}",
            marginal="box"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("🎯 Scenario Analysis")
        
        # Run scenario analysis
        scenarios = scenario_analysis(selected_model, features)
        
        # Display results
        cols = st.columns(3)
        for i, (scenario, result) in enumerate(scenarios.items()):
            with cols[i]:
                color = "success" if result['probability'] < 0.5 else "danger"
                st.markdown(f"""
                    <div style='text-align: center; padding: 20px; border-radius: 10px; 
                         background: {"#d4edda" if color == "success" else "#f8d7da"}'>
                        <h4>{scenario}</h4>
                        <h3>{"✅" if result['probability'] < 0.5 else "⚠️"}</h3>
                        <p>Risk: {result['probability']:.2%}</p>
                    </div>
                """, unsafe_allow_html=True)
        
        # Monte Carlo simulation
        st.subheader("🎲 Monte Carlo Risk Simulation")
        if st.button("🚀 Run Simulation"):
            with st.spinner("Running 1000 simulations..."):
                results = []
                for _ in range(1000):
                    # Add random noise to features
                    sim_features = [f + np.random.normal(0, 0.1) for f in features]
                    sim_features = [max(0, min(1, f)) for f in sim_features]  # Bound between 0-1
                    _, sim_prob = predict_bankruptcy(selected_model, sim_features)
                    results.append(sim_prob[1])
                
                fig = px.histogram(
                    x=results, 
                    nbins=50,
                    title="Monte Carlo Simulation - Bankruptcy Probability Distribution",
                    labels={'x': 'Bankruptcy Probability', 'y': 'Frequency'}
                )
                fig.add_vline(x=np.mean(results), line_dash="dash", 
                             annotation_text=f"Mean: {np.mean(results):.2%}")
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"📊 Simulation Complete! Average Risk: {np.mean(results):.2%}")

    with tab4:
        st.subheader("📈 Model Performance Comparison")
        
        # Model comparison chart
        comparison_fig = plot_model_comparison(model_scores)
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Detailed metrics
        st.subheader("📋 Detailed Performance Metrics")
        
        for model_name, scores in model_scores.items():
            with st.expander(f"🤖 {model_name} Details"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{scores['accuracy']:.3f}")
                with col2:
                    st.metric("ROC AUC", f"{scores['roc_auc']:.3f}")
                with col3:
                    st.metric("CV Score (avg)", f"{np.mean(scores['cv_scores']):.3f}")
                
                # Cross-validation scores
                fig = px.bar(
                    x=[f"Fold {i+1}" for i in range(len(scores['cv_scores']))],
                    y=scores['cv_scores'],
                    title=f"{model_name} Cross-Validation Scores"
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("🔍 Comprehensive Risk Assessment")
        
        # Risk categories
        risk_categories = {
            "Financial Health": (financial_flexibility, credibility),
            "Market Position": (competitiveness, industrial_risk),
            "Operational Efficiency": (operating_risk, management_risk)
        }
        
        for category, (metric1, metric2) in risk_categories.items():
            risk_score = (metric1 + (1-metric2)) / 2 if category == "Financial Health" else (1-metric1 + 1-metric2) / 2
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{category}**")
                progress_color = "🟢" if risk_score > 0.7 else "🟡" if risk_score > 0.4 else "🔴"
                st.progress(risk_score)
            with col2:
                st.metric("Score", f"{risk_score:.2f}", f"{progress_color}")
        
        # Recommendations
        st.subheader("💡 AI Recommendations")
        recommendations = []
        
        if industrial_risk > 0.6:
            recommendations.append("🏭 Consider diversifying across industries to reduce sector-specific risks")
        if management_risk > 0.6:
            recommendations.append("👥 Strengthen management processes and decision-making frameworks")
        if financial_flexibility < 0.4:
            recommendations.append("💰 Improve cash flow management and secure additional financing options")
        if credibility < 0.5:
            recommendations.append("🏆 Focus on building market trust through transparent communication")
        if competitiveness < 0.5:
            recommendations.append("⚡ Invest in innovation and competitive advantage development")
        if operating_risk > 0.6:
            recommendations.append("⚙️ Optimize operational processes and reduce inefficiencies")
        
        if not recommendations:
            recommendations.append("✨ Company shows strong financial health across all metrics!")
        
        for rec in recommendations:
            st.info(rec)

    with tab6:
        st.subheader("📋 Interactive Data Explorer")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Dataset Overview")
            st.dataframe(data.head(10))
            
            # Interactive scatter plot
            x_axis = st.selectbox("Select X-axis:", data.columns[:-1])
            y_axis = st.selectbox("Select Y-axis:", data.columns[:-1])
            
            fig = px.scatter(
                data, 
                x=x_axis, 
                y=y_axis, 
                color='class',
                title=f"{x_axis} vs {y_axis}",
                color_discrete_map={0: 'green', 1: 'red'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Quick Stats")
            st.write("**Dataset Info:**")
            st.write(f"• Total samples: {len(data)}")
            st.write(f"• Bankruptcy cases: {data['class'].sum()}")
            st.write(f"• Healthy companies: {len(data) - data['class'].sum()}")
            st.write(f"• Bankruptcy rate: {data['class'].mean():.1%}")
            
            # Correlation heatmap
            st.subheader("🔥 Feature Correlations")
            corr_matrix = data.corr()
            fig = px.imshow(
                corr_matrix,
                title="Correlation Heatmap",
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig, use_container_width=True)

    # Footer with enhanced styling
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🔄 **Real-time Updates**: Enable for live predictions")
    with col2:
        st.success("🤖 **AI Powered**: Using advanced ML algorithms")
    with col3:
        st.warning("⚠️ **Disclaimer**: For educational purposes only")
    
    # Enhanced feedback section
    with st.expander("💬 Feedback & Export"):
        feedback_tab1, feedback_tab2 = st.tabs(["💭 Feedback", "📤 Export"])
        
        with feedback_tab1:
            rating = st.select_slider(
                "Rate this app:", 
                options=[1, 2, 3, 4, 5], 
                format_func=lambda x: "⭐" * x,
                value=5
            )
            feedback_text = st.text_area("Share your thoughts:")
            if st.button("Submit Feedback"):
                st.success("Thank you for your feedback! 🙏")
        
        with feedback_tab2:
            if st.button("📊 Export Prediction Report"):
                report_data = {
                    'Company_Profile': features,
                    'Risk_Score': probability[1],
                    'Prediction': 'Bankruptcy Risk' if prediction == 1 else 'Financially Stable',
                    'Model_Used': model_choice,
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                st.download_button(
                    label="📥 Download Report (JSON)",
                    data=str(report_data),
                    file_name=f"bankruptcy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

    st.markdown("""
        <div style='text-align: center; padding: 20px; color: #666;'>
            <p>🚀 Made with ❤️ by FinShield | Enhanced AI Bankruptcy Predictor </p>
            <p style='font-size: 12px;'>Powered by XGBoost, Random Forest & Logistic Regression</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()