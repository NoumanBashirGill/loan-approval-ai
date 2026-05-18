"""
LOAN APPROVAL AI MODEL - COMPLETE WEB APPLICATION
Ready to Run: python app.py
"""

# ============================================
# IMPORT ALL LIBRARIES
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import os
import base64
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION (Must be first Streamlit command)
# ============================================
st.set_page_config(
    page_title="Loan Approval AI System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CREATE DATASET
# ============================================
@st.cache_resource
def create_and_train_model():
    """Create dataset and train the model"""
    
    np.random.seed(42)
    n = 5000

    # Features
    income = np.random.normal(50000, 15000, n)
    loan_amount = np.random.normal(20000, 10000, n)
    credit_score = np.random.randint(300, 850, n)
    employment_years = np.random.randint(0, 30, n)
    dti = np.random.uniform(0, 50, n)  # Debt-to-income ratio

    # Target variable (loan approval logic)
    approved = ((credit_score > 650) & 
               (income > loan_amount * 0.3) & 
               (dti < 40)).astype(int)

    # Create DataFrame
    df = pd.DataFrame({
        'income': income,
        'loan_amount': loan_amount,
        'credit_score': credit_score,
        'employment_years': employment_years,
        'dti': dti,
        'approved': approved
    })

    # Prepare data
    X = df.drop('approved', axis=1)
    y = df['approved']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest with hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5]
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', verbose=0)
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_

    # Evaluate model
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Store metrics
    metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_
    }

    return best_model, scaler, metrics, X_test, y_test, y_pred, y_pred_proba

# ============================================
# LOAD OR TRAIN MODEL
# ============================================
st.sidebar.markdown("## 🚀 Loan Approval AI System")
st.sidebar.markdown("---")

# Check if model exists, otherwise train
model_path = 'loan_approval_model.pkl'
scaler_path = 'scaler.pkl'

if os.path.exists(model_path) and os.path.exists(scaler_path):
    with st.spinner('Loading trained model...'):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        # For metrics, we need to retrain or load from cache
        model, scaler, metrics, X_test, y_test, y_pred, y_pred_proba = create_and_train_model()
    st.sidebar.success("✅ Model loaded successfully!")
else:
    with st.spinner('Training AI model on 5000 loan records... This may take a few seconds.'):
        model, scaler, metrics, X_test, y_test, y_pred, y_pred_proba = create_and_train_model()
        # Save model for future use
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
    st.sidebar.success("✅ Model trained and saved successfully!")

# ============================================
# PREDICTION FUNCTION
# ============================================
def predict_loan(income, loan_amount, credit_score, employment_years, dti):
    """Predict loan approval for new application"""
    input_data = np.array([[income, loan_amount, credit_score, employment_years, dti]])
    input_scaled = scaler.transform(input_data)
    
    probability = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]
    
    result = "APPROVED" if prediction == 1 else "REJECTED"
    confidence = probability if prediction == 1 else 1 - probability
    
    return {
        'decision': result,
        'confidence': f"{confidence*100:.1f}%",
        'probability_score': probability,
        'prediction': prediction
    }

# ============================================
# MAIN UI
# ============================================
st.title("🏦 Loan Approval Prediction System")
st.markdown("### AI-Powered Credit Decision Engine")
st.markdown("---")

# Create two columns for layout
col1, col2 = st.columns([1.2, 0.8], gap="large")

with col1:
    st.markdown("### 📋 Applicant Information")
    st.markdown("Enter the applicant's financial details below:")
    
    # Create input form
    with st.form("loan_application_form"):
        col_a, col_b = st.columns(2)
        
        with col_a:
            income = st.number_input(
                "💰 Annual Income ($)",
                min_value=0,
                max_value=500000,
                value=55000,
                step=5000,
                help="Applicant's gross annual income"
            )
            
            loan_amount = st.number_input(
                "💵 Loan Amount Requested ($)",
                min_value=1000,
                max_value=200000,
                value=25000,
                step=1000,
                help="Total loan amount requested"
            )
            
            credit_score = st.slider(
                "📊 Credit Score",
                min_value=300,
                max_value=850,
                value=680,
                step=5,
                help="Credit score range 300-850"
            )
        
        with col_b:
            employment_years = st.number_input(
                "💼 Years of Employment",
                min_value=0,
                max_value=50,
                value=5,
                step=1,
                help="Number of years at current job"
            )
            
            dti = st.slider(
                "📉 Debt-to-Income Ratio (DTI %)",
                min_value=0,
                max_value=60,
                value=35,
                step=1,
                help="Monthly debt payments / Monthly income * 100"
            )
        
        st.markdown("---")
        submitted = st.form_submit_button("🔍 PREDICT LOAN DECISION", use_container_width=True)
    
    # Prediction result
    if submitted:
        with st.spinner('Analyzing application...'):
            result = predict_loan(income, loan_amount, credit_score, employment_years, dti)
        
        # Display result with styling
        st.markdown("### 📊 Loan Decision Result")
        
        if result['decision'] == "APPROVED":
            st.success(f"## ✅ LOAN {result['decision']}")
            st.balloons()
        else:
            st.error(f"## ❌ LOAN {result['decision']}")
        
        # Create metrics row
        col_met1, col_met2, col_met3 = st.columns(3)
        with col_met1:
            st.metric("Decision Confidence", result['confidence'])
        with col_met2:
            st.metric("Risk Score", f"{100 - result['probability_score']*100:.1f}%")
        with col_met3:
            if result['decision'] == "APPROVED":
                st.metric("Recommendation", "✅ Proceed with offer")
            else:
                st.metric("Recommendation", "⚠️ Review manually")
        
        # Show factors affecting decision
        st.markdown("#### 🔍 Key Factors Analyzed:")
        
        # Determine which factors are good/bad
        factors = []
        if credit_score > 650:
            factors.append(("✅ Credit Score", f"{credit_score} (Good)", "positive"))
        else:
            factors.append(("⚠️ Credit Score", f"{credit_score} (Below 650)", "negative"))
        
        income_to_loan_ratio = income / loan_amount if loan_amount > 0 else 0
        if income_to_loan_ratio > 0.3:
            factors.append(("✅ Income-to-Loan Ratio", f"{income_to_loan_ratio:.2f}x (Healthy)", "positive"))
        else:
            factors.append(("⚠️ Income-to-Loan Ratio", f"{income_to_loan_ratio:.2f}x (Low)", "negative"))
        
        if dti < 40:
            factors.append(("✅ Debt-to-Income", f"{dti}% (Good)", "positive"))
        else:
            factors.append(("⚠️ Debt-to-Income", f"{dti}% (High)", "negative"))
        
        for factor, value, status in factors:
            if status == "positive":
                st.markdown(f"- {factor}: {value}")
            else:
                st.markdown(f"- {factor}: {value}")

with col2:
    st.markdown("### 📈 Model Performance")
    st.markdown("Trained on 5,000 historical loan records")
    
    # Display metrics
    st.metric("Model Accuracy", f"{metrics['accuracy']*100:.2f}%")
    st.metric("ROC-AUC Score", f"{metrics['roc_auc']:.4f}")
    st.metric("Cross-Validation Score", f"{metrics['cv_score']*100:.2f}%")
    
    st.markdown("---")
    st.markdown("### 🎯 Best Parameters")
    for param, value in metrics['best_params'].items():
        st.markdown(f"- **{param}**: `{value}`")
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.markdown(f"- Total Records: 5,000")
    st.markdown(f"- Features: 5")
    st.markdown(f"- Target: Loan Approved (Yes/No)")
    st.markdown(f"- Split: 80% Train / 20% Test")

# ============================================
# VISUALIZATIONS SECTION
# ============================================
st.markdown("---")
st.markdown("## 📊 Model Analytics & Insights")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Confusion Matrix", "🎯 ROC Curve", "📊 Feature Importance", "📋 Sample Predictions"])

with tab1:
    # Confusion Matrix
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                xticklabels=['Rejected', 'Approved'],
                yticklabels=['Rejected', 'Approved'])
    ax_cm.set_title('Confusion Matrix - Loan Approval', fontsize=14, fontweight='bold')
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    st.pyplot(fig_cm)
    
    # Calculate metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("True Negatives", tn)
    with col_m2:
        st.metric("False Positives", fp)
    with col_m3:
        st.metric("False Negatives", fn)
    with col_m4:
        st.metric("True Positives", tp)

with tab2:
    # ROC Curve
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    ax_roc.plot(fpr, tpr, 'b-', linewidth=2, label=f'Random Forest (AUC = {metrics["roc_auc"]:.3f})')
    ax_roc.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
    ax_roc.fill_between(fpr, tpr, alpha=0.2)
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title('ROC Curve - Model Performance', fontsize=14, fontweight='bold')
    ax_roc.legend(loc='lower right')
    ax_roc.grid(True, alpha=0.3)
    st.pyplot(fig_roc)
    
    st.info(f"**AUC Interpretation:** The model has {metrics['roc_auc']*100:.1f}% probability of distinguishing between approved and rejected loans. A value above 0.8 indicates excellent performance.")

with tab3:
    # Feature Importance
    feature_names = ['Income', 'Loan Amount', 'Credit Score', 'Employment Years', 'DTI']
    importances = model.feature_importances_
    
    fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
    sorted_idx = np.argsort(importances)
    ax_fi.barh(range(len(importances)), importances[sorted_idx], color='steelblue')
    ax_fi.set_yticks(range(len(importances)))
    ax_fi.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax_fi.set_xlabel('Feature Importance Score', fontsize=12)
    ax_fi.set_title('Feature Importance Analysis', fontsize=14, fontweight='bold')
    ax_fi.grid(True, alpha=0.3)
    st.pyplot(fig_fi)
    
    # Show importance values
    st.markdown("#### Feature Ranking:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        st.progress(imp, text=f"{name}: {imp*100:.1f}%")

with tab4:
    st.markdown("### Sample Test Case Predictions")
    
    # Sample test cases
    sample_cases = [
        {"Income": 75000, "Loan": 15000, "Credit": 720, "Employment": 8, "DTI": 25, "Expected": "Approved"},
        {"Income": 55000, "Loan": 25000, "Credit": 580, "Employment": 2, "DTI": 45, "Expected": "Rejected"},
        {"Income": 120000, "Loan": 50000, "Credit": 780, "Employment": 12, "DTI": 20, "Expected": "Approved"},
        {"Income": 45000, "Loan": 20000, "Credit": 680, "Employment": 4, "DTI": 55, "Expected": "Rejected"},
    ]
    
    sample_results = []
    for case in sample_cases:
        result = predict_loan(case["Income"], case["Loan"], case["Credit"], case["Employment"], case["DTI"])
        sample_results.append({
            **case,
            "Prediction": result['decision'],
            "Confidence": result['confidence']
        })
    
    df_samples = pd.DataFrame(sample_results)
    
    # Color coding for predictions
    def color_prediction(val):
        if val == 'APPROVED':
            return 'background-color: #90EE90'
        return 'background-color: #FFCCCC'
    
    st.dataframe(df_samples.style.applymap(color_prediction, subset=['Prediction']), use_container_width=True)
    
    st.caption("Note: These are sample predictions. Your actual results may vary based on the specific combination of inputs.")

# ============================================
# SIDEBAR - ADDITIONAL INFO
# ============================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Details")
st.sidebar.markdown(f"- **Algorithm:** Random Forest Classifier")
st.sidebar.markdown(f"- **Training Size:** 4,000 samples")
st.sidebar.markdown(f"- **Testing Size:** 1,000 samples")
st.sidebar.markdown(f"- **Features:** 5 input variables")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Approval Criteria")
st.sidebar.markdown("""
- ✅ Credit Score > 650
- ✅ Income > 30% of Loan
- ✅ DTI < 40%
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Deployment")
st.sidebar.markdown("""
- **Framework:** Streamlit
- **ML Library:** Scikit-learn
- **Model Format:** Pickle (.pkl)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📞 Support")
st.sidebar.markdown("For questions or issues, contact the AI team.")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🏦 Loan Approval AI System | Powered by Random Forest Algorithm | Trained on 5,000 Loan Records</p>
        <p style='font-size: 12px'>This is a demonstration system. Final loan decisions should involve human review.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================
# RUN COMMAND (for reference)
# ============================================
# To run this application:
# streamlit run app.py
#
# Or if using python directly:
# python -m streamlit run app.py