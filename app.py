#!/usr/bin/env python3
"""
NBA Lineup Efficiency Predictor - Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from utils.ee_utils import DataProcessor
from utils.ml_utils import BayesianNetworkModel
import os
import json

# Page configuration
st.set_page_config(
    page_title="NBA Lineup Efficiency Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-result {
        background: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .status-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'model' not in st.session_state:
    st.session_state.model = BayesianNetworkModel()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'data_stats' not in st.session_state:
    st.session_state.data_stats = {}

def update_status():
    """Update system status"""
    return {
        'data_loaded': st.session_state.data_loaded,
        'model_trained': st.session_state.model_trained,
        'data_count': len(st.session_state.data_processor.processed_data) if st.session_state.data_loaded else 0,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def load_data():
    """Load existing data"""
    try:
        df = st.session_state.data_processor.load_sample_data()
        st.session_state.data_loaded = True
        st.session_state.data_stats = st.session_state.data_processor.get_data_stats()
        st.success("Data loaded successfully!")
        return True
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return False

def download_new_data():
    """Download new data from NBA API"""
    try:
        df = st.session_state.data_processor.download_nba_data()
        st.session_state.data_loaded = True
        st.session_state.data_stats = st.session_state.data_processor.get_data_stats()
        st.success("New data downloaded successfully!")
        return True
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return False

def train_model():
    """Train the Bayesian Network model"""
    if not st.session_state.data_loaded:
        st.error("Please load data first!")
        return False
    
    try:
        with st.spinner("Training model..."):
            processed_df, preprocessing_metadata = st.session_state.data_processor.preprocess_data(
                st.session_state.data_processor.processed_data
            )
            
            results = st.session_state.model.train_model(processed_df, preprocessing_metadata)
            st.session_state.model_trained = True
            
            st.success(f"Model trained successfully! Accuracy: {results['accuracy']:.3f}")
            return True
    except Exception as e:
        st.error(f"Error training model: {e}")
        return False

def create_visualization():
    """Create talent distribution visualization"""
    if not st.session_state.data_loaded:
        return None
    
    try:
        df = st.session_state.data_processor.processed_data
        
        # Create talent distribution plot
        talent_cols = ['scoring_talent', 'playmaking_talent', 'rebounding_talent', 'defensive_talent']
        talent_data = []
        
        for col in talent_cols:
            counts = df[col].value_counts()
            for talent, count in counts.items():
                talent_data.append({
                    'Talent Type': col.replace('_talent', '').title(),
                    'Level': talent,
                    'Count': count
                })
        
        talent_df = pd.DataFrame(talent_data)
        
        fig = px.bar(
            talent_df, 
            x='Talent Type', 
            y='Count', 
            color='Level',
            title='Talent Distribution Across Lineups',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig.update_layout(
            xaxis_title="Talent Type",
            yaxis_title="Number of Lineups",
            legend_title="Talent Level"
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        return None

# Main app
def main():
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1>🏀 NBA Lineup Efficiency Predictor</h1>
        <p class="lead">Using Bayesian Networks to predict NBA lineup performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status Section
    st.header("System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    status = update_status()
    
    with col1:
        st.markdown(f"""
        <div class="status-card">
            <strong>Data Loaded:</strong><br>
            <span style="color: {'green' if status['data_loaded'] else 'red'}">
                {'✅ Yes' if status['data_loaded'] else '❌ No'}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="status-card">
            <strong>Model Trained:</strong><br>
            <span style="color: {'green' if status['model_trained'] else 'red'}">
                {'✅ Yes' if status['model_trained'] else '❌ No'}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="status-card">
            <strong>Data Points:</strong><br>
            {status['data_count']}
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="status-card">
            <strong>Last Update:</strong><br>
            {status['timestamp']}
        </div>
        """, unsafe_allow_html=True)
    
    # Control Section
    st.header("Data & Model Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📥 Load Existing Data", use_container_width=True):
            load_data()
    
    with col2:
        if st.button("🔄 Download New Data", use_container_width=True):
            download_new_data()
    
    with col3:
        if st.button("🤖 Train Model", use_container_width=True):
            train_model()
    
    with col4:
        if st.button("📊 Generate Visualization", use_container_width=True):
            fig = create_visualization()
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    # Prediction Section
    st.header("Lineup Efficiency Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Lineup Talent Assessment")
        st.markdown("Rate your lineup's talent levels")
        
        scoring = st.selectbox(
            "Scoring Talent",
            ["Low", "Medium", "High", "Very High"],
            index=2
        )
        
        playmaking = st.selectbox(
            "Playmaking Talent", 
            ["Low", "Medium", "High", "Very High"],
            index=1
        )
        
        rebounding = st.selectbox(
            "Rebounding Talent",
            ["Low", "Medium", "High", "Very High"],
            index=1
        )
        
        defensive = st.selectbox(
            "Defensive Talent",
            ["Low", "Medium", "High", "Very High"],
            index=1
        )
        
        if st.button("🎯 Predict Lineup Efficiency", use_container_width=True):
            if not st.session_state.model_trained:
                st.error("Please train the model first!")
            else:
                with st.spinner("Making prediction..."):
                    features = {
                        'scoring': scoring,
                        'playmaking': playmaking,
                        'rebounding': rebounding,
                        'defensive': defensive
                    }
                    
                    result = st.session_state.model.predict(features)
                    
                    if result['success']:
                        prediction = result['prediction']
                        
                        st.markdown("""
                        <div class="prediction-result">
                        """, unsafe_allow_html=True)
                        
                        st.subheader("Prediction Result")
                        
                        # Main prediction
                        efficiency_color = {
                            'Very Poor': 'red',
                            'Poor': 'orange', 
                            'Good': 'blue',
                            'Excellent': 'green'
                        }[prediction['predicted_efficiency']]
                        
                        st.markdown(f"""
                        <h3 style='color: {efficiency_color}'>
                            Predicted Efficiency: <strong>{prediction['predicted_efficiency']}</strong>
                        </h3>
                        <p>Confidence: <strong>{(prediction['probability'] * 100):.1f}%</strong></p>
                        """, unsafe_allow_html=True)
                        
                        # Probability breakdown
                        st.subheader("Probability Breakdown")
                        
                        prob_data = []
                        for eff, prob in prediction['all_probabilities'].items():
                            prob_data.append({
                                'Efficiency': eff,
                                'Probability': prob * 100
                            })
                        
                        prob_df = pd.DataFrame(prob_data)
                        
                        fig_prob = px.bar(
                            prob_df,
                            x='Efficiency',
                            y='Probability',
                            color='Efficiency',
                            color_discrete_map={
                                'Very Poor': 'red',
                                'Poor': 'orange',
                                'Good': 'blue', 
                                'Excellent': 'green'
                            },
                            text='Probability'
                        )
                        
                        fig_prob.update_traces(
                            texttemplate='%{text:.1f}%',
                            textposition='outside'
                        )
                        fig_prob.update_layout(
                            yaxis_title="Probability (%)",
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_prob, use_container_width=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.error(f"Prediction failed: {result['message']}")
    
    with col2:
        # Visualization area
        if st.session_state.data_loaded:
            st.subheader("Data Overview")
            
            # Show data statistics
            if st.session_state.data_stats:
                st.write("**Efficiency Distribution:**")
                eff_data = st.session_state.data_stats.get('efficiency_distribution', {})
                eff_df = pd.DataFrame(list(eff_data.items()), columns=['Efficiency', 'Count'])
                st.dataframe(eff_df, use_container_width=True, hide_index=True)
        
        # About section
        st.markdown("---")
        st.subheader("About This System")
        st.markdown("""
        This system uses Bayesian Networks to predict NBA lineup efficiency based on:
        
        - **Scoring Talent:** Points per game adjusted by true shooting percentage
        - **Playmaking Talent:** Assists per game  
        - **Rebounding Talent:** Rebounds per game
        - **Defensive Talent:** Combined steals and blocks per game
        
        The model is trained on NBA lineup data and uses probabilistic inference
        to predict lineup performance.
        """)

if __name__ == "__main__":
    main()