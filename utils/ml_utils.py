"""
Machine learning utilities for Bayesian Network model.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BayesianNetworkModel:
    """Bayesian Network model for NBA lineup efficiency prediction."""
    
    def __init__(self):
        self.model_trained = False
        self.model = None
        self.preprocessing_metadata = None
        self.accuracy = 0.0
        
    def train_model(self, df: pd.DataFrame, preprocessing_metadata: Dict) -> Dict:
        """
        Train Bayesian Network model on the processed data.
        Returns training results.
        """
        logger.info("Training Bayesian Network model...")
        
        try:
            # For demonstration, we'll use a simple probabilistic approach
            # In a real implementation, you might use libraries like pgmpy
            
            # Calculate conditional probabilities
            feature_cols = preprocessing_metadata['feature_columns']
            target_col = preprocessing_metadata['target_column']
            
            # Create probability tables
            self.probability_tables = self._build_probability_tables(df, feature_cols, target_col)
            self.preprocessing_metadata = preprocessing_metadata
            
            # Evaluate model
            X = df[feature_cols]
            y = df[target_col]
            
            # Simple prediction based on maximum probability
            predictions = []
            for _, row in X.iterrows():
                pred = self._predict_single(row)
                predictions.append(pred)
            
            self.accuracy = accuracy_score(y, predictions)
            self.model_trained = True
            
            results = {
                'accuracy': self.accuracy,
                'classification_report': classification_report(y, predictions, output_dict=True),
                'probability_tables': {k: len(v) for k, v in self.probability_tables.items()}
            }
            
            logger.info(f"Model training completed with accuracy: {self.accuracy:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def _build_probability_tables(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Dict:
        """Build probability tables for Bayesian Network."""
        probability_tables = {}
        
        # Calculate prior probabilities for target
        target_probs = df[target_col].value_counts(normalize=True).to_dict()
        probability_tables['prior'] = target_probs
        
        # Calculate conditional probabilities for each feature given target
        for feature in feature_cols:
            conditional_probs = {}
            for target_val in df[target_col].unique():
                subset = df[df[target_col] == target_val]
                feature_probs = subset[feature].value_counts(normalize=True).to_dict()
                conditional_probs[target_val] = feature_probs
            probability_tables[feature] = conditional_probs
        
        return probability_tables
    
    def _predict_single(self, features: pd.Series) -> int:
        """Predict single instance using Bayesian inference."""
        if not self.model_trained:
            raise ValueError("Model not trained")
        
        # Calculate posterior probabilities for each target class
        posterior_probs = {}
        
        for target_val, prior_prob in self.probability_tables['prior'].items():
            likelihood = 1.0
            
            for feature, feature_val in features.items():
                if (feature in self.probability_tables and 
                    target_val in self.probability_tables[feature] and
                    feature_val in self.probability_tables[feature][target_val]):
                    
                    likelihood *= self.probability_tables[feature][target_val][feature_val]
                else:
                    # Laplace smoothing for unseen combinations
                    likelihood *= 0.001
            
            posterior_probs[target_val] = prior_prob * likelihood
        
        # Normalize probabilities
        total = sum(posterior_probs.values())
        if total > 0:
            for key in posterior_probs:
                posterior_probs[key] /= total
        
        # Return prediction with highest probability
        return max(posterior_probs.items(), key=lambda x: x[1])[0]
    
    def predict(self, features_dict: Dict) -> Dict[str, Any]:
        """
        Predict lineup efficiency given talent levels.
        
        Args:
            features_dict: Dictionary with talent levels
                {
                    'scoring': 'Low'|'Medium'|'High'|'Very High',
                    'playmaking': 'Low'|'Medium'|'High'|'Very High', 
                    'rebounding': 'Low'|'Medium'|'High'|'Very High',
                    'defensive': 'Low'|'Medium'|'High'|'Very High'
                }
        
        Returns:
            Dictionary with prediction results
        """
        if not self.model_trained:
            return {'success': False, 'message': 'Model not trained'}
        
        try:
            # Convert input to encoded features
            talent_mapping = self.preprocessing_metadata['talent_mapping']
            efficiency_mapping = self.preprocessing_metadata['efficiency_mapping']
            reverse_efficiency_mapping = {v: k for k, v in efficiency_mapping.items()}
            
            encoded_features = {
                'scoring_talent_encoded': talent_mapping[features_dict['scoring']],
                'playmaking_talent_encoded': talent_mapping[features_dict['playmaking']],
                'rebounding_talent_encoded': talent_mapping[features_dict['rebounding']],
                'defensive_talent_encoded': talent_mapping[features_dict['defensive']]
            }
            
            # Calculate probabilities for all target classes
            posterior_probs = {}
            features_series = pd.Series(encoded_features)
            
            for target_val, prior_prob in self.probability_tables['prior'].items():
                likelihood = 1.0
                
                for feature, feature_val in features_series.items():
                    if (feature in self.probability_tables and 
                        target_val in self.probability_tables[feature] and
                        feature_val in self.probability_tables[feature][target_val]):
                        
                        likelihood *= self.probability_tables[feature][target_val][feature_val]
                    else:
                        likelihood *= 0.001
                
                posterior_probs[target_val] = prior_prob * likelihood
            
            # Normalize probabilities
            total = sum(posterior_probs.values())
            if total > 0:
                for key in posterior_probs:
                    posterior_probs[key] /= total
            
            # Get prediction
            predicted_class = max(posterior_probs.items(), key=lambda x: x[1])[0]
            predicted_efficiency = reverse_efficiency_mapping[predicted_class]
            
            # Convert all probabilities to readable format
            all_probabilities = {
                reverse_efficiency_mapping[k]: v for k, v in posterior_probs.items()
            }
            
            return {
                'success': True,
                'prediction': {
                    'predicted_efficiency': predicted_efficiency,
                    'probability': posterior_probs[predicted_class],
                    'all_probabilities': all_probabilities
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'success': False, 'message': f'Prediction error: {str(e)}'}
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        if not self.model_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'probability_tables': self.probability_tables,
            'preprocessing_metadata': self.preprocessing_metadata,
            'accuracy': self.accuracy
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.probability_tables = model_data['probability_tables']
        self.preprocessing_metadata = model_data['preprocessing_metadata']
        self.accuracy = model_data['accuracy']
        self.model_trained = True
        
        logger.info(f"Model loaded from {filepath}")