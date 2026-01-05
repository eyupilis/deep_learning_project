"""
TEFAS Fund Analysis - Risk-Return Scorer ANN
==============================================
CRISP-DM: Modeling Phase

This module provides a simple feedforward ANN for risk-return scoring.

Why ANN for Scoring (Module 3)?
-------------------------------
- Learns non-linear risk-return relationships
- Auxiliary signal only - final selection is rule-based
- Simple architecture, no black-box complexity
- Interpretable through feature importance approximation

Important Note:
- This model does NOT make investment decisions
- It provides a numerical score for rule-based filtering
- Final portfolio selection uses explicit constraints
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskReturnScorer:
    """
    Simple ANN to predict a risk-adjusted return score.
    
    Used in Module 3 as an auxiliary signal for portfolio construction.
    The final portfolio selection remains rule-based.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [32, 16],
        activation: str = "relu",
        output_activation: str = "linear",
        seed: int = 42,
    ):
        """
        Initialize RiskReturnScorer.
        
        Args:
            input_dim: Number of input features
            hidden_layers: Sizes of hidden layers
            activation: Hidden layer activation
            output_activation: Output layer activation
            seed: Random seed
        """
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.output_activation = output_activation
        
        self.model = self._build_model()
        self.scaler = StandardScaler()
        self.history = None
    
    def _build_model(self) -> Model:
        """Build the scoring network."""
        inputs = layers.Input(shape=(self.input_dim,), name="features")
        
        x = inputs
        for i, units in enumerate(self.hidden_layers):
            x = layers.Dense(units, activation=self.activation, name=f"hidden_{i}")(x)
            x = layers.Dropout(0.3, name=f"dropout_{i}")(x)
        
        # Single output: risk-adjusted score
        outputs = layers.Dense(
            1,
            activation=self.output_activation,
            name="score",
        )(x)
        
        model = Model(inputs=inputs, outputs=outputs, name="risk_return_scorer")
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"],
        )
        
        logger.info(f"Built RiskReturnScorer: {self.input_dim} -> {self.hidden_layers} -> 1")
        
        return model
    
    def _create_target(
        self,
        returns: np.ndarray,
        risk_scores: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Create risk-adjusted return target for training.
        
        Target = alpha * normalized_return - (1 - alpha) * normalized_risk
        
        Args:
            returns: Annualized returns
            risk_scores: Risk scores (higher = riskier)
            alpha: Weight for return vs risk (0.5 = balanced)
            
        Returns:
            Risk-adjusted target scores
        """
        # Normalize both to [0, 1]
        ret_norm = (returns - returns.min()) / (returns.max() - returns.min() + 1e-8)
        risk_norm = (risk_scores - risk_scores.min()) / (risk_scores.max() - risk_scores.min() + 1e-8)
        
        # Higher return is good, higher risk is bad
        target = alpha * ret_norm - (1 - alpha) * risk_norm
        
        return target
    
    def fit(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        risk_scores: np.ndarray,
        epochs: int = 100,
        batch_size: int = 16,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        verbose: int = 1,
    ) -> keras.callbacks.History:
        """
        Train the scorer.
        
        Args:
            features: Feature matrix (samples x features)
            returns: Annualized returns for target creation
            risk_scores: Risk scores for target creation
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Validation fraction
            early_stopping_patience: Early stopping patience
            verbose: Verbosity
            
        Returns:
            Training history
        """
        # Scale features
        X = self.scaler.fit_transform(features)
        
        # Create target
        y = self._create_target(returns, risk_scores)
        
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
            ),
        ]
        
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
        )
        
        return self.history
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict risk-adjusted scores.
        
        Args:
            features: Feature matrix
            
        Returns:
            Array of scores (higher = better risk-adjusted)
        """
        X = self.scaler.transform(features)
        return self.model.predict(X, verbose=0).flatten()
    
    def rank(self, features: np.ndarray) -> np.ndarray:
        """
        Rank funds by predicted score.
        
        Args:
            features: Feature matrix
            
        Returns:
            Array of ranks (0 = best)
        """
        scores = self.predict(features)
        return np.argsort(np.argsort(-scores))  # Descending rank
    
    def summary(self) -> None:
        """Print model summary."""
        self.model.summary()


def create_risk_scorer(
    input_dim: int,
    config: Optional[dict] = None,
) -> RiskReturnScorer:
    """
    Factory function to create RiskReturnScorer with config.
    
    Args:
        input_dim: Number of features
        config: Configuration dict
        
    Returns:
        Configured RiskReturnScorer
    """
    if config is None:
        config = {}
    
    return RiskReturnScorer(
        input_dim=input_dim,
        hidden_layers=config.get("hidden_layers", [32, 16]),
        activation=config.get("activation", "relu"),
        output_activation=config.get("output_activation", "linear"),
    )


if __name__ == "__main__":
    print("RiskReturnScorer module loaded successfully")
    print("\nThis ANN is an AUXILIARY scoring tool.")
    print("Final portfolio selection is RULE-BASED, not model-driven.")
