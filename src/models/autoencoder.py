"""
TEFAS Fund Analysis - Autoencoder Models
==========================================
CRISP-DM: Modeling Phase

This module provides Autoencoder architectures for:
1. Module 1: Anomaly detection via reconstruction error
2. Module 2: Embedding extraction for correlation mapping

Why Autoencoder?
----------------
- Unsupervised learning: No labeled anomalies needed
- Reconstruction error naturally identifies outliers
- Bottleneck layer provides compressed representation
- Well-established in financial anomaly detection literature
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


class RiskAutoencoder:
    """
    Autoencoder for fund risk anomaly detection.
    
    Used in Module 1 to identify funds with unusual risk profiles.
    Funds with high reconstruction error are flagged as anomalies.
    """
    
    def __init__(
        self,
        input_dim: int,
        encoding_dims: List[int] = [32, 16, 8],
        latent_dim: int = 4,
        activation: str = "relu",
        seed: int = 42,
    ):
        """
        Initialize RiskAutoencoder.
        
        Args:
            input_dim: Number of input features
            encoding_dims: Hidden layer sizes for encoder
            latent_dim: Size of latent (bottleneck) layer
            activation: Activation function
            seed: Random seed
        """
        set_seed(seed)
        
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.latent_dim = latent_dim
        self.activation = activation
        
        self.model = self._build_model()
        self.encoder = self._build_encoder()
        self.history = None
    
    def _build_model(self) -> Model:
        """Build the full autoencoder model."""
        # Input layer
        inputs = layers.Input(shape=(self.input_dim,), name="input")
        
        # Encoder
        x = inputs
        for i, dim in enumerate(self.encoding_dims):
            x = layers.Dense(
                dim,
                activation=self.activation,
                name=f"encoder_{i}",
            )(x)
            x = layers.BatchNormalization(name=f"bn_enc_{i}")(x)
        
        # Latent space
        latent = layers.Dense(
            self.latent_dim,
            activation=self.activation,
            name="latent",
        )(x)
        
        # Decoder (mirror of encoder)
        x = latent
        for i, dim in enumerate(reversed(self.encoding_dims)):
            x = layers.Dense(
                dim,
                activation=self.activation,
                name=f"decoder_{i}",
            )(x)
            x = layers.BatchNormalization(name=f"bn_dec_{i}")(x)
        
        # Output layer
        outputs = layers.Dense(
            self.input_dim,
            activation="linear",
            name="output",
        )(x)
        
        model = Model(inputs=inputs, outputs=outputs, name="risk_autoencoder")
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"],
        )
        
        logger.info(f"Built RiskAutoencoder: {self.input_dim} -> {self.latent_dim} -> {self.input_dim}")
        
        return model
    
    def _build_encoder(self) -> Model:
        """Extract encoder part for embedding extraction."""
        latent_layer = self.model.get_layer("latent")
        encoder = Model(
            inputs=self.model.input,
            outputs=latent_layer.output,
            name="encoder",
        )
        return encoder
    
    def fit(
        self,
        X: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        verbose: int = 1,
    ) -> keras.callbacks.History:
        """
        Train the autoencoder.
        
        Args:
            X: Training data (samples x features)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction for validation
            early_stopping_patience: Patience for early stopping
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            ),
        ]
        
        self.history = self.model.fit(
            X, X,  # Autoencoder: input = target
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
        )
        
        logger.info(
            f"Training complete. Final loss: {self.history.history['loss'][-1]:.6f}"
        )
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct input data."""
        return self.model.predict(X, verbose=0)
    
    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction error (MSE) for each sample.
        
        High error indicates anomaly.
        
        Args:
            X: Input data
            
        Returns:
            Array of reconstruction errors per sample
        """
        reconstructed = self.predict(X)
        mse = np.mean((X - reconstructed) ** 2, axis=1)
        return mse
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Extract latent representations."""
        return self.encoder.predict(X, verbose=0)
    
    def detect_anomalies(
        self,
        X: np.ndarray,
        threshold_percentile: float = 95,
    ) -> Tuple[np.ndarray, float]:
        """
        Detect anomalies based on reconstruction error.
        
        Args:
            X: Input data
            threshold_percentile: Percentile threshold for anomaly
            
        Returns:
            Tuple of (boolean anomaly mask, threshold value)
        """
        errors = self.reconstruction_error(X)
        threshold = np.percentile(errors, threshold_percentile)
        anomalies = errors > threshold
        
        n_anomalies = anomalies.sum()
        logger.info(
            f"Detected {n_anomalies} anomalies ({n_anomalies/len(X)*100:.1f}%) "
            f"at {threshold_percentile}th percentile (threshold={threshold:.6f})"
        )
        
        return anomalies, threshold
    
    def summary(self) -> None:
        """Print model summary."""
        self.model.summary()
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")


class EmbeddingAutoencoder:
    """
    Autoencoder for extracting fund embeddings from return time series.
    
    Used in Module 2 to learn compressed representations of fund behavior.
    The latent space reveals hidden correlations and clustering structure.
    """
    
    def __init__(
        self,
        input_dim: int,
        encoding_dims: List[int] = [64, 32],
        latent_dim: int = 2,
        activation: str = "relu",
        seed: int = 42,
    ):
        """
        Initialize EmbeddingAutoencoder.
        
        Args:
            input_dim: Number of time steps (return series length)
            encoding_dims: Hidden layer sizes
            latent_dim: Embedding dimension (2 for visualization)
            activation: Activation function
            seed: Random seed
        """
        set_seed(seed)
        
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.latent_dim = latent_dim
        self.activation = activation
        
        self.model = self._build_model()
        self.encoder = self._build_encoder()
        self.history = None
    
    def _build_model(self) -> Model:
        """Build the embedding autoencoder."""
        inputs = layers.Input(shape=(self.input_dim,), name="input")
        
        # Encoder
        x = inputs
        for i, dim in enumerate(self.encoding_dims):
            x = layers.Dense(dim, activation=self.activation, name=f"enc_{i}")(x)
            x = layers.Dropout(0.2, name=f"drop_enc_{i}")(x)
        
        # Latent embedding
        latent = layers.Dense(
            self.latent_dim,
            activation="linear",  # Linear for interpretable embeddings
            name="embedding",
        )(x)
        
        # Decoder
        x = latent
        for i, dim in enumerate(reversed(self.encoding_dims)):
            x = layers.Dense(dim, activation=self.activation, name=f"dec_{i}")(x)
            x = layers.Dropout(0.2, name=f"drop_dec_{i}")(x)
        
        outputs = layers.Dense(self.input_dim, activation="linear", name="output")(x)
        
        model = Model(inputs=inputs, outputs=outputs, name="embedding_autoencoder")
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
        )
        
        logger.info(f"Built EmbeddingAutoencoder: {self.input_dim} -> {self.latent_dim}")
        
        return model
    
    def _build_encoder(self) -> Model:
        """Extract encoder for embedding generation."""
        embedding_layer = self.model.get_layer("embedding")
        return Model(
            inputs=self.model.input,
            outputs=embedding_layer.output,
            name="embedding_encoder",
        )
    
    def fit(
        self,
        X: np.ndarray,
        epochs: int = 150,
        batch_size: int = 16,
        validation_split: float = 0.2,
        early_stopping_patience: int = 15,
        verbose: int = 1,
    ) -> keras.callbacks.History:
        """Train the autoencoder."""
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
            ),
        ]
        
        self.history = self.model.fit(
            X, X,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
        )
        
        return self.history
    
    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """
        Get 2D embeddings for visualization.
        
        Args:
            X: Return series (funds x time_steps)
            
        Returns:
            2D embeddings (funds x latent_dim)
        """
        return self.encoder.predict(X, verbose=0)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct return series."""
        return self.model.predict(X, verbose=0)


def create_risk_autoencoder(
    input_dim: int,
    config: Optional[dict] = None,
) -> RiskAutoencoder:
    """
    Factory function to create RiskAutoencoder with config.
    
    Args:
        input_dim: Number of features
        config: Configuration dict (from config.py)
        
    Returns:
        Configured RiskAutoencoder
    """
    if config is None:
        config = {}
    
    return RiskAutoencoder(
        input_dim=input_dim,
        encoding_dims=config.get("encoding_dims", [32, 16, 8]),
        latent_dim=config.get("latent_dim", 4),
        activation=config.get("activation", "relu"),
    )


def create_embedding_autoencoder(
    input_dim: int,
    config: Optional[dict] = None,
) -> EmbeddingAutoencoder:
    """
    Factory function to create EmbeddingAutoencoder with config.
    
    Args:
        input_dim: Length of return series
        config: Configuration dict
        
    Returns:
        Configured EmbeddingAutoencoder
    """
    if config is None:
        config = {}
    
    return EmbeddingAutoencoder(
        input_dim=input_dim,
        encoding_dims=config.get("encoding_dims", [64, 32]),
        latent_dim=config.get("latent_dim", 2),
        activation=config.get("activation", "relu"),
    )


if __name__ == "__main__":
    print("Autoencoder module loaded successfully")
    print("\nAvailable classes:")
    print("- RiskAutoencoder: For anomaly detection (Module 1)")
    print("- EmbeddingAutoencoder: For embedding extraction (Module 2)")
