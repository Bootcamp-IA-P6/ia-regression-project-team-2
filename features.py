import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class HouseFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        if 'plantaN' in X_copy.columns and 'ascensor' in X_copy.columns:
            X_copy['plantaN'] = pd.to_numeric(X_copy['plantaN'], errors='coerce').fillna(0)
            X_copy['ascensor'] = pd.to_numeric(X_copy['ascensor'], errors='coerce').fillna(0)
            X_copy['accessibility_index'] = (X_copy['plantaN'] ** 2) * (1 - X_copy['ascensor'])
        else:
            X_copy['accessibility_index'] = 0
            
        if 'habitaciones' in X_copy.columns and 'metros' in X_copy.columns:
            X_copy['rooms_per_meter'] = X_copy['habitaciones'] / (X_copy['metros'] + 1e-5)
            X_copy['avg_room_size'] = X_copy['metros'] / (X_copy['habitaciones'] + 1e-5)
            
        return X_copy
