import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
import joblib
import logging
from typing import Dict, Tuple, Any, Optional
from config.settings import Config

class OilPricePredictor:
    
    def __init__(self):
        self.models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=1000,
                learning_rate=0.03,
                max_depth=6,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=Config.RANDOM_STATE,
                early_stopping_rounds=50,
                objective='reg:squarederror',
                eval_metric='rmse'
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.03,
                max_depth=6,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=Config.RANDOM_STATE,
                objective='regression',
                metric='rmse',
                verbosity=-1
            )
        }
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        self.scaler = RobustScaler()
        self.logger = logging.getLogger(__name__)
    
    def add_lagged_features(self, df: pd.DataFrame, col: str, max_lag: int = 10) -> pd.DataFrame:
        for lag in range(1, max_lag + 1):
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        return df
    
    def add_rolling_features(self, df: pd.DataFrame, col: str, 
                           windows: Tuple[int, ...] = (5, 10, 20)) -> pd.DataFrame:
        for w in windows:
            df[f"{col}_roll_mean_{w}"] = df[col].rolling(w).mean()
            df[f"{col}_roll_std_{w}"] = df[col].rolling(w).std()
            df[f"{col}_roll_vol_{w}"] = df[col].rolling(w).std() / df[col].rolling(w).mean()
        return df
    
    def prepare_data(self, features: pd.DataFrame, 
                     target_column: str = 'bz_price',
                     signal_col: str = 'uncertainty_index',
                     max_lag: int = 10) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:

        df = features.copy()
        df = self.add_lagged_features(df, signal_col, max_lag)
        df = self.add_rolling_features(df, signal_col)
        df['ret_1d'] = np.log(df[target_column]).diff().shift(-1)
        df = df.dropna()
        
        price_cols = [col for col in df.columns if any(x in col.lower() for x in ['price', 'ret_1d'])]
        X_df = df.drop(columns=price_cols)
        y = df['ret_1d'].values
        
        X = X_df.values
        return X, y, df
    
    def train_models(self, X: np.ndarray, y: np.ndarray, 
                     df: pd.DataFrame, verbose: bool = True) -> Dict[str, Dict]:
        n = len(X)
        split_idx = int(n * Config.TRAIN_TEST_SPLIT)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        tscv = TimeSeriesSplit(n_splits=5)
        results = {}
        
        for name, model in self.models.items():
            self.logger.info(f"Training {name}...")
            
            cv_scores = []
            best_iter = 0
            
            for train_idx, val_idx in tscv.split(X_train_scaled):
                if name == 'xgboost':
                    dtrain = xgb.DMatrix(X_train_scaled[train_idx], label=y_train[train_idx])
                    dval = xgb.DMatrix(X_train_scaled[val_idx], label=y_train[val_idx])
                    cv_model = xgb.train(
                        model.get_params(),
                        dtrain,
                        num_boost_round=1000,
                        evals=[(dval, 'val')],
                        early_stopping_rounds=50,
                        verbose_eval=False
                    )
                    best_iter = cv_model.best_iteration
                else:  # LightGBM
                    train_data = lgb.Dataset(X_train_scaled[train_idx], label=y_train[train_idx])
                    val_data = lgb.Dataset(X_train_scaled[val_idx], label=y_train[val_idx])
                    cv_model = lgb.train(
                        model.params,
                        train_data,
                        num_boost_round=1000,
                        valid_sets=[val_data],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                    )
                    best_iter = cv_model.best_iteration
                
                cv_pred = cv_model.predict(X_train_scaled[val_idx], num_iteration=cv_model.best_iteration)
                cv_scores.append(mean_squared_error(y_train[val_idx], cv_pred))
            
            if name == 'xgboost':
                dtrain_final = xgb.DMatrix(X_train_scaled, label=y_train)
                final_model = xgb.train(
                    model.get_params(),
                    dtrain_final,
                    num_boost_round=best_iter,
                    verbose_eval=False
                )
                y_pred_train = final_model.predict(xgb.DMatrix(X_train_scaled))
                y_pred_test = final_model.predict(xgb.DMatrix(X_test_scaled))
            else:
                train_data_final = lgb.Dataset(X_train_scaled, label=y_train)
                final_model = lgb.train(
                    model.params,
                    train_data_final,
                    num_boost_round=best_iter,
                    callbacks=[lgb.log_evaluation(0)]
                )
                y_pred_train = final_model.predict(X_train_scaled, num_iteration=best_iter)
                y_pred_test = final_model.predict(X_test_scaled, num_iteration=best_iter)
            
            results[name] = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'cv_rmse': np.sqrt(np.mean(cv_scores)),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'model': final_model,
                'best_iteration': best_iter
            }
            
            if verbose:
                self.logger.info(f"{name} - Test RMSE: {results[name]['test_rmse']:.4f}, "
                               f"R²: {results[name]['test_r2']:.4f}, Iterations: {best_iter}")
            
            if hasattr(final_model, 'feature_importances_'):
                self.feature_importance = final_model.feature_importances_
            elif hasattr(final_model, 'feature_name_'):
                self.feature_importance = final_model.feature_importance('gain')
        
        best_name = min(results.keys(), key=lambda x: results[x]['test_rmse'])
        self.best_model = results[best_name]['model']
        self.best_model_name = best_name
        
        self.logger.info(f"Best model: {best_name} (Test RMSE: {results[best_name]['test_rmse']:.4f})")
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        
        if self.best_model_name == 'xgboost':
            return self.best_model.predict(xgb.DMatrix(X_scaled))
        else:
            return self.best_model.predict(X_scaled)
    
    def feature_importance_df(self, feature_names: list) -> pd.DataFrame:
        if self.feature_importance is None:
            return None
        imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        return imp_df
    
    def save_model(self, filepath: str):
        if self.best_model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model_name': self.best_model_name,
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        model_data = joblib.load(filepath)
        self.best_model_name = model_data['model_name']
        self.best_model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_importance = model_data.get('feature_importance')
        self.logger.info(f"Model loaded from {filepath}")
