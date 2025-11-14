# -*- coding: utf-8 -*-


"""Importing libraries and loading dataset"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

def load_data(path='ipl.csv'):
    df = pd.read_csv(path)
    return df

def quick_explore(df):
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nDtypes:\n", df.dtypes)
    print("\nSample rows:\n", df.head())

"""Data Processing"""

def preprocess(df,
               date_col='date',
               batting_col='bat_team',
               bowling_col='bowl_team',
               over_col='overs',
               runs_col='runs',
               wickets_col='wickets',
               runs_last5_col='runs_last_5',
               wickets_last5_col='wickets_last_5',
               total_col_candidates=['total','total_score','inning_total']):
    """
    Returns:
      df_processed: preprocessed DataFrame ready for modeling
    """

    #  Normalize column names (simple)
    df = df.copy()

    # Try to find the column that stores final innings total
    total_col = None
    for cand in total_col_candidates:
        if cand in df.columns:
            total_col = cand
            break
    if total_col is None:
        raise ValueError(f"Could not find a total score column in {total_col_candidates}. Please rename your total column accordingly.")

    # Convert date column to datetime if present
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Drop rows with NA in essential columns
    essential_cols = [batting_col, bowling_col, over_col, runs_col, wickets_col, runs_last5_col, wickets_last5_col, total_col]
    missing_ess = [c for c in essential_cols if c not in df.columns]
    if missing_ess:
        raise ValueError(f"Missing expected columns: {missing_ess}. Adjust the column names or CSV.")

    df = df.dropna(subset=essential_cols)

    # Filter consistent teams:
    # Common IPL teams list — adjust if your dataset uses different names.
    consistent_teams = [
        'Chennai Super Kings','Mumbai Indians','Royal Challengers Bangalore','Kolkata Knight Riders',
        'Rajasthan Royals','Delhi Daredevils','Kings XI Punjab','Sunrisers Hyderabad',
        'Deccan Chargers','Pune Warriors','Gujarat Lions','Rising Pune Supergiant',
        'Rising Pune Supergiants','Delhi Capitals'
    ]
    df = df[df[batting_col].isin(consistent_teams) & df[bowling_col].isin(consistent_teams)].copy()

    # Remove first 5 overs to focus on mid/late innings situations (as in your doc)
    # If overs is fractional (e.g., 5.4), keep only rows where overs >= 5.0
    df = df[df[over_col] >= 5.0]

    # Extract year from date if possible for train/test split
    if date_col in df.columns:
        df['year'] = df[date_col].dt.year
    else:
        # If no date column, try to create 'year' column from other info or fail
        raise ValueError("Date column missing or unparsable; cannot create train/test split by seasons.")

    # Build features and target
    feature_cols = [batting_col, bowling_col, over_col, runs_col, wickets_col, runs_last5_col, wickets_last5_col]
    X = df[feature_cols].copy()
    y = df[total_col].astype(int).copy()

    # Keep year for splitting later
    X['year'] = df['year'].values
    X.index = df.index

    return X, y, batting_col, bowling_col

#Build preprocessing + models pipeline
def build_and_train(X_train, y_train, X_test, y_test, cat_cols=['bat_team','bowl_team']):
    # ColumnTransformer with OneHotEncoder for teams, passthrough numeric
    numeric_cols = [c for c in X_train.columns if c not in cat_cols + ['year']]
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        remainder='passthrough'  # numeric columns keep their order after categorical encodings
    )

    # Models to evaluate
    models = {
        'LinearRegression': LinearRegression(),
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'AdaBoost(LR)': AdaBoostRegressor(estimator=LinearRegression(), n_estimators=50, random_state=42)
    }

    results = {}
    trained_pipelines = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[('pre', preprocessor), ('model', model)])
        pipe.fit(X_train.drop(columns=['year']), y_train)
        preds = pipe.predict(X_test.drop(columns=['year']))
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse}
        trained_pipelines[name] = pipe
        print(f"{name} -> MAE: {mae:.3f}  RMSE: {rmse:.3f}")

    return results, trained_pipelines

  # Evaluate and choose final model
def evaluate_results(results):
    # Display results as DataFrame
    res_df = pd.DataFrame(results).T.sort_values('RMSE')
    print(res_df)   # ✅ works in VS Code
    return res_df


  # Prediction helper (using chosen pipeline
def make_prediction(pipeline, batting_team, bowling_team, overs, runs, wickets, runs_last_5, wickets_last_5, year=None):
    """
    Given pipeline (trained) and feature values, return predicted innings total.
    """
    data = {
        'bat_team': [batting_team],
        'bowl_team': [bowling_team],
        'overs': [overs],
        'runs': [runs],
        'wickets': [wickets],
        'runs_last_5': [runs_last_5],
        'wickets_last_5': [wickets_last_5]
    }
    if year is not None:
        data['year'] = [year]
    else:
        data['year'] = [None]

    df = pd.DataFrame(data)
    # If pipeline expects column order same as training (without 'year' column), drop 'year'
    if 'year' in df.columns:
        df_input = df.drop(columns=['year'])
    else:
        df_input = df
    pred = pipeline.predict(df_input)[0]
    return int(round(pred))

  # Putting it all together
def run_full_pipeline(csv_path='ipl.csv'):
    print("Loading data...")
    df = load_data(csv_path)
    quick_explore(df)

    print("\nPreprocessing...")
    X, y, bat_col, bowl_col = preprocess(df,
                                         date_col='date',
                                         batting_col='bat_team',
                                         bowling_col='bowl_team',
                                         over_col='overs',
                                         runs_col='runs',
                                         wickets_col='wickets',
                                         runs_last5_col='runs_last_5',
                                         wickets_last5_col='wickets_last_5',
                                         total_col_candidates=['total','total_score','inning_total','total_runs','score','inning_total'])
    # Split train/test by year: train <= 2016, test == 2017 (as per your doc)
    X_train = X[X['year'].between(2008, 2016)].copy()
    y_train = y.loc[X_train.index]
    X_test = X[X['year'] == 2017].copy()
    y_test = y.loc[X_test.index]

    if X_test.shape[0] == 0 or X_train.shape[0] == 0:
        print("Warning: after filtering by years, training or test set is empty. Check 'date' parsing and year ranges.")
        # fallback: random split
        X_train, X_test, y_train, y_test = train_test_split(X.drop(columns=['year']), y, test_size=0.2, random_state=42)
    else:
        # drop 'year' column for model training as it's not a predictive feature for scores (we used it only for split)
        pass

    print(f"\nTraining set size: {X_train.shape[0]} rows. Test set size: {X_test.shape[0]} rows.")

    # Train models
    print("\nTraining models...")
    results, pipelines = build_and_train(X_train, y_train, X_test, y_test, cat_cols=[bat_col, bowl_col])

    print("\nEvaluation summary:")
    res_df = evaluate_results(results)

    # Choose best model (lowest RMSE)
    best_model_name = res_df.index[0]
    best_pipeline = pipelines[best_model_name]
    print(f"\nBest model: {best_model_name}")

    return {
        'results': res_df,
        'pipelines': pipelines,
        'best_model_name': best_model_name,
        'best_pipeline': best_pipeline,
        'X_test': X_test,
        'y_test': y_test
    }
  # Example usage and sample predictions
if __name__ == '__main__':
    # Run the pipeline (this will print progress and results)
    artifacts = run_full_pipeline('ipl.csv')

    # Example predictions (format: batting_team, bowling_team, overs, runs, wickets, runs_last_5, wickets_last_5)
    best_pipe = artifacts['best_pipeline']
    examples = [
        ('Kolkata Knight Riders', 'Delhi Daredevils', 15.0, 120, 3, 40, 1),
        ('Sunrisers Hyderabad', 'Royal Challengers Bangalore', 12.0, 80, 4, 30, 2),
        ('Mumbai Indians', 'Kings XI Punjab', 16.0, 140, 2, 45, 1),
        ('Rajasthan Royals', 'Chennai Super Kings', 14.0, 105, 3, 32, 1)
    ]

    print("\nExample Predictions (using best model):")
    for ex in examples:
        pred = make_prediction(best_pipe, *ex, year=2019)
        print(f"{ex[0]} vs {ex[1]} at over {ex[2]} -> Predicted final score: {pred}")

    # Optionally: plot residuals for best model on test set
    X_test = artifacts['X_test'].drop(columns=['year'])
    y_test = artifacts['y_test']
    preds = best_pipe.predict(X_test)
    residuals = y_test.values - preds

    plt.figure(figsize=(8,5))
    sns.histplot(residuals, kde=True)
    plt.title('Residuals distribution (actual - predicted)')
    plt.xlabel('Residual')
    plt.show()

    # Print metrics for best model
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"\nBest model test MAE: {mae:.3f}, RMSE: {rmse:.3f}")

from sklearn.model_selection import GridSearchCV

def tune_models(X_train, y_train):
    print("\n Hyperparameter tuning started...\n")

    # Drop 'year' column if present
    if 'year' in X_train.columns:
        X_train = X_train.drop(columns=['year'])

    # Define preprocessing
    cat_cols = ['bat_team', 'bowl_team']
    numeric_cols = [c for c in X_train.columns if c not in cat_cols]
    preprocessor = ColumnTransformer(
        transformers=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols)],
        remainder='passthrough'
    )

    # 1 Decision Tree Grid Search
    tree_params = {
        'model__max_depth': [5, 10, 15, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }

    tree_pipe = Pipeline(steps=[('pre', preprocessor),
                                ('model', DecisionTreeRegressor(random_state=42))])

    tree_grid = GridSearchCV(tree_pipe, tree_params, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    tree_grid.fit(X_train, y_train)
    print(" Best Decision Tree Parameters:", tree_grid.best_params_)

    # 2 Random Forest Grid Search
    rf_params = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [5, 10, None],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2]
    }

    rf_pipe = Pipeline(steps=[('pre', preprocessor),
                              ('model', RandomForestRegressor(random_state=42))])

    rf_grid = GridSearchCV(rf_pipe, rf_params, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    print(" Best Random Forest Parameters:", rf_grid.best_params_)

    return tree_grid.best_estimator_, rf_grid.best_estimator_

"""Save trained model"""

import joblib

joblib.dump(best_pipe, 'ipl_score_model.pkl')
print(" Model saved as 'ipl_score_model.pkl'")

"""Runing Model"""

import joblib
import pandas as pd

#  Load the saved model pipeline
ipl_score_model = joblib.load('ipl_score_model.pkl')

#  Define prediction function
def predict_score(batting_team, bowling_team, overs, runs, wickets, runs_last_5, wickets_last_5):
    # Prepare input as DataFrame
    input_data = pd.DataFrame({
        'bat_team': [batting_team],
        'bowl_team': [bowling_team],
        'overs': [overs],
        'runs': [runs],
        'wickets': [wickets],
        'runs_last_5': [runs_last_5],
        'wickets_last_5': [wickets_last_5]
    })

    #  Model is a pipeline (includes preprocessing)
    predicted_score = ipl_score_model.predict(input_data)[0]

    print(f"🏏 Predicted Final Score: {int(predicted_score)} runs")

"""Example"""

predict_score('Mumbai Indians', 'Chennai Super Kings', overs=11, runs=90, wickets=3, runs_last_5=65, wickets_last_5=2)