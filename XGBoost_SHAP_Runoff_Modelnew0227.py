import pandas as pd
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.cluster import KMeans
import xgboost as xgb
import logging
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from bayes_opt import BayesianOptimization
import shap
import re
from datetime import datetime
import matplotlib.dates as mdates

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

#%% Function definitions
#% Load meteorological station data
def load_meteo_data(station_list, meteo_dir, variables):
    meteo_dfs = []
    for station in station_list:
        file_path = meteo_dir / f"{station}.csv"
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df['station'] = station
        df = df[['date', 'station'] + variables]
        meteo_dfs.append(df)
    return pd.concat(meteo_dfs, ignore_index=True)

def weighted_average_meteo(df, stations, stationinfo_file, prefix, meteo_vars, dist_type='center'):
    distance_df = pd.read_excel(stationinfo_file)
    distances = dict(zip(distance_df['No'].astype(str), zip(distance_df['DIST2CENTER'], distance_df['DIST2OUTLET'])))
    dist_key = 0 if dist_type == 'center' else 1
    weighted_features = {var: [] for var in meteo_vars}
    grouped = df.groupby('date')
    for date, group in grouped:
        station_dists = [distances.get(station, (1, 1))[dist_key] for station in group['station']]
        weights = np.array([1 / d if d > 0 else 1 for d in station_dists])
        weights = weights / weights.sum()
        for var in meteo_vars:
            weighted_value = np.average(group[var], weights=weights)
            weighted_features[var].append(weighted_value)
    result = pd.DataFrame({'date': list(grouped.groups.keys()), **{f"{prefix}_{var}": vals for var, vals in weighted_features.items()}})
    result = result.set_index("date")
    result = result.interpolate(method='linear').ffill().fillna(0)
    return result

#% Load satellite precipitation time series
def merge_precip(base_path, precip_type):
    folder_path = os.path.join(base_path, precip_type)
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"{folder_path} does not exist")
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {folder_path}")
    precip_dataframes = []
    for file in csv_files:
        subbasin_id = file.split('.')[0]
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        if 'Precipitation' not in df.columns:
            raise KeyError(f"Missing 'Precipitation' column in file {file}")
        df_temp = df[['Precipitation']].rename(columns={'Precipitation': f'sub_p_{subbasin_id}'})
        if not all(col in df.columns for col in ['Year', 'Month', 'Day']):
            raise KeyError(f"Missing 'Year', 'Month', or 'Day' columns in file {file}")
        df_temp['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
        df_temp = df_temp.set_index('Date')
        # Append current sub-basin precipitation data to list
        precip_dataframes.append(df_temp[[f'sub_p_{subbasin_id}']])
    df_combined = pd.concat(precip_dataframes, axis=1)
    return df_combined

# Sub-basin precipitation time series weighting
def precip_weighting(subinfo_file, precip_df, aggmethod='A'):
    # Load sub-basin info
    sub_info = pd.read_excel(subinfo_file)
    dist_to_outlet = sub_info['distance2outlet'].values
    dist_to_center = sub_info['distance2center'].values
    lat = sub_info['Lat'].values
    long = sub_info['Long'].values
    # Aggregate precipitation into basin-scale features
    if aggmethod == 'A':
        # Method 1: Area-weighted average
        sub_areas = sub_info['area'].values
        precip_watershed = (precip_df * sub_areas).sum(axis=1) / sub_areas.sum()
        result = pd.DataFrame({'Weighted_P': precip_watershed})
    elif aggmethod == 'D2C':
        # Method 2: Distance to center weighted average
        precip_watershed = (precip_df * dist_to_center).sum(axis=1) / dist_to_center.sum()
        result = pd.DataFrame({'Weighted_P': precip_watershed})
    elif aggmethod == 'D2O':
        # Method 3: Distance to outlet weighted average
        precip_watershed = (precip_df * dist_to_outlet).sum(axis=1) / dist_to_outlet.sum()
        result = pd.DataFrame({'Weighted_P': precip_watershed})
    elif aggmethod == 'Mean':
        # Method 4: Arithmetic mean
        precip_watershed = precip_df.sum(axis=1) / len(sub_info)
        result = pd.DataFrame({'Weighted_P': precip_watershed})
    elif aggmethod == 'Cluster':
        # Method 5: Geographical clustering (Optional)
        # Cluster using Lat/Long
        coords = np.column_stack((lat, long))
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_labels = kmeans.fit_predict(coords)
        # Cluster precipitation
        precip_clusters = {}
        for cluster in range(3):
            cluster_mask = (cluster_labels == cluster)
            cluster_areas = dist_to_outlet * cluster_mask
            if cluster_areas.sum() > 0:
                precip_clusters[f'PRECIP_cluster_{cluster}'] = (precip_df.iloc[:, cluster_mask] * cluster_areas[cluster_mask]).sum(axis=1) / cluster_areas[cluster_mask].sum()
            else:
                precip_clusters[f'PRECIP_cluster_{cluster}'] = 0
        result = pd.DataFrame(precip_clusters)
    return result

#% Function: Feature expansion (lagged features, cumulative features, time series features)
# Add lagged features
def add_lagged_features(df, lag_days):
    lagged_df = df.copy()
    cols_to_lag = [col for col in lagged_df.columns]
    for col in cols_to_lag:
        for lag in lag_days:
            lagged_df[f'{col}_t-{lag}'] = df[col].shift(lag)
    return lagged_df

# Add cumulative features (precipitation only)
def add_cumulative_features(df, cum_periods):
    cum_df = df.copy()
    cols_to_cum = [col for col in cum_df.columns]
    for col in cols_to_cum:
        for period in cum_periods:
            cum_df[f'{col}_cum{period}'] = df[col].rolling(window=period, min_periods=1).sum()
    return cum_df

# Create time series features
def create_features(df):
    df = df.copy()
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    day_of_year = df.index.dayofyear
    df['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year/365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year/365)
    return df

#% Function: Define scoring functions
def nash_sutcliffe_efficiency(true_value, pred_value, weights=None):
    mean_y_true = np.mean(true_value)
    if weights is None:
        weights = np.ones_like(true_value)
    numerator = np.sum(weights * (true_value - pred_value) ** 2)
    denominator = np.sum(weights * (true_value - mean_y_true) ** 2)
    return 1 - (numerator / denominator) if denominator != 0 else 0

def weighted_nse_scorer(true_value, pred_value):
    weights = np.abs(true_value)**2
    return nash_sutcliffe_efficiency(true_value, pred_value, weights)

nse_scorer = make_scorer(weighted_nse_scorer, greater_is_better=True)

def custom_nse_eval(y_true, y_pred):
    # Handle potential NaN or Inf values
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return 'nse', 999  # Return large value to indicate poor performance
    weights = np.abs(y_true) ** 2 # Calculate weights
    nse = nash_sutcliffe_efficiency(y_true, y_pred, weights) # Calculate NSE
    # XGBoost assumes evaluation metric is "smaller is better", but NSE is "larger is better", so return negative
    return -nse

#% Function: Visualization
# Feature importance
def plot_feature_importance(feature_importance: pd.Series, output_dir: str, title: str = "Feature Importance", figsize: tuple = (12, 8), fontsize: int = 12):
    # Create horizontal bar chart
    plt.figure(figsize=figsize)
    bars = feature_importance.sort_values(ascending=True).plot(kind='barh')
    # Annotate importance values on bars
    for index, value in enumerate(feature_importance.sort_values(ascending=True)):
        plt.text(value, index, f'{value:.3f}', va='center', ha='left', fontsize=fontsize)
    # Set chart properties
    plt.xlabel("Importance", fontsize=fontsize + 2, labelpad=10)
    plt.ylabel("Variable Name", fontsize=fontsize + 2, labelpad=10)
    plt.tick_params(axis='both', rotation=0, labelsize=fontsize + 1)
    plt.grid(False)
    plt.tight_layout(
        pad=2.0,          # Overall padding
        w_pad=3.0,        # Horizontal spacing
        h_pad=2.0,        # Vertical spacing
        rect=[0, 0, 1, 1]
    )
    # Save chart
    save_path = f"{output_dir}/{title.lower().replace(' ', '_')}.png"
    plt.savefig(save_path)
    plt.show()

# Residual plot
def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, output_dir: str, title: str = "Residual Plot", figsize: tuple = (10, 6), alpha: float = 0.5, max_features: int = None):
    # Create residual scatter plot
    plt.figure(figsize=figsize)
    residuals = y_pred - y_true
    plt.scatter(y_true, residuals, alpha=alpha)
    plt.axhline(0, color='red', linestyle='--')
    # Set chart properties
    plt.xlabel("True Values", fontsize=14, labelpad=10)
    plt.ylabel("Residuals (Simulated - Observed)", fontsize=14, labelpad=10)
    plt.tick_params(axis='both', rotation=0, labelsize=12)
    plt.title(title, fontsize=16, pad=10)
    plt.grid(False)
    # Ensure compact layout
    plt.tight_layout()
    # Save chart
    filename = f"{title.lower().replace(' ', '_')}"
    if max_features is not None:
        filename = f"{max_features}-{filename}"
    save_path = f"{output_dir}/{filename}.png"
    plt.savefig(save_path)
    plt.show()

# Calculate yearly evaluation metrics
def calculate_yearly_metrics(y_true, y_pred, years, indices_per_year):
    yearly_metrics = {}
    for year, (start, end) in zip(years, indices_per_year):
        y_true_year = y_true[start:end]
        y_pred_year = y_pred[start:end]
        nse, kge, corr, pbias, bias, rmse = calculate_metrics(y_true_year, y_pred_year)
        yearly_metrics[year] = {
            'NSE': nse,
            'KGE': kge,
            'CC': corr,
            'PBIAS': pbias,
            'BIAS': bias,
            'RMSE': rmse
        }
    return yearly_metrics

# Plot test set time series and output evaluation metrics
def plot_time_series_test(y_true: pd.Series, y_pred: np.ndarray, output_dir: str, title: str = "Test Set True vs Predicted (2014-2017)",
                          figsize: tuple = (12, 7), fontsize: int = 12, max_features: int = None):
    # Calculate overall test set metrics
    metrics_test = print_metrics(y_true, y_pred, "Test Set")
    # Define years and index ranges
    years = [2014, 2015, 2016, 2017]
    indices_per_year = [(0, 365), (365, 730), (730, 1096), (1096, 1461)]
    yearly_metrics = calculate_yearly_metrics(y_true.values, y_pred, years, indices_per_year)
    # Create metrics text
    metrics_text = ""
    for year in years:
        metrics = yearly_metrics[year]
        metrics_text += (f"Y{year}   "
                         f"NSE: {metrics['NSE']:.4f}, "
                         f"KGE: {metrics['KGE']:.4f}, "
                         f"CC: {metrics['CC']:.4f}, "
                         f"PBIAS: {metrics['PBIAS']:.4f}, "
                         f"BIAS: {metrics['BIAS']:.4f}, "
                         f"RMSE: {metrics['RMSE']:.4f}\n")

    # Plot time series
    plt.figure(figsize=figsize)
    plt.plot(y_true.index, y_true, label="Observed")
    plt.plot(y_true.index, y_pred, label="Simulated")
    # Set year ticks and boundaries
    year_boundaries = [0, 365, 730, 1096, 1461]
    year_labels = ['2014', '2015', '2016', '2017']
    plt.xticks([y_true.index[i] for i in year_boundaries[:-1]], year_labels)
    for boundary in year_boundaries[1:-1]:
        plt.axvline(x=y_true.index[boundary], color='gray', linestyle='--', alpha=0.5)
    # Add metrics text box
    plt.text(0.2, -0.25, metrics_text.strip(), transform=plt.gca().transAxes, fontsize=fontsize,
             verticalalignment='center', linespacing=1.5, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.text(0.01, -0.25, metrics_test, transform=plt.gca().transAxes,
             fontsize=fontsize, verticalalignment='center', linespacing=1.5, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # Set chart properties
    plt.xlabel("Year", fontsize=14, labelpad=10)
    plt.ylabel("Runoff (m³/s)", fontsize=14, labelpad=10)
    plt.tick_params(axis='both', rotation=0, labelsize=12)
    plt.legend(loc='upper right', ncol=1, fontsize=12)
    plt.xlim(datetime(2014, 1, 1), datetime(2017, 12, 31))
    plt.grid(False)
    plt.tight_layout()
    # Save chart
    filename = f"{title.lower().replace(' ', '_')}"
    if max_features is not None:
        filename = f"{max_features}-{filename}"
    save_path = f"{output_dir}/{filename}.png"
    plt.savefig(save_path)
    plt.show()
    # Return overall test set metrics (to store in all_metrics_str)
    return metrics_test

# Plot train set time series and output evaluation metrics
def plot_train_time_series_train(y_true: pd.Series, y_pred: np.ndarray, fold_info: list, output_dir: str,
                                 title: str = "Train Set Time Series with Cross-Validation Metrics", figsize: tuple = (14, 10), fontsize: int = 12, max_features: int = None):
    # Calculate overall train set metrics
    metrics_train = print_metrics(y_true, y_pred, "Train Set")
    # Create cross-validation metrics text
    metrics_text = ""
    for fold in fold_info:
        train_metrics = fold['train_metrics']
        test_metrics = fold['test_metrics']
        metrics_text += (f"Fold {fold['fold']} (Test: Y{fold['fold']+2013}):\n"
                         f"Train: NSE={train_metrics['NSE']:.4f}, KGE={train_metrics['KGE']:.4f}, CC={train_metrics['CC']:.4f}, "
                         f"PBIAS={train_metrics['PBIAS']:.4f}, BIAS={train_metrics['BIAS']:.4f}, RMSE={train_metrics['RMSE']:.4f}\n"
                         f"Test: NSE={test_metrics['NSE']:.4f}, KGE={test_metrics['KGE']:.4f}, CC={test_metrics['CC']:.4f}, "
                         f"PBIAS={test_metrics['PBIAS']:.4f}, BIAS={test_metrics['BIAS']:.4f}, RMSE={test_metrics['RMSE']:.4f}\n\n")
    # Plot time series
    plt.figure(figsize=figsize)
    plt.plot(y_true.index, y_true, label="Observed")
    plt.plot(y_true.index, y_pred, label="Simulated")
    # Add metrics text box
    plt.text(0.25, -0.15, metrics_text.strip(), transform=plt.gca().transAxes, fontsize=fontsize,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.text(0.05, -0.40, metrics_train, transform=plt.gca().transAxes,
             fontsize=fontsize, verticalalignment='center', linespacing=1.5, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    # Set chart properties
    plt.xlabel("Year", fontsize=14, labelpad=10)
    plt.ylabel("Runoff (m³/s)", fontsize=14, labelpad=10)
    plt.tick_params(axis='both', rotation=0, labelsize=12)
    plt.legend(loc='upper left', ncol=1, fontsize=12)
    plt.xlim(datetime(2008, 1, 1), datetime(2013, 12, 31))
    plt.grid(False)
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.18)  # Manual fine-tuning
    plt.tight_layout()
    # Save chart
    filename = f"{title.lower().replace(' ', '_')}"
    if max_features is not None:
        filename = f"{max_features}-{filename}"
    save_path = f"{output_dir}/{filename}.png"
    plt.savefig(save_path)
    plt.show()
    # Return overall train set metrics
    return metrics_train

# Scatter plot
def plot_scatter_with_fit(
    y_train: pd.Series,
    y_train_pred: np.ndarray,
    y_test: pd.Series,
    y_test_pred: np.ndarray,
    output_dir: str,
    title: str = "Train & Test Set Observed vs Simulated Scatter Plot",
    figsize: tuple = (7, 6),
    fontsize: int = 14,
    max_features: int = None
):
    # Create scatter plot
    plt.figure(figsize=figsize)
    # Plot train and test scatters
    plt.scatter(y_train, y_train_pred, color='mediumseagreen', label='Train', alpha=0.6, s=40, edgecolors='k')
    plt.scatter(y_test, y_test_pred, color='darkorange', label='Test', alpha=0.6, s=40, edgecolors='k')
    # Train set fit line
    coeffs_train = np.polyfit(y_train, y_train_pred, deg=1)
    plt.plot(
        np.array([min(y_train), max(y_train)]),
        coeffs_train[0] * np.array([min(y_train), max(y_train)]) + coeffs_train[1],
        color='seagreen', linestyle='-', linewidth=2,
        label=f'Train fit (y = {coeffs_train[0]:.2f}x + {coeffs_train[1]:.2f})'
    )
    # Test set fit line
    coeffs_test = np.polyfit(y_test, y_test_pred, deg=1)
    plt.plot(
        np.array([min(y_test), max(y_test)]),
        coeffs_test[0] * np.array([min(y_test), max(y_test)]) + coeffs_test[1],
        color='orangered', linestyle='-', linewidth=2,
        label=f'Test fit (y = {coeffs_test[0]:.2f}x + {coeffs_test[1]:.2f})'
    )
    # Ideal fit line y = x
    min_val = min(y_train.min(), y_test.min())
    max_val = max(y_train.max(), y_test.max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        'r--', linewidth=2, label='y = x'
    )
    # Set chart properties
    plt.xlabel('Observed', fontsize=fontsize, labelpad=10)
    plt.ylabel('Simulated', fontsize=fontsize, labelpad=10)
    plt.tick_params(axis='both', rotation=0, labelsize=12)
    plt.legend(loc='upper left', ncol=1, fontsize=12)
    plt.xlim(0, None)   # x-axis starts from 0, right is auto
    plt.ylim(0, None)
    plt.grid(False)
    plt.tight_layout()
    # Save chart
    filename = f"{title.lower().replace(' ', '_').replace('&', 'and')}"
    if max_features is not None:
        filename = f"{max_features}-{filename}"
    save_path = f"{output_dir}/{filename}.png"
    plt.savefig(save_path)
    plt.show()

#% Function: Use 2008-2013 data for preliminary feature screening based on XGBoost, then remove highly correlated features
def simple_feature_selection(X, y, output_dir, seed=42, min_features=10, max_features=20, n_splits=3, corr_threshold=0.7):
    # Input validation
    if not isinstance(X, pd.DataFrame) or not isinstance(y, (pd.Series, np.ndarray)):
        raise ValueError("X must be pd.DataFrame, y must be pd.Series or np.ndarray")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Sample size mismatch: X ({X.shape[0]}) and y ({y.shape[0]})")
    if X.isna().any().any() or pd.Series(y).isna().any():
        raise ValueError("X or y contains missing values")
    if X.shape[1] < min_features:
        raise ValueError(f"Number of features ({X.shape[1]}) is less than min_features ({min_features})")
    if not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError("X index must be pd.DatetimeIndex")
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Set up logging
    logging.basicConfig(filename=output_dir / 'simple_feature_selection.log', level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info("Starting simple feature selection (XGBoost + Cross-Validation + Multicollinearity handling, 2008-2013 data)")
    # Limit data to 2008-2013
    X_train = X[X.index.year <= 2013]
    y_train = y[X.index.year <= 2013]
    if len(X_train) == 0:
        raise ValueError("2008-2013 data is empty, please check the index")
    # Define grid search parameters
    param_grid = {
        'n_estimators': [700, 800, 1000, 1200, 1500],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.03, 0.05]
    }
    # Initialize XGBoost model
    xgb_model = xgb.XGBRegressor(random_state=seed, n_jobs=4)
    # Initialize cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    # Grid search
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=tscv,
        scoring=nse_scorer,
        n_jobs=4,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"Best Parameters: {best_params}")
    logging.info(f"Best Parameters: {best_params}")
    # Initialize XGBoost with best parameters
    xgb_model = xgb.XGBRegressor(**best_params, random_state=seed, n_jobs=4)
    # Cross-validation to calculate average feature importance
    importances = np.zeros(X_train.shape[1])
    for fold, (train_idx, _) in enumerate(tscv.split(X_train)):
        X_fold = X_train.iloc[train_idx]
        y_fold = y_train.iloc[train_idx]
        xgb_model.fit(X_fold, y_fold)
        importances += xgb_model.feature_importances_
        logging.info(f"Fold {fold + 1} feature importance calculation complete")
    importances /= n_splits
    # Feature ranking
    feature_ranking = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
    initial_features = feature_ranking.head(max_features).index.tolist()
    # Handle multicollinearity
    corr_matrix = X_train[initial_features].corr().abs()
    high_corr_pairs = [(i, j, corr_matrix.loc[i, j]) for i in corr_matrix.columns for j in corr_matrix.columns
                       if i < j and corr_matrix.loc[i, j] > corr_threshold]
    to_remove = set()
    for i, j, _ in high_corr_pairs:
        if feature_ranking[i] > feature_ranking[j]:
            to_remove.add(j)
        else:
            to_remove.add(i)
    selected_features = [f for f in initial_features if f not in to_remove]
    selected_importances = feature_ranking[selected_features]
    # Ensure feature count is between min_features and max_features
    target_n_features = min(max_features, max(min_features, len(selected_features)))
    selected_features = selected_features[:target_n_features]
    selected_importances = selected_importances[:target_n_features]
    # Print feature info
    print(f"Selected {len(selected_features)} features: {selected_features}")
    logging.info(f"Selected {len(selected_features)} features: {selected_features}")
    corr = X_train[selected_features].corrwith(y_train)
    print("\nFeature correlation with runoff (2008-2013):")
    print(corr.sort_values(ascending=False))
    variances = X_train[selected_features].var()
    print("\nFeature variances (2008-2013):")
    print(variances.sort_values(ascending=False))
    # Report multicollinearity
    print(f"\nInitial highly correlated (> {corr_threshold}) feature pairs: {len(high_corr_pairs)} pairs")
    if high_corr_pairs:
        print("Highly correlated feature pairs (Feature1, Feature2, Correlation Coefficient):")
        for pair in high_corr_pairs:
            print(f"{pair[0]} - {pair[1]}: {pair[2]:.4f}")
    print(f"Removed features: {list(to_remove)}")
    logging.info(f"Initial highly correlated (> {corr_threshold}) feature pairs: {len(high_corr_pairs)} pairs")
    logging.info(f"Removed features: {list(to_remove)}")
    return selected_features

#% Function: Calculate evaluation metrics
def calculate_metrics(y_true, y_pred):
    nse = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2) # NSE (Nash-Sutcliffe Efficiency)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) # RMSE
    pearson_corr, _ = pearsonr(y_true, y_pred) # Pearson correlation coefficient
    bias = np.mean(y_pred - y_true) # Bias
    pbias = 100 * np.sum(y_pred - y_true) / np.sum(y_true) if np.sum(y_true) != 0 else np.nan # PBIAS (Percent Bias)
    # KGE (Kling-Gupta Efficiency)
    # Correlation component
    r = pearson_corr
    # Ratio of means
    mean_ratio = np.mean(y_pred) / np.mean(y_true) if np.mean(y_true) != 0 else np.nan
    # Standard deviation ratio (Variability ratio component alpha for KGE)
    cv_pred = np.std(y_pred)
    cv_true = np.std(y_true)
    cv_ratio = cv_pred / cv_true if cv_true != 0 else np.nan
    # KGE calculation
    kge = 1 - np.sqrt((r - 1)**2 + (mean_ratio - 1)**2 + (cv_ratio - 1)**2) if not np.isnan([r, mean_ratio, cv_ratio]).any() else np.nan
    return nse, kge, pearson_corr, pbias, bias, rmse

# Evaluation metrics text print
def print_metrics(y_true, y_pred, dataset_name):
    nse, kge, pearson_corr, pbias, bias, rmse = calculate_metrics(y_true, y_pred)
    print(f"{dataset_name}:")
    print(f"NSE: {nse:.4f}")
    print(f"KGE: {kge:.4f}")
    print(f"CC: {pearson_corr:.4f}")
    print(f"PBIAS: {pbias:.4f}")
    print(f"BIAS: {bias:.4f}")
    print(f"RMSE: {rmse:.4f}")
    metrics_str = (f"NSE: {nse:.4f}\n"
                   f"KGE: {kge:.4f}\n"
                   f"CC: {pearson_corr:.4f}\n"
                   f"PBIAS: {pbias:.4f}\n"
                   f"BIAS: {bias:.4f}\n"
                   f"RMSE: {rmse:.4f}")
    return metrics_str

#% Function: Used for Bayesian Optimization
def xgb_evaluate(n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda):
    params = {
        'n_estimators': int(n_estimators),
        'learning_rate': learning_rate,
        'max_depth': int(max_depth),
        'min_child_weight': int(min_child_weight),
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'gamma': gamma,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'random_state': 42
    }
    model = xgb.XGBRegressor(**params)
    tscv = TimeSeriesSplit(n_splits=3)
    nse_scores = []
    for train_idx, test_idx in tscv.split(X_train):
        X_train_fold, X_test_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_train_fold, y_test_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]
        sample_weights_fold = sample_weights_train[:train_size].iloc[train_idx]
        model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights_fold)
        y_pred = model.predict(X_test_fold)
        nse = nash_sutcliffe_efficiency(y_test_fold, y_pred)
        nse_scores.append(nse)
    return np.mean(nse_scores)

#% Function: Cross-validation function
def train_with_cv(X, y, sample_weights, params, cv_splits, early_stopping_rounds=50):
    """Train model using custom 4-fold time series cross-validation"""
    models = []
    nse_scores = []
    kge_scores = []
    corr_scores = []
    pbias_scores = []
    bias_scores = []
    rmse_scores = []
    fold_info = []
    for fold, split in enumerate(cv_splits):
        train_start, train_end = split['train_range']
        test_start, test_end = split['test_range']
        # Get train and test sets
        X_train_fold = X.iloc[train_start:train_end]
        y_train_fold = y.iloc[train_start:train_end]
        X_test_fold = X.iloc[test_start:test_end]
        y_test_fold = y.iloc[test_start:test_end]
        # Extract sample weights for current fold
        sample_weights_fold = sample_weights[train_start:train_end]
        # Verify sample weights match train set size
        assert len(sample_weights_fold) == len(X_train_fold), f"Fold {fold}: Sample weight size {len(sample_weights_fold)} does not match train set size {len(X_train_fold)}"
        # Print validation set statistics
        print(f"Fold {fold+1}: y_test_fold mean={np.mean(y_test_fold):.4f}, std={np.std(y_test_fold):.4f}, min={np.min(y_test_fold):.4f}, max={np.max(y_test_fold):.4f}")
        # Initialize model
        model = xgb.XGBRegressor(**params, early_stopping_rounds=early_stopping_rounds, eval_metric=custom_nse_eval)
        # Train model
        model.fit(
            X_train_fold, y_train_fold,
            sample_weight=sample_weights_fold,
            eval_set=[(X_test_fold, y_test_fold)],
            verbose=False
        )
        models.append(model)
        # Evaluate
        y_train_pred = model.predict(X_train_fold)
        y_test_pred = model.predict(X_test_fold)
        # Train set metrics
        train_nse, train_kge, train_corr, train_pbias, train_bias, train_rmse = calculate_metrics(y_train_fold, y_train_pred)
        # Validation set metrics
        test_nse, test_kge, test_corr, test_pbias, test_bias, test_rmse = calculate_metrics(y_test_fold, y_test_pred)
        nse_scores.append(test_nse)
        kge_scores.append(test_kge)
        corr_scores.append(test_corr)
        pbias_scores.append(test_pbias)
        bias_scores.append(test_bias)
        rmse_scores.append(test_rmse)
        # Save fold info
        fold_info.append({
            'fold': fold + 1,
            'train_range': (train_start, train_end),
            'test_range': (test_start, test_end),
            'train_pred': y_train_pred,
            'test_pred': y_test_pred,
            'train_metrics': {'NSE': train_nse, 'KGE': train_kge, 'CC': train_corr,
                              'PBIAS': train_pbias, 'BIAS': train_bias, 'RMSE': train_rmse},
            'test_metrics': {'NSE': test_nse, 'KGE': test_kge, 'CC': test_corr,
                              'PBIAS': test_pbias, 'BIAS': test_bias, 'RMSE': test_rmse}
        })
        print(f"Fold {fold+1}: Train NSE={train_nse:.4f}, KGE={train_kge:.4f}, CC={train_corr:.4f}, PBIAS={train_pbias:.4f}, BIAS={train_bias:.4f}, RMSE={train_rmse:.4f}")
        print(f"Fold {fold+1}: Test NSE={test_nse:.4f}, KGE={test_kge:.4f}, CC={test_corr:.4f}, PBIAS={test_pbias:.4f}, BIAS={test_bias:.4f}, RMSE={test_rmse:.4f}")
    # Print cross-validation results
    print("\nCross-Validation Results:")
    print(f"Mean NSE: {np.mean(nse_scores):.4f} ± {np.std(nse_scores):.4f}")
    print(f"Mean KGE: {np.mean(kge_scores):.4f} ± {np.std(kge_scores):.4f}")
    print(f"Mean CC: {np.mean(corr_scores):.4f} ± {np.std(corr_scores):.4f}")
    print(f"Mean PBIAS: {np.mean(pbias_scores):.4f} ± {np.std(pbias_scores):.4f}")
    print(f"Mean BIAS: {np.mean(bias_scores):.4f} ± {np.std(bias_scores):.4f}")
    print(f"Mean RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    return nse_scores, kge_scores, corr_scores, pbias_scores, bias_scores, rmse_scores, fold_info

#% Function: Feature grouping
def generate_feature_groups(variable_names):
    # Define grouping rules (keywords and corresponding group names)
    group_rules = {
        '$\mathrm{P_{cum}}$': lambda x: re.search(r'P_.*cum', x),        # P_cum7, P_cum15 etc.
        '$\mathrm{P_{lag}}$': lambda x: re.search(r'P_.*t-', x),         # P_t-1, P_t-2 etc.
        'RH': lambda x: 'RH_' in x or x.startswith('RH'),  # contains rh_ or starts with RH
        'MON': lambda x: 'month_' in x or x.startswith('month'),
        'SD': lambda x: 'SD_' in x or x.startswith('SD'),  # contains SD_ or starts with SD
        'T': lambda x: any(s in x for s in ['Tmax_', 'Tmin_', 'Tmean_']) or x.startswith(('Tmax', 'Tmin', 'Tmean')),  # Temperature
        'WS': lambda x: 'WS_' in x or x.startswith('WS')  # Wind speed
    }
    # Initialize grouping dictionary
    groups = {key: [] for key in group_rules}
    # Iterate through column names and assign to groups
    for var in variable_names:
        for group_name, condition in group_rules.items():
            if condition(var):
                groups[group_name].append(var)
                break  # Assuming each variable belongs to only one group
    # Remove empty groups
    groups = {k: v for k, v in groups.items() if v}
    return groups

# Function: Convert column names to LaTeX format, using content after _ as subscript
def to_latex_subscript(col_name):
    match = re.match(r'([A-Za-z]+)_(.+)', col_name)
    if match:
        prefix, subscript = match.groups()
        return f'$\\mathrm{{{prefix}_{{{subscript}}}}}$'
    return col_name  # Keep unchanged if no _

#%% Data Loading, Aggregation, Feature Engineering
# Define paths (Please update these English paths to your actual directory structure)
METEO_DIR = Path(r'\Input\Gauge') # Meteorological station data folder
STATIONINFO_FILE = r'\Input\Station_Info.xlsx'
BASE_PRECIP_DIR = r'\Input\FY' # Satellite precip time series base folder
SUBINFO_FILE = r'\Input\Subbasin_Info.xlsx'
FLOW_FILE = Path(r'\Input\Runoff_Data.xlsx')  # Runoff data path
OUTPUT_DIR = Path(r'\OutputsData')  # Output path
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Inner and outer basin station IDs
INNER_STATIONS = ['59072', '59082', '59087', '59088', '57996', '59280']
OUTER_STATIONS = ['57972', '57974', '59096', '59097', '59271']

# Define variable names
METEO_VARS = ['rh', 'sund', 'tmean', 'wsmean','tmin','tmax']

# Load inner and outer basin data
InnerDF = load_meteo_data(INNER_STATIONS, METEO_DIR, METEO_VARS)
OuterDF = load_meteo_data(OUTER_STATIONS, METEO_DIR, METEO_VARS)

# Distance-weighted aggregation of hydrological elements (center dist for inner, outlet dist for outer)
InnerMeteo = weighted_average_meteo(InnerDF, INNER_STATIONS, STATIONINFO_FILE, 'inner', METEO_VARS, dist_type='center')
OuterMeteo = weighted_average_meteo(OuterDF, OUTER_STATIONS, STATIONINFO_FILE, 'outer', METEO_VARS, dist_type='outlet')

# Read specific type of precipitation sequence
PRECIP_TYPE = 'FYmm75'  # Can be replaced with 'FYm', 'FYmm85', or other types
PrecipDF = merge_precip(BASE_PRECIP_DIR, PRECIP_TYPE)

# Weighting to obtain basin-scale or regional precipitation sequence
Precip = precip_weighting(SUBINFO_FILE, PrecipDF, aggmethod='D2O')  # Can be replaced with 'A', 'Mean', 'Cluster', etc.

# Load runoff data
FlowDF = pd.read_excel(FLOW_FILE)
FlowDF['date'] = pd.to_datetime(FlowDF[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1))
FlowDF = FlowDF.set_index('date')['Flow'].to_frame() # Set index

# Feature expansion (Lagged and Cumulative features)
InnerMeteo_lagged = add_lagged_features(InnerMeteo, [1, 2, 3, 6, 7, 10, 11, 12, 17, 25])
OuterMeteo_lagged = add_lagged_features(OuterMeteo, [1, 2, 3, 6, 7, 10, 11, 12, 17, 25])
Precip_lagged = add_lagged_features(Precip, [1, 2, 3, 6, 7, 10, 11, 12, 17, 25])
Precip_cum = add_cumulative_features(Precip, [3, 5, 7, 15, 30])

# Combine all features
CombinedDF = Precip_lagged.copy()
CombinedDF = CombinedDF.merge(Precip_cum.drop(columns=CombinedDF.columns.intersection(Precip_cum.columns)), left_index=True, right_index=True, how='inner')
CombinedDF = CombinedDF.merge(InnerMeteo_lagged, left_index=True, right_index=True, how='inner')
CombinedDF = CombinedDF.merge(OuterMeteo_lagged, left_index=True, right_index=True, how='inner')  # Optional

# Remove NaNs (missing values caused by lag and cumulative calculations)
CombinedDF = CombinedDF.dropna()

# Add time series features based on index
FeatureDF = create_features(CombinedDF)

#%% Feature Selection and Model Training
# Data alignment, extract time series after removing NaN values
CommonDates = FeatureDF.index.intersection(FlowDF.index)
X_all = FeatureDF.loc[CommonDates]
y = FlowDF.loc[CommonDates, 'Flow']
max_features_values = [20]
selected_features_results = {}
all_metrics_str = {}

for max_features in max_features_values:
    print(f"Running feature selection, max_features={max_features}")
    # Feature selection
    selected_features = simple_feature_selection(X_all, y, OUTPUT_DIR, seed=42, min_features=10,
                                             max_features=max_features, n_splits=3, corr_threshold=0.9)
    # Store results
    selected_features_results[max_features] = selected_features
    print(f"Selected {len(selected_features)} features when max_features={max_features}")
    FinalDF = pd.concat([X_all[selected_features], y], axis=1)
    CleanedDF = FinalDF.dropna()
    OUTPUT_PATH = OUTPUT_DIR / f'{max_features}-cleaned_nan.csv'
    CleanedDF.to_csv(OUTPUT_PATH)
    
    # Prepare training data
    X = CleanedDF.iloc[:, :-1]  # Features
    X_original = X
    y = CleanedDF.iloc[:, -1]   # Target values
    
    # Split train and test sets
    test_size = 1461   # 1461 days in total for 2014-2017
    train_size = X.shape[0] - test_size  # 2008-2013
    X_train_original = X[:train_size]  # 2008-2013
    y_train = y[:train_size]
    X_test_original = X[train_size:]  # 2014-2017
    y_test = y[train_size:]
    
    # Standardize features
    scaler = StandardScaler()
    # Fit scaler on X_train_original to learn mean and std of training data
    scaler.fit(X_train_original) 
    # Transform train and test features
    X_train_scaled_np = scaler.transform(X_train_original) 
    X_test_scaled_np = scaler.transform(X_test_original) 
    # Convert NumPy arrays back to Pandas DataFrames to preserve column names and indices
    X_train = pd.DataFrame(X_train_scaled_np, columns=X_train_original.columns, index=X_train_original.index)
    X_test = pd.DataFrame(X_test_scaled_np, columns=X_test_original.columns, index=X_test_original.index)
    X = pd.concat([X_train, X_test], axis=0)
    
    # Generate sample weights for the entire dataset (based on complete y)
    sample_weights_train = np.abs(y_train)
    sample_weights_train = sample_weights_train / np.mean(sample_weights_train)
    
    # Bayesian Optimization
    print("Running model training...")
    xgb_bo = BayesianOptimization(
        f=xgb_evaluate,
        pbounds={
            'n_estimators': (500, 2000),
            'learning_rate': (0.01, 0.05),
            'max_depth': (3, 5),
            'min_child_weight': (1, 10),
            'subsample': (0.3, 1),
            'colsample_bytree': (0.3, 1),
            'gamma': (0, 5),
            'reg_alpha': (0, 5),
            'reg_lambda': (0, 5)
        },
        random_state=42,
    )
    xgb_bo.maximize(n_iter=150, init_points=5)
    
    # Get best parameters
    best_params = xgb_bo.max['params']
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_child_weight'] = int(best_params['min_child_weight'])
    print("Best Parameters:", best_params)
    
    # Define train and test sets for 4-fold cross-validation (based on indices)
    cv_splits = [
        {'train_range': (0, train_size), 'test_range': (train_size , train_size + 365)},  # 2008-2013 vs 2014
        {'train_range': (0, train_size + 365), 'test_range': (train_size + 365, train_size + 730)},  # 2008-2014 vs 2015
        {'train_range': (0, train_size + 730), 'test_range': (train_size + 730, train_size + 1096)},  # 2008-2015 vs 2016
        {'train_range': (0, train_size + 1096), 'test_range': (train_size + 1096, train_size + 1461)}  # 2008-2016 vs 2017
    ]
    
    # Train final model with cross-validation
    nse_scores, kge_scores, corr_scores, pbias_scores, bias_scores, rmse_scores, fold_info = train_with_cv(
        X, y, sample_weights_train, best_params, cv_splits=cv_splits, early_stopping_rounds=100
    )
    
    # Final model training: Re-fit final_model on fixed training set (2008-2013) with best_params to prevent leakage
    print("\nTraining final model (on fixed train set 2008-2013)...")
    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(
        X_train, y_train,
        sample_weight=sample_weights_train[:train_size],  # Only use training period weights
        verbose=False
    )
    
    # Predictions
    y_train_pred = final_model.predict(X_train)  # Train set predictions
    y_test_pred = final_model.predict(X_test)    # Test set predictions

    #%% Best model evaluation and visualization
    # Plot feature importance
    feature_importance = pd.Series(final_model.feature_importances_, index=X.columns)
    feature_importance = feature_importance.sort_values(ascending=False)
    print("Final model feature importance:")
    print(feature_importance)
    
    # Plot horizontal bar chart with annotated importance values
    plot_feature_importance(feature_importance=feature_importance, output_dir=OUTPUT_DIR, title=f"{max_features}-feature importance")
    
    # Plot residual plot
    plot_residuals(y_true=y_test, y_pred=y_test_pred, output_dir=OUTPUT_DIR, title="Test Set Residual Plot", max_features=max_features)
    
    # Plot train set time series chart
    metrics_train = plot_train_time_series_train(y_true=y_train, y_pred=y_train_pred, fold_info=fold_info, output_dir=OUTPUT_DIR, title="Train (2008-2013)", max_features=max_features)
    
    # Plot test set time series chart
    metrics_test = plot_time_series_test(y_true=y_test, y_pred=y_test_pred, output_dir=OUTPUT_DIR, title="Test (2014-2017)", max_features=max_features)
    all_metrics_str[max_features] = metrics_test
    
    # Scatter plot
    plot_scatter_with_fit(y_train=y_train, y_train_pred=y_train_pred, y_test=y_test, y_test_pred=y_test_pred, output_dir=OUTPUT_DIR, title="scatter_plot", max_features=max_features)

    #%% SHAP Analysis
    # Ensure X and X_original have consistent shapes, indices, and column names
    assert X.shape == X_original.shape, "Shape mismatch between X and X_original" 
    assert set(X.columns) == set(X_original.columns), "Column mismatch between X and X_original" 
    X_original = X_original[X.columns] # Reorder columns in X_original to match X
    
    # Simplify column names
    X.columns = [col.replace("Weighted_P", "P").replace("inner_", "").replace("outer_", "").replace("tmean", "Tmean").replace("tmin", "Tmin").replace("tmax", "Tmax").replace("rh", "RH").replace("sund", "SD").replace("wsmean", "WS") for col in X.columns]
    X_original.columns = [col.replace("Weighted_P", "P").replace("inner_", "").replace("outer_", "").replace("tmean", "Tmean").replace("tmin", "Tmin").replace("tmax", "Tmax").replace("rh", "RH").replace("sund", "SD").replace("wsmean", "WS") for col in X_original.columns]

    # Update column names with LaTeX subscripts
    X.columns = [to_latex_subscript(col) for col in X.columns]
    X_original.columns = [to_latex_subscript(col) for col in X_original.columns]
    print(X.columns)
    print(X_original.columns)
    
    # Create SHAP explainer (for XGBoost model)
    explainer = shap.TreeExplainer(final_model)
    # Calculate SHAP values (based on standardized X, as the model was trained on it)
    shap_values = explainer.shap_values(X)

    # Fig 1. Full sequence shap.summary_plot: Feature importance summary (using X_original to display raw feature values)
    shap.summary_plot(shap_values, X_original, show=False)
    # Get current figure and set size
    fig = plt.gcf()
    fig.set_size_inches(8,6)
    plt.title("SHAP Summary Plot")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{max_features}-shap_summary_plot_original.png", bbox_inches='tight')  # Save as image
    plt.close()

    # Fig 2. Rolling SHAP value changes for each feature with time on x-axis
    end_date = pd.to_datetime("2017-12-31")
    # Build date index
    date_index = pd.date_range(end=end_date, periods=X_original.shape[0], freq="D")
    start_date = date_index[0]
    # Replace index
    X_original.index = date_index
    # SHAP calculation + rolling
    window_size = 90
    shap_abs = np.abs(shap_values)
    shap_rolling = pd.DataFrame(
        shap_abs, index=X_original.index, columns=X_original.columns
    ).rolling(window=window_size).mean()
    
    # Plotting
    plt.figure(figsize=(14, 7))
    for col in shap_rolling.columns:
        plt.plot(shap_rolling.index, shap_rolling[col], label=col)
    plt.title(f"Rolling SHAP Value Changes Over Time (Absolute Values, Window={window_size})", fontsize=21, pad=10)
    plt.ylabel("Rolling Mean of |SHAP Value|", fontsize=19, labelpad=15)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=15)
    plt.xlabel("Year", fontsize=19, labelpad=15)
    
    # Time axis styling
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(7,)))
    ax.set_xlim(start_date+pd.Timedelta(days=window_size-1), end_date)  # Adjust date range based on data
    ax.tick_params(axis='x', which='major', length=4, width=1, labelsize=17, direction='out')
    ax.tick_params(axis='x', which='minor', length=2, width=0.5, direction='out')
    ax.tick_params(axis='y', which='major', length=4, width=1, labelsize=17, direction='out')
    
    # Add border lines: set all spines visible and adjust styling
    for spine in ax.spines.values():
        spine.set_visible(True)  # Ensure border is visible
        spine.set_linewidth(1)   # Set border width
        spine.set_color('black') # Set border color
    plt.xticks(rotation=0, fontsize=17)
    plt.yticks(fontsize=17)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{max_features}-shap_rolling_time_plot_original.png")
    plt.close()

    # Fig 3. shap.summary_plot split by quarter (Top 10 features)
    season_names = ['Q1', 'Q2', 'Q3', 'Q4']
    season_months = [(3, 4, 5), (6, 7, 8), (9, 10, 11), (12, 1, 2)]
    # Create large figure, 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(17, 14))
    plt.subplots_adjust(wspace=5)  # Set horizontal spacing
    labels = ['(a)', '(b)', '(c)', '(d)']
    # Add numbering to each subplot
    for ax, label in zip(axes.flat, labels):
        ax.text(-0.45, 1.0, label, transform=ax.transAxes, fontsize=14, fontweight='normal', va='top', color='k')
    axes = axes.flatten()
    
    for idx, (season_name, months) in enumerate(zip(season_names, season_months)):
        # Filter samples for the current quarter
        season_mask = X.index.month.isin(months)
        X_season = X[season_mask]
        X_original_season = X_original[season_mask]
        if len(X_season) == 0:
            print(f"No data for {season_name}")
            plt.sca(axes[idx])
            plt.text(0.5, 0.5, "No Data", ha='center', va='center')
            continue
        try:
            # Calculate SHAP values for current quarter
            shap_values_season = explainer.shap_values(X_season)
            # Plot shap.summary_plot, showing only top 10 features
            plt.sca(axes[idx])
            shap.summary_plot(shap_values_season, X_original_season, max_display=10, show=False, plot_size=(8, 6), color_bar_label="", alpha=0.7) 
            # Get color bar object
            cbar = plt.gcf().axes[-1]  
            cbar.tick_params(labelsize=10)  
            plt.xlabel("SHAP value", fontsize=12, labelpad=6)  
            plt.tick_params(axis='y', labelsize=10, labelcolor='black', pad=-15)
            plt.grid(False)
        except Exception as e:
            print(f"Error drawing SHAP plot for {season_name}: {e}")
            plt.sca(axes[idx])
            plt.text(0.5, 0.5, "Plot Failed", ha='center', va='center')
    # Adjust layout to prevent color bar overlap
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{max_features}-shap_summary_seasons_top10.png"), dpi=600, bbox_inches='tight')
    plt.close(fig)

    # Fig 4. Grouped rolling SHAP change plot
    variable_names = X_original.columns
    # Group features
    groups = generate_feature_groups(variable_names)
    variable_names = X_original.columns.tolist()
    # Calculate mean absolute SHAP value for each group
    group_importance = {}
    group_shap_data = {}
    for group_name, vars_in_group in groups.items():
        indices = [variable_names.index(var) for var in vars_in_group]
        group_shap = np.abs(shap_values[:, indices])
        group_importance[group_name] = np.mean(group_shap)
        group_shap_s = np.mean(np.abs(shap_values[:, indices]), axis=1) # Average after taking absolute value
        group_shap_data[group_name] = group_shap_s
        
    # Display sorted importance
    sorted_importance = dict(sorted(group_importance.items(), key=lambda x: x[1], reverse=True))
    print("\nSorted Group Importance:")
    for group, importance in sorted_importance.items():
        print(f"{group}: {importance:.4f}")
        
    # Convert to DataFrame, use date_index as index
    shap_df = pd.DataFrame(group_shap_data, index=date_index).abs()  # Use absolute value as importance
    # Calculate 90-day rolling mean importance
    rolling_importance = shap_df.rolling(window=90).mean()
    # Rename columns
    new_columns = ['$\mathrm{P_{cum}}$', '$\mathrm{P_{lag}}$', 'RH','MON', 'SD', 'T', 'WS']  
    rolling_importance.columns = new_columns
    
    # Improved rolling line chart
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = ['#234f8c','#aa2b46', '#e58760', '#f6dea4', '#c9dee5', '#6cb3da', '#3b5da3', '#427ab2', '#f09148', '#ff9896', '#dbdb8d', '#c59d94', '#afc7e8', '#9bbf8a', '#82afda', '#f79059', '#b9be3c', '#c2bdde', '#fa8878']
    
    plt.figure(figsize=(14, 7))
    for i, group in enumerate(rolling_importance.columns):
        plt.plot(rolling_importance.index, rolling_importance[group], label=group, color=colors[i],
            linewidth=2.5, alpha=1.0)
    plt.xlabel("Year", fontsize=19, labelpad=15)
    plt.ylabel("Rolling Mean |SHAP| Value", fontsize=19, labelpad=15)
    
    # Time axis styling
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(7,)))
    ax.set_ylim(-5.0, 600)  # Adjust min/max values based on data
    ax.set_xlim(start_date+pd.Timedelta(days=window_size-1), end_date)  
    ax.tick_params(axis='x', which='major', length=4, width=1, labelsize=17, direction='out')
    ax.tick_params(axis='x', which='minor', length=2, width=0.5, direction='out')
    ax.tick_params(axis='y', which='major', length=4, width=1, labelsize=17, direction='out')
    
    for spine in ax.spines.values():
        spine.set_visible(True)  
        spine.set_linewidth(1)   
        spine.set_color('black') 
    plt.xticks(rotation=0, fontsize=17)
    plt.yticks(fontsize=17)
    plt.legend(bbox_to_anchor=(0.6, 0.9), fontsize=15, loc="lower center", ncol=7, frameon=False)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{max_features}-shap_rolling_importance_improve.png", bbox_inches='tight')  
    plt.close()

    #%% Export Simulation Results
    # Calculate maximum length for alignment
    max_length = max(len(y_train), len(y_test))
    
    # Export train and test sets, observed and predicted results
    df_output = pd.DataFrame({
        'Train_Observed': np.append(y_train.values, [np.nan] * (max_length - len(y_train))),  # Pad with NaN
        'Train_Predicted': np.append(y_train_pred, [np.nan] * (max_length - len(y_train_pred))),
        'Test_Observed': np.append(y_test.values, [np.nan] * (max_length - len(y_test))),  # Align test data
        'Test_Predicted': np.append(y_test_pred, [np.nan] * (max_length - len(y_test_pred)))
    })
    save_path = os.path.join(OUTPUT_DIR, f"{max_features}-model_predictions.csv")
    df_output.to_csv(save_path, index=False)  
    
    # Export test set results only
    test_output = pd.DataFrame({
        'Test_Observed': y_test.values,  
        'Test_Predicted': y_test_pred
    })
    save_path = os.path.join(OUTPUT_DIR, f"{max_features}-test_prediction.csv")
    test_output.to_csv(save_path, index=False)  

#%% Save final configurations
# Save test set evaluation metrics for different max_features conditions
metrics_df = pd.DataFrame.from_dict(all_metrics_str, orient='index') 
metrics_file = r'\OutputsData\metrics_of_different_maxfeatures.xlsx'
metrics_df.to_excel(metrics_file, index=False)

# Save selected features under different max_features conditions
max_len_selected = max(len(features) for features in selected_features_results.values())
selected_columns = [f'Feature_{i+1}' for i in range(max_len_selected)]
selected_data = {model: features + [''] * (max_len_selected - len(features)) for model, features in selected_features_results.items()}
df_selected = pd.DataFrame.from_dict(selected_data, orient='index', columns=selected_columns)
features_file = r'\OutputsData\selected_features_of_different_maxfeatures.xlsx'
df_selected.to_excel(features_file, index_label='Max_Features')