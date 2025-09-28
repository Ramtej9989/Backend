# Statistical analysis functions for data insight dashboard
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_summary_statistics(df, numeric_only=True):
    """
    Calculate comprehensive summary statistics for dataframe columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe to analyze
    numeric_only : bool
        Whether to include only numeric columns
        
    Returns:
    --------
    dict
        Dictionary of column statistics
    """
    if numeric_only:
        df_numeric = df.select_dtypes(include="number")
    else:
        df_numeric = df
    
    # Handle empty dataframe
    if df_numeric.empty:
        return {}
    
    # Initialize results dictionary
    results = {}
    
    # Calculate statistics for each column
    for col in df_numeric.columns:
        col_stats = {}
        
        # Basic statistics
        col_stats["mean"] = float(df_numeric[col].mean()) if not pd.isna(df_numeric[col].mean()) else None
        col_stats["median"] = float(df_numeric[col].median()) if not pd.isna(df_numeric[col].median()) else None
        col_stats["std"] = float(df_numeric[col].std()) if not pd.isna(df_numeric[col].std()) else None
        col_stats["min"] = float(df_numeric[col].min()) if not pd.isna(df_numeric[col].min()) else None
        col_stats["max"] = float(df_numeric[col].max()) if not pd.isna(df_numeric[col].max()) else None
        col_stats["sum"] = float(df_numeric[col].sum()) if not pd.isna(df_numeric[col].sum()) else None
        
        # Additional statistics
        col_stats["count"] = int(df_numeric[col].count())
        col_stats["missing"] = int(df_numeric[col].isna().sum())
        col_stats["missing_percent"] = float(df_numeric[col].isna().mean() * 100)
        
        # Quartiles
        q1 = float(df_numeric[col].quantile(0.25)) if not pd.isna(df_numeric[col].quantile(0.25)) else None
        q3 = float(df_numeric[col].quantile(0.75)) if not pd.isna(df_numeric[col].quantile(0.75)) else None
        col_stats["q1"] = q1
        col_stats["q3"] = q3
        col_stats["iqr"] = float(q3 - q1) if q1 is not None and q3 is not None else None
        
        # Distribution characteristics
        if col_stats["count"] > 2:  # Need at least 3 values for skew/kurtosis
            col_stats["skew"] = float(df_numeric[col].skew()) if not pd.isna(df_numeric[col].skew()) else None
            col_stats["kurtosis"] = float(df_numeric[col].kurtosis()) if not pd.isna(df_numeric[col].kurtosis()) else None
        
        results[col] = col_stats
    
    return results

def compute_correlation_matrix(df, min_periods=None):
    """
    Compute correlation matrix with appropriate handling of missing values
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with numeric columns
    min_periods : int, optional
        Minimum number of observations required per pair of columns
        
    Returns:
    --------
    pandas.DataFrame
        Correlation matrix
    """
    try:
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=["number"])
        
        if numeric_df.empty or numeric_df.shape[1] < 2:
            return pd.DataFrame()
            
        # Compute correlations
        corr_matrix = numeric_df.corr(method='pearson', min_periods=min_periods)
        return corr_matrix
    except Exception as e:
        print(f"Error in correlation calculation: {str(e)}")
        return pd.DataFrame()

def detect_outliers(df, method='iqr', contamination=0.05):
    """
    Detect outliers in numeric columns using specified method
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    method : str
        Method to use ('iqr', 'zscore', or 'isolation_forest')
    contamination : float
        Expected proportion of outliers (for isolation_forest)
        
    Returns:
    --------
    dict
        Dictionary with outlier indices and counts per column
    """
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return {}
    
    results = {"outlier_indices": {}, "outlier_counts": {}}
    
    if method == 'iqr':
        # IQR method
        for col in numeric_df.columns:
            q1 = numeric_df[col].quantile(0.25)
            q3 = numeric_df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = numeric_df[(numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)].index
            results["outlier_indices"][col] = outliers.tolist()
            results["outlier_counts"][col] = len(outliers)
            
    elif method == 'zscore':
        # Z-score method
        for col in numeric_df.columns:
            z_scores = np.abs(stats.zscore(numeric_df[col], nan_policy='omit'))
            outliers = numeric_df.index[z_scores > 3].tolist()
            results["outlier_indices"][col] = outliers
            results["outlier_counts"][col] = len(outliers)
            
    elif method == 'isolation_forest':
        # Isolation Forest - detect outliers across all features
        try:
            model = IsolationForest(contamination=contamination, random_state=42)
            # Fill NaN values to make algorithm work (not ideal but necessary)
            filled_df = numeric_df.fillna(numeric_df.mean())
            preds = model.fit_predict(filled_df)
            
            # -1 represents outliers in isolation forest
            outlier_indices = numeric_df.index[preds == -1].tolist()
            results["global_outliers"] = outlier_indices
            results["global_outlier_count"] = len(outlier_indices)
        except Exception as e:
            print(f"Error in isolation forest: {str(e)}")
    
    return results

def calculate_missing_data_stats(df):
    """
    Calculate detailed statistics about missing data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
        
    Returns:
    --------
    dict
        Dictionary with missing data statistics
    """
    results = {}
    
    # Overall missing data
    total_cells = df.size
    total_missing = df.isna().sum().sum()
    results["total_missing"] = int(total_missing)
    results["missing_percent"] = float(total_missing / total_cells * 100)
    
    # Missing data by column
    missing_by_column = df.isna().sum().to_dict()
    results["missing_by_column"] = {col: int(count) for col, count in missing_by_column.items()}
    
    # Missing data percentage by column
    missing_percent_by_column = (df.isna().mean() * 100).to_dict()
    results["missing_percent_by_column"] = {col: float(percent) for col, percent in missing_percent_by_column.items()}
    
    # Identify columns with high missing rates (>50%)
    high_missing = {col: percent for col, percent in missing_percent_by_column.items() if percent > 50}
    results["high_missing_columns"] = high_missing
    
    # Count rows with at least one missing value
    rows_with_na = df[df.isna().any(axis=1)]
    results["rows_with_missing"] = len(rows_with_na)
    results["rows_with_missing_percent"] = float(len(rows_with_na) / len(df) * 100)
    
    return results

def identify_skewed_features(df, threshold=0.5):
    """
    Identify features with significant skew
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    threshold : float
        Absolute skew threshold to consider feature as skewed
        
    Returns:
    --------
    dict
        Dictionary of skewed features and their skew values
    """
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return {}
    
    skew_values = numeric_df.skew().to_dict()
    skewed_features = {col: float(skew) for col, skew in skew_values.items() 
                      if abs(skew) > threshold and not pd.isna(skew)}
    
    return {
        "skewed_features": skewed_features,
        "positive_skew": {col: skew for col, skew in skewed_features.items() if skew > 0},
        "negative_skew": {col: skew for col, skew in skewed_features.items() if skew < 0}
    }

def calculate_variance_inflation_factor(df):
    """
    Calculate Variance Inflation Factor to detect multicollinearity
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with numeric columns
        
    Returns:
    --------
    dict
        Dictionary of VIF values per feature
    """
    try:
        # Select numeric columns and drop any with missing values
        numeric_df = df.select_dtypes(include=["number"])
        numeric_df = numeric_df.dropna()
        
        # Need at least 2 columns for VIF to make sense
        if numeric_df.shape[1] < 2:
            return {}
            
        # Calculate VIF for each feature
        vif_data = {}
        for i, col in enumerate(numeric_df.columns):
            try:
                X = numeric_df.values
                vif = variance_inflation_factor(X, i)
                vif_data[col] = float(vif)
            except Exception:
                vif_data[col] = None
                
        return {
            "vif_values": vif_data,
            "high_collinearity": {col: vif for col, vif in vif_data.items() if vif and vif > 5}
        }
    except Exception as e:
        print(f"Error calculating VIF: {str(e)}")
        return {}

def perform_pca_analysis(df, n_components=2):
    """
    Perform PCA analysis on the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with numeric features
    n_components : int
        Number of principal components to extract
        
    Returns:
    --------
    dict
        Dictionary with PCA results
    """
    try:
        # Select numeric columns and drop rows with missing values
        numeric_df = df.select_dtypes(include=["number"])
        numeric_df = numeric_df.dropna()
        
        if numeric_df.empty or numeric_df.shape[1] < 2:
            return {}
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Apply PCA
        n_components = min(n_components, numeric_df.shape[1], numeric_df.shape[0])
        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(scaled_data)
        
        # Prepare results
        results = {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "components": pca.components_.tolist(),
            "transformed_data": pcs.tolist()
        }
        
        # Feature importance
        feature_importance = {}
        for i, component in enumerate(pca.components_):
            feature_importance[f"PC{i+1}"] = {col: float(abs(importance)) 
                                             for col, importance in zip(numeric_df.columns, component)}
        
        results["feature_importance"] = feature_importance
        
        return results
    except Exception as e:
        print(f"Error in PCA analysis: {str(e)}")
        return {}

def perform_cluster_analysis(df, n_clusters=3):
    """
    Perform KMeans clustering on the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with numeric features
    n_clusters : int
        Number of clusters to form
        
    Returns:
    --------
    dict
        Dictionary with clustering results
    """
    try:
        # Select numeric columns and drop rows with missing values
        numeric_df = df.select_dtypes(include=["number"])
        numeric_df = numeric_df.dropna()
        
        if numeric_df.empty or numeric_df.shape[0] < n_clusters:
            return {}
            
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_data)
        
        # Calculate cluster statistics
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_df = numeric_df[labels == i]
            cluster_stats[f"cluster_{i}"] = {
                "size": len(cluster_df),
                "percentage": float(len(cluster_df) / len(numeric_df) * 100),
                "mean": cluster_df.mean().to_dict(),
                "std": cluster_df.std().to_dict()
            }
            
        # Get distances from cluster centers
        distances = []
        for i, point in enumerate(scaled_data):
            cluster = labels[i]
            center = kmeans.cluster_centers_[cluster]
            distance = np.linalg.norm(point - center)
            distances.append(float(distance))
            
        # Find exemplars (points closest to cluster centers)
        exemplars = {}
        for i in range(n_clusters):
            cluster_points = np.where(labels == i)[0]
            if len(cluster_points) > 0:
                cluster_distances = [distances[p] for p in cluster_points]
                exemplar_idx = cluster_points[np.argmin(cluster_distances)]
                exemplars[f"cluster_{i}"] = int(exemplar_idx)
        
        return {
            "labels": labels.tolist(),
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "cluster_stats": cluster_stats,
            "distances": distances,
            "exemplars": exemplars,
            "inertia": float(kmeans.inertia_)
        }
    except Exception as e:
        print(f"Error in cluster analysis: {str(e)}")
        return {}
