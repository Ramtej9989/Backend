# Visualization functions for data insight dashboard
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
import pandas as pd
import io
import base64
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Define Power BI-like colors
POWER_BI_COLORS = ["#01B8AA", "#374649", "#FD625E", "#F2C80F", "#5F6B6D", "#8AD4EB", "#FE9666", "#A66999"]

# Helper function to convert figure to base64
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def create_correlation_heatmap(df):
    """
    Create correlation heatmap with Power BI styling
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with numeric columns
        
    Returns:
    --------
    str
        Base64 encoded image of the heatmap
    """
    # Select numeric columns
    df_num = df.select_dtypes(include=["number"])
    
    if df_num.empty or df_num.shape[1] < 2:
        return None
        
    # Calculate correlation matrix
    corr = df_num.corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Custom colormap similar to Power BI
    cmap = LinearSegmentedColormap.from_list(
        "power_bi", 
        ["#FD625E", "#FFFFFF", "#01B8AA"]
    )
    
    # Plot heatmap
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, 
               annot=True, fmt=".2f", linewidths=0.5,
               cbar_kws={"shrink": .8})
               
    plt.title("Correlation Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Convert to base64 and return
    result = fig_to_base64(plt.gcf())
    plt.close()
    
    return result

def create_pca_visualization(df):
    """
    Create PCA visualization with Power BI styling
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with numeric columns
        
    Returns:
    --------
    str
        Base64 encoded image of the PCA plot
    """
    # Select numeric columns and drop NAs
    df_num = df.select_dtypes(include=["number"]).dropna()
    
    if df_num.empty or df_num.shape[1] < 2 or df_num.shape[0] < 3:
        return None
    
    try:
        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_num)
        
        # Apply PCA
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(scaled_data)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.scatter(pcs[:,0], pcs[:,1], alpha=0.7, s=40, 
                   c=POWER_BI_COLORS[0], edgecolor='w', linewidth=0.5)
        
        # Add variance explained
        var_explained = pca.explained_variance_ratio_
        plt.xlabel(f"PC1 ({var_explained[0]:.2%} variance)", fontsize=12)
        plt.ylabel(f"PC2 ({var_explained[1]:.2%} variance)", fontsize=12)
        
        plt.title("Dimension Reduction (PCA)", fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Convert to base64 and return
        result = fig_to_base64(plt.gcf())
        plt.close()
        
        return result
    except Exception as e:
        print(f"Error in PCA visualization: {str(e)}")
        return None

def create_cluster_visualization(df, n_clusters=3):
    """
    Create cluster visualization with Power BI styling
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with numeric columns
    n_clusters : int
        Number of clusters to form
        
    Returns:
    --------
    str
        Base64 encoded image of the cluster plot
    """
    # Select numeric columns and drop NAs
    df_num = df.select_dtypes(include=["number"]).dropna()
    
    if df_num.empty or df_num.shape[1] < 2 or df_num.shape[0] < n_clusters:
        return None
    
    try:
        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_num)
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(scaled_data)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_data)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        
        # Plot each cluster with its own color
        for i in range(n_clusters):
            plt.scatter(pcs[labels==i, 0], pcs[labels==i, 1], 
                      alpha=0.7, s=50, 
                      color=POWER_BI_COLORS[i % len(POWER_BI_COLORS)],
                      edgecolor='w', linewidth=0.5,
                      label=f'Cluster {i+1}')
        
        # Add cluster centers
        centers = pca.transform(kmeans.cluster_centers_)
        plt.scatter(centers[:, 0], centers[:, 1], s=200, marker='X', 
                   c='black', alpha=0.8, label='Centroids')
        
        # Add labels and title
        var_explained = pca.explained_variance_ratio_
        plt.xlabel(f"PC1 ({var_explained[0]:.2%} variance)", fontsize=12)
        plt.ylabel(f"PC2 ({var_explained[1]:.2%} variance)", fontsize=12)
        plt.title("Data Segments (KMeans Clustering)", fontsize=14, fontweight='bold')
        plt.legend(loc='best', frameon=True, fancybox=True, framealpha=0.7)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Convert to base64 and return
        result = fig_to_base64(plt.gcf())
        plt.close()
        
        return result
    except Exception as e:
        print(f"Error in cluster visualization: {str(e)}")
        return None

def create_distribution_plots(df, columns=None, max_cols=6):
    """
    Create distribution plots for numeric columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    columns : list, optional
        Specific columns to plot (if None, use all numeric columns)
    max_cols : int
        Maximum number of columns to plot
        
    Returns:
    --------
    str
        Base64 encoded image of the distribution plots
    """
    # Select numeric columns
    df_num = df.select_dtypes(include=["number"])
    
    if df_num.empty:
        return None
    
    # Filter columns if specified
    if columns:
        cols = [c for c in columns if c in df_num.columns]
        if not cols:
            return None
        df_num = df_num[cols]
    
    # Limit to max_cols columns with highest variance
    if df_num.shape[1] > max_cols:
        variances = df_num.var()
        top_cols = variances.sort_values(ascending=False).index[:max_cols]
        df_num = df_num[top_cols]
    
    # Create figure with subplots
    n_cols = df_num.shape[1]
    fig, axes = plt.subplots(n_cols, 1, figsize=(8, n_cols * 2))
    
    # Convert to array if there's only one axis
    if n_cols == 1:
        axes = [axes]
    
    # Plot each column's distribution
    for i, col in enumerate(df_num.columns):
        sns.histplot(df_num[col].dropna(), kde=True, ax=axes[i], 
                    color=POWER_BI_COLORS[i % len(POWER_BI_COLORS)])
        axes[i].set_title(f"{col} Distribution", fontweight='bold')
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Convert to base64 and return
    result = fig_to_base64(fig)
    plt.close(fig)
    
    return result

def create_categorical_charts(df, column, chart_type='bar'):
    """
    Create charts for categorical columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    column : str
        Column to visualize
    chart_type : str
        Type of chart ('bar' or 'pie')
        
    Returns:
    --------
    str
        Base64 encoded image of the chart
    """
    if column not in df.columns:
        return None
    
    # Get value counts
    counts = df[column].value_counts()
    if counts.empty:
        return None
    
    # Limit to top 10 categories for readability
    if len(counts) > 10:
        others_sum = counts.iloc[10:].sum()
        counts = counts.iloc[:10]
        counts['Others'] = others_sum
    
    if chart_type == 'bar':
        # Create bar chart
        plt.figure(figsize=(8, 5))
        ax = counts.plot(kind="bar", color=POWER_BI_COLORS, edgecolor='none')
        ax.set_title(f"{column} Distribution", fontsize=14, fontweight='bold')
        ax.set_ylabel("Count")
        ax.set_xlabel("")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
    elif chart_type == 'pie':
        # Create pie chart
        plt.figure(figsize=(8, 6))
        wedges, texts, autotexts = plt.pie(
            counts, 
            labels=counts.index, 
            autopct='%1.1f%%',
            startangle=90, 
            colors=POWER_BI_COLORS[:len(counts)],
            wedgeprops=dict(width=0.6, edgecolor='w'),
            textprops={'fontsize': 10}
        )
        plt.setp(autotexts, size=9, weight="bold")
        plt.title(f"{column} Distribution", fontsize=14, fontweight='bold')
        plt.tight_layout()
    
    # Convert to base64 and return
    result = fig_to_base64(plt.gcf())
    plt.close()
    
    return result

def create_time_series_visualization(df, date_column, value_column):
    """
    Create time series visualization
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    date_column : str
        Column containing dates
    value_column : str
        Column containing values to plot
        
    Returns:
    --------
    str
        Base64 encoded image of the time series plot
    """
    if date_column not in df.columns or value_column not in df.columns:
        return None
    
    # Try to convert date column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        try:
            df = df.copy()
            df[date_column] = pd.to_datetime(df[date_column])
        except:
            return None
    
    # Sort by date and plot
    try:
        df_sorted = df.sort_values(by=date_column)
        
        plt.figure(figsize=(10, 5))
        plt.plot(df_sorted[date_column], df_sorted[value_column], 
                marker='o', markersize=3, linestyle='-', linewidth=2,
                color=POWER_BI_COLORS[0])
        
        plt.title(f"{value_column} over Time", fontsize=14, fontweight='bold')
        plt.xlabel(date_column, fontsize=12)
        plt.ylabel(value_column, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert to base64 and return
        result = fig_to_base64(plt.gcf())
        plt.close()
        
        return result
    except Exception as e:
        print(f"Error in time series visualization: {str(e)}")
        return None

def create_box_plots(df, columns=None, max_cols=6):
    """
    Create box plots for numeric columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    columns : list, optional
        Specific columns to plot (if None, use all numeric columns)
    max_cols : int
        Maximum number of columns to plot
        
    Returns:
    --------
    str
        Base64 encoded image of the box plots
    """
    # Select numeric columns
    df_num = df.select_dtypes(include=["number"])
    
    if df_num.empty:
        return None
    
    # Filter columns if specified
    if columns:
        cols = [c for c in columns if c in df_num.columns]
        if not cols:
            return None
        df_num = df_num[cols]
    
    # Limit to max_cols columns with highest variance
    if df_num.shape[1] > max_cols:
        variances = df_num.var()
        top_cols = variances.sort_values(ascending=False).index[:max_cols]
        df_num = df_num[top_cols]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Melt the dataframe for seaborn boxplot
    melted_df = pd.melt(df_num)
    sns.boxplot(x='variable', y='value', data=melted_df, 
              palette=POWER_BI_COLORS[:len(df_num.columns)])
    
    plt.title("Value Distribution by Feature", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("")
    plt.tight_layout()
    
    # Convert to base64 and return
    result = fig_to_base64(plt.gcf())
    plt.close()
    
    
