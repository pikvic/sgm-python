import re
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import request
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

root = Path()
uploads = root / 'uploads'
downloads = root / 'downloads'

sns.set_style("whitegrid")
sns.set_context("talk")

def check_and_load_file(filename):
    error = False
    try:
        df = pd.read_csv(uploads / filename)
        error = False
    except Exception as e:
        error = True
    if error:
        try:
            df = pd.read_excel(uploads / filename)
            error = False
        except Exception as e:
            error = True 
    if error:
        return {'error': 'Wrong file format!'}
    return df


def run_stats(filename, params):
    results = []
    df = check_and_load_file(filename)
    if isinstance(df, dict) and 'error' in df:
        return df
    column = params['column'] - 1
    data = data.iloc[:, column]
    stats = data.describe()
    stats.index.name = "Stats"
    output = 'result.csv'
    if params['transpose']:
        stats.T.to_csv(downloads / output)
    else:
        stats.to_csv(downloads / output)
    results.append(request.url_root + "downloads/" + output)
    if params['show_graph']:
        fig, ax = plt.subplots(2, 1, figsize=(6, 10))
        sns.distplot(data, ax=ax[0])
        sns.boxplot(data, ax=ax[1])
        figname = f'Column_{column + 1}.png'
        fig.savefig(downloads / figname, bbox_inches = 'tight')
        results.append(request.url_root + "downloads/" + figname)
        plt.close(fig)
    return results

def run_kmeans(filename, params):
    results = []
    df = check_and_load_file(filename)
    if isinstance(df, dict) and 'error' in df:
        return df

    if params['exclude']:
        pattern = r'^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$'
        if not re.search(pattern, params['exclude']):
            return {'error': 'Wrong exclude columns pattern!'}

        res = re.findall(r'\d+(?:-\d+)*', params['exclude'])
        columns = set()
        for r in res:
            if '-' in r:
                left, right = r.split('-')
                if left < right:
                    columns = columns | set(range(int(left) - 1, int(right)))
            else:
                columns.add(int(r) - 1)
        columns = list({i for i in range(len(df.columns))} - columns)
        data = df.iloc[:, columns]
    else:
        data = df.iloc[:, :]        
    
    if params['normalize']:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=params['n_clusters'], random_state=params['random_state'])
    labels = kmeans.fit_predict(data) + 1
    if params['add_result_columns']:
        df['Cluster'] = labels
        df.to_csv(downloads / 'kmeans_result.csv', index=False)
    else:
        new_df = pd.DataFrame(labels, columns=['Cluster'])
        new_df.to_csv(downloads / 'kmeans_result.csv', index=False)
    results.append(request.url_root + "downloads/" + 'kmeans_result.csv')
    if params['show_stats']:
        centers = kmeans.cluster_centers_
        inertia = kmeans.inertia_
    if params['show_graph']:
        pca = PCA(n_components=2)
        components = pca.fit_transform(data)
        fig, ax = plt.subplots()
        scatter = ax.scatter(components[:, 0], components[:, 1], c=labels)
        ax.set_title('Clustering In Principal Components')
        ax.set_xlabel('Principal Component 0')
        ax.set_ylabel('Principal Component 1')
        ax.legend(*scatter.legend_elements(), title="Clusters")
        fig.savefig(downloads / 'kmeans_figure.png', bbox_inches = 'tight')
        results.append(request.url_root + "downloads/" + 'kmeans_figure.png')
    return results

def run_pca(filename, params):
    results = []
    df = check_and_load_file(filename)
    if isinstance(df, dict) and 'error' in df:
        return df

    if params['exclude']:
        pattern = r'^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$'
        if not re.search(pattern, params['exclude']):
            return {'error': 'Wrong exclude columns pattern!'}

        res = re.findall(r'\d+(?:-\d+)*', params['exclude'])
        columns = set()
        for r in res:
            if '-' in r:
                left, right = r.split('-')
                if left < right:
                    columns = columns | set(range(int(left) - 1, int(right)))
            else:
                columns.add(int(r) - 1)
        columns = list({i for i in range(len(df.columns))} - columns)
        data = df.iloc[:, columns]
    else:
        data = df.iloc[:, :]   

    if params['normalize']:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    pca = PCA()
    pca.fit(data)
    
    components = pd.DataFrame(pca.components_, columns=data.columns)
    components.index.name = 'Component'
    components.to_csv(downloads / 'components.csv')
    results.append(request.url_root + "downloads/" + 'components.csv')

    pd.DataFrame({'Explained Variance': pca.explained_variance_,
              'Explained Variance Ratio': pca.explained_variance_ratio_}).to_csv(downloads / 'variance.csv', index=False)
    results.append(request.url_root + "downloads/" + 'variance.csv')
    
    params['show_graph'] = True
    if params['show_graph']:
        fig, ax = plt.subplots()
        ax.bar(list(range(1, pca.n_components_ + 1)), pca.explained_variance_ratio_)
        ax.set_title('Principal Component Analysis')
        ax.set_xlabel('Number of components')
        ax.set_ylabel('Explained Variance Ratio')
        fig.savefig(downloads / 'pca_figure_1.png', bbox_inches = 'tight')
        results.append(request.url_root + "downloads/" + 'pca_figure_1.png')
        fig, ax = plt.subplots()
        ax.plot(list(range(1, pca.n_components_ + 1)), np.cumsum(pca.explained_variance_ratio_))
        ax.set_title('Principal Component Analysis')
        ax.set_xlabel('Number of components')
        ax.set_ylabel('Cumulative Explained Variance Ratio')
        fig.savefig(downloads / 'pca_figure_2.png', bbox_inches = 'tight')
        results.append(request.url_root + "downloads/" + 'pca_figure_2.png')
    return results

def run_linear(filename, params):
    results = []
    output = 'result.csv'
    df = check_and_load_file(filename)
    if isinstance(df, dict) and 'error' in df:
        return df
    data = df.iloc[:, 2:]
    x_column_id = params['x_column'] - 1
    y_column_id = params['y_column'] - 1
    x_column_id = x_column_id if x_column_id < len(data.columns) else 0
    y_column_id = y_column_id if y_column_id < len(data.columns) else 1

    x = data.iloc[:, x_column_id].values.reshape(-1, 1)
    y = data.iloc[:, y_column_id].values.reshape(-1, 1)

    lr = LinearRegression()
    lr.fit(x, y)
    yhat = lr.predict(x)

    res = {
        'Parameter': ['Coefficient', 'Intercept', 'Mean Squared Error'],
        'Value': [lr.coef_[0][0], lr.intercept_[0], mean_squared_error(y, yhat)]
        }
    pd.DataFrame(res).to_csv(downloads / output, index=False)
    results.append(request.url_root + "downloads/" + output)
    
    fig, ax = plt.subplots()
    sns.regplot(x, y, yhat, ax=ax)
    ax.set(title=f'y = {lr.coef_[0][0]:.4f}x + {lr.intercept_[0]:.4f}')
    ax.set(xlabel=data.columns[x_column_id], ylabel=data.columns[y_column_id])
    figname = f'Regression_Column{x_column_id + 1}_Column{y_column_id + 1}.png'
    fig.savefig(downloads / figname, bbox_inches = 'tight')
    results.append(request.url_root + "downloads/" + figname)

    return results

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def run_hca(filename, params):
    results = []
    df = check_and_load_file(filename)
    if isinstance(df, dict) and 'error' in df:
        return df

    if params['exclude']:
        pattern = r'^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$'
        if not re.search(pattern, params['exclude']):
            return {'error': 'Wrong exclude columns pattern!'}

        res = re.findall(r'\d+(?:-\d+)*', params['exclude'])
        columns = set()
        for r in res:
            if '-' in r:
                left, right = r.split('-')
                if left < right:
                    columns = columns | set(range(int(left) - 1, int(right)))
            else:
                columns.add(int(r) - 1)
        columns = list({i for i in range(len(df.columns))} - columns)
        data = df.iloc[:, columns]
    else:
        data = df.iloc[:, :]   

    if params['normalize']:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model.fit(data)

    distances = pd.DataFrame(model.distances_, columns=['distance'])
    distances.index.name = 'Cluster Number'
    distances.to_csv(downloads / 'distances.csv')
    results.append(request.url_root + "downloads/" + 'distances.csv')

    fig, ax = plt.subplots()
    plot_dendrogram(model, truncate_mode='level', p=params['levels'], ax=ax)
    ax.set_title('Hierarchical Clustering Dendrogram')
    ax.set_xlabel('Number of points in node (or index of point if no parenthesis).')
    ax.set_ylabel('Distance')
    fig.savefig(downloads / 'hca_figure_1.png', bbox_inches = 'tight')
    results.append(request.url_root + "downloads/" + 'hca_figure_1.png')
    return results

