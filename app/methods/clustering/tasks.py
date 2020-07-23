import re
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

import app.core.config as config
from app.core.tasks import validate_input_and_get_dataframe, get_or_create_dir

def run_kmeans(params):
    # common
    results = []
    job_id = params['job_id']
    res = validate_input_and_get_dataframe(params['url'], job_id)
    if not res['success']:
        return res
    df = res['dataframe']
    root = get_or_create_dir(config.DOWNLOAD_DIR, job_id)

    # specific
    try:
        if params['exclude']:
            pattern = r'^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$'
            if not re.search(pattern, params['exclude']):
                return {'success': False, 'error': 'Wrong exclude columns pattern!'}

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
    except:
        return {'success': False, 'error': 'Error while excluding columns!'}    

    if params['normalize']:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=params['nclusters'], random_state=params['randomstate'])
    labels = kmeans.fit_predict(data) + 1

    try:
        file_path = root / 'kmeans_result.csv'
        if params['addresultcolumns']:
            df['Cluster'] = labels
            df.to_csv(file_path, index=False)
        else:
            new_df = pd.DataFrame(labels, columns=['Cluster'])
            new_df.to_csv(file_path, index=False)
        results.append(str(file_path))
    except:
        return {'success': False, 'error': 'Error while saving result!'}    

    if params['showstats']:
        centers = kmeans.cluster_centers_
        inertia = kmeans.inertia_
    print(params)
    if params['showgraph']:
        try:
            file_path = root / 'kmeans_figure.png'
            pca = PCA(n_components=2)
            components = pca.fit_transform(data)
            fig, ax = plt.subplots()
            scatter = ax.scatter(components[:, 0], components[:, 1], c=labels)
            ax.set_title('Clustering In Principal Components')
            ax.set_xlabel('Principal Component 0')
            ax.set_ylabel('Principal Component 1')
            ax.legend(*scatter.legend_elements(), title="Clusters")
            fig.savefig(file_path, bbox_inches = 'tight')
            results.append(str(file_path))
        except:
            return {'success': False, 'error': 'Error while showing graph!'}
    return {'ready': True, 'results': results}


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

def run_hca(params):
    # common
    results = []
    job_id = params['job_id']
    res = validate_input_and_get_dataframe(params['url'], job_id)
    if not res['success']:
        return res
    df = res['dataframe']
    root = get_or_create_dir(config.DOWNLOAD_DIR, job_id)

    # specific
    try:
        if params['exclude']:
            pattern = r'^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$'
            if not re.search(pattern, params['exclude']):
                return {'success': False, 'error': 'Wrong exclude columns pattern!'}

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
    except:
        return {'success': False, 'error': 'Error while excluding columns!'}    

    if params['normalize']:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model.fit(data)
    try:
        distances = pd.DataFrame(model.distances_, columns=['distance'])
        distances.index.name = 'Cluster Number'
        file_path = root / 'distances.csv'
        distances.to_csv(file_path)
        results.append(str(file_path))
    except:
        return {'success': False, 'error': 'Error while saving result!'}
    params['levels'] = 4
    try:
        fig, ax = plt.subplots()
        plot_dendrogram(model, truncate_mode='level', p=params['levels'], ax=ax)
        ax.set_title('Hierarchical Clustering Dendrogram')
        ax.set_xlabel('Number of points in node (or index of point if no parenthesis).')
        ax.set_ylabel('Distance')
        file_path = root / 'hca_figure_1.png'
        fig.savefig(file_path, bbox_inches = 'tight')
        results.append(str(file_path))
    except Exception as e:
        return {'success': False, 'error': f'Error while showing graph! Exception: {e}'}
    return {'ready': True, 'results': results}