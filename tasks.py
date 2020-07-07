import logging
import re
import seaborn as sns
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import config

logging.basicConfig(filename='tasks.log', filemode='w', level=logging.DEBUG)

def clear_files_for_job(job_id):
    path = config.UPLOAD_DIR / job_id
    if path.exists():
        for f in path.iterdir():
            f.unlink()
        path.rmdir()
    path = config.DOWNLOAD_DIR / job_id
    if path.exists():
        for f in path.iterdir():
            f.unlink()
        path.rmdir()

def get_or_create_dir(root, job_id):
    logging.DEBUG("Get or create dir")
    path = root / job_id
    if not path.exists():
        path.mkdir()
    return path

def upload_file(url, job_id):
    logging.DEBUG(f"Upload file: {url}, {job_id}")
    try:
        root = get_or_create_dir(config.UPLOAD_DIR, job_id)
        logging.DEBUG(f'Root: {root}')
        logging.DEBUG(f'Dirs: {list(config.UPLOAD_DIR.iterdir())}')
        res = requests.get(url)
        logging.DEBUG(f'Res: {res}')

        if res.ok:
            name = url.split('/')[-1]
            logging.DEBUG(f'Name: {name}')
            with open(root / name, 'wb') as f:
                f.write(res.content)
            logging.DEBUG(f'Writed!')
        else:
            logging.DEBUG(f'Exception!')
            raise Exception
    except:
        logging.DEBUG(f'Except')
        return {'success': False, 'error': f'Can not upload file from {url}'}
    logging.DEBUG(f'Success')
    return {'success': True, 'file_path': f'{root / name}'}

def check_file(filename):
    error = False
    file_format = None
    try:
        df = pd.read_csv(filename)
        error = False
        file_format = 'csv'
    except Exception as e:
        error = True
    if error:
        try:
            df = pd.read_excel(filename)
            error = False
            file_format = 'excel'
        except Exception as e:
            error = True 
    if error:
        return {'success': False, 'error': 'Wrong file format! Excel or csv excepted.'}
    return {'success': True, 'file_format': file_format}

def get_dataframe(filename, file_format):
    try:
        if file_format == 'excel':
            df = pd.read_excel(filename)
        elif file_format == 'csv':
            df = pd.read_csv(filename)
    except:
        return {'success': False, 'error': 'Can not get dataframe!'}
    return {'success': True, 'dataframe': df}

def validate_input_and_get_dataframe(url, job_id):
    logging.DEBUG("Validate Input")
    res = upload_file(url, job_id)
    if not res['success']:
        return res
    filename = res['file_path']
    res = check_file(filename)
    if not res['success']:
        return res
    res = get_dataframe(filename, res['file_format'])
    if not res['success']:
        return res
    return res

def run_stats(params):
    # common
    logging.DEBUG("Run stats started")
    results = []
    job_id = params['job_id']
    res = validate_input_and_get_dataframe(params['url'], job_id)
    if not res['success']:
        return res
    df = res['dataframe']
    root = get_or_create_dir(config.DOWNLOAD_DIR, job_id)

    # specific
    try:
        column = params['column'] - 1
        data = df.iloc[:, column]
    except:
        return {'success': False, 'error': 'Wrong column parameter!'}
    stats = data.describe()
    stats.index.name = "Stats"
    
    try:
        output = 'result.csv'
        file_path = root / output
        if params['transpose']:
            stats.T.to_csv(file_path)
        else:
            stats.to_csv(file_path)
        results.append(str(file_path))
    except:
        return {'success': False, 'error': 'Error while saving result!'}

    if params['showgraph']:
        try:
            sns.set_style("whitegrid")
            sns.set_context("talk")
            fig, ax = plt.subplots(2, 1, figsize=(6, 10))
            sns.distplot(data, ax=ax[0])
            sns.boxplot(data, ax=ax[1])
            figname = f'Column_{column + 1}.png'
            file_path = root / figname
            fig.savefig(file_path, bbox_inches = 'tight')
            results.append(str(file_path))
            plt.close(fig)
        except:
            return {'success': False, 'error': 'Error while showing graph!'}
    return {'ready': True, 'results': results}

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


def run_pca(params):
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

    columns = data.columns
    if params['normalize']:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    try:
        file_path = root / 'components.csv'
        pca = PCA()
        pca.fit(data)
        components = pd.DataFrame(pca.components_, columns=columns)
        components.index.name = 'Component'
        file_path = root / 'components.csv'
        components.to_csv(file_path)
        results.append(str(file_path))

        file_path = root / 'variance.csv'
        pd.DataFrame({'Explained Variance': pca.explained_variance_,
                'Explained Variance Ratio': pca.explained_variance_ratio_}).to_csv(file_path, index=False)
        results.append(str(file_path))
    except:
        return {'success': False, 'error': 'Error while saving result!'}    

    params['showgraph'] = True
    if params['showgraph']:
        try:
            fig, ax = plt.subplots()
            ax.bar(list(range(1, pca.n_components_ + 1)), pca.explained_variance_ratio_)
            ax.set_title('Principal Component Analysis')
            ax.set_xlabel('Number of components')
            ax.set_ylabel('Explained Variance Ratio')
            file_path = root / 'pca_figure_1.png'
            fig.savefig(file_path, bbox_inches = 'tight')
            results.append(str(file_path))
            fig, ax = plt.subplots()
            ax.plot(list(range(1, pca.n_components_ + 1)), np.cumsum(pca.explained_variance_ratio_))
            ax.set_title('Principal Component Analysis')
            ax.set_xlabel('Number of components')
            ax.set_ylabel('Cumulative Explained Variance Ratio')
            file_path = root / 'pca_figure_2.png'
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

    try:
        fig, ax = plt.subplots()
        plot_dendrogram(model, truncate_mode='level', p=params['levels'], ax=ax)
        ax.set_title('Hierarchical Clustering Dendrogram')
        ax.set_xlabel('Number of points in node (or index of point if no parenthesis).')
        ax.set_ylabel('Distance')
        file_path = root / 'hca_figure_1.png'
        fig.savefig(file_path, bbox_inches = 'tight')
        results.append(str(file_path))
    except:
        return {'success': False, 'error': 'Error while showing graph!'}
    return {'ready': True, 'results': results}

def run_linear(params):
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
        data = df.iloc[:, :]
        x_column_id = params['xcolumn'] - 1
        y_column_id = params['ycolumn'] - 1
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
        file_path = root / 'result.csv'
        pd.DataFrame(res).to_csv(file_path, index=False)
        results.append(str(file_path))
    except:
        return {'success': False, 'error': 'Error while saving result!'}

    try:
        fig, ax = plt.subplots()
        sns.regplot(x, y, yhat, ax=ax)
        ax.set(title=f'y = {lr.coef_[0][0]:.4f}x + {lr.intercept_[0]:.4f}')
        ax.set(xlabel=data.columns[x_column_id], ylabel=data.columns[y_column_id])
        figname = f'Regression_Column{x_column_id + 1}_Column{y_column_id + 1}.png'
        file_path = root / figname
        fig.savefig(file_path, bbox_inches = 'tight')
        results.append(str(file_path))
    except:
        return {'success': False, 'error': 'Error while showing graph!'}

    return {'ready': True, 'results': results}