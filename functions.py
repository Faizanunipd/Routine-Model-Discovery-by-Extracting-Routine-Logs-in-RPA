import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator

import hdbscan
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances_argmin_min

import pm4py
from pm4py import read_xes
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

import warnings
warnings.filterwarnings('ignore')

# Set the maximum number of rows to display
pd.set_option('display.max_rows', 1000)  # Change 1000 to your desired value





def get_log_stats(log):
    # event log columns
    print("\nEvent log have total {} columns and {} log activities".format(log.shape[1], log.shape[0]))
    print("======================================================")
    
    print("\nEvent Log have the follwing columns:")
    print("====================================")
    print(list(log.columns))
    

def read_log(xes_file_path):
    # Import the XES file as an event log
    event_log = read_xes(xes_file_path)
    return event_log


def split_log(log, test_size=0.30, shuffle=False):
    train_log, test_log = train_test_split(log, test_size=test_size, random_state=42, shuffle=shuffle)
    return train_log, test_log


def train_evaluate_petri_net(training_log, testing_log):
    net, im, fm = pm4py.discover_petri_net_inductive(training_log, noise_threshold=0.2)
    return net, im, fm


def log_preprocessing(event_log):
    # Forward fill NaT values with the previous row's value
    event_log['time:timestamp'].fillna(method='ffill', inplace=True)
    event_log['time:timestamp'] = event_log['time:timestamp'].apply(lambda x : str(x))

    event_log['time:timestamp'] = event_log['time:timestamp'].apply(lambda x : x.split('.')[0])
    event_log['time:timestamp'] = event_log['time:timestamp'].apply(lambda x : x.split('+')[0])

    format_string = '%Y-%m-%d %H:%M:%S'
    event_log['time:timestamp'] = event_log['time:timestamp'].apply(lambda x : datetime.strptime(x, format_string))
    return event_log


def create_session(log):
    event_log = log_preprocessing(log)
    
    customers = list(event_log['case:concept:name'].unique())
    if 'lifecycle:transition' in event_log.columns:
        session_df = pd.DataFrame(columns=['case:concept:name', 'time:timestamp', 'lifecycle:transition', 'concept:name', 'Session'])
    else:
        session_df = pd.DataFrame(columns=['case:concept:name', 'time:timestamp', 'concept:name', 'Session'])

    session_number = 1

    for customer in customers:
        customer_trace = event_log[event_log['case:concept:name'] == customer]
        customer_trace = customer_trace.sort_values('time:timestamp')
        
        # Initialize session number
        # session_number = 1
        customer_trace['Session'] = 0

        # Iterate over the rows of customer_trace
        for i in range(len(customer_trace)):
            # For each activity, check if it's a "submit_button"
            if i == 0:
                customer_trace.at[customer_trace.index[i], 'Session'] = session_number
            else:
                # If activity is "submit_button", start a new session
                if customer_trace.iloc[i - 1]['concept:name'] == 'clickOK':
                    session_number += 1
                
                # Assign session number to the current activity
                customer_trace.at[customer_trace.index[i], 'Session'] = session_number

        # Concatenate this customer's trace into the final session_df
        session_df = pd.concat([session_df, customer_trace], ignore_index=True)
        session_number += 1  # will be remove in future

    # session_df.to_csv('saved_logs/log1.csv', index=False)
    return session_df


def freq_encoding(session_log):
    # Perform one-hot encoding
    one_hot_encoded = pd.get_dummies(session_log['concept:name'], prefix='activity')
    
    # Replace frequency with 1 where frequency is not 0
    one_hot_encoded = one_hot_encoded.applymap(lambda x: 1 if x > 0 else 0)
    
    if 'lifecycle:transition' in session_log.columns:
        df_encoded = pd.concat([session_log[['case:concept:name', 'lifecycle:transition', 'Session']], one_hot_encoded], axis=1)
        df_grouped = df_encoded.groupby(['case:concept:name', 'lifecycle:transition', 'Session']).sum().reset_index()
    else:
        df_encoded = pd.concat([session_log[['case:concept:name', 'Session']], one_hot_encoded], axis=1)
        df_grouped = df_encoded.groupby(['case:concept:name', 'Session']).sum().reset_index()

    df_grouped.to_csv('saved_logs/log2.csv', index=False)
    return df_grouped


def clustering_preprocessing(encoded_log):
    # Filter columns
    activity_columns = [col for col in encoded_log.columns if col.startswith('activity_')]
    
    # Select only the columns that start with 'activity_'
    features = encoded_log[activity_columns]

    # Standardize features
    # X = StandardScaler().fit_transform(features)
    # MinMax scaling
    # scaler = MinMaxScaler()
    # X = scaler.fit_transform(features)

    # Convert the transformed ndarray back to a DataFrame
    transformed_df = pd.DataFrame(features, columns=features.columns, index=features.index)
    return transformed_df


def elbow_clustering(encoded_log):
    features_log = clustering_preprocessing(encoded_log)
    
    # Compute K-Means for different values of k
    k_values = range(1, 15)  # Adjust the range of k as needed
    silhouette_scores = []
    inertia_scores = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features_log)
        if len(np.unique(labels)) > 3:
            silhouette_scores.append(silhouette_score(features_log, labels))
        inertia_scores.append(kmeans.inertia_)
    
    # Plot Inertia
    plt.plot(k_values, inertia_scores, 'rx-', label='Inertia')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Inertia vs Number of Clusters')
    plt.legend()
    plt.show()
    
    # Find the elbow point using KneeLocator
    knee_locator = KneeLocator(k_values, inertia_scores, curve="convex", direction="decreasing")
    optimal_k = knee_locator.knee
    if optimal_k is None:
        print("Use Default Value, No Knee Located!")
        optimal_k = 1
    
    print(f"Optimal number of clusters (k) based on the elbow method (Inertia): {optimal_k}")
    
    return optimal_k


def KMeans_Clusteirng(encoded_log, k=2):

    features_log = clustering_preprocessing(encoded_log)
    k = elbow_clustering(features_log)

    # Apply DBSCAN clustering
    kmeans_model = KMeans(n_clusters=k, random_state=42 )  
    encoded_log['KMeans_Cluster'] = kmeans_model.fit_predict(features_log)

    labels = kmeans_model.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    print('\nEstimated number of clusters: %d' % n_clusters_, "\n")
    # print('Estimated number of noise points: %d' % n_noise_)
    return encoded_log


def calculate_minpts(data, imgpath=""):
    avg_distances = []  # Store the average distances for each k
    dim = data.shape[1]  
    k_values = range(1, dim+1)
    # k_values = range(1, min(50, len(data)//2)) 
    
    # Iterate over values of k to calculate average distances
    for k in k_values:
        nbrs = NearestNeighbors(n_neighbors=k).fit(data)
        distances, _ = nbrs.kneighbors(data)
        avg_distance = np.mean(distances, axis=1)  # Average distance to k-neighbors
        avg_distances.append(np.mean(avg_distance))

    # Plot the average k-distance curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, avg_distances, marker='o', linestyle='-', color='blue', label='Average k-Distance')
    plt.xlabel('k (MinPts)', fontsize=12)
    plt.ylabel('Average k-Distance', fontsize=12)
    plt.title('Determining MinPts for DBSCAN Using k-Distance', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.show()

    # Use KneeLocator to identify the optimal MinPts (k)
    kneedle = KneeLocator(k_values, avg_distances, curve="convex", direction="increasing")
    optimal_k = kneedle.knee
    optimal_k = optimal_k if optimal_k != None else dim
    print(f"Optimal MinPts (k) identified: {optimal_k}")
    return optimal_k


def epsEstimate(enc, distance_method="sum", k=20, imgpath=""):
    # Calculate the distances to the k-th nearest neighbor
    print(distance_method)
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nbrs.fit(enc)
    distances, _ = nbrs.kneighbors(enc)
    print(distances[0])
    
    if distance_method=="sum":
        k_distances = np.sum(distances, axis=1)
        print("k-Distances Shape", k_distances.shape)
        print("k-Distances:", k_distances)

        # Min-Max Scaling
        min_val = np.min(k_distances)
        max_val = np.max(k_distances)

        if max_val - min_val == 0:
            # Avoid division by zero
            print("Warning: All distances have the same value. Returning zeros.")
            return np.zeros_like(k_distances)
        
        scaled_distances = (k_distances - min_val) / (max_val - min_val)
        k_distances = scaled_distances
        k_distances.sort()
    else:
        # Get the distances to the k-th nearest neighbor (k-distance)
        k_distances = np.mean(distances, axis=1)
        k_distances.sort()

    
    # Plot k-distance graph for visual inspection
    plt.figure(figsize=(10, 5))
    plt.plot(k_distances)
    plt.xlabel('Data Points sorted by distance to {}-th nearest neighbor'.format(k))
    plt.ylabel('{}-distance'.format(k))
    plt.title("K-distance Graph for Epsilon Estimation")
    plt.grid(True)
    plt.show()

    # # Detect the elbow point using KneeLocator
    knee_locator = KneeLocator(range(len(k_distances)), k_distances, curve="convex", direction="increasing")
    elbow_index = knee_locator.knee
    elbow_value = k_distances[elbow_index]
    elbow_value = elbow_value if elbow_index != None else 0.1
    print(f"Optimal epsilon (eps) detected at index {elbow_index} with value: {elbow_value}")
    return elbow_value


def DBSACN_Clusteirng(encoded_log, distance_method="sum", epsilon_value=0.7, minimum_samples=15):
    features_log = clustering_preprocessing(encoded_log)

    # Estimate eps and minimum samples
    minimum_samples = calculate_minpts(features_log)
    # minimum_samples = 8
    epsilon_value = epsEstimate(features_log, distance_method, minimum_samples)
    # minimum_samples = minPointsEstimate(features_log, epsilon_value)

    print(f"Epsilon value: {epsilon_value} and Minimum Points are: {minimum_samples}")

    # Apply DBSCAN clustering
    # dbscan_model = DBSCAN(eps=epsilon_value, min_samples=minimum_samples)  # Adjust eps and min_samples based on your data
    dbscan_model = hdbscan.HDBSCAN(min_cluster_size=int(features_log.shape[0]*0.05))
    encoded_log['DBSCAN_Cluster'] = dbscan_model.fit_predict(features_log)

    labels = dbscan_model.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    if n_clusters_ == 0:
        encoded_log['DBSCAN_Cluster'] = encoded_log['DBSCAN_Cluster'].replace({-1: 0})
        print("\nNo Cluster found, all points were noisy")

    print(encoded_log['DBSCAN_Cluster'].value_counts())

    return encoded_log


def remove_noise_samples(encoded_log):
    features_log = clustering_preprocessing(encoded_log)
    if 'DBSCAN_Cluster' in encoded_log.columns:
        cluster_column = 'DBSCAN_Cluster'
    else:
        cluster_column = 'KMeans_Cluster'

    features_log[cluster_column] = encoded_log[cluster_column]
    
    # Find noise indices
    noise_indices = encoded_log[encoded_log[cluster_column] == -1].index

    if len(noise_indices) >= 1:
    
        # Find the nearest cluster for each noise point
        nearest_cluster_indices, _ = pairwise_distances_argmin_min(
            features_log.iloc[noise_indices, :-1],  # Exclude the cluster label column
            features_log[features_log[cluster_column] != -1].iloc[:, :-1]  # Exclude noise points
        )
        
        # Merge noise samples into the nearest cluster
        encoded_log.loc[noise_indices, cluster_column] = encoded_log.loc[
            encoded_log[cluster_column] != -1, cluster_column
        ].iloc[nearest_cluster_indices].values
        
        # Check if there are still noise points
        remaining_noise_indices = encoded_log[encoded_log[cluster_column] == -1].index
        if len(remaining_noise_indices) > 0:
            print(f"\nThere are still {len(remaining_noise_indices)} noise points remaining.\n")
        else:
            print("\nAll noise points have been assigned to the nearest cluster.\n")

        print("\n", encoded_log[cluster_column].value_counts(), "\n")
    return encoded_log


def __label_activity(group):
    # print(group)
    if len(group) == 1:
        group.loc[group.index[0], 'Activity'] = 'start'
        new_activity = group.iloc[0].copy()  # Copy the first activity
        new_activity['Activity'] = 'end'
        return pd.concat([group, pd.DataFrame([new_activity])], ignore_index=True)
    else:
        group.loc[group.index[0], 'Activity_Status'] = 'start'
        group.loc[group.index[-1], 'Activity_Status'] = 'end'
        return group
    
    
def assign_clusters(session_log, cluster_log, cluster_map):
    if 'DBSCAN_Cluster' in cluster_log.columns:
        cluster_column = 'DBSCAN_Cluster'
    else:
        cluster_column = 'KMeans_Cluster'

    if 'lifecycle:transition' in session_log.columns:
        # Perform inner join
        merged_log = pd.merge(session_log, cluster_log, on=['case:concept:name', 'Session', 'lifecycle:transition'], how='inner')
        merged_log = merged_log[['case:concept:name', 'Session', 'lifecycle:transition', 'time:timestamp', 'concept:name', cluster_column]]
    else:
        # Perform inner join
        merged_log = pd.merge(session_log, cluster_log, on=['case:concept:name', 'Session'], how='inner')
        merged_log = merged_log[['case:concept:name', 'Session', 'time:timestamp', 'concept:name', cluster_column]]

    merged_log[cluster_column] = merged_log[cluster_column].replace(cluster_map)
    # merged_log.to_csv("saved_logs/log4.csv", index=False)
    merged_log.rename({'concept:name':'log_activity', cluster_column:'abstract_activity'}, axis=1, inplace=True)

    merged_log = merged_log.sort_values(by=['case:concept:name', 'Session', 'time:timestamp'])
    
    # Group the DataFrame by 'CustomerID' and 'Session', then apply the function
    new_df = merged_log.groupby(['case:concept:name', 'Session']).apply(__label_activity).reset_index(drop=True)
    
    # Select only the 'start' and 'end' activities
    new_df = new_df[new_df['Activity_Status'].isin(['start', 'end'])]

    activity_log = merged_log[['case:concept:name', 'time:timestamp','log_activity', 'abstract_activity']]
    activity_log.rename({'log_activity':'concept:name'}, axis=1, inplace=True)
    
    abstract_log = new_df[['case:concept:name', 'time:timestamp','abstract_activity', 'Activity_Status']]
    abstract_log.rename({'abstract_activity':'concept:name'}, axis=1, inplace=True)

    activity_log.to_csv('saved_logs/log5.csv', index=False)
    return activity_log, abstract_log


def clustering(event_log, params):
    encoding = params.get('encoding')
    clustering_tech = params.get('clustering')
    distance_method = params.get('method')
    print(clustering_tech)
    print(encoding)
    eps = params.get('eps')
    minSample = params.get('minSample')
    k = params.get('k')
    print("\n")
    if encoding == "freq":
        session_log = create_session(event_log)
        encoded_log = freq_encoding(session_log)
    else:
        session_log = create_session(event_log)
        encoded_log = freq_encoding(session_log)

    if clustering_tech == "DBSCAN":
        encoded_log = DBSACN_Clusteirng(encoded_log, distance_method, eps, minSample)
        clusters = encoded_log['DBSCAN_Cluster'].unique()
        # print(f"{encoded_log['DBSCAN_Cluster'].value_counts()}\n")

        encoded_log = remove_noise_samples(encoded_log)
        clusters = encoded_log['DBSCAN_Cluster'].unique()
    else:
        # k = f.elbow_clustering(encoded_log)
        encoded_log = KMeans_Clusteirng(encoded_log, k)
        clusters = encoded_log['KMeans_Cluster'].unique()
        print(f"{encoded_log['KMeans_Cluster'].value_counts()}\n")

    cluster_map = {c: f"routine_{c+1}" for c in clusters}
    return session_log, encoded_log, cluster_map