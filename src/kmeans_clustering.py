import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib


FEATURE_CSV = 'frame_features.csv'
N_CLUSTERS = 2

def load_features():
    df = pd.read_csv(FEATURE_CSV)
    features = df[['mean_edge', 'std_edge']].values
    labels = df['class'].values
    return df, features, labels

def plot_clusters(df, cluster_labels):
    df['cluster'] = cluster_labels
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x='mean_edge',
        y='std_edge',
        hue='cluster',
        palette='Set2',
        style='class',
        s=60
    )
    plt.title("K-Means Clustering on Frame Edge Features")
    plt.savefig('cluster_plot.png')
    plt.show()

def map_clusters_to_classes(true_labels, cluster_labels):
    """Simple mapping based on majority vote in each cluster."""
    label_mapping = {}
    clusters = np.unique(cluster_labels)
    for cluster in clusters:
        indices = np.where(cluster_labels == cluster)[0]
        cluster_true_labels = true_labels[indices]
        majority_label = pd.Series(cluster_true_labels).mode()[0]
        label_mapping[cluster] = majority_label
    return label_mapping

def main():
    df, features, true_labels = load_features()

    # Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)

    df['cluster'] = cluster_labels
    df.to_csv(FEATURE_CSV, index=False)
    print(f"Cluster assignments saved to {FEATURE_CSV}")

    plot_clusters(df.copy(), cluster_labels)

    cluster_to_class = map_clusters_to_classes(true_labels, cluster_labels)
    predicted_labels = [cluster_to_class[c] for c in cluster_labels]

    cm = confusion_matrix(true_labels, predicted_labels, labels=['Basketball', 'Football'])
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=['Basketball', 'Football'],
                yticklabels=['Basketball', 'Football'],
                cmap='Blues')
    plt.title("Confusion Matrix: K-Means Predicted vs Ground Truth")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(kmeans, 'kmeans_model.pkl')

    

if __name__ == "__main__":
    main()
