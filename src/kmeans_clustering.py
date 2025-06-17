# აუცილებელი ბიბლიოთეკების ჩატვირთვა
import pandas as pd
from sklearn.cluster import KMeans                 # KMeans ალგორითმი კლასტერიზაციისთვის
from sklearn.preprocessing import StandardScaler   # მახასიათებლების ნორმალიზაცია
from sklearn.metrics import confusion_matrix, accuracy_score  # შეფასების მეტრიკები
import matplotlib.pyplot as plt                    # გრაფიკების აგება
import seaborn as sns                              # უფრო ლამაზი ვიზუალიზაციები
import numpy as np
import joblib                                      # მოდელების შენახვა .pkl ფაილში

# ფაილის სახელები და პარამეტრები
FEATURE_CSV = 'frame_features.csv'     # ფაილი სადაც შენახულია კადრების მახასიათებლები
N_CLUSTERS = 2                         # რამდენ კლასტერად უნდა დაიყოს მონაცემები

# მახასიათებლების ჩატვირთვა
def load_features():
    df = pd.read_csv(FEATURE_CSV)  # მახასიათებლების წაკითხვა CSV ფაილიდან
    features = df[['mean_edge', 'std_edge']].values  # გამოსახულების სობელის საშუალო და სტანდარტული გადახრა
    labels = df['class'].values  # ჭეშმარიტი ლეიბლები (Basketball/Football)
    return df, features, labels

# კლასტერების ვიზუალიზაცია
def plot_clusters(df, cluster_labels):
    df['cluster'] = cluster_labels  # ახალ სვეტში ვამატებთ რომელ კლასტერს მიეკუთვნება თითოეული ჩანაწერი
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x='mean_edge',
        y='std_edge',
        hue='cluster',     # ფერით გამოყოფა კლასტერების მიხედვით
        palette='Set2',
        style='class',     # წერტილის სტილი კლასის მიხედვით (Basketball/Football)
        s=60
    )
    plt.title("K-Means კლასტერიზაცია გამოსახულების მახასიათებლებზე")
    plt.savefig('cluster_plot.png')  # შენახვა
    plt.show()

# კლასტერების გაწერა კონკრეტულ კლასებზე (უმრავლესობის ხმის მიხედვით)
def map_clusters_to_classes(true_labels, cluster_labels):
    """კლასტერების მინიჭება კონკრეტულ კლასებზე უმრავლესობის პრინციპით"""
    label_mapping = {}
    clusters = np.unique(cluster_labels)
    for cluster in clusters:
        indices = np.where(cluster_labels == cluster)[0]  # იმ ელემენტების ინდექსები, რომლებიც ეკუთვნის ამ კლასტერს
        cluster_true_labels = true_labels[indices]        # ამ ელემენტების ჭეშმარიტი ლეიბლები
        majority_label = pd.Series(cluster_true_labels).mode()[0]  # ყველაზე ხშირად განმეორებადი კლასი
        label_mapping[cluster] = majority_label
    return label_mapping

# მთავარი ფუნქცია
def main():
    df, features, true_labels = load_features()  # მონაცემების ჩატვირთვა

    # მახასიათებლების სტანდარტიზაცია (z-score normalization)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # KMeans კლასტერიზაციის შესრულება
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)

    # კლასტერის ლეიბლების შენახვა ფაილში
    df['cluster'] = cluster_labels
    df.to_csv(FEATURE_CSV, index=False)
    print(f"კლასტერები შენახულია ფაილში: {FEATURE_CSV}")

    # კლასტერების ვიზუალიზაცია
    plot_clusters(df.copy(), cluster_labels)

    # კლასტერების შესაბამისობის პოვნა რეალურ კლასებზე
    cluster_to_class = map_clusters_to_classes(true_labels, cluster_labels)
    predicted_labels = [cluster_to_class[c] for c in cluster_labels]

    # შერჩევის სიზუსტის შეფასება (Confusion Matrix + Accuracy)
    cm = confusion_matrix(true_labels, predicted_labels, labels=['Basketball', 'Football'])
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=['Basketball', 'Football'],
                yticklabels=['Basketball', 'Football'],
                cmap='Blues')
    plt.title("Confusion Matrix: K-Means პროგნოზი vs ჭეშმარიტი კლასი")
    plt.xlabel("პროგნოზი")
    plt.ylabel("ჭეშმარიტი")
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"\nსიზუსტე: {accuracy * 100:.2f}%")

    # მოდელის და სკეილერის შენახვა შემდეგისთვის
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(kmeans, 'kmeans_model.pkl')

# სკრიპტის გაშვების წერტილი
if __name__ == "__main__":
    main()
