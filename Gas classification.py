import warnings
import time
timestamp = time.strftime("%Y%m%d-%H%M%S")
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.spatial import ConvexHull
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier, RandomForestRegressor,
    AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
)
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    mean_squared_error, r2_score
)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import umap
# After training and testing, add visualization code
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Load the data
file_path = "gas_test_2.xls"  # Please ensure the file path is correct
df = pd.read_excel(file_path)

# Separate the data and labels
X = df.iloc[:, 2:].values  # Feature data
y = df.iloc[:, 0].values  # Label data

# Feature selection (select the top 16 features)
selector = SelectKBest(score_func=f_classif, k=16)
X_new = selector.fit_transform(X, y)

# Define dimensionality reduction methods
pca_2d = PCA(n_components=2)
X_reduced_pca = pca_2d.fit_transform(X)
explained_variance_pca = pca_2d.explained_variance_ratio_ * 100  # Calculate the percentage of explained variance
print("PCA")
print(explained_variance_pca)

lda_2d = LDA(n_components=2)
X_reduced_lda = lda_2d.fit_transform(X, y)
explained_variance_lda = lda_2d.explained_variance_ratio_ * 100  # Calculate the percentage of explained variance for LDA               
print("LDA")
print(explained_variance_lda)

tsne_2d = TSNE(n_components=2, random_state=42)
X_reduced_tsne = tsne_2d.fit_transform(X)

umap_model = umap.UMAP(n_components=2, random_state=42)
X_reduced_umap = umap_model.fit_transform(X)

# LDA + t-SNE Dimensionality Reduction Combination
num_classes = len(np.unique(y))
lda_n_components = min(num_classes - 1, X.shape[1])
lda_2d_for_tsne = LDA(n_components=lda_n_components).fit_transform(X, y)
X_reduced_lda_tsne = TSNE(n_components=2, random_state=42).fit_transform(lda_2d_for_tsne)

# t-SNE + LDA Dimensionality Reduction Combination
tsne_3d = TSNE(n_components=3, random_state=42)
X_reduced_tsne_3d = tsne_3d.fit_transform(X)
lda_for_tsne_3d = LDA(n_components=2)
X_reduced_tsne_lda = lda_for_tsne_3d.fit_transform(X_reduced_tsne_3d, y)

# PCA + t-SNE Dimensionality Reduction Combination
X_reduced_pca_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_reduced_pca)

# PCA + LDA Dimensionality Reduction Combination
X_reduced_pca_lda = LDA(n_components=2).fit_transform(X_reduced_pca, y)

# PCA + UMAP Dimensionality Reduction Combination
X_reduced_pca_umap = umap_model.fit_transform(X_reduced_pca)

# t-SNE + UMAP Dimensionality Reduction Combination
X_reduced_tsne_umap = umap_model.fit_transform(X_reduced_tsne)

# LDA + UMAP Dimensionality Reduction Combination
X_reduced_lda_umap = umap_model.fit_transform(X_reduced_lda)
# t-SNE + PCA Dimensionality Reduction Combination
X_reduced_tsne_pca = pca_2d.fit_transform(X_reduced_tsne)
# LDA + PCA Dimensionality Reduction Combination
X_reduced_lda_pca = pca_2d.fit_transform(X_reduced_lda)  # First perform LDA dimensionality reduction, then use PCA for further dimensionality reduction


# Reduced-dimensional data dictionary
reduced_data = {
    'PCA': (X_reduced_pca, explained_variance_pca),
    't-SNE': (X_reduced_tsne, None),
    'LDA': (X_reduced_lda, explained_variance_lda),
    'UMAP': (X_reduced_umap, None),
    'LDA + t-SNE': (X_reduced_lda_tsne, None),
    't-SNE + LDA': (X_reduced_tsne_lda, None),
    'PCA + t-SNE': (X_reduced_pca_tsne, None),
    'PCA + LDA': (X_reduced_pca_lda, None),
    'PCA + UMAP': (X_reduced_pca_umap, None),
    't-SNE + UMAP': (X_reduced_tsne_umap, None),
    'LDA + UMAP': (X_reduced_lda_umap, None),
    't-SNE + PCA': (X_reduced_tsne_pca, None),  # t-SNE + PCA Dimensionality Reduction Combination
    'LDA + PCA': (X_reduced_lda_pca, None)  # LDA + PCA Dimensionality Reduction Combination

}



from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

# Set color   
scatter_colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(y))))
# Use KMeans for clustering
# Plot scatter plots for each dimensionality reduction method individually
for title, (data, explained_variance) in reduced_data.items():
    plt.figure(figsize=(8, 6))
    plt.title(f'{title} of the Top 16 Selected Features')

    # Use KMeans for clustering
    n_clusters = len(np.unique(y))
    print("n_clusters")
    print(n_clusters)
    # Use GaussianMixture for clustering
    
    # gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    # clusters = gmm.fit_predict(data)
    # probabilities = gmm.predict_proba(data)  # 计算每个数据点在各个簇中的概率

    n_clusters = 10  # Assume you manually set the number of clusters to 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    print(title)
    
    print("Data shape after PCA:", data.shape)
    print("Data shape after GMM clustering:", clusters.shape)
    print(f"{title} - Data shape after KMeans clustering:", data.shape)  # Confirm data shape
    print("Clustering labels shape:", clusters.shape)
    print("Unique number of clusters:", len(np.unique(clusters)))  # Confirm unique number of clusters
    # print("probabilities:", probabilities)  # Check the number of unique tags

    for idx in range(n_clusters):
        # Extract the points of the current cluster
        cluster_points = data[clusters == idx]
        # cluster_probs = probabilities[clusters == idx, idx]  # Get the probability of belonging to the current cluster
        # print(cluster_points)
        # Ensure that the number of points in each cluster does not exceed the number of points in the original data
        print(f"Cluster {idx} - Number of data points:", len(cluster_points))
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'gas_type {idx}', color=scatter_colors[idx])#,alpha=0.5 * cluster_probs + 0.2)  # 使用概率调整透明度)

        # # Get the mean and covariance matrix of each cluster for drawing ellipses
        # mean = kmeans.cluster_centers_[idx]
        # covar = np.cov(cluster_points, rowvar=False)
        # # Ensure covariance matrix is 2D for the ellipse calculations
        # if covar.ndim == 1:
        #     covar = np.expand_dims(covar, axis=0)  # Expand to 2D if needed
        # Use the mean and covariance matrix of each cluster to draw ellipses

        # Extract the data points of the current cluster
        cluster_points = data[clusters == idx]
        if len(cluster_points) < 2:
            print(f"Cluster {idx} - Number of data points is less than 2, skip ellipse drawing.")
            continue

        # Calculate the mean and covariance matrix of the cluster
        mean = np.mean(cluster_points, axis=0)
        covar = np.cov(cluster_points, rowvar=False)
        if covar.ndim == 1:
         covar = np.expand_dims(covar, axis=0)  # Expand to 2D if needed

        # Draw ellipses to represent the cluster distribution       
        eigenvalues, eigenvectors = np.linalg.eigh(covar)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 4 * np.sqrt(eigenvalues)  # Scale the ellipse to better cover the area. Originally it was 2
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                          edgecolor='black', facecolor=scatter_colors[idx], alpha=0.2)
        # ellipse = Ellipse(mean, width, height, angle, edgecolor='black', facecolor=scatter_colors[idx], alpha=0.2)
        plt.gca().add_patch(ellipse)

    if explained_variance is not None:
        plt.xlabel(f'{title} Component 1 ({explained_variance[0]:.2f}%)')
        plt.ylabel(f'{title} Component 2 ({explained_variance[1]:.2f}%)')
    else:
        plt.xlabel(f'{title} Component 1')
        plt.ylabel(f'{title} Component 2')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'class_save_fig/{title}_{timestamp}.png', dpi=300)
    # Save the original data to a CSV file
    original_data = pd.DataFrame(data, columns=[f'{title}_Component_{i + 1}' for i in range(data.shape[1])])
    original_data['Cluster'] = clusters
    original_data.to_csv(f'class_save_fig/{title}_original_data_{timestamp}.csv', index=False)
    # plt.show()

# Initialize the variable to store the best result
best_method = None
best_accuracy = 0
best_y_pred = None
best_y_test = None
accuracies = {}
best_reduced_data = None  # Data used to store the best dimensionality reduction method
# 4. Categorical Training: The data is divided into a training set and a test set
# Use RandomForest to perform classification training and evaluation on data after each dimensionality reduction method
print("\nAccuracy of classification after each dimensionality reduction method (RandomForest):")
for method, (data, _) in reduced_data.items():
    # Check if the number of samples in the reduced data matches the number of labels
    if data.shape[0] != len(y):
        print(f"Error: The number of samples in the reduced data ({data.shape[0]}) does not match the number of labels ({len(y)}).")
        continue
    # Split the dataset into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=42, stratify=y)
# 5. Categorical Training: Use Random Forest to train a classifier on the training set and evaluate it on the test set
    rf_model = RandomForestClassifier(max_depth=5, min_samples_split=5, n_estimators=50, random_state=42)
    rf_model.fit(X_train, y_train)
    # Predict the test set and evaluate
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[method] = accuracy  # Record the accuracy of each dimensionality reduction method
    # Output the classification report and accuracy
    print(f"\n{method} Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"{method} Accuracy: {accuracy:.4f}")
    # Determine if it is the best dimensionality reduction method
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_method = method
        best_y_pred = y_pred
        best_y_test = y_test
        best_reduced_data = data  # Store the data of the best dimensionality reduction method
# Output the best dimensionality reduction method
print(f"\nBest Dimensionality Reduction Method: {best_method}，Accuracy: {best_accuracy:.4f}")      


# Draw a bar chart of the accuracy of different dimensionality reduction methods
plt.figure(figsize=(10, 6))
bars=plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
plt.xlabel("Dimensionality Reduction Method")
plt.ylabel("Accuracy")
plt.title("Comparison of Classification Accuracy Across Dimensionality Reduction Methods")
plt.xticks(rotation=45)
# Add the accuracy value on top of each bar
for bar, accuracy in zip(bars, accuracies.values()):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{accuracy:.2f}',
             ha='center', va='bottom', alpha=0.7, fontsize=10)
plt.tight_layout()
plt.savefig('class_save_fig/different_dimension.png', dpi=300)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    best_reduced_data, y, test_size=0.3, random_state=42, stratify=y
)

# Create a classification model dictionary with 15 models added
classification_models = {
    "Random Forest": RandomForestClassifier(max_depth=5, min_samples_split=5, n_estimators=50, random_state=42),
    "SVM": SVC(kernel='linear', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "MLP": MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=300, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=50, random_state=42),
    "Quadratic Discriminant": QuadraticDiscriminantAnalysis(),
    "Nearest Centroid": NearestCentroid(),
    "Gaussian Process": GaussianProcessClassifier(random_state=42),
    "Bagging": BaggingClassifier(n_estimators=50, random_state=42)
}

from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score

# Ensure that each type of gas exists in the training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    best_reduced_data, y, test_size=0.3, random_state=42, stratify=y
)

# Create a classification model dictionary with 15 models added
classification_models = {
    "Random Forest": RandomForestClassifier(max_depth=5, min_samples_split=5, n_estimators=50, random_state=42),
    "SVM": SVC(kernel='linear', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "MLP": MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=300, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=50, random_state=42),
    "Quadratic Discriminant": QuadraticDiscriminantAnalysis(),
    "Nearest Centroid": NearestCentroid(),
    "Gaussian Process": GaussianProcessClassifier(random_state=42),
    "Bagging": BaggingClassifier(n_estimators=50, random_state=42)
}

# Initialize lists to store model names, RMSE, R², and accuracy values
model_names = []
rmse_values = []
r2_values = []
accuracy_values=[]
# Initialize variables to store the best model
best_model_name = None
best_model = None
best_accuracy = 0
best_r2 = -float('inf')
best_rmse = float('inf')
best_y_pred = None

# Iterate over all models and evaluate accuracy, R², and RMSE       
for model_name, model in classification_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    model_names.append(model_name)
    rmse_values.append(rmse)
    r2_values.append(r2)
    accuracy_values.append(accuracy)
    
    if r2 > best_r2:
        best_accuracy = accuracy
        best_r2 = r2
        best_rmse = rmse
        best_model_name = model_name
        best_model = model
        best_y_pred = y_pred
    
    conf_matrix = confusion_matrix(y_test, y_pred, normalize='true')
    conf_matrix_normalized = (conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True) * 100).round(1)
    conf_matrix_normalized[conf_matrix==0]= np.nan
    labels = np.array([
        [f"{conf_matrix_normalized[i, j]:.1f}%\n({conf_matrix[i, j]})" if conf_matrix[i, j] != 0 else ""
         for j in range(conf_matrix.shape[0])]
        for i in range(conf_matrix.shape[1])])
    print(labels)
    print(conf_matrix)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap='Blues', xticklabels=np.unique(y_pred),
                yticklabels=np.unique(y_pred), cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.show()

# Output the result of the best model
print(f"Best model: {best_model_name}")
print(f"Accuracy: {best_accuracy:.4f}")
print(f"R² Score: {best_r2:.4f}")
print(f"RMSE: {best_rmse:.4f}")


print(y_test)
print(best_y_pred)

# Confusion matrix of the best model
conf_matrix = confusion_matrix(y_test, best_y_pred, normalize='true')
# Normalize the confusion matrix to show percentages        
conf_matrix_normalized = (conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True) * 100).round(1)
labels = np.array([
        [f"{conf_matrix_normalized[i, j]:.1f}%\n({conf_matrix[i, j]})" if conf_matrix[i, j] != 0 else ""
         for j in range(conf_matrix.shape[0])]
        for i in range(conf_matrix.shape[1])    ])
print(labels)
print(conf_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap='Blues', xticklabels=np.unique(best_y_pred),yticklabels=np.unique(best_y_pred), cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.show()

fig, ax1 = plt.subplots(figsize=(12, 6))

bars = ax1.bar(model_names, rmse_values, color='skyblue', label='RMSE')
ax1.set_xlabel('Models')
ax1.set_ylabel('RMSE')
ax1.set_xticklabels(model_names, rotation=45)

ax2 = ax1.twinx()
ax2.plot(model_names, r2_values, color='salmon', marker='o', label='R²')
ax2.set_ylabel('R² Score')

r2_max = max(r2_values) + 1
r2_min = min(r2_values) - 0.1
ax2.set_yticks(np.arange(r2_min, r2_max, 0.5))  

for bar, rmse in zip(bars, rmse_values):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{rmse:.2f}', ha='center', va='bottom',alpha=0.7,fontsize=10)

for i, r2 in enumerate(r2_values):
    ax2.text(i, r2, f'{r2:.2f}', ha='center', va='bottom', color='salmon', alpha=0.7,fontsize=10)
plt.title('Comparison of RMSE and R² Across Different Models')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
accuracy_bars = plt.bar(model_names, accuracy_values, color='lightgreen')
plt.xlabel('Models', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Accuracy Comparison Across Different Models', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)


for bar, accuracy in zip(accuracy_bars, accuracy_values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{accuracy:.2f}',
             ha='center', va='bottom', alpha=0.7, fontsize=10)

plt.tight_layout()  #   
plt.show()