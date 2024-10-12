# SCT_ML_2
SKILLCRAFT TECHNOLOGY INTERNSHIP
# Customer Segmentation Using K-Means Clustering

## Project Overview
This project uses the K-Means clustering algorithm to segment customers based on various features such as gender, age, annual income, and spending score. The dataset used for this project is the "Mall_Customers.csv," which contains data about customers from a shopping mall.

The project involves:
1. Data cleaning and preprocessing (handling missing values and encoding categorical data).
2. Feature selection and scaling of data.
3. Determining the optimal number of clusters using the Elbow Method.
4. Performing K-Means clustering.
5. Visualizing the clusters.
6. Calculating the silhouette score to evaluate the clustering performance.

## Libraries Used
- `pandas`: For data manipulation.
- `numpy`: For numerical operations.
- `matplotlib`: For data visualization.
- `sklearn`: For machine learning utilities such as clustering and data preprocessing.

## Getting Started

### 1. Install Required Libraries
If you don't have the required libraries installed, you can install them using pip:
```bash
pip install pandas numpy matplotlib scikit-learn
```

### 2. Load the Dataset
Make sure you have the dataset named `Mall_Customers.csv` in the same directory as this script or update the file path accordingly.

```python
df = pd.read_csv('Mall_Customers.csv')
```

### 3. Data Cleaning and Preprocessing

- **Handling Missing Values**: 
  Missing values in numeric columns are filled using the median, and categorical columns are filled using the mode.

- **Encoding Categorical Data**: 
  The 'Gender' column is encoded as `Male: 1, Female: 0`.

- **Feature Selection**: 
  We select features relevant for clustering, including 'Gender', 'Age', 'Annual Income', and 'Spending Score'.

- **Scaling**: 
  Data is scaled using `StandardScaler` to ensure all features contribute equally to the clustering.

### 4. Elbow Method for Optimal Clusters
We use the Elbow Method to determine the optimal number of clusters by plotting the within-cluster sum of squares (WCSS) for different numbers of clusters. The "elbow" point indicates the optimal number of clusters.

```python
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled_cleaned)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
```

### 5. K-Means Clustering
The optimal number of clusters (in this case, 5 clusters) is used to perform K-Means clustering on the scaled and imputed data.

```python
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled_cleaned)
```

### 6. Visualizing the Clusters
The clusters are visualized using a scatter plot, where the x-axis represents 'Annual Income' and the y-axis represents 'Spending Score'. The cluster centroids are marked with red crosses.

```python
plt.scatter(X_scaled[:, 1], X_scaled[:, 2], c=y_kmeans, s=50, cmap='viridis')
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 1], centroids[:, 2], c='red', s=200, alpha=0.5, marker='X')
plt.show()
```

### 7. Evaluating the Clustering Performance (Silhouette Score)
The silhouette score is calculated to evaluate the clustering quality. A higher silhouette score indicates better-defined clusters.

```python
sil_score = silhouette_score(X_scaled_cleaned, y_kmeans)
print(f"Silhouette Score: {sil_score}")
```

### 8. Output
The dataset is augmented with the cluster labels and displayed:

```python
df['Cluster'] = y_kmeans
print(df.head())
```

## Files
- `Mall_Customers.csv`: Input dataset containing customer data.
- `customer_segmentation.py`: Python script performing customer segmentation using K-Means clustering.

## Example Output
```
Dataset Preview:
   CustomerID  Gender  Age  Annual Income (k$)  Spending Score (1-100)
0           1       M   19                  15                     39
1           2       M   21                  15                     81
2           3       F   20                  16                     6
3           4       F   23                  16                    77

Missing values after filling:
CustomerID                    0
Gender                         0
Age                            0
Annual Income (k$)             0
Spending Score (1-100)         0
dtype: int64

Clustered Data:
   CustomerID  Gender  Age  Annual Income (k$)  Spending Score (1-100)  Cluster
0           1       1   19                  15                     39        0
1           2       1   21                  15                     81        2
2           3       0   20                  16                      6        1
3           4       0   23                  16                     77        4
```

## Conclusion
This project demonstrates customer segmentation using the K-Means clustering algorithm. It includes steps for data cleaning, feature selection, scaling, finding optimal clusters, clustering, and visualizing the results.
