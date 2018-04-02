# Cluster-Analysis

Scenario: Variation in names leads to difficulty in identifying a unique person and hence deduplication 
of records is an unsolved challenge. The problem becomes more complicated in cases where 
data is coming from multiple sources. Following variations are same as Vladimir Frometa: 
 
Vladimir Antonio Frometa Garo 
Vladimir A Frometa Garo 
Vladimir Frometa 
Vladimir Frometa G 
Vladimir A Frometa 
Vladimir A Frometa G 

Problem Statement: Train a model to identify unique patients in the sample dataset

**Data Preprocessing**

We have processed a function which takes a input as person date of birth and return its age in days.
Used basic label encoding to gender feature and one-hot encoding to first-name and last-name feature.
Converting the age in days leads to high range of values compare to other features in the dataset.So we need to normalize each feature value of a feature vector in order to not get conditioned by features with wider range of possible values when computing distances.

Standard-scaler has been used for scaling the feature.

**Machine Learning Models**


This problem can be dealt with many unsupervised techniques in machine learning like K-means, K-medians and linear and non-linear dimensionality reduction techniques.
Apart from unsupervised learning techniques we can also use cosine similarity between the last name and the whole name of patient which we will get after adding the first name and last name, followed by computing levenshtein distance between them. After computing the levenshtein distance, our dataset has one new feature which shows distance between the names. But this approach will computationally expensive as we dealt with large dataset.
The approach we are utilizing here is **K-means clustering**, in which we are willing to cluster all similar names in the cluster based on the features we have.

**K-Means Clustering Alorithm**

The K-means clustering is unsupervised algorithm used to group different object into clusters.The K-means clustering involves following steps:
1.Define the number of clusters
2.Determine the centroid coordinate
3.Determine the distance of each object to the centroids
4.Group the objects based on minimum distance


In order to find optimum cluster in our dataset, we can use Elbow Method.
In Elbow method,The idea of the Elbow method is to run k-means clustering on the dataset for a range of values of k (say, k from 1 to 40 in the examples above), and for each value of k calculate the sum of squared errors. then the "elbow" on the arm is the value of k that is the best. The idea is that we want a small SSE, but that the SSE tends to decrease toward 0 as we increase k.So, our goal is to choose a small value of k that still has a low SSE, and the elbow usually represents where we start to have diminishing returns by increasing k.
Using elbow method, I found approximately 30 clusters are optimum to describe out dataset

To prevent from random initialization trap, we have used “K-means++” initialization in our model
As we can see that, After 30 clusters the sum of squared errors (SSE) do not seems to vary anymore.

**Measuring Cluster Quality**

Once, we have finalised the optimum number of clusters by Elbow method. Dimensional reduction technique (Principal Component Analysis) is used to visualise the unique clusters in 2 dimensional space.
Goodness of clustering is evaluated by considering how well the clusters are separated and how compact the cluster are, e.g., levenshtein distance between the centroid and their points in the individual distance


