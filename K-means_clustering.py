import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#K-means from sklearn
from sklearn.cluster import KMeans

def prepro (X):
    W =[]
    for i in X:
        lines = i.strip().split('/')
        temp =(28-int(lines[0]))+(30*(int(lines[1])-1))+((100-int(lines[2])+17)*365)+59
        W.append(temp)                         
    final = pd.Series(W)
    return final
    
df = pd.read_csv('data_info.csv')
df['dob'] = prepro(df['dob'])
dataset_X = df.iloc[:,0:4].values
#first_name = df.iloc[:,0].values
#last_name = df.iloc[:,3].values
dataset_X[:,1] -= np.mean(dataset_X[:,1], axis = 0)
dataset_X[:,1] /= np.std(dataset_X[:,1], axis = 0)
dataset_X[:,1] = pd.to_numeric(dataset_X[:,1])
labelencoder = LabelEncoder()
dataset_X[:,2] = labelencoder.fit_transform(dataset_X[:,2])
dataset_X[:,0] = labelencoder.fit_transform(dataset_X[:,0])
dataset_X[:,3] = labelencoder.fit_transform(dataset_X[:,3])
dataset_X[:,0] = pd.to_numeric(dataset_X[:,0])
dataset_X[:,3] = pd.to_numeric(dataset_X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [0,3])
dataset_X = onehotencoder.fit_transform(dataset_X).toarray()

from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=15, kernel = 'rbf')
red_dataset_X = kpca.fit_transform(dataset_X)

wcss = []
for i in range(1,70):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(red_dataset_X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,70), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 30, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(red_dataset_X)
