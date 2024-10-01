import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Country-data.csv')

features = data.drop('country', axis=1)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)

pca_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['country'] = data['country']

explained_variance = pca.explained_variance_ratio_
print(f'Explained Variance by Component: {explained_variance}')

plt.figure(figsize=(12, 8))
palette = sns.color_palette("husl", len(pca_df['country'].unique()))
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', hue='country', data=pca_df, palette=palette, s=100)
plt.title('PCA of Countries with Distinct Colors')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.show()
