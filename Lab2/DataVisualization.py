# import libraries
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
import sklearn
import sklearn.decomposition

# load dataset
X = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# Other
sns.histplot(data=X, x="DEATH_EVENT")
plt.savefig('Otherplot.pdf')

# Create and Save Pairplot
sns.pairplot(data=X, hue="DEATH_EVENT")
plt.savefig('Pairplot.pdf')



# Format for PCA plot
y = X["DEATH_EVENT"]
X.drop("DEATH_EVENT", axis=1, inplace=True)

# Construct PCA to use for plot
pca = sklearn.decomposition.PCA(n_components=2)

# Perform the PCA transformation
X_2D = pca.fit_transform(X)

# Plot the scatterplot
sns.scatterplot(x=X_2D[:,0], y=X_2D[:,1], hue=y)

# Save the scatterplot
plt.savefig('PCAplot.pdf')