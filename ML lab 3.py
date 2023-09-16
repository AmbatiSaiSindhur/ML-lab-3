#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
df=pd.read_csv(r"C:\class\projects\sem 5\Machine Learning\Custom_CNN_Features.csv")
data=df.drop(['Filename'], axis=1)
data


# In[58]:


import numpy as np
a = data[data['Label'] == 0]
b= data[data['Label'] == 1]
print(f'spread A: {a}')
print(f'spread B: {b}')
intraa = np.var(a[['f2', 'f3']], ddof=1)
intrab = np.var(b[['f4', 'f5']], ddof=1)
meanb= np.mean(b[['f4', 'f5']], axis=0)
meana= np.mean(a[['f2', 'f3']], axis=0)
distance = np.linalg.norm(meana - meanb)
print(f'distance between A and B: {distance}')


# In[59]:


grouped = data.groupby('Label')

# Calculate the class centroids (mean) for each class
centroids = {}
for label, group_data in grouped:
    class_mean = group_data[['f2', 'f3']].mean(axis=0)
    centroids[label] = class_mean
# Print the class centroids
for label, centroid in centroids.items():
    print(f'Label {label} Centroid: {centroid.values}')
print(grouped)


# In[61]:


grouped = data.groupby('Label')

# Calculate the standard deviation for each class
standarddeviations = {}
for clabel, group_data in grouped:
    class_std = group_data[['f2', 'f3']].std(axis=0)
    standarddeviations[clabel] = class_std

# Print the standard deviations for each class
for clabel, std_deviation in standarddeviations.items():
    print(f'SD for Label {clabel}: {std_deviation.values}')


# In[70]:


grouped = data.groupby('Label')

# Calculate the mean vectors (centroids) for each class
centroids = {}
for clabel, group_data in grouped:
    mean = group_data[['f2', 'f3']].mean(axis=0)
    centroids[clabel] = mean

# Calculate the distance between mean vectors of different classes
clabels = list(centroids.keys())
num_classes = len(clabels)
distances = {}

for i in range(num_classes):
    for j in range(i + 1, num_classes):
        label1 = clabels[i]
        class_label2 = clabels[j]
        distance = np.linalg.norm(centroids[label1] - centroids[class_label2])
        distances[(label1, class_label2)] = distance

# Print the distances between mean vectors
for (label1, class_label2), distance in distances.items():
    print(f'Distance between Label {label1} and Label {class_label2}: {distance}')


# In[83]:


import numpy as np
import matplotlib.pyplot as plt


featuredata = df['f16']

# Define the number of bins (buckets) for the histogram
num_bins = 5

# Calculate the histogram data (hist_counts) and bin edges (bin_edges)
hist_counts, bin_edges = np.histogram(featuredata, bins=num_bins)

# Calculate the mean and variance of 'Feature1'
mean_feature1 = np.mean(featuredata)
variance_feature1 = np.var(featuredata, ddof=1)  # Use ddof=1 for sample variance

# Plot the histogram
plt.hist(featuredata, bins=num_bins, edgecolor='red', alpha=0.2)
plt.ylabel('Frequency (F)')
plt.title('Histogram ')
plt.xlabel('Feature')
plt.grid(True)

# Show the histogram and statistics
plt.show()

# Print the mean and variance of 'Feature1'
print(f'Mean : {mean_feature1}')
print(f'Variance : {variance_feature1}')


# In[44]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance


vector1 = np.array([df['f10'][0], df['f15'][0]])
vector2 = np.array([df['f10'][3], df['f15'][3]])

# Define a range of values for 'r'
r_values = range(1, 11)

# Calculate Minkowski distances for different 'r' values
distances = [distance.minkowski(vector1, vector2, p=r) for r in r_values]

# Create a plot to observe the nature of the graph
plt.plot(r_values, distances, marker='o', linestyle='-')
plt.ylabel('Minkowski Distance')
plt.title('Minkowski Distance vs. r')
plt.xlabel('r')
plt.grid(True)
plt.show()


# In[84]:


import numpy as np
from sklearn.model_selection import train_test_split

classes = [0, 1]
sdata = df[df['Label'].isin(classes)]

# Define your features (X) and target (y)
X = sdata[['f2', 'f3']]
y = sdata['Label']

# Split the dataset into a train set (70%) and a test set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Now, you have your train and test sets for binary classification
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# In[40]:


import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Assuming you have already split your data into X_train and y_train
# If not, please refer to the previous code for splitting the data.

# Create a k-NN classifier with k=3
neigh = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier to your training data
neigh.fit(X_train, y_train)
accuracy = neigh.score(X_test, y_test)

# Print the accuracy report
print("Accuracy:", accuracy)


# In[41]:


test_vect = [[0.009625,0.003646 ]]  # Replace with the feature values you want to classify

# Use the predict() function to classify the test vector
predicted_class = neigh.predict(test_vect)

# Print the predicted class
print("Predicted Class:", predicted_class[0])


# In[45]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Define your feature vectors (X_train, X_test) and class labels (y_train, y_test)
# Assuming you have already split your data into training and test sets
# If not, please refer to the previous code for splitting the data.

# Create arrays to store accuracy values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k_values = range(1, 12)
knn_accuracies = []
nn_accuracies = []

# Iterate through different values of k
for k in k_values:
    # Train k-NN classifier with k=3
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)

    # Predict using k-NN
    knn_predictions = knn_classifier.predict(X_test)

    # Calculate accuracy for k-NN
    knn_accuracy = accuracy_score(y_test, knn_predictions)
    knn_accuracies.append(knn_accuracy)

    # Train NN classifier with k=1
    nn_classifier = KNeighborsClassifier(n_neighbors=1)
    nn_classifier.fit(X_train, y_train)

    # Predict using NN
    nn_predictions = nn_classifier.predict(X_test)

    # Calculate accuracy for NN
    nn_accuracy = accuracy_score(y_test, nn_predictions)
    nn_accuracies.append(nn_accuracy)

# Plot the accuracy results
plt.figure(figsize=(10, 6))
plt.plot(k_values, knn_accuracies, label='k-NN (k=3)', marker='o')
plt.plot(k_values, nn_accuracies, label='NN (k=1)', marker='o')
plt.title('Accuracy vs. k for k-NN and NN')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.show()


# In[85]:


from sklearn.metrics import confusion_matrix, classification_report

# Train a k-NN classifier with k=3 on the training data
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# Predict class labels for the training and test data
y_train_pred = neigh.predict(X_train)
y_test_pred = neigh.predict(X_test)

# Calculate confusion matrices for training and test data
confusion_matrix_train = confusion_matrix(y_train, y_train_pred)
confusion_matrix_test = confusion_matrix(y_test, y_test_pred)

# Generate classification reports for training and test data
report_train = classification_report(y_train, y_train_pred)
report_test = classification_report(y_test, y_test_pred)

# Print confusion matrices and classification reports
print("Confusion Matrix (training data):")
print(confusion_matrix_train)
print("\nClassification Report (training data):")
print(report_train)

print("\nConfusion Matrix (test data):")
print(confusion_matrix_test)
print("\nClassification Report (test data):")
print(report_test)

