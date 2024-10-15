import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
 
# Load dataset
digits = datasets.load_digits()

# View data
print(digits)
print(digits.DESCR)
print(digits.data)
print(digits.target)

# Visualize data image
plt.gray() 
plt.matshow(digits.images[100])
plt.show()

# Verify target label
print(digits.target[100])

# Build model via KMeans method
model = KMeans(n_clusters=10, random_state=42)
model.fit(digits.data)

# Visualize after KMeans method
fig = plt.figure(figsize = (8, 3))
fig.suptitle('Cluser Center Images', fontsize = 14, fontweight='bold')

# Display cluster centers
for i in range(10):
  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)
  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
plt.show()


# Testing the model - draw numbers in test.html and copy array below (6260)
new_samples = np.array([
[0.00,0.00,3.74,6.86,6.71,2.14,0.00,0.00,0.00,1.91,7.63,5.03,6.40,6.41,0.00,0.00,0.00,3.66,7.63,0.23,0.84,1.15,0.00,0.00,0.00,3.81,7.63,0.00,1.83,3.05,1.68,0.00,0.00,3.82,7.63,5.34,7.63,7.62,7.63,1.75,0.00,3.81,7.62,7.62,3.59,1.91,7.62,2.29,0.00,3.74,7.62,7.63,1.22,4.42,7.62,0.92,0.00,0.92,6.41,7.62,7.62,7.62,4.58,0.00],
[0.00,0.46,5.72,7.62,6.87,1.15,0.00,0.00,0.00,5.04,7.40,3.51,7.17,4.81,0.00,0.00,0.54,7.63,3.89,0.00,6.10,5.34,0.00,0.00,0.30,6.41,1.83,0.92,7.32,4.27,0.00,0.00,0.00,0.00,0.00,4.20,7.32,0.69,0.00,0.00,0.00,0.00,1.30,7.47,4.20,0.00,0.00,0.00,0.00,2.14,7.17,7.40,2.90,2.29,2.29,0.23,0.31,7.17,7.62,7.62,7.62,7.62,7.62,2.29],
[0.00,0.00,2.14,6.71,6.71,4.27,0.00,0.00,0.00,0.23,7.09,5.87,5.03,7.62,2.44,0.00,0.00,2.52,7.62,1.61,0.08,6.48,3.97,0.00,0.00,3.74,7.63,0.15,0.00,0.38,0.15,0.00,0.00,4.42,7.02,0.46,2.90,3.05,1.45,0.00,0.00,4.57,7.09,6.03,7.62,7.62,7.25,0.31,0.00,4.27,7.62,7.55,1.76,3.13,7.62,0.77,0.00,1.07,7.09,7.47,5.41,6.79,6.71,0.23],
[0.00,0.00,2.67,7.25,7.62,4.58,0.23,0.00,0.00,1.68,7.55,5.42,4.81,7.47,4.66,0.00,0.00,4.12,7.48,0.08,0.00,4.43,7.40,0.23,0.00,4.88,6.79,0.00,0.00,2.60,7.62,1.45,0.00,5.34,6.10,0.00,0.00,2.29,7.62,1.53,0.00,5.34,6.10,0.00,0.00,2.44,7.62,1.37,0.00,4.58,7.24,4.35,3.81,6.25,6.94,0.23,0.00,0.69,5.11,6.86,6.86,6.56,1.91,0.00]
])

new_labels = model.predict(new_samples)
print(new_labels)

# Map the cluster labels
for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')


