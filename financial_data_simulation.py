import numpy as np
import pandas as pd
from scipy.stats import skewnorm, norm, rankdata
from copulas.multivariate import GaussianMultivariate
import matplotlib.pyplot as plt
import seaborn as sns

# Function to generate skewed data with specific mean, standard deviation, and skewness
def generate_data(m, mean, std_dev, skewness):
    # Generate skewed data
    data = skewnorm.rvs(a=skewness, loc=0, scale=1, size=m)
    # Adjust for the desired mean and standard deviation
    data = std_dev * (data - np.mean(data)) / np.std(data) + mean
    return data

# Parameters for the four indices
params = {
    'DJI': {'m': 4729, 'mean': 0.02, 'std_dev': 1.2, 'skewness': 16.5},
    'RUT': {'m': 4729, 'mean': -0.02, 'std_dev': 7.1, 'skewness': 9.9},
    'VIX': {'m': 4729, 'mean': 0.01, 'std_dev': 1.5, 'skewness': 9.5},
    'N225': {'m': 4729, 'mean': 0.02, 'std_dev': 1.6, 'skewness': 10.5}
}

# Generate data for each index
indices_data = {index: generate_data(**params) for index, params in params.items()}

# Convert to DataFrame for visualization
data_df = pd.DataFrame(indices_data)

# Saving the pairplot of the generated data
sns.pairplot(data_df)
plt.savefig('generated_data_visualization.png')  # Save the visualization
plt.close()

# Function to scale data between new min and max values
def scale_data(data, new_min=-10, new_max=10):
    min_val = data.min()
    max_val = data.max()
    scaled_data = (data - min_val) * (new_max - new_min) / (max_val - min_val) + new_min
    return scaled_data

# Scaling each column in the DataFrame
scaled_indices_data = data_df.apply(lambda x: scale_data(x))

# Saving the pairplot of the scaled data
sns.pairplot(scaled_indices_data)
plt.savefig('scaled_data_visualization.png')  # Save the visualization
plt.close()

# Function to transform data to a uniform distribution [0, 1] using the PIT method
def to_uniform(data):
    uniform_data = (rankdata(data) - 1) / len(data)
    return uniform_data

# Transforming each data column to the uniform space [0, 1]
uniform_indices_data = data_df.apply(lambda x: to_uniform(x))

# Saving the pairplot of the uniform data
sns.pairplot(uniform_indices_data)
plt.savefig('uniform_data_visualization.png')  # Save the visualization
plt.close()

# Initialize and fit a Gaussian copula to the transformed data
copula = GaussianMultivariate()
copula.fit(uniform_indices_data)

# Function to inverse transform from uniform to original space, assuming normal marginals
def inverse_transform(uniform_data, original_data):
    mean = original_data.mean()
    std = original_data.std()
    return norm.ppf(uniform_data, loc=mean, scale=std)

# Generate samples from the copula
N = 4729  # Desired number of samples
sampled_u = copula.sample(N)

# Inverse transform for each column back to the original space
sampled_data = pd.DataFrame()
for column in uniform_indices_data.columns:
    sampled_data[column] = inverse_transform(sampled_u[column], data_df[column])

# Save sampled data to CSV
sampled_data.to_csv('sampled_data.csv', index=False)

# Optionally, save a visualization of the sampled data
sns.pairplot(sampled_data)
plt.savefig('sampled_data_visualization.png')  # Save the visualization
plt.close()
