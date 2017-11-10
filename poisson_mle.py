
# coding: utf-8

# In[4]:


get_ipython().magic('matplotlib inline')
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats


# We first produce a sample:

# In[8]:


TRUE_LAMBDA = 6
X = np.random.poisson(TRUE_LAMBDA, 100)


# For our sample, we estimate a value for $\lambda$ using MLE:

# In[25]:


def poisson_lambda_MLE(X):
    return np.sum(X)/len(X)

lambda_mle = poisson_lambda_MLE(X)


# We finally plot the sample and the resulting distribution:

# In[32]:


# TODO
figure = plt.figure(figsize=(13,6))
ax = figure.add_subplot(1,1,1)
ax.plot(X)
plt.title("Sample")
plt.show()


# In[37]:


# TODO
figure = plt.figure(figsize=(13,6))
ax = figure.add_subplot(1,1,1)
x = np.arange(-10,30,1)
ax.plot(x,scipy.stats.poisson.pmf(x,lambda_mle))
plt.title("Resulting Poisson distribution")
plt.show()

