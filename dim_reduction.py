import torch
from read_vectors import *
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

word = "bank"

tensors = read_vectors(word)
tup = torch.pca_lowrank(tensors)

k = 2
reduced = torch.matmul(tensors, tup[2][:, :k])


plt.figure(figsize=(20, 20))
plt.scatter(reduced[:, 0], reduced[:, 1], color="blue")
plt.show()

tensors1 = np.asarray(tensors, dtype='float32')
x_reduced = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(tensors1)

plt.figure(figsize=(20, 20))
plt.scatter(x_reduced[:, 0], x_reduced[:, 1], color="blue")
plt.show()