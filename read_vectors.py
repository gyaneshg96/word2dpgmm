import torch
import os
from os import listdir


acc = 'cuda:0' if torch.cuda.is_available() else 'cpu'
tensors = []
def read_vectors(word):
    directory = "wordvectors/bert/"+word
    for f in listdir(directory):
        if not f[-3:] == "txt":
            temp = torch.load(f, map_location=torch.device(acc))
            tensors.append(temp)
    return torch.stack(tensors)
