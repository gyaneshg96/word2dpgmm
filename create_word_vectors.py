# from transformers import BertTokenizer, BertModel
from transformers import DistilBertTokenizer, DistilBertModel
import spacy
import os
import codecs
import torch
from time import time

wordset = set()

## above is for word entailment tasks

# def readWords(filename):
#     with open(filename,"r") as file:
#         for line in file:
#             words = line.split(" ")
#             for w in words:
#                 ww = w.split("-")
#                 wordset.add(ww[0])

# readWords("eacl2012-data/positive-examples.txtinput")
# readWords("eacl2012-data/negative-examples.txtinput")

wordset = set()

# these have already been added

# wordset.add("rock")
# wordset.add("light")
# wordset.add("apple")
# wordset.add("pop")
# wordset.add("cell")
# wordset.add("bank")

wordset.add("animal")
wordset.add("bird")
wordset.add("bug")
wordset.add("chick")
wordset.add("right")
wordset.add("left")
wordset.add("direction")

dataset = "dataset/"

#either do it by context or by sentences
contextfiles = {}
mode = 0o777
for w in wordset:
    path = "wordvectors/bert/"+w
    print(path)
    if not os.path.isdir(path):
        os.mkdir(path, mode)
    contextfiles[w] = open(path+'/sentences.txt',"w")

nlp = spacy.load("en_core_web_sm")

dataset = "dataset/"
threshold = 5

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

model.to('cuda:0')

def getwordvector(i, sentence):
    encoded_text = tokenizer(sentence, return_tensors='pt')
    encoded_text.to('cuda:0')
    outputs = model(**encoded_text)
    return outputs.last_hidden_state[0][i+1].data

wordict = {}
start = time()
for c in range(1,2):
    print(c)
    x = 0
    with codecs.open(dataset+str(c)+".txt", encoding="utf-8", errors="ignore") as file:
        for line in file:
            doc = nlp(line)
            print(x)
            x += 1
            docs = list(doc.sents)
            for d in docs:
                if len(d) < threshold and len(d) > 15:
                    continue
                tokens = tokenizer.tokenize(str(d))
                for i in range(1,len(tokens)-1):
                    sword = tokens[i].lower()
                    if sword in wordset:
                        try:
                            vec = getwordvector(i,str(d))
                            if sword in wordict:
                                if wordict[sword] > 3000:
                                    continue
                                wordict[sword] += 1
                            else:
                                wordict[sword] = 1
                            contextfiles[sword].write(str(d)+"|"+str(wordict[sword])+"\n")
                            torch.save(vec, "wordvectors/bert/"+sword+"/"+str(wordict[sword])+".pt")
                        except (RuntimeError):
                            pass
print(time.time() - start)
for w in wordset:
    contextfiles[w].close()







