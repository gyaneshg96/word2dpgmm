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
wordset.add("apple")
# wordset.add("pop")
# wordset.add("cell")
# wordset.add("bank")

# wordset.add("animal")
# wordset.add("bird")
wordset.add("bug")
wordset.add("chick")
# wordset.add("right")
# wordset.add("left")
# wordset.add("direction")

wordset.add("airplane")
wordset.add("vehicle")
wordset.add("chocolate")
wordset.add("food")

dataset = "dataset/"

#either do it by context or by sentences
contextfiles = {}
mode = 0o777
for w in wordset:
    path = "wordvectors/bert_context/"+w
    print(path)
    if not os.path.isdir(path):
        os.mkdir(path, mode)
    contextfiles[w] = open(path+'/sentences.txt',"w")

nlp = spacy.load("en_core_web_sm")

dataset = "dataset/"
threshold = 5
contextsize = 5

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

model = model.to('cuda:0')
model.eval()

def getwordvector(i, sentence):
    encoded_text = tokenizer(sentence, return_tensors='pt')
    encoded_text.to('cuda:0')
    with torch.no_grad():
        outputs = model(**encoded_text)
        return outputs.last_hidden_state[0][i+1].data

wordict = {}
for w in wordset:
    wordict[w] = 0
start = time()
for c in range(1,2):
    # print(c)
    x = 0
    with codecs.open(dataset+str(c)+".txt", encoding="utf-8", errors="ignore") as file:
        for line in file:
            doc = nlp(line)
            if x%100==0:
                print(x, wordict)
            flag = True
            for k in wordict:
                if wordict[k] <= 10000:
                    flag = False
                    break
            if flag:
                break
            x += 1
            docs = list(doc.sents)
            for d in docs:
                tokens = tokenizer.tokenize(str(d).lower())
                l = len(tokens)
                if l < 5 :
                    continue 
                for i in range(l):
                    sword = tokens[i]
                    if sword in wordset:
                        try:
                            if wordict[sword] > 10000:
                                    continue
                            wordict[sword] += 1
                            if i >= contextsize and l-i >= contextsize:
                                ind = tokens[i-contextsize:i+contextsize+1].index(sword)
                                context = ' '.join(tokens[i-contextsize:i+contextsize+1]) 
                            elif i >= contextsize:
                                ind = tokens[i-contextsize:].index(sword)
                                context = ' '.join(tokens[i-contextsize:]) 
                            elif l-i >= contextsize:
                                ind = tokens[:i+contextsize+1].index(sword)
                                context = ' '.join(tokens[:i+contextsize+1])
                            else:
                                ind = tokens.index(sword)
                                context = ' '.join(tokens)
                            # context = ' '.join(tokens[minpos:maxpos+1]) 
                            vec = getwordvector(ind,context)
                            contextfiles[sword].write(context+"|"+str(wordict[sword])+"\n")
                            torch.save(vec, "wordvectors/bert_context/"+sword+"/"+str(wordict[sword])+".pt")
                        except (RuntimeError):
                            pass
print(time() - start)
for w in wordset:
    contextfiles[w].close()







