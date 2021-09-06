!pip install transformers
from transformers import BertModel, BertConfig, BertForSequenceClassification, BertTokenizer
import os, sys
import torch
import torch.nn.functional as nnf
from transformers import pipeline
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

drive.mount('/content/gdrive')
from pickle import load

#np.random.seed(0)

#sentence = "My name is Madhu working for an App development company based in India with 7 years of experience. I have a background in creating apps like Uber, Health and Fitness apps, Camera filter apps, etc across a variety of platforms such as Android, IOS, Angular, Ionic, Swift, Unity, and other platforms as well. I can work with you to come up with a game plan for creating your MVP or if you already have an app I can also help you with maintenance and updates of your existing apps."
sentence = "Last year saw the arrival of DeFi Summer 1.0, and now we’re right in the middle of DeFi Summer 2.0. Or maybe DeFi Autumn? Either way, the universe of crypto-based lending, saving, and trading has been demolishing all sorts of records and sending the tokens of blockchains that support smart contracts like Ethereum and Solana soaring to new heights."

output_dir = '/content/gdrive/MyDrive/Machine Learning/datos/Spam/modelos/model_saveTMP1'
#output_dir = output_dir + str(num)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Loading model to %s" % output_dir)

params_dir = output_dir

model = BertForSequenceClassification.from_pretrained(params_dir)
tokenizer = BertTokenizer.from_pretrained(params_dir)
print(model.config)
print(params_dir)

model.eval()
inputs = tokenizer(sentence, return_tensors="pt")
etiqueta = torch.tensor([1]).unsqueeze(0)  
with torch.no_grad():
    outputs = model(**inputs, labels=etiqueta)
print(outputs['logits'])
prediction = nnf.softmax(outputs['logits'], dim=1)

print(prediction)



prob = round(prediction.topk(1, dim = 1)[0].double().item()*100, 2)
print(type(prob))
print(prob)
if prediction.topk(1, dim = 1)[1] == 1:
  print("YES (Danger!!!!) with prob = ", prob, "%")
else:
  print("NO with prob = ", prob, "%")



