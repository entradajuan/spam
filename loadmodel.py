!pip install transformers
import os
from google.colab import drive
from transformers import BertTokenizer, BertConfig
from transformers import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup


drive.mount('/content/gdrive')
from pickle import load

#np.random.seed(0)

sentence = "My name is Madhu working for an App development company based in India with 7 years of experience. I have a background in creating apps like Uber, Health and Fitness apps, Camera filter apps, etc across a variety of platforms such as Android, IOS, Angular, Ionic, Swift, Unity, and other platforms as well. I can work with you to come up with a game plan for creating your MVP or if you already have an app I can also help you with maintenance and updates of your existing apps."


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
