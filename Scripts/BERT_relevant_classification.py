import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import keras
import string
import re
import nltk
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from nltk.tokenize import RegexpTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig, TFBertForSequenceClassification
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# Import dataset
df = pd.read_excel('tweets.xlsx') # add the excel file with the random sample of tweets annotated according to their relevance to climate change
df = df[['Text','Relevance']]
print(df)
print(df['Relevance'].value_counts())

# Balance dataset 
df_0_class = df[df['Relevance']==0]
df_1_class = df[df['Relevance']==1]
df_0_class_undersampled = df_0_class.sample(df_1_class.shape[0]) # implement undersampling to balance the classes in the dataset
balanced_tweets = pd.concat([df_0_class_undersampled, df_1_class], axis=0)
print(balanced_tweets['Relevance'].value_counts())

# Split dataset
tweets = balanced_tweets.Text.values
labels = balanced_tweets.Relevance.values
print(tweets, labels)

x_train, x_test, y_train, y_test = train_test_split(tweets, labels, random_state = 7, stratify = labels) # generate train and test sets
print(x_train.shape)
print(x_test.shape)


# BERT initialization
tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased') # initialize the BERT model ("neuralmind/bert-base-portuguese-cased" for portuguese, "dccuchile/bert-base-spanish-wwm-cased" for spanish and "bert-base-uncased" for english)

print('Actual text', tweets[0])
print('Tokens', tokenizer.tokenize(tweets[0]))
print('Token to ids', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tweets[0])))

# Encode/tokenize each tweet in the list
max_len = 0

for tweet in tweets:
  max_len = max(max_len, len(tweet))

print('Max sentence length: ', max_len)

def mask_inputs_for_bert(tweets, max_len):
  input_ids = []
  attention_masks = []
  i = 0
  for tweet in tweets:
    if (i < 3):
      print("Tweet", tweet)
    encoded_dict = tokenizer.encode_plus(tweet, add_special_tokens = True, max_length = max_len, pad_to_max_length = True, return_attention_mask = True)
    if (i < 3):
      print("dict", encoded_dict['input_ids'])
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
    i += 1
  input_ids = tf.convert_to_tensor(input_ids)
  attention_masks = tf.convert_to_tensor(attention_masks)
  return input_ids, attention_masks

x_train, train_mask = mask_inputs_for_bert(x_train, max_len)
x_test, test_mask = mask_inputs_for_bert(x_test, max_len)
y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)

print('x_train_shape', x_train.shape)
print('train_mask_shape', train_mask.shape)
print('x_test_shape', x_test.shape)
print('test_mask_shape', test_mask.shape)
print('y_train_shape', y_train.shape)
print('y_test_shape', y_test.shape)

# BERT implementation
bert_model = TFBertForSequenceClassification.from_pretrained('neuralmind/bert-base-portuguese-cased', num_labels = 2) # BERT model ("neuralmind/bert-base-portuguese-cased" for portuguese, "dccuchile/bert-base-spanish-wwm-cased" for spanish and "bert-base-uncased" for english)

callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath = '/content/', save_weights_only = True, monitor = 'val_loss', mode = 'min', save_best_only = True)] # add the path to directory in filepath

print('\nBert Model', bert_model.summary())

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
optimizer = tf.keras.optimizers.Adam(learning_rate = 2e-6, epsilon = 1e-08)

bert_model.compile(loss = loss, optimizer = optimizer, metrics = [metric])

history = bert_model.fit([x_train, train_mask], y_train, batch_size = 4, epochs = 40, validation_split = 0.1, callbacks = callbacks)

#Plot the training parameters
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc = 'upper left')
plt.show()

# BERT evaluation
print('Model evaluation')
bert_model.evaluate([x_test, test_mask], y_test)
y_predicted = bert_model.predict([x_test, test_mask])
pred_labels = np.argmax(y_predicted.logits, axis=1)
f1 = f1_score(y_test, pred_labels)
print('F1 score: ', f1)
print('Classification Report')
print(classification_report(y_test, pred_labels, target_names = ["Not Relevant", "Relevant"])) # report with the classification metrics used to evaluate the model
c1 = confusion_matrix(y_test, pred_labels) # confusion matrix showing true and false positives and negatives 
print('confusion_matrix: ', c1)
