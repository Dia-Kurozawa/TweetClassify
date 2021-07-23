import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.random.set_seed(1)
import pandas as pd
from transformers import BertTokenizer, TFBertModel
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, KFold
import nltk
import os
from TweetNormalizer import normalizeTweet
from eda import *

dataset_file = '/home/shaohuang9/HSUintern' #the file contains dataset
os.chdir(dataset_file)
data = pd.read_excel('PHM COVID19 - enlarged and refined.xlsx')
model_name = 'bert-base-uncased'
max_len = 300
lr = 5e-5
batch_size = 16
epochs = 2
num_of_fold = 5
alpha = 0.05

data = data.iloc[:,1:].drop_duplicates(ignore_index=True)
tokenizer = BertTokenizer.from_pretrained(model_name)
for i in range(len(data)):
    data.text.values[i] = normalizeTweet(data.text.values[i])
    data.label.values[i] = data.label.values[i] - 1

x_train_ori, x_test, y_train_ori, y_test = train_test_split(data.text.values, data.label.values, shuffle=True, test_size=0.1, random_state=2477)
def list_encode_tensor(x):
    for i in range(len(x)):
        x[i] = tokenizer.encode(x[i], add_special_tokens=True, max_length=max_len, padding='max_length')
    return tf.convert_to_tensor(x)
x_test = list_encode_tensor(x_test.tolist())

def EDA(x_train_ori, y_train_ori, alpha=0.05):
    data_train = pd.DataFrame(x_train_ori, columns=['text'])
    data_train['label'] = y_train_ori
    data_aug = data_train[~data_train.label.isin([2])] 
    for index, row in data_aug.iterrows(): 
        if row['label']==0: num_aug = 6
        if row['label']==1: num_aug = 2
        if row['label']==3: num_aug = 1
        aug_sentences = eda(row['text'], alpha_sr=alpha, num_aug=num_aug)
        for x in eda(row['text'], alpha_ri=alpha, num_aug=num_aug):
            aug_sentences.append(x)
        for x in eda(row['text'], alpha_rs=alpha, num_aug=num_aug):
            aug_sentences.append(x)
        for x in eda(row['text'], p_rd=alpha, num_aug=num_aug):
            aug_sentences.append(x)
        for aug_sentence in aug_sentences:
            data_aug = data_aug.append({"label":row['label'],"text":aug_sentence},ignore_index=True)
    data_augment = pd.concat([data_train, data_aug], ignore_index=True)
    data_augment = data_augment.drop_duplicates(ignore_index=True)
    for i in range(100):
        data_augment = data_augment.sample(frac=1).reset_index(drop=True)
    x_train, y_train = data_augment.text.values.tolist(), data_augment.label.values.tolist()
    x_train = list_encode_tensor(x_train)
    y_train = to_categorical(y_train, num_classes=4)
    return x_train, y_train

kf = KFold(n_splits=num_of_fold, random_state=1, shuffle=True)
fold_index = list(kf.split(x_train_ori))
df_vote = pd.DataFrame(columns=range(num_of_fold+1))

for i in range(num_of_fold):
    x_train_fold, x_val_fold = x_train_ori[fold_index[i][0]], x_train_ori[fold_index[i][1]]
    y_train_fold, y_val_fold = y_train_ori[fold_index[i][0]], y_train_ori[fold_index[i][1]]
    x_val = list_encode_tensor(x_val_fold.tolist())
    y_val = to_categorical(y_val_fold, num_classes=4)
    x_train, y_train = EDA(x_train_fold, y_train_fold, alpha)
    
    bert_model = TFBertModel.from_pretrained(model_name, output_hidden_states=True)
    bert_in = Input(shape=(max_len,), dtype='int32')
    bert_layer = bert_model(bert_in) # (batch_size, max_sent_length, hidden_size)
    GAP = GlobalAveragePooling1D()(bert_layer[2][1]+bert_layer[2][-1]) # (batch_size, hidden_size)
    output = Dense(4, activation='softmax')(GAP) # (batch_size, number_of_classes)
    
    model = Model(bert_in, output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr), 
        metrics=['accuracy']
    )
    model.fit(
        x_train, y_train, batch_size=batch_size, epochs=epochs,
        validation_data=(x_val, y_val)
    )
    df_vote.iloc[:,i] = np.argmax(model.predict(x_test), axis=1)

for i in range(len(df_vote)):
    df_vote.iloc[i,-1] = np.argmax(np.bincount(df_vote.iloc[i,:-1]))
df_vote.iloc[:,-1] = df_vote.iloc[:,-1].astype(int)
print(classification_report(y_test, df_vote.iloc[:,-1], target_names=['1','2','3','4']))
print(confusion_matrix(y_test, df_vote.iloc[:,-1]))
