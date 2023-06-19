#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import GATConv
import pandas as pd
import numpy as np
import string

import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import re


# In[2]:


import dgl
import dgl.function as fn
from dgl.nn import GATConv


# In[3]:


from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[4]:


from sklearn.metrics.pairwise import cosine_similarity


# In[5]:


import scipy.sparse as sp


# In[343]:


df = pd.read_csv('Filtered_data.csv', encoding= 'latin')

df


# In[344]:


df = df.dropna()


# In[345]:


def remove_pattern(text, pattern_regex):
    r = re.findall(pattern_regex, text)
    for i in r:
        text = re.sub(i, '', text)
    
    return text


# In[346]:


df['Clean_Resumes'] = np.vectorize(remove_pattern)(df['text'], '@[\w]*')

df

clean_Resumes =[]

for index, row in df.iterrows():
    words_without_links = [word for word in row.Clean_Resumes.split() if 'http' not in word]
    clean_Resumes.append(' '.join(words_without_links)) 

df['Clean_Resumes'] = clean_Resumes
df.head(10)

df = df[df['Clean_Resumes']!= ''] #removes empty string

df

def clean_text(text):
    text = text.lower()
    text = re.sub('!','', text)
    text = re.sub('\[.*?\]','', text)
    text = re.sub('➢','',text)
    text = re.sub('•','',text)
    text = re.sub('●','', text)
    text = re.sub('⚫', '', text)
    text = re.sub('https?://\S+|www.\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

df.Clean_Resumes = df.Clean_Resumes.apply(lambda x: clean_text(x))
df.head()

my_stop_words = stopwords.words('english')

cleaned_resumes = []

for index, row in df.iterrows():
    words_without_stopwords = [word for word in row.Clean_Resumes.split() if word not in my_stop_words]
    cleaned_resumes.append(' '.join(words_without_stopwords))

df['Absolute_Clean_Resumes'] = cleaned_resumes
df.head(10)

Tokenized_Resume = df['Absolute_Clean_Resumes'].apply(lambda x: x.split())
Tokenized_Resume.head(10)

word_lemmatizer = WordNetLemmatizer()

Tokenized_Resume = Tokenized_Resume.apply(lambda x: [word_lemmatizer.lemmatize(i) for i in x])
Tokenized_Resume.head(10)

unique_words_per_row = []

for sublist in Tokenized_Resume:
    unique_words = set(sublist)
    unique_words_per_row.append(unique_words)

pos_tags = []
for resume in unique_words_per_row:
    pos_tags.append(pos_tag(resume))

pos_tags

filtered_tokens = []
for tags in pos_tags:
    tokens = [word for word, tag in tags if tag in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB', 'RBR', 'RBS']]
    filtered_tokens.append(' '.join(tokens))

filtered_tokens

df['Absolute_Clean_Resumes'] = filtered_tokens
df


# In[347]:


df['Clean_job_description'] = np.vectorize(remove_pattern)(df['job_description'], '@[\w]*')

Clean_job_description =[]

for index, row in df.iterrows():
    words_without_links = [word for word in row.Clean_job_description.split() if 'http' not in word]
    Clean_job_description.append(' '.join(words_without_links))

df['Clean_job_description'] = Clean_job_description

df = df[df['Clean_job_description']!= '']

df

df.Clean_job_description = df.Clean_job_description.apply(lambda x: clean_text(x))
df.head()

Clean_job_description = []
for index, row in df.iterrows():
    words_without_stopwords = [word for word in row.Clean_job_description.split() if word not in my_stop_words]
    Clean_job_description.append(' '.join(words_without_stopwords))

df['Clean_job_description'] = Clean_job_description
df.head(10)

Tokenized_JD = df['Clean_job_description'].apply(lambda x: x.split())
Tokenized_JD.head(10)

Tokenized_JD = Tokenized_JD.apply(lambda x: [word_lemmatizer.lemmatize(i) for i in x])

unique_words_per_row = []

for sublist in Tokenized_JD:
    unique_words = ' '.join(set(sublist))
    unique_words_per_row.append(unique_words)

df['Clean_job_description'] = unique_words_per_row

df


# In[604]:


resumes = df['Absolute_Clean_Resumes'].tolist()

job_descriptions = df['Clean_job_description'].tolist()

labels = df['labels'].tolist()


# In[605]:


concatenated_resumes = []

for i in range(len(resumes)):
    concatenated_resume = resumes[i] + " " + labels[i]
    concatenated_resumes.append(concatenated_resume)


# In[606]:


resumes = concatenated_resumes


# In[607]:


concatenated_jds = []
for i in range(len(job_descriptions)):
    concatenated_jd = job_descriptions[i] + " " + labels[i]
    concatenated_jds.append(concatenated_jd)


# In[608]:


job_descriptions = concatenated_jds


# In[609]:


word_to_vec = {}
with open(r'D:\Resume Classification\glove.6B.300d.txt', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], dtype="float32")
        word_to_vec[word] = vector


# In[610]:


job_desc_features = []
resume_features = []

for description in job_descriptions:
    tokens = word_tokenize(description)
    embedding_matrix = [word_to_vec[token] for token in tokens if token in word_to_vec]
    if embedding_matrix:
        mean_embedding = np.mean(embedding_matrix, axis=0)
        job_desc_features.append(mean_embedding)

for resume in resumes:
    tokens = word_tokenize(resume)
    embedding_matrix = [word_to_vec[token] for token in tokens if token in word_to_vec]
    if embedding_matrix:
        mean_embedding = np.mean(embedding_matrix, axis=0)
        resume_features.append(mean_embedding)

job_desc_features = np.array(job_desc_features)
resume_features = np.array(resume_features)


# In[611]:


job_desc_features.shape


# In[612]:


resume_features.shape


# In[613]:


adj = cosine_similarity(resume_features, job_desc_features)


# In[614]:


from sklearn.preprocessing import LabelEncoder

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)


# In[615]:


encoded_labels


# In[616]:


X = job_desc_features
y = encoded_labels


# In[617]:


adj_sparse = sp.csr_matrix(adj)


# In[618]:


g = dgl.from_scipy(adj_sparse)


# In[619]:


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.attention_layer1 = nn.Linear(in_features, out_features)
        self.attention_layer2 = nn.Linear(in_features, out_features)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = torch.matmul(z2, self.a)
        return {'e': F.leaky_relu(a, negative_slope=0.2)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, X):
        h1 = F.leaky_relu(self.attention_layer1(X), negative_slope=0.2)

        g.ndata['z'] = h1
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        h1_prime = g.ndata.pop('h')
        
        h2 = F.leaky_relu(self.attention_layer2(h1_prime), negative_slope=0.2)

        g.ndata['z'] = h2
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        h2_prime = g.ndata.pop('h')

        # Concatenate attention heads
        h_prime = torch.cat([h1_prime, h2_prime], dim=1)

        return h_prime


# In[620]:


class GATClassifier(nn.Module):
    def __init__(self, in_features, num_classes, num_heads, dropout):
        super(GATClassifier, self).__init__()
        self.num_heads = num_heads

        self.attention_heads = nn.ModuleList()
        for _ in range(num_heads):
            self.attention_heads.append(GraphAttentionLayer(in_features, in_features, dropout, alpha=0.2))
        
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Average pooling layer
        self.dropout = nn.Dropout(dropout)  # Dropout layer
        self.out_layer = nn.Linear(in_features * num_heads * 2, num_classes)

    def forward(self, g, X):
        head_outputs = []
        for attn_head in self.attention_heads:
            head_output = attn_head(g, X)
            head_outputs.append(head_output)

        h_cat = torch.cat(head_outputs, dim=1)
        h_cat = h_cat.unsqueeze(2)  # Add an extra dimension for sequence_length
        h_cat_pooled = self.pooling(h_cat).squeeze(2)  # Apply average pooling
        h_cat_pooled = self.dropout(h_cat_pooled)  # Apply dropout
        
        batch_size = h_cat_pooled.size(0)
        h_cat_pooled = h_cat_pooled.view(batch_size, -1)  # Reshape to (batch_size, in_features * num_heads * 2)
        output = self.out_layer(h_cat_pooled)
        
        softmax_output = F.softmax(output, dim=1)

        return output


# In[621]:


X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(encoded_labels, dtype=torch.long)


# In[622]:


X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.4, random_state=40)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=40)


# In[623]:


# Convert the training data to DGL graph format
train_adj = cosine_similarity(X_train, X_train)
train_adj_sparse = sp.csr_matrix(train_adj)
train_graph = dgl.from_scipy(train_adj_sparse)
train_graph.ndata['feat'] = torch.tensor(X_train, dtype=torch.float32)
train_graph.ndata['label'] = torch.tensor(y_train, dtype=torch.long)


# In[624]:


val_adj = cosine_similarity(X_val, X_val)
val_adj_sparse = sp.csr_matrix(val_adj)
val_graph = dgl.from_scipy(val_adj_sparse)
val_graph.ndata['feat'] = torch.tensor(X_val, dtype=torch.float32)
val_graph.ndata['label'] = torch.tensor(y_val, dtype=torch.long)


# In[625]:


test_adj = cosine_similarity(X_test, X_test)
test_adj_sparse = sp.csr_matrix(test_adj)
test_graph = dgl.from_scipy(test_adj_sparse)
test_graph.ndata['feat'] = torch.tensor(X_test, dtype=torch.float32)
test_graph.ndata['label'] = torch.tensor(y_test, dtype=torch.long)


# In[626]:


num_heads = 8
dropout = 0.3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[627]:


model = GATClassifier(in_features=job_desc_features.shape[1], num_classes=len(label_encoder.classes_), num_heads=num_heads, dropout=dropout).to(device)


# In[628]:


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


# In[629]:


num_epochs = 150


# In[630]:


for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    logits = model(train_graph, train_graph.ndata['feat'])
    loss = criterion(logits, train_graph.ndata['label'])
    loss.backward()
    optimizer.step()

    # Evaluation on the training set
    model.eval()
    with torch.no_grad():
        train_logits = model(train_graph, train_graph.ndata['feat'])
        train_predictions = torch.argmax(train_logits, dim=1)
        train_true_labels = train_graph.ndata['label']
        train_accuracy = torch.sum(train_predictions == train_graph.ndata['label']).item() / len(train_graph.ndata['label'])

    # Evaluation on the validation set
    model.eval()
    with torch.no_grad():
        val_logits = model(val_graph, val_graph.ndata['feat'])
        val_predictions = torch.argmax(val_logits, dim=1)
        val_true_labels = val_graph.ndata['label']
        val_accuracy = torch.sum(val_predictions == val_graph.ndata['label']).item() / len(val_graph.ndata['label'])

    # Evaluation on the testing set
    model.eval()
    with torch.no_grad():
        test_logits = model(test_graph, test_graph.ndata['feat'])
        test_predictions = torch.argmax(test_logits, dim=1)
        test_true_labels = test_graph.ndata['label']
        test_accuracy = torch.sum(test_predictions == test_graph.ndata['label']).item() / len(test_graph.ndata['label'])

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Val Accuracy: {val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")


# In[631]:


train_predictions


# In[632]:


train_true_labels


# In[633]:


test_predictions


# In[634]:


test_true_labels


# # Assign features and labels to the graph
# g.ndata['feat'] = X_tensor
# g.ndata['label'] = y_tensor

# num_heads = 8
# dropout = 0.3
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# In[ ]:





# model = GATClassifier(in_features=job_desc_features.shape[1], num_classes=len(label_encoder.classes_), num_heads=num_heads, dropout=dropout).to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss()

# num_epochs = 100

# # Training loop
# for epoch in range(num_epochs):
#     model.train()
#     optimizer.zero_grad()
#     logits = model(g, g.ndata['feat'])
#     loss = criterion(logits, g.ndata['label'])
#     loss.backward()
#     optimizer.step()
#     
#     # Evaluation
#     model.eval()
#     with torch.no_grad():
#         logits = model(g, g.ndata['feat'])
#         predictions = torch.argmax(logits, dim=1)
#         true_labels = g.ndata['label']
#         accuracy = torch.sum(predictions == true_labels).item() / len(true_labels)
#     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
# 

# In[ ]:


predictions


# In[ ]:


true_labels


# In[ ]:




