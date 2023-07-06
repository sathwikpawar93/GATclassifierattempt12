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

import dgl
import dgl.function as fn
from dgl.nn import GATConv

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics.pairwise import cosine_similarity

import scipy.sparse as sp
from collections import defaultdict

import timeit

from dgl import DGLGraph
import time

from sklearn.model_selection import StratifiedKFold


from tensorflow.keras.layers import Embedding

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


# In[2]:


start_time = timeit.default_timer()


# In[3]:


df = pd.read_csv('Filtered_data.csv', encoding= 'latin')

df


# In[4]:


def remove_pattern(text, pattern_regex):
    r = re.findall(pattern_regex, str(text))
    for i in r:
        text = re.sub(i, '', str(text))
    return text


# In[5]:


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


# In[6]:


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


# In[7]:


resumes = df['Absolute_Clean_Resumes'].tolist()

job_descriptions = df['Clean_job_description'].tolist()

labels = df['labels'].tolist()


# In[8]:


label_job=df['labels'].tolist()


# In[25]:


from sklearn.preprocessing import LabelEncoder

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

encoded_label_job = label_encoder.fit_transform(label_job)


# In[9]:


concatenated_jds = []
for i in range(len(job_descriptions)):
    concatenated_jd = job_descriptions[i] + " " + labels[i]
    concatenated_jds.append(concatenated_jd)

job_descriptions = concatenated_jds


# In[10]:


# Load GloVe word embeddings
word_to_vec = {}
with open(r'D:\Resume Classification\glove.6B.300d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], dtype=np.float32)
        word_to_vec[word] = vector


# In[15]:


combine_texts = resumes+job_descriptions


# In[16]:


# Create a set of unique words
unique_words = set()
for text in combine_texts:
    tokens = word_tokenize(text)
    unique_words.update(tokens)


# In[14]:


word_embeddings = np.zeros((len(unique_words), 300))  # Assuming GloVe embeddings have size 300
word_to_index = {}
for idx, word in enumerate(unique_words):
    if word in word_to_vec:
        word_embeddings[idx] = word_to_vec[word]
    word_to_index[word] = idx


# In[42]:


# Construct the co-occurrence matrix with a sliding window of size 3
co_occurrence_matrix = defaultdict(lambda: defaultdict(int))
for text in combine_texts:
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        center_word = tokens[i]
        window_start = max(0, i - 2)
        window_end = min(len(tokens), i + 3)
        for j in range(window_start, window_end):
            if i != j:
                context_word = tokens[j]
                co_occurrence_matrix[word_to_index[center_word]][word_to_index[context_word]] += 1


# In[18]:


vocab_size = len(unique_words)
co_occurrence_matrix_array = np.zeros((vocab_size, vocab_size), dtype=np.float32)
for i in range(vocab_size):
    for j in range(vocab_size):
        co_occurrence_matrix_array[i, j] = co_occurrence_matrix[i][j]


# In[21]:


from scipy.sparse import csr_matrix

co_occurrence_matrix = csr_matrix(co_occurrence_matrix_array)


# In[19]:


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        if h.dtype == torch.float64:
            h = h.float()
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        if h.dtype == torch.float64:
            h = h.float()
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

    def forward(self, h):
        if h.dtype == torch.float64:
            h = h.float()
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h


# In[22]:


g = dgl.DGLGraph()
g.add_nodes(vocab_size)
g.add_edges(co_occurrence_matrix.nonzero()[0], co_occurrence_matrix.nonzero()[1])


# In[23]:


in_dim = 300  # Assuming GloVe embeddings have size 300
hidden_dim = 64
out_dim = 10  # Specify the number of output classes
num_heads = 8
mgat_model = GAT(g, in_dim, hidden_dim, out_dim, num_heads)


# In[26]:


features = torch.FloatTensor(word_embeddings) 
encoded_label_job = torch.tensor(encoded_label_job)


# In[27]:


from sklearn.model_selection import StratifiedKFold

# Define the number of folds (k)
k = 3

# Create the StratifiedKFold object
skf = StratifiedKFold(n_splits=k, shuffle=True)

# Create lists to store the evaluation metrics for each fold
fold_train_losses = []
fold_train_accuracies = []
fold_test_losses = []
fold_test_accuracies = []
fold_train_aucs = []
fold_train_f1s = []
fold_test_aucs = []
fold_test_f1s = []

for fold, (train_index, test_index) in enumerate(skf.split(job_descriptions, encoded_label_job)):

    # Split the data into train and test sets for the current fold
    train_documents = [job_descriptions[i] for i in train_index]
    test_documents = [job_descriptions[i] for i in test_index]
    train_labels = [encoded_label_job[i] for i in train_index]
    test_labels = [encoded_label_job[i] for i in test_index]

    # Create the GAT model for each fold
    net = GAT(g,
              in_dim=300,
              hidden_dim=64,
              out_dim=10,
              num_heads=8)

    # Create optimizer for each fold
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    # Main training loop for each fold
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    train_aucs = []
    train_f1s = []
    test_aucs = []
    test_f1s = []
    dur = []
    for epoch in range(80):
        if epoch >= 3:
            t0 = time.time()
            
        net.train()

        # Perform forward pass through the GAT model
        node_embeddings = net(torch.tensor(word_embeddings))

        # Create document embeddings for training data
        train_document_embeddings = []
        for document in train_documents:
            uni_words = set()
            uni_words.update(document.split())
            uni_words=list(uni_words)
            word_embeds = []
            for uni_word in uni_words:
                if uni_word in word_to_index:
                    word_index_value = word_to_index[uni_word]
                    word_embed = node_embeddings[word_index_value]
                    word_embeds.append(word_embed)

            # Compute the document embedding
            if len(word_embeds) > 0:
                word_embeds = torch.stack(word_embeds)  # Convert word embeddings to tensor
                document_embedding = torch.mean(word_embeds, dim=0)  # Apply average pooling

                # Apply dropout to the document embedding
                dropout = nn.Dropout(p=0.3)  # Adjust the dropout probability as needed
                document_embedding = dropout(document_embedding)

                train_document_embeddings.append(document_embedding)
            
        train_document_embeddings = torch.stack(train_document_embeddings)

        # Apply softmax for classification
        probabilities = F.softmax(train_document_embeddings, dim=1)
        _, predicted_classes = torch.max(probabilities, dim=1)

        # Compute the loss and accuracy for the training set
        loss = loss_function(probabilities, torch.tensor(train_labels))
        accuracy = (predicted_classes == torch.tensor(train_labels)).float().mean()

        # Perform backward propagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        print("Fold {:02d}, Epoch {:05d} | Train Loss {:.4f} | Train Accuracy {:.4f} | Time(s) {:.4f}".format(
            fold + 1, epoch, loss.item(), accuracy.item(), np.mean(dur)))

        # Store train loss and accuracy for each epoch
        train_losses.append(loss.item())
        train_accuracies.append(accuracy.item())
        
        net.eval()

        # Create document embeddings for test data
        test_document_embeddings = []
        for document in test_documents:
            uni_words = set()
            uni_words.update(document.split())
            uni_words = list(uni_words)
            word_embeds = []
            for uni_word in uni_words:
                if uni_word in word_to_index:
                    word_index_value = word_to_index[uni_word]
                    word_embed = node_embeddings[word_index_value]
                    word_embeds.append(word_embed)

            # Compute the document embedding
            if len(word_embeds) > 0:
                word_embeds = torch.stack(word_embeds)
                document_embedding = torch.mean(word_embeds, dim=0)
                test_document_embeddings.append(document_embedding)

        test_document_embeddings = torch.stack(test_document_embeddings)

        # Apply softmax for classification
        test_probabilities = F.softmax(test_document_embeddings, dim=1)
        _, test_predicted_classes = torch.max(test_probabilities, dim=1)

        # Compute the loss and accuracy for the test set
        test_loss = loss_function(test_probabilities, torch.tensor(test_labels))
        test_accuracy = (test_predicted_classes == torch.tensor(test_labels)).float().mean()
        
        # Calculate AUC for training set
        train_auc = roc_auc_score(torch.tensor(train_labels), probabilities.detach().numpy(), multi_class='ovr')
        # Calculate F1 score for training set
        train_f1 = f1_score(torch.tensor(train_labels), predicted_classes.detach().numpy(), average='weighted')


        # Calculate AUC for test set
        test_auc = roc_auc_score(torch.tensor(test_labels), test_probabilities.detach().numpy(), multi_class='ovr')
        # Calculate F1 score for test set
        test_f1 = f1_score(torch.tensor(test_labels), test_predicted_classes.detach().numpy(), average='weighted')

        # Store AUC and F1 score for each fold
        train_aucs.append(train_auc)
        train_f1s.append(train_f1)
        test_aucs.append(test_auc)
        test_f1s.append(test_f1)


        if epoch >= 3:
            dur.append(time.time() - t0)

        print("Fold {:02d} | Test Loss {:.4f} | Test Accuracy {:.4f}".format(
            fold + 1, test_loss.item(), test_accuracy.item()))

        # Store test loss and accuracy for each fold
        test_losses.append(test_loss.item())
        test_accuracies.append(test_accuracy.item())

    # Store the evaluation metrics for the current fold
    fold_train_losses.append(train_losses)
    fold_train_accuracies.append(train_accuracies)
    fold_test_losses.append(test_losses)
    fold_test_accuracies.append(test_accuracies)
    fold_train_aucs.append(train_aucs)
    fold_train_f1s.append(train_f1s)
    fold_test_aucs.append(test_aucs)
    fold_test_f1s.append(test_f1s)
    
    # Save the trained model for the current fold
    torch.save(net.state_dict(), f"gat_model_fold{fold+1}.pt")
    
# Print the AUC and F1 score for each fold
for fold in range(k):
    print("Fold {:02d} | Train AUC: {:.4f} | Train F1: {:.4f}".format(
        fold + 1, np.mean(fold_train_aucs[fold]), np.mean(fold_train_f1s[fold])))
    print("Fold {:02d} | Test AUC: {:.4f} | Test F1: {:.4f}".format(
        fold + 1, np.mean(fold_test_aucs[fold]), np.mean(fold_test_f1s[fold])))
    
# Calculate the average AUC and F1 score across all folds
avg_train_auc = np.mean([np.mean(auc) for auc in fold_train_aucs])
avg_train_f1 = np.mean([np.mean(f1) for f1 in fold_train_f1s])
avg_test_auc = np.mean([np.mean(auc) for auc in fold_test_aucs])
avg_test_f1 = np.mean([np.mean(f1) for f1 in fold_test_f1s])

# Print the average evaluation metrics
print("Average Train AUC: {:.4f}".format(avg_train_auc))
print("Average Train F1: {:.4f}".format(avg_train_f1))
print("Average Test AUC: {:.4f}".format(avg_test_auc))
print("Average Test F1: {:.4f}".format(avg_test_f1))


# In[28]:


# Load the saved model for prediction
fold_number = 2  # Specify the fold number for the desired model
model_path = f"gat_model_fold{fold_number}.pt"

net = GAT(g,
          in_dim=300,
          hidden_dim=64,
          out_dim=10,
          num_heads=8)

net.load_state_dict(torch.load(model_path))
net.eval()


# In[32]:


def predict_labels(model, documents):
    node_embeddings = model(torch.tensor(word_embeddings))

    document_embeddings = []
    for document in documents:
        uni_words = set(document.split())
        word_embeds = []
        for uni_word in uni_words:
            if uni_word in word_to_index:
                word_index_value = word_to_index[uni_word]
                word_embed = node_embeddings[word_index_value]
                word_embeds.append(word_embed)

        if len(word_embeds) > 0:
            word_embeds = torch.stack(word_embeds)
            document_embedding = torch.mean(word_embeds, dim=0)
            document_embeddings.append(document_embedding)

    document_embeddings = torch.stack(document_embeddings)

    probabilities = F.softmax(document_embeddings, dim=1)
    _, predicted_classes = torch.max(probabilities, dim=1)

    return predicted_classes.numpy()


# In[33]:


resume_documents = resumes


# In[34]:


predicted_labels = []
fold_predicted_labels = predict_labels(net, resume_documents)
predicted_labels.append(fold_predicted_labels)

predicted_labels


# In[37]:


predicted_labels[0]


# In[38]:


predicted_labels_inverse = label_encoder.inverse_transform(predicted_labels[0])


# In[39]:


predicted_labels_inverse


# In[40]:


accuracy = accuracy_score(encoded_labels, predicted_labels[0])


# In[41]:


accuracy


# In[ ]:




