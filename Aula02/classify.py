#!/usr/bin/env python
# coding: utf-8

# In[137]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[166]:


def limiar(v):
    if v >= 0: return 2
    else: return 1


# In[167]:


def linear(v):
    result = 0
    if v <= -0.5: result = 0
    elif v >= 0.5: result = 1
    else: result = v+0.5
        
    if result >= 0.5: return 2
    else: return 1


# In[168]:


def sigmoid(v):
    result = np.tanh(v)
    if result >= 0: return 2
    else: return 1


# # Dataset 1

# In[203]:


dataset = pd.read_csv('Aula2-exec1.csv')


# In[204]:


dataset.head()


# In[205]:


dataset.tail()


# In[206]:


dataset.plot(x='V1', y='V2', kind='scatter', c='V3', colorbar=False, colormap='RdBu')
plt.figure(1)


# In[235]:


weights = np.array([-1, -1, 1])


# In[223]:


data = dataset.iloc[:, :-1]
data['bias'] = 1

classes = dataset.iloc[:, -1]


# In[236]:


values = np.sum(weights * data, axis=1)


# In[237]:


limiar_acc = 0
linear_acc = 0
sigmoid_acc = 0
for i, value in enumerate(values):
    predicted_class = limiar(value)
    if predicted_class == dataset.iloc[i, -1]: limiar_acc += 1
        
    predicted_class = linear(value)
    if predicted_class == dataset.iloc[i, -1]: linear_acc += 1
        
    predicted_class = sigmoid(value)
    if predicted_class == dataset.iloc[i, -1]: sigmoid_acc += 1
        
limiar_acc = limiar_acc / dataset.shape[0]
print('Limiar Accuracy:', limiar_acc)

linear_acc = linear_acc / dataset.shape[0]
print('Linar Accuracy:', linear_acc)

sigmoid_acc = sigmoid_acc / dataset.shape[0]
print('Sigmoid Accuracy:', sigmoid_acc)


# # Dataset 2

# In[275]:


dataset = pd.read_csv('Aula2-exec2.csv')


# In[197]:


dataset.head()


# In[198]:


dataset.tail()


# In[289]:


dataset.plot(x='V1', y='V2', kind='scatter', c='V5', colorbar=False, colormap='RdBu')
plt.figure(2)


# In[290]:


dataset.plot(x='V1', y='V3', kind='scatter', c='V5', colorbar=False, colormap='RdBu')
plt.figure(3)


# In[291]:


dataset.plot(x='V1', y='V4', kind='scatter', c='V5', colorbar=False, colormap='RdBu')
plt.figure(4)


# In[292]:


dataset.plot(x='V2', y='V3', kind='scatter', c='V5', colorbar=False, colormap='RdBu')
plt.figure(5)


# In[293]:


dataset.plot(x='V2', y='V4', kind='scatter', c='V5', colorbar=False, colormap='RdBu')
plt.figure(6)


# In[282]:


dataset.plot(x='V3', y='V4', kind='scatter', c='V5', colorbar=False, colormap='RdBu')
plt.figure(7)


# In[297]:


weights = np.array([0, 0, -1, -1, 1])


# In[241]:


data = dataset.iloc[:, :-1]
data['bias'] = 1

classes = dataset.iloc[:, -1]


# In[298]:


values = np.sum(weights * data, axis=1)


# In[299]:


limiar_acc = 0
linear_acc = 0
sigmoid_acc = 0
for i, value in enumerate(values):
    predicted_class = limiar(value)
    if predicted_class == dataset.iloc[i, -1]: limiar_acc += 1
        
    predicted_class = linear(value)
    if predicted_class == dataset.iloc[i, -1]: linear_acc += 1
        
    predicted_class = sigmoid(value)
    if predicted_class == dataset.iloc[i, -1]: sigmoid_acc += 1
        
limiar_acc = limiar_acc / dataset.shape[0]
print('Limiar Accuracy:', limiar_acc)

linear_acc = linear_acc / dataset.shape[0]
print('Linar Accuracy:', linear_acc)

sigmoid_acc = sigmoid_acc / dataset.shape[0]
print('Sigmoid Accuracy:', sigmoid_acc)


# # Comentário

# Para o segundo dataset, qualquer combinação das entradas já é linearmente separável, portanto, não é necessário adicionar pesos para as demais entradas.
# 
# Caso os pesos fossem mantidos em [-1, -1, 0, 0], a acurácia também seria de 100%.

plt.show()