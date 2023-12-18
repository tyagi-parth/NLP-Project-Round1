#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy
import nltk

from nltk.corpus import stopwords


# In[2]:


file = open("pride-and-prejudice-text-format.txt",encoding='utf-8')
wordslists = file.read().splitlines()
wordslists = [i for i in wordslists if i != ' ']
text = " "
text = text.join(wordslists)


# In[3]:


text[:2000]


# In[4]:


# STEP 2 : Normalize the text i.e convert our text T to a cleaner standard format.
#removing all punctuationns from our text file
punctuations = '''!()-[]{};:'"\,<>./‘’?“”@#$%^&*_~'''
cleantext = ""
for char in text:
    if char not in punctuations:
        cleantext = cleantext + char
        
#Converting the text into lower case         
cleantext = cleantext.lower()


# In[5]:


#remove unwanted spaces
import re
from nltk.tokenize import word_tokenize
res = re.sub(' +', ' ', cleantext)
cleantext = str(res)

print(cleantext)
tokens = word_tokenize(cleantext)
stop_words = set(stopwords.words('english'))
# tokens = word_tokenize(cleantext)
tokens_final = [i for i in tokens if not i in stop_words] # tokenising with removing stopwords
finaltext = "  "
finaltext = finaltext.join(tokens_final)


# In[6]:


formatted_text = finaltext.replace('chapter','#parth#chapter')
formatted_text[:2000]


# In[7]:


chapter_splitted_text = formatted_text.split("#parth#")


# In[8]:


print(chapter_splitted_text)


# In[9]:


chapter1 = chapter_splitted_text[1]
from tabulate import tabulate
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Generate matrix of word vectors
vectors = vectorizer.fit_transform(chapter_splitted_text)

feature_names = vectorizer.get_feature_names_out()

dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names,index = [f"Chapter {i+1}" for i in range(len(chapter_splitted_text))])
# display(df)
# df.style.format(precision=7).format_index(str, axis=1)
df = df.round(7)

def highlight_non_zero(val):
    color = 'background-color: lightgreen' if val != 0 else ''
    return color

# Styling for better visual representation
styled_df = df.style\
    .format(precision=7)\
    .set_table_styles([{
        'selector': 'th',
        'props': [('background-color', 'black')]
    }])\
    .map(highlight_non_zero)

# Display the styled DataFrame
styled_df


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(vectors[45:55], vectors[45:55])

# Visualize the similarity matrix as a gradient table
plt.figure(figsize=(10,8))
sns.heatmap(similarity_matrix, annot=True, cmap="YlGnBu", xticklabels=range(45, 55), yticklabels=range(45,55))
plt.title("Chapter Similarity Matrix")
plt.xlabel("Chapter")
plt.ylabel("Chapter")
plt.show()


# In[12]:


ch1 = -1
ch2 = -1
mx = -1
for i in range(len(chapter_splitted_text)):
    for j in range(0,i + 1):
        if i == j:
            continue
        if mx < similarity_matrix[i][j]:
            mx = similarity_matrix[i][j]
            ch1 = i + 1
            ch2 = j + 1
            # print(ch1,ch2,mx)
print("Maximum similarity is between Chapter " + str(ch1) + " and Chapter " + str(ch2))


# In[23]:


#PART 1
from spacy import displacy
NER = spacy.load("en_core_web_sm")
named_entity_text = NER(text)

entity_types_of_interest = [
    "PER",
    "ORG",
    "LOC",
    "GPE",
    "FAC",
    "VEH",
]

entities = []
for word in named_entity_text.ents:
    if word.label_ in entity_types_of_interest:
        entities.append((word.text, word.label_))

print(tabulate(entities, headers=["Entity", "Entity Type"], tablefmt="double_outline "))


# In[ ]:





# In[14]:


# Evaluating the method


# In[15]:


# take a random passage of length
import numpy as np
randomPassageIndex = 54323
randomPassageIndex2 = 23456
passage = text[randomPassageIndex:randomPassageIndex + 3000] + " " + text[randomPassageIndex2:randomPassageIndex2 + 3000]

print(passage)


# In[16]:


# Manual Named Entities recognized
# Total entities recognized(manually) = 65


# In[17]:


passageEntityText = NER(passage)
spacy.displacy.render(passageEntityText,style = "ent")


# In[18]:


# Entities recognized by the model = 53
# Correctly identified entities = 48
correct_entity = 48
actualEntities = 65
predictedEntities = len(passageEntityText.ents)
# precision = correctly identified entities/predcitedEntities
# recall = correctly identified entities/actualEntites

precision = correct_entity/predictedEntities
recall = correct_entity/actualEntities

F_Score = 2*precision*recall/(precision + recall)

print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F-Score: " + str(F_Score))


# In[ ]:




