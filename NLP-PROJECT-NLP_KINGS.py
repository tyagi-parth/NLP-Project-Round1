#!/usr/bin/env python
# coding: utf-8

# In[39]:


import nltk

from nltk.corpus import stopwords
import string
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
from nltk.tokenize import word_tokenize
import re
from tabulate import tabulate
import random


# In[2]:


nltk.download('stopwords')


# In[3]:


nltk.download('punkt')


# In[4]:


nltk.download('averaged_perceptron_tagger')


# In[5]:


# STEP 1: Import the text, let’s call it as T (book that you have downloaded and in TxT format) 
file = open("pride-and-prejudice-text-format.txt",encoding='utf-8')
wordslists = file.read().splitlines()
wordslists = [i for i in wordslists if i != ' ']
text = " "
text = text.join(wordslists)


# In[6]:


type(file)


# In[7]:


text[:2000]


# In[8]:


len(text)

# #### Preprocessing
# In[9]:


# STEP 2 : Normalize the text i.e convert our text T to a cleaner standard format.
#removing all punctuationns from our text file
punctuations = '''!()-[]{};:'"\,<>./‘’?“”@#$%^&*_~'''
cleantext = ""
for char in text:
    if char not in punctuations:
        cleantext = cleantext + char
        
#Converting the text into lower case         
cleantext = cleantext.lower()


# In[10]:


cleantext[:2000]


# In[11]:


#remove unwanted spaces
res = re.sub(' +', ' ', cleantext)
cleantext = str(res)

cleantext[:2000]


# In[12]:


# STEP 3 : Tokenise T and Remove the stop words from T
# Tokenizing 
tokens = word_tokenize(cleantext)
tokens[:50] #first 50 tokens


# In[13]:


type(tokens)


# In[14]:


len(tokens)


# In[15]:


# Removing stopwords and storing it into finaltext
stop_words = set(stopwords.words('english'))
# tokens = word_tokenize(cleantext)
tokens_final = [i for i in tokens if not i in stop_words] # tokenising with removing stopwords
finaltext = "  "
finaltext = finaltext.join(tokens_final)


# In[16]:


finaltext[:2000]


# In[17]:


#STEP 4 : Analyse the frequency distribution of tokens in T
#With stopwords
freq = nltk.FreqDist(tokens)
freq = {k: v for k, v in sorted(freq.items(), key=lambda item: item[1],reverse=True)}
x = list(freq.keys())[:40]
y = list(freq.values())[:40]
plt.figure(figsize=(12,5))
plt.plot(x,y,c='r',lw=4,ls='-.')
plt.grid()
plt.xticks(rotation=90)
plt.title('Token Frequency (with stopwords)',size=17)
plt.xlabel('Words',size=14)
plt.ylabel('Count',size=14)
plt.show()


# In[18]:


#STEP 4 : Analyse the frequency distribution of tokens in T
#Without stopwords
freq = nltk.FreqDist(finaltext)
freq = {k: v for k, v in sorted(freq.items(), key=lambda item: item[1],reverse=True)}
x = list(freq.keys())[:40]
y = list(freq.values())[:40]
plt.figure(figsize=(12,5))
plt.plot(x,y,c='r',lw=4,ls='-.')
plt.grid()
plt.xticks(rotation=90)
plt.title('Token Frequency (without stopwords)',size=17)
plt.xlabel('Words',size=14)
plt.ylabel('Count',size=14)
plt.show()


# In[19]:


# Create a word cloud on the Tokens in T 
#With stopwords
wordcloud = WordCloud(width = 800, height = 600, 
                background_color ='white', 
                min_font_size = 10,stopwords = {},colormap='winter').generate(cleantext) 

plt.figure(figsize = (12,8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 


# In[20]:


# Create a word cloud on the Tokens in T 
# Without stopwords
wordcloud = WordCloud(width = 800, height = 600, 
                background_color ='white', 
                min_font_size = 10,stopwords = {},colormap='winter').generate(finaltext) 

plt.figure(figsize = (12,8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 


# # #### PoS Tagging and Frequency Distribution of Tags on Text

# In[21]:


# using Penn Treebank Tag Set
tagged_text = nltk.pos_tag(tokens)
tagged_text[:20]


# In[22]:


type(tagged_text)


# In[23]:


from collections import Counter
counts = Counter( tag for word,  tag in tagged_text)
print(counts)


# In[24]:


freq_tags = nltk.FreqDist(counts)
freq_tags = {k: v for k, v in sorted(freq_tags.items(), key=lambda item: item[1],reverse=True)}
x = list(freq_tags.keys())[:40]
y = list(freq_tags.values())[:40]
plt.figure(figsize=(12,5))
plt.plot(x,y,c='r',lw=4,ls='-.')
plt.grid()
plt.xticks(rotation=90)
plt.title('TAGs Frequency',size=17)
plt.xlabel('Tags',size=14)
plt.ylabel('Count',size=14)
plt.show()


# In[25]:


# Get the largest chapter C from the book
# replace chapter with a intentional delimiter
formatted_text = cleantext.replace('chapter','#parth#chapter')
formatted_text[:2000]


# In[26]:


chapter_splitted_text = formatted_text.split("#parth#")
largest_chapter = ""
largest_chapter_len = 0
for i in chapter_splitted_text:
    chapter = i.split("\n\n")
    if len(i) > largest_chapter_len:
        largest_chapter = str(i)
        largest_chapter_len = len(i)
print(largest_chapter_len)
print(largest_chapter)


# In[27]:


chapter_words = largest_chapter.split()
print(chapter_words)


# In[28]:


bigrams = [(w1, w2) for w1, w2 in zip(chapter_words,chapter_words[1:])]
print(bigrams)


# In[57]:


listOfBigrams = []
bigramCounts = {}
unigramCounts = {}
nbyn = {}

for i in range(len(chapter_words)):
    if i < len(chapter_words) - 1:

        listOfBigrams.append((chapter_words[i], chapter_words[i + 1]))

        if (chapter_words[i], chapter_words[i+1]) in bigramCounts:
            bigramCounts[(chapter_words[i], chapter_words[i + 1])] += 1
        else:
            bigramCounts[(chapter_words[i], chapter_words[i + 1])] = 1

    if chapter_words[i] in unigramCounts:
        unigramCounts[chapter_words[i]] += 1
    else:
        unigramCounts[chapter_words[i]] = 1 

listOfProb = {}

bigram_of_every_word = {}

for bigram in listOfBigrams:
    word1 = bigram[0]
    word2 = bigram[1]
    listOfProb[bigram] = (bigramCounts.get(bigram))/(unigramCounts.get(word1))
    if word1 in bigram_of_every_word:
        bigram_of_every_word[word1].append([word2,str((bigramCounts.get(bigram))/(unigramCounts.get(word1)))])
    else:
        bigram_of_every_word[word1] = [word2,str((bigramCounts.get(bigram))/(unigramCounts.get(word1)))]

bigram_prob_table = []

# print(bigram_of_every_word['and'])

for bigrams in listOfBigrams:
    temp_list = [(str(bigrams)),str(bigramCounts[bigrams]),str(listOfProb[bigrams])]
    bigram_prob_table.append(temp_list)
	# print(str(bigrams) + ' : ' + str(bigramCounts[bigrams]) + ' : ' + str(listOfProb[bigrams]) + '\n')
print (tabulate(bigram_prob_table, headers=["Bigram", "Count", "Probability"]))



# In[64]:


shannon_game_text = random.choice(chapter_splitted_text)
type(shannon_game_text)


# In[65]:


shannon_game_words = shannon_game_text.split()
sentence = shannon_game_words[10:15]
print(sentence)


# In[66]:


prev_word = sentence[-1]
print(prev_word)


# In[67]:


predicted_next_word = prev_word
if prev_word not in bigram_of_every_word:
    print('Word not found')
else:
    next_words = bigram_of_every_word[prev_word]
    next_letters_sorted = sorted(next_words, key=lambda x: x[1],reverse=True)
    predicted_next_word = next_letters_sorted[0]
    print('The predicted next word is : ' + predicted_next_word)
    print('Actual word is : ' + shannon_game_words[16])


# In[ ]:




