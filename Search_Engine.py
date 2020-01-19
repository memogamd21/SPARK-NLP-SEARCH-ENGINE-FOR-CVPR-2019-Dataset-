#!/usr/bin/env python
# coding: utf-8

# In[1]:


from preprocess import *
import math
import GUI as gui
from collections import Counter
import operator


# In[2]:


resultspp = 30


# In[3]:


#In this cell we are trying to extract the info of the abstract from the paper itself
import io 
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage

def extract_text_from_pdf(pdf_path):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle)
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(pdf_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh, 
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()
    # close open handles
    converter.close()
    fake_file_handle.close()
 
    if text:
        return text


# In[ ]:


#txt = extract_text_from_pdf("D:\\Desktop\\Fall2019\\BigData\\cvpr-2019-papers\\CVPR2019\\CVPR2019\\papers\\Aafaq_Spatio-Temporal_Dynamics_and_Semantic_Attribute_Enriched_Visual_Encoding_for_Video_CVPR_2019_paper.pdf")


# In[ ]:


# y = wordninja.split(txt)
# type(y)


# In[ ]:


#!pip install wordninja


# In[ ]:


#import wordninja


# In[10]:


import os
import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from pyspark import SparkContext
from pyspark import HiveContext
from preprocess import *
import math
import GUI as gui
from collections import Counter
import operator
import webbrowser
stop_words = set(stopwords.words('english')) 
abstracts = {}
props = []
for filename in os.listdir('D:\\Desktop\\Fall2019\\BigData\\cvpr-2019-papers\\CVPR2019\\CVPR2019\\abstracts'):
    #txt = extract_text_from_pdf('D:\\Desktop\\Fall2019\\BigData\\cvpr-2019-papers\\CVPR2019\\CVPR2019\\papers\\' + filename)
    ff = open('D:\\Desktop\\Fall2019\\BigData\\cvpr-2019-papers\\CVPR2019\\CVPR2019\\abstracts\\' + filename, 'r')
    #index_abstract = txt.find("Abstract")+8
    #index_intro = txt.find("1.Introduction")
    #abstract = txt[index_abstract:index_intro]
    #introduction = txt[index_intro+14 : txt.find("2.RelatedWork")]
    lines = ff.readline()
    word_tokens = word_tokenize(lines)
    #word_tokens1 = wordninja.split(abstract)
    #word_tokens2 = wordninja.split(introduction)
    #word_tokens = word_tokens1 + word_tokens2
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    abstracts[filename[:-4]] = filtered_sentence
    props.append(filtered_sentence)


# In[11]:


sc = SparkContext()
sql = HiveContext(sc)
absts = sc.parallelize(abstracts)


# In[12]:


keyvalueflat = absts.flatMapValues(lambda word: word)


# In[22]:


def invert(k,v):
    return (v,k)
invertkey = keyvalueflat.map(invert)


# In[23]:


def to_list(a):
    return [a]

def append(a, b):
    a.append(b)
    return a

def extend(a, b):
    a.extend(b)
    return a


# In[ ]:


abstracts_per_word = invertkey.combineByKey(to_list,append,extend)
inverted_i = abstracts_per_word.collect()
inverted_index = {}


# In[ ]:


for w in inverted_i:
    word = w[0].encode('ascii')
    for u in w[1]:
        user = u.encode('ascii')
        inverted_index.setdefault(word, {})[user] = inverted_index.setdefault(word, {}).get(user, 0) + 1


# In[ ]:


df = {}
idf = {}


# In[ ]:


for key in inverted_index.keys():
    df[key] = len(inverted_index[key].keys())
    idf[key] = math.log(1293 / df[key], 2)


# In[ ]:


def tf_idf(w, doc):
    return inverted_index[w][doc] * idf[w]


# In[ ]:


def innrproduct_similarities(query):
    similarity = {}
    for w in query:
        wq = idf.get(w, 0)
        if wq != 0:
            for doc in inverted_index[w].keys():
                similarity[doc] = similarity.get(doc, 0) + tf_idf(w, doc) * wq
    return similarity


# In[ ]:


def lengthofpaper(pid):
    words_accounted_for = []
    length = 0
    for w in abstracts[pid]:
        if w not in words_accounted_for:
            length += tf_idf(w, pid) ** 2
            words_accounted_for.append(w)
    return math.sqrt(length)


# In[ ]:


def lengthofquery(query):
    length = 0
    cnt = Counter()
    for w in query:
        cnt[w] += 1
    for w in cnt.keys():
        length += (cnt[w]*idf.get(w, 0)) ** 2
    return math.sqrt(length)


# In[ ]:


def cosine_similarities(query):
    similarity = innrproduct_similarities(query)
    for doc in similarity.keys():
        similarity[doc] = similarity[doc] / lengthofpaper(doc) / lengthofquery(query)
    return similarity


# In[ ]:


def rankeddocs(similarities):
    return sorted(similarities.items(), key=operator.itemgetter(1), reverse=True)


# In[ ]:


def new_query():
    query = gui.ask()
    if query is None:
        exit()
    query_tokens = preprocess(query)
    ranked_similarities = rankeddocs(cosine_similarities(query_tokens))
    handle_show_query(ranked_similarities, query_tokens, resultspp)


def handle_show_query(ranked_similarities, query_tokens, n):
    choice = gui.display_query_results(ranked_similarities[:n], query_tokens)

    if choice == 'Show more results':
        handle_show_query(ranked_similarities, query_tokens, n + resultspp)
    else:
        if choice is None:
            new_query()
        else:
            open_pdf(choice)


def open_pdf(doc):
    os.startfile('D:\\Desktop\\Fall2019\\BigData\\cvpr-2019-papers\\CVPR2019\\CVPR2019\\papers\\'+ doc.split()[0] + '.pdf')
new_query()

sc.stop()


# In[ ]:




