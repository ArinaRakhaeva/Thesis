import json
import requests 
import pandas as pd
import numpy as np 

import spacy
import en_core_web_sm
import regex as re
from gensim.models.phrases import Phrases, Phraser
from time import time  # To time our operations
from collections import defaultdict  # For word frequency

import multiprocessing
from gensim.models import Word2Vec

# Data
dictionary_inflation = ['CPI', 'deflation', 'inflation','disinflation', 'inflationary', 'recession','stagflation','consumption basket','gas','oil','petrol','fuel','electricity','price','cost','income','revenue','wage','expenditure','payment','rent','purchasing power','tariff','sale']

# Functions
def items_in_dictionary(dictionary, string):
    for term in dictionary:
        if term in string:
            return string
    return None

def download_json(url, destination):
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(destination, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded JSON file to {destination}")
    else:
        print(f"Failed to download JSON file. Status code: {response.status_code}")

def cleaning(doc):
    txt = [token.lemma_ for token in doc if not token.is_stop]
    if len(txt) > 2:
        return ' '.join(txt)

month = "1"
year = "2022"

# Paths
json_url = "https://api.nytimes.com/svc/archive/v1/" + year + "/" + month + ".json?api-key=9QigMRakLuhywBq0jduwbEZEyKV3XoiJ"
output_file = 'output.json'

# Download
download_json(json_url, output_file)

json_data = open('output.json')
data = json.load(json_data)
snippets_with_ids = []
lead_paragraph_with_ids = []
abstract_with_ids = []

for idx, doc in enumerate(data["response"]["docs"]):
    try:
        snippet = doc.get("snippet", "")
        cleaned_snippets = items_in_dictionary(dictionary_inflation, snippet)
        snippets_with_ids.append({"id": idx, "snippet": cleaned_snippets})
    except:
        continue
    try: 
        lead_paragraph = doc.get("lead_paragraph", "")
        cleaned_lead_paragraph = items_in_dictionary(dictionary_inflation, lead_paragraph)
        lead_paragraph_with_ids.append({"id": idx, "lead_paragraph": cleaned_lead_paragraph})
    except:
        continue
    try:
        abstract = doc.get("abstract", "")
        cleaned_abstract = items_in_dictionary(dictionary_inflation, abstract)
        abstract_with_ids.append({"id": idx, "abstract": cleaned_abstract})
    except:
        continue

#for item in snippets_with_ids:
#    print(f"ID: {item['id']}, Snippet: {item['snippet']}")

snippet_df = pd.DataFrame(snippets_with_ids, index = None)
#print(snippet_df)
lead_paragraph_df = pd.DataFrame(lead_paragraph_with_ids, index = None)
abstract_df = pd.DataFrame(abstract_with_ids, index = None)

# Clean dfs
merge_snippet = pd.concat([snippet_df,abstract_df], axis=1)
merge_snippet = merge_snippet.drop(merge_snippet.columns[0], axis=1)
df = merge_snippet.dropna(subset=['snippet', 'abstract'], how='all')
df=df.drop_duplicates()
df["abstract"] = np.where((df["snippet"] == df["abstract"]), 0, df["abstract"])
df["snippet_abs"] = np.where(((df["snippet"] != df["abstract"]) & (df["abstract"]==0)), df["snippet"], df["abstract"])
df=df.drop(columns=['snippet', 'abstract'])
# Removing non-alphabetic characters
df["snippet_abs"]  = df.snippet_abs.str.replace("'", "")
df["snippet_abs"]  = df.snippet_abs.str.replace("[^A-Za-z']+", " ", regex=True).str.lower()
#print(df["snippet_abs"])

lp_df = lead_paragraph_df.drop_duplicates (subset=['lead_paragraph'], keep=False)
lp_df = lp_df.drop(lp_df.columns[0], axis=1)
lp_df = lp_df.dropna(subset=['lead_paragraph'], how='all')
# Removing non-alphabetic characters
lp_df['lead_paragraph'] = lp_df.lead_paragraph.str.replace("'", "")
lp_df['lead_paragraph'] = lp_df.lead_paragraph.str.replace("[^A-Za-z']+", " ", regex=True).str.lower()

# Lemmatazing, removing stopwords
nlp = spacy.load("en_core_web_sm")
nlp = en_core_web_sm.load()

# Snippet db
snippet_abs = [nlp(row) for row in df.snippet_abs]
snippet_abs_txt = [cleaning(doc) for doc in snippet_abs]
df_clean = pd.DataFrame({'clean_snippet_abs': snippet_abs_txt})
df_clean["year"] = year
df_clean["month"] = month
df_clean = df_clean.dropna().drop_duplicates() 

# Lead paragraphs db
lead_paragraph = [nlp(row) for row in lp_df.lead_paragraph]
lead_paragraph_txt = [cleaning(doc) for doc in lead_paragraph]
lp_df_clean = pd.DataFrame({'clean_lp': lead_paragraph_txt})
lp_df_clean["year"] = year
lp_df_clean["month"] = month
lp_df_clean = lp_df_clean.dropna().drop_duplicates() 

# Detect common phrases (bigrams) from a list of sentences
row_snippet_abs = [row.split() for row in df_clean['clean_snippet_abs']]
phrases = Phrases(row_snippet_abs, min_count=30, progress_per=10000)
bigram = Phraser(phrases)
sentences = bigram[row_snippet_abs]

# Sanity check
word_freq = defaultdict(int)
for sent in sentences:
    for i in sent:
        word_freq[i] += 1
len(word_freq)
#print(sorted(word_freq, key=word_freq.get, reverse=True)[:10])

# Word2Vec
cores = multiprocessing.cpu_count()

w2v_model = Word2Vec(min_count=20,
                     window=2,
                     vector_size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)
# Building vocab for w2v
t = time()
w2v_model.build_vocab(sentences, progress_per=10000)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

# Training the model
t = time()
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

# Check 
print(w2v_model.wv.most_similar(positive=["price"]))