# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: small_projects
#     language: python
#     name: small_projects
# ---

# %%
import numpy as np
import requests
import os
import tarfile

DATA_DIR = '../data'


# %%
def download_and_extract_data(url, data_dir=DATA_DIR):
    # Check if data folder exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data = requests.get(url)
    with open(f'{data_dir}/cisi.tar.gz', 'wb') as f:
        f.write(data.content)

    # Extract the data
    tar = tarfile.open(f'{data_dir}/cisi.tar.gz')
    tar.extractall(f'{data_dir}/cisi')
    tar.close()

    # Delete the .tar.gz file
    os.remove(f'{data_dir}/cisi.tar.gz')



# %%
# Download the data .tar.gz and store in the data folder
# https://ir.dcs.gla.ac.uk/resources/test_collections/cisi/cisi.tar.gz
download_and_extract_data('https://ir.dcs.gla.ac.uk/resources/test_collections/cisi/cisi.tar.gz')

# %%
import re

def load_documents(filename, schema_mapping):
    documents = []
    with open(filename, 'r') as file:
        content = file.read()
        current_doc = {}

        for i, line in enumerate(content.split('\n')):
            pattern = re.compile(r'\.\w+')
            isKey = pattern.match(line)
            if isKey:
                raw_key = isKey.group()
                if raw_key in schema_mapping:
                    key_config = schema_mapping[raw_key].copy()
                    if 'default' not in key_config:
                        key_config["default"] = ""
                    if 'delimiter' not in key_config:
                        key_config["delimiter"] = None
                    if key_config["alias"] == 'id':
                        if current_doc:
                            documents.append(current_doc)
                        current_doc = {}
                        current_doc[key_config["alias"]] = int(line.split(' ')[1])
                    else:
                        current_doc[key_config["alias"]] = key_config["default"][:] # Requires [:] to copy the list
                else:
                    key_config = None
            elif key_config is not None:
                if key_config["default"] == []:
                    if key_config["delimiter"]:
                        line = line.split(key_config["delimiter"])
                    current_doc[key_config["alias"]].append(line)
                else:
                    current_doc[key_config["alias"]] += line.strip() + ' '

    return documents



# %%
doc_mapping = {
    '.I': {
        "alias": 'id',
        "default": ""
    },
    '.T': {
        "alias": 'title',
        "default": ""
    },
    '.A': {
        "alias": 'author',
        "default": []
    },
    '.W': {
        "alias": 'abstract',
        "default": ""
    },
    '.X': {
        "alias": 'xrefs',
        "default": [],
        "delimiter": "\t"
    }
}
docs = load_documents(f'{DATA_DIR}/cisi/CISI.ALL', doc_mapping) 

# %%
query_mapping = {
    '.I': {
        "alias": 'id',
        "default": ""
    },
    '.W': {
        "alias": 'query',
        "default": ""
    }
}
queries = load_documents(f'{DATA_DIR}/cisi/CISI.QRY', query_mapping) 

# %%
# Results are in the form of <query_id> <document_id> ordered by relevance
relevant_results = np.loadtxt(f'{DATA_DIR}/cisi/CISI.REL')

# %% [markdown]
# # Basic Vector Space Model Implementation
#
# A Vector Space model is a Text Retrieval system. We have preselected a Vector Space Model to solve this text retrieval task which means we have already made a few decisions before we even are getting started. A Vector Space Model has the following characteristics to solve TR tasks:
# - The relevant documents are ranked by relative relevance to the query rather than being classified as relevant or non-relevant.
# - The documents are considered relevant based on the relative similarity of the query and the document.
# - The documents are represented as vectors in a multi-dimensional space. The query is also represented as a vector in the same space.
#
# The factors that we have to consider when implementing a Vector Space Model are:
# 1. The similarity measure to compare the query and the documents.
# 2. The dimensions to span the vector space.
# 3. The encoding of the documents and the query as vectors.
#
#
# # Evaluation
#
# Important properties of a good search are relevance and speed. We will evaluate the performance of our Vector Space Model based on the following metrics:
#
# In terms of relevance:
#    - Mean Average Precision (MAP)
#
# In terms of speed:
#     - Average Query Computational Time

# %%
# Implementing the Timing Function
import time

def time_this(iter, func, *args, **kwargs):
  start_time = time.time()

  # Your code or function here
  # for example:
  for i in range(iter):
      func(*args, **kwargs)

  end_time = time.time()
  avg_time = round((end_time - start_time) / iter,6)
  print(f"Avg. Time taken for {iter} iterations:", avg_time, "seconds")
  return avg_time



# %%
def precision_recall_at_k(k, results, relevant_docs):
    # k cannot be larger than the number of results
    assert k <= len(results)

    # Get the top k results
    top_k_results = results[:k]
    # Get the number of relevant documents in the top k results
    num_rel_docs = len(set(top_k_results).intersection(relevant_docs))
    # k result is relevant or not 
    k_is_relevant = is_relevant(results[k-1], relevant_docs)

    # Return the precision at k
    assert num_rel_docs <= k

    return num_rel_docs / k, num_rel_docs / len(relevant_docs), k_is_relevant

def is_relevant(doc_id, relevant_results):
    # Check if the document is in the relevant results
    return doc_id in relevant_results

def average_precision_to_k(k,results, relevant_docs):
    # For results larger than k we assume that the precision is 0
    ap_num = 0
    for i in range(1,k+1):
        p_i, _, k_is_relevant = precision_recall_at_k(i, results, relevant_docs)
        if k_is_relevant:
            # If the recall increased, add the precision at k to the AP otherwise add 0
            ap_num += p_i
    total_rel_docs = len(relevant_docs)

    return ap_num / total_rel_docs if total_rel_docs > 0 else 0


# %%
# # We're using Numpy for the implementation as it is much faster than Python lists
# list_1 = list(range(1000000))
# list_2 = list(range(1000000))

# np_1 = np.array(list_1)
# np_2 = np.array(list_2)

# def add_lists(list_1, list_2):
#     return [x + y for x, y in zip(list_1, list_2)]

# def add_numpy_arrays(list_1, list_2):
#     return np.add(list_1, list_2)

# list_perf = time_this(100, add_lists, list_1, list_2)
# np_perf = time_this(100, add_numpy_arrays, np_1, np_2)

# print("Numpy Operations can be", round(list_perf / np_perf, 2), "times faster than Python list operations")

# %% [markdown]
# # First Iteration
#
# For the first implementation we are going to be using the following:
# 1. Dot-Product as the similarity measure.
# 2. The dimensions / tokens are going to be the unique words in the corpus. 
# 3. The encoding of the documents and the query are going to be bit vectors (i.e. a word being present or not).
#
# To solve this task we will have to implement the following equation in our ranking function:
#
# $$f(d,q) = \sum^N_{i=1} x_i y_i = \sum_{w\ \in\ q \cap d} b(w, q) b(w,d), \ where \ b(w,q),b(w,d) \in \{0,1\}$$
#
#

# %%
from collections import defaultdict

def basic_tokenizer(text):
    # Each document is a string of words separated by spaces (does not apply to all languages)
    # Remove punctuation and symbols.
    pattern = re.compile(r'[^\w\s]')
    text = pattern.sub('', text)

    # Split by spaces and remove empty strings. 
    # Ensure all words are lowercase and return only unique words.
    return set([word.lower() for word in text.split(' ') if word != ''])

def basic_inverted_index(tokenized_docs):
    # Create the defaultdict
    inv_index = defaultdict(dict)

    # Loop through the documents
    for doc_id, doc in tokenized_docs.items():
        for token in doc:
            if "doc_ids" not in inv_index[token]:
                inv_index[token]["doc_ids"] = []
            if doc_id not in inv_index[token]['doc_ids']:
                inv_index[token]['doc_ids'].append(doc_id)
            
    return inv_index

def basic_ranking_function(query, inv_index):
    # Implementation of a basic ranking function
    # Takes a query and an inverted index and returns a ranked list of documents
    # according to the basic scoring function
    tokenized_query = basic_tokenizer(query)

    scores = defaultdict(int)
    for word in tokenized_query:
        for doc_id in inv_index[word]['doc_ids']:
            scores[doc_id] += 1
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def basic_search(query, inv_index, docs_by_id):
    # Implementation of a basic search
    # Takes a query and an inverted index and returns a ranked list of documents

    ranks = basic_ranking_function(query, inv_index)
    return [(docs_by_id[doc_id], score) for doc_id, score in ranks]



# %%
docs_by_id = {doc['id']: doc for doc in docs}

# Preprocess the documents (add title and abstract) into a dictionary by ID
prepared_docs = {doc['id']: doc['title'] + ' ' + doc['abstract'] for doc in docs}
tokenized_docs = {doc_id: basic_tokenizer(doc) for doc_id, doc in prepared_docs.items()}

# Create the inverted index
inv_index = basic_inverted_index(tokenized_docs)

# %%
time_this(1000, basic_search, queries[0]['query'], inv_index, docs_by_id)

# %%
sample_query = queries[0]['query']
print("Query: " + sample_query)
results = basic_search(sample_query, inv_index, docs_by_id)
for result in results[:3]:
    print("ID: " + str(result[0]["id"]))
    print("Title: " + result[0]["title"])
    print("Snippet: " + result[0]["abstract"][:100] + "...")
    print("Score: " + str(result[1]))

# %%
sample_relevant_results = relevant_results[relevant_results[:, 0] == 1][:, 1].astype(int)
result_ids = [result[0]['id'] for result in results]

# %%
k = 1000
print(f"Average Precision at {k}", average_precision_to_k(k,result_ids, sample_relevant_results))

# %%
