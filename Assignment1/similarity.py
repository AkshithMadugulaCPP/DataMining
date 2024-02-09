# -------------------------------------------------------------------------
# AUTHOR: Ashith Madugula
# FILENAME: similarity.py
# SPECIFICATION: Print the highest cosine similarity for document-term matrix
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: 20mins
# -----------------------------------------------------------*/

# Importing some Python libraries
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Defining the documents
doc1 = "soccer is my favorite sport"
doc2 = "I like sports and my favorite one is soccer"
doc3 = "support soccer at the olympic games"
doc4 = "I do like soccer, my favorite sport in the olympic games"

# Use the following words as terms to create your document-term matrix
# [soccer, favorite, sport, like, one, support, olympic, games]
# --> Add your Python code here

docs = {
    "doc1":doc1, 
    "doc2":doc2, 
    "doc3":doc3, 
    "doc4":doc4}
counts = {}
terms = ["soccer", "favorite", "sport", "like","one", "support", "olympic", "games"]
for key in docs:
    counts[key] = []
    doc = docs[key]
    for term in terms:
        counts[key].append(doc.count(term))

#counts would be
"""
'doc1': [1, 1, 1, 0, 0, 0, 0, 0]
'doc2': [1, 1, 1, 1, 1, 0, 0, 0]
'doc3': [1, 0, 0, 0, 0, 1, 1, 1]
'doc4': [1, 1, 1, 1, 0, 0, 1, 1]        
"""
# so the document-term matrix would be the list of values
document_term_matrix = list(counts.values())




# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors only
# Use cosine_similarity([X, Y, Z]) to calculate the pairwise similarities between multiple vectors
# --> Add your Python code here
maximum = 0
for first in counts.keys():
    for second in counts.keys():
        if first!=second:
            cos_similarity = cosine_similarity([counts[first]], [counts[second]])[0][0]
            if cos_similarity> maximum:
                maximum = cos_similarity
                keys = first,second

# Print the highest cosine similarity following the information below
# The most similar documents are: doc1 and doc2 with cosine similarity = x
# --> Add your Python code here
print(f"The most similar documents are: {keys[0]} and {keys[1]} with cosine similarity = {maximum}")