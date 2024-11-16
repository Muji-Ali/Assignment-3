import os
import json
import re
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter

# Initialize the Porter Stemmer
stemmer = PorterStemmer()

# Set base_path to the location of your ANALYST folder
base_path = '/Users/muji/Downloads/ANALYST'  # Update this path as needed

# Function to process content of document
def process_content(content):
    soup = BeautifulSoup(content, 'html.parser')
    tokens = []

    important_texts = {
        "title": soup.title.string if soup.title else "",
        "headings": " ".join(h.get_text() for h in soup.find_all(['h1', 'h2', 'h3'])),
        "bold": " ".join(b.get_text() for b in soup.find_all(['b', 'strong']))
    }

    for tag, text in important_texts.items():
        for word in word_tokenize(text):
            if re.match(r'^\w+$', word):
                tokens.append((stemmer.stem(word.lower()), tag))

    for word in word_tokenize(soup.get_text()):
        if re.match(r'^\w+$', word):
            tokens.append((stemmer.stem(word.lower()), "general"))

    return tokens

# Function to build inverted index
def build_inverted_index():
    inverted_index = defaultdict(lambda: defaultdict(int))

    for domain_folder in os.listdir(base_path):
        domain_path = os.path.join(base_path, domain_folder)
        if not os.path.isdir(domain_path):
            continue
        
        for file_name in os.listdir(domain_path):
            file_path = os.path.join(domain_path, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    content = data.get("content", "")
                    doc_id = file_name

                    tokens = process_content(content)

                    term_counts = Counter()
                    for token, tag in tokens:
                        weight = 3 if tag in ["title", "headings", "bold"] else 1
                        term_counts[token] += weight

                    for token, frequency in term_counts.items():
                        inverted_index[token][doc_id] += frequency
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing file {file_path}: {e}")
                continue  # Skip this file if there's an error

    return inverted_index

# Build the inverted index
inverted_index = build_inverted_index()

# Save the inverted index to a JSON file
output_path = './inverted_index.json'
with open(output_path, 'w', encoding='utf-8') as out_file:
    json.dump(inverted_index, out_file)

print(f"Inverted index saved to {output_path}")

# Analytics Section

index_path = '/Users/muji/Downloads/Assignment-3/Assignment-3/inverted_index.json'

# Read the inverted index from the file
with open(index_path, 'r', encoding='utf-8') as file:
    inverted_index = json.load(file)

# 1. Number of Indexed Documents
document_ids = set()
for postings in inverted_index.values():
    document_ids.update(postings.keys())  # Update with document IDs
num_documents = len(document_ids)

# # 2. Number of Unique Tokens
try:
    num_tokens = len(inverted_index)  # Number of unique terms in the index
    print(f"Number of unique tokens: {num_tokens}")
except Exception as e:
    print(f"Error calculating the number of unique tokens: {e}")

# 3. Total Size of the Index on Disk (in KB)
index_size_kb = os.path.getsize(index_path) / 1024  # Size in KB

# Print Analytics
print(f"Number of indexed documents: {num_documents}")
print(f"Number of unique tokens: {num_tokens}")
print(f"Total index size on disk: {index_size_kb:.2f} KB")
