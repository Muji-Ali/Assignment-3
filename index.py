import os
import json
import re
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter

# Initialize the Porter Stemmer
stemmer = PorterStemmer()

base_path = '/Users/muji/Downloads/ANALYST'

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

    # Ensure base_path is correct and exists
    if not os.path.exists(base_path):
        print(f"Base path does not exist: {base_path}")
        return None

    # Go through each domain directory
    for domain_folder in os.listdir(base_path):
        domain_path = os.path.join(base_path, domain_folder)
        if not os.path.isdir(domain_path):
            continue

        # Process each file in the domain
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
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                continue

    return inverted_index

# Build the inverted index
inverted_index = build_inverted_index()

if inverted_index is None:
    print("Failed to build inverted index. Check the data or logic.")
else:
    # Save the inverted index to a JSON file
    output_path = './inverted_index.json'
    with open(output_path, 'w', encoding='utf-8') as out_file:
        json.dump(inverted_index, out_file)

    print(f"Inverted index saved to {output_path}")

    # Analytics Section
    # 1. Number of Indexed Documents
    num_documents = len(inverted_index)
    print(f"Number of Indexed Documents: {num_documents}")

    # 2. Number of Unique Tokens
    unique_tokens = set()
    for tokens in inverted_index.values():
        unique_tokens.update(tokens)
    num_unique_tokens = len(unique_tokens)
    print(f"Number of Unique Tokens: {num_unique_tokens}")

    # 3. Total Size of the Index
    index_size = os.path.getsize(output_path) / 1024  # Size in KB
    print(f"Total Size of Index: {index_size:.2f} KB")

