import os
import json
import re
from nltk.stem import PorterStemmer
from collections import defaultdict
import math
import sys

# Initialize the stemmer
stemmer = PorterStemmer()

# Helper function: Tokenize and stem content
def tokenize_and_stem(text):
    # Remove non-alphanumeric characters and split
    tokens = re.findall(r'\b\w+\b', text.lower())
    # Stem each token
    return [stemmer.stem(token) for token in tokens]

# Function to build the inverted index
def build_inverted_index(input_folder):
    inverted_index = defaultdict(list)
    doc_lengths = {}
    doc_id = 0

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.json'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        content = data.get("content", "")
                        url = data.get("url", "")
                        
                        if not content:
                            continue

                        doc_id += 1
                        tokens = tokenize_and_stem(content)
                        term_freq = defaultdict(int)

                        # Calculate term frequency for the document
                        for token in tokens:
                            term_freq[token] += 1

                        # Normalize term frequency
                        doc_length = sum(term_freq.values())
                        doc_lengths[doc_id] = doc_length
                        for token, freq in term_freq.items():
                            tf = freq / doc_length
                            inverted_index[token].append((doc_id, tf))
                    
                    except json.JSONDecodeError:
                        print(f"Error reading file: {filepath}", file=sys.stderr)

    return inverted_index, doc_id, len(inverted_index)

# Function to save index to disk
def save_index_to_disk(index, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(index, f)

# Function to generate analytics
def generate_analytics(index, num_docs, output_file):
    index_size = os.path.getsize(output_file) / 1024  # in KB
    return {
        "Number of indexed documents": num_docs,
        "Number of unique tokens": len(index),
        "Total index size (KB)": round(index_size, 2)
    }

# Main function to run the process
def main(input_folder, output_file):
    print("Building the inverted index...")
    inverted_index, num_docs, num_tokens = build_inverted_index(input_folder)

    print(f"Saving the index to {output_file}...")
    save_index_to_disk(inverted_index, output_file)

    print("Generating analytics...")
    analytics = generate_analytics(inverted_index, num_docs, output_file)
    
    print("\nAnalytics Report:")
    for key, value in analytics.items():
        print(f"{key}: {value}")

# Example usage
if __name__ == "__main__":
    input_folder = "path/to/analyst_dataset"  # Change to your dataset folder
    output_file = "inverted_index.json"
    main(input_folder, output_file)
