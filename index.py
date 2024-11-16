import os
import json
from lxml import html
from collections import defaultdict
from string import punctuation
import re


class PorterStemmer:
    #Simple Porter Stemmer implementation
    def __init__(self):
        self.suffixes = ['ing', 'ed', 'ly', 'es', 's', 'er', 'ion']

    def stem(self, word):
        for suffix in self.suffixes:
            if word.endswith(suffix) and len(word) > len(suffix):
                return word[:-len(suffix)]
        return word


class InvertedIndexBuilder:
    def __init__(self, input_folder, output_folder, batch_size=100):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.inverted_index = defaultdict(list)
        self.doc_id_map = {}
        self.batch_size = batch_size
        self.processed_files = 0
        self.partial_index_count = 0
        self.global_unique_tokens = set()  # Track unique tokens globally
        self.stemmer = PorterStemmer()
        os.makedirs(output_folder, exist_ok=True)

    def tokenize(self, text):
        """Custom tokenizer to split text into words.
           Remove punctuation and split by whitespace"""

        return [token.lower() for token in re.split(r'\W+', text) if token]

    def preprocess_content(self, content):
        """Preprocess HTML content to ensure compatibility with lxml."""
        if isinstance(content, str):
            return content.encode('utf-8')
        return content

    def parse_html(self, content):
        """Parse HTML content and extract tokens."""
        try:
            content = self.preprocess_content(content)
            tree = html.fromstring(content)
            text_segments = tree.xpath("//body//text()")
            tokens = []
            for segment in text_segments:
                tokens.extend(self.tokenize(segment))
            return tokens
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return []

    def process_file(self, file_path, doc_id):
        """Process a single HTML file and update the index."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                content = data.get("content", "")
                tokens = self.parse_html(content)
                stemmed_tokens = [self.stemmer.stem(token) for token in tokens]

                # Add tokens to the global unique set
                self.global_unique_tokens.update(stemmed_tokens)

                # Calculate term frequencies
                tf = defaultdict(int)
                for token in stemmed_tokens:
                    tf[token] += 1

                max_tf = max(tf.values(), default=1)
                for token, freq in tf.items():
                    normalized_tf = freq / max_tf
                    self.inverted_index[token].append({"docID": doc_id, "tf": normalized_tf})

                print(f"Processed file {file_path} with {len(tokens)} tokens.")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    def write_partial_index(self):
        """Write the current inverted index to a partial file."""
        try:
            file_path = os.path.join(self.output_folder, f"partial_index_{self.partial_index_count}.json")
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(self.inverted_index, file, indent=4)
            print(f"Partial index written to {file_path}.")
            self.partial_index_count += 1
        except Exception as e:
            print(f"Error writing partial index: {e}")

    def build_index(self):
        """Build the inverted index by processing all files."""
        doc_id = 0
        for root, _, files in os.walk(self.input_folder):
            for file in files:
                if not file.endswith(".json"):
                    continue

                file_path = os.path.join(root, file)
                self.doc_id_map[doc_id] = file_path
                self.process_file(file_path, doc_id)
                doc_id += 1
                self.processed_files += 1

                if self.processed_files >= self.batch_size:
                    self.write_partial_index()
                    self.inverted_index.clear()
                    self.processed_files = 0

        if self.inverted_index:
            self.write_partial_index()

    def save_doc_id_map(self):
        """Save the document ID map."""
        try:
            file_path = os.path.join(self.output_folder, "doc_id_map.json")
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(self.doc_id_map, file, indent=4)
            print(f"Document ID map saved to {file_path}.")
        except Exception as e:
            print(f"Error saving document ID map: {e}")

    def generate_analytics(self):
        """Generate and print analytics"""
        try:
            num_docs = len(self.doc_id_map)
            unique_tokens = len(self.global_unique_tokens)
            total_size_kb = sum(
                os.path.getsize(os.path.join(self.output_folder, f))
                for f in os.listdir(self.output_folder)
                if f.startswith("partial_index_")
            ) / 1024
            analytics = {
                "Number of indexed documents": num_docs,
                "Number of unique tokens": unique_tokens,
                "Total size of the index (KB)": total_size_kb,
            }
            print("Analytics:", analytics)
            return analytics
        except Exception as e:
            print(f"Error generating analytics: {e}")
            return {}


#Testing
input_folder = "/Users/joshnguyen/PycharmProjects/CS121Assignment3M1/DEV"
output_folder = "/Users/joshnguyen/PycharmProjects/CS121Assignment3M1/output"

index_builder = InvertedIndexBuilder(input_folder, output_folder, batch_size=100)
index_builder.build_index()
index_builder.save_doc_id_map()

# Generate Report
analytics = index_builder.generate_analytics()
