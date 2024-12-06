# -*- coding: utf-8 -*-
#!/env/bin/python3
# Standard Library
import os
import numpy as np
import pandas as pd
from opensearchpy import OpenSearch
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Connect to OpenSearch instance
username = "x"
pwd = "x"

dbclient = OpenSearch(
    hosts=[{'host': 'opensearch-ds.ifi.uni-heidelberg.de', 'port': 443}],
    http_auth=(username, pwd),  # Replace with your credentials
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False
)

# Function to get articles and their word count from OpenSearch index
def get_word_count(client, index_name='kic_apothekenumschau_articles', scroll_time='2m', page_size=100):
    data = []
    response = client.search(
        index=index_name,
        scroll=scroll_time,
        size=page_size,
        body={"_source": ["title", "text"]}
    )

    scroll_id = response['_scroll_id']
    hits = response['hits']['hits']

    while hits:
        for hit in hits:
            title = hit['_source'].get('title', None)
            text = hit['_source'].get('text', '')
            word_count = len(text.split())
            data.append({'Title': title, 'Word Count': word_count, 'Text': text})

        response = client.scroll(scroll_id=scroll_id, scroll=scroll_time)
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']

    return pd.DataFrame(data)

# Function to filter articles based on word count and specific phrases
def filter_dataframe(df):
    min_word_count = 40
    phrase1 = "Medikamente eines Wirkstoffs Wirkstoffe von A bis Z Medikamente mit dem Bestandteil"
    phrase2 = "Beiträge von"

    # Filter by word count and remove specific phrases
    df_filtered = df[(df['Word Count'] >= min_word_count) & 
                     (~df['Text'].str.contains(phrase1, na=False)) & 
                     (~df['Text'].str.contains(phrase2, na=False))]

    return df_filtered

def get_stats(df, df_filtered, filename="./data/stats.txt"):
    with open(filename, "w") as file:
        file.write(f"Number of rows before filtering and removing duplicates: {len(df)}\n")
        file.write(f"Number of rows after filtering and removing duplicates: {len(df_filtered)}\n")
        file.write(f"Mean word count: {df_filtered['Word Count'].mean()}\n")
        file.write(f"Maximum word count: {df_filtered['Word Count'].max()}\n")
        file.write(f"Minimum word count: {df_filtered['Word Count'].min()}\n")
        file.write(f"Median word count: {df_filtered['Word Count'].median()}\n")

# Initialize a text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=70)

# Define the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")

# Function to embed chunks
def embed_chunks(chunks):
    return [embedding_model.embed_query(chunk) for chunk in chunks]

# Batch processing function for large datasets
def process_in_batches(df_filtered, batch_size=1000, max_chroma_batch_size=5461, persist_directory="./data/"):
    total_docs = 0

    # Iterate in batches
    for i in range(0, len(df_filtered), batch_size):
        batch_df = df_filtered.iloc[i:i + batch_size].copy()

        # Chunk the text column
        batch_df.loc[:, 'chunks'] = batch_df['Text'].apply(lambda x: text_splitter.split_text(x))

        # Embed the chunks
        batch_df.loc[:, 'chunk_embeddings'] = batch_df['chunks'].apply(embed_chunks)

        # Prepare documents with metadata
        docs = [Document(page_content=chunk, metadata={"title": title})
                for title, chunks in zip(batch_df['Title'], batch_df['chunks'])
                for chunk in chunks]

        # Insert documents in smaller sub-batches if necessary
        for j in range(0, len(docs), max_chroma_batch_size):
            sub_batch_docs = docs[j:j + max_chroma_batch_size]
            
            # Store the sub-batch in the vector store
            vectorstore = Chroma.from_documents(
                sub_batch_docs,
                embedding=embedding_model,
                persist_directory=persist_directory
            )
            print(f"Processed sub-batch {j // max_chroma_batch_size + 1} of batch {i // batch_size + 1}: {len(sub_batch_docs)} documents added")

        total_docs += len(docs)

    print(f"Total documents processed: {total_docs}")


# Get the data from OpenSearch
df = get_word_count(dbclient)

# Filter the DataFrame
df_filtered = filter_dataframe(df).drop_duplicates().reset_index(drop=True)

# Print stats of data
get_stats(df, df_filtered)

# Run the batch processing
process_in_batches(df_filtered, batch_size=1000, persist_directory="./data/")
