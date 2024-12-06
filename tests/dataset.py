from opensearchpy import OpenSearch
import pandas as pd

# Connect to the OpenSearch instance
client = OpenSearch(
    hosts=[{'host': 'opensearch-ds.ifi.uni-heidelberg.de', 'port': 443}],
    http_auth=('ryousaf', 'i5am3SHER4locked'),  # Replace with your credentials
    use_ssl=True,
    verify_certs=False,  # Set to True if you have valid certificates
    ssl_show_warn=False
)

# Function to get word count from OpenSearch index
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

# Filter DataFrame based on word count and specific phrases
def filter_dataframe(df, min_word_count=39, phrase="Medikamente eines Wirkstoffs Wirkstoffe von A bis Z Medikamente mit dem Bestandteil"):
    df_filtered = df[df['Word Count'] >= min_word_count]
    df_filtered = df_filtered[~df_filtered['Text'].str.contains(phrase, na=False)]
    return df_filtered

# Additional filtering to remove rows with "Beiträge von"
def remove_biographical_entries(df):
    phrase = "Beiträge von"
    initial_count = len(df)
    df_filtered = df[~df['Text'].str.contains(phrase, na=False)]
    deleted_count = initial_count - len(df_filtered)
    return df_filtered, deleted_count

# Get the data from OpenSearch
df = get_word_count(client)

# Filter the DataFrame
df = df.sort_values(by='Word Count')
filtered_df = filter_dataframe(df)
filtered_df, deleted_count = remove_biographical_entries(filtered_df)

# Remove duplicates
df_filtered_no_duplicates = filtered_df.drop_duplicates()

# Reset index
df_final = df_filtered_no_duplicates.reset_index(drop=True)

# Save the final DataFrame to Excel
df_final.to_excel("filtered_no_duplicates.xlsx", index=False)

# Print final statistics
print("Number of rows after filtering and removing duplicates: ", len(df_final))
print("Mean word count: ", df_final['Word Count'].mean())
print("Maximum word count: ", df_final['Word Count'].max())
print("Minimum word count: ", df_final['Word Count'].min())
print("Median word count: ", df_final['Word Count'].median())

# Print the number of "Beiträge von" rows deleted
print(f"Number of rows with 'Beiträge von' deleted: {deleted_count}")