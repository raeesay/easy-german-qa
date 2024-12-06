from opensearchpy import OpenSearch
from bert_score import score
import sacrebleu

# Connect to the OpenSearch instance with the correct headers
client = OpenSearch(
    hosts=[{'host': 'opensearch-ds.ifi.uni-heidelberg.de', 'port': 443}],
    http_auth=('ryousaf', 'i5am3SHER4locked'),  # Replace with your credentials
    use_ssl=True,
    verify_certs=False,  # Set to True if you have valid certificates
    ssl_show_warn=False
)

def search_title(index, title):
    # Define the search query
    query = {
        "query": {
            "match": {
                "title": title
            }
        }
    }

    hits = []

    try:
        # Execute the search query
        response = client.search(index=index, body=query)
        
        # Check if we got any hits
        if response['hits']['total']['value'] > 0:
            for hit in response['hits']['hits']:
                hits.append(hit['_source']['text'])  # Save each hit to the list
                print(f"Title: {hit['_source']['title']}")
                print(f"Text: {hit['_source']['text']}")
                print(f"URL: {hit['_source']['url']}")
                print("-" * 80)
        else:
            print("No results found")
    except Exception as e:
        print(f"Error executing search query: {e}")

    return hits

# BERTScore
def calculate_bertscore(refs, hyps):
    P, R, F1 = score(refs, hyps, lang='de')
    return P.mean().item(), R.mean().item(), F1.mean().item()

# BLEU Score
def calculate_bleu(refs, hyps):
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return bleu.score

# Call the function and save the results to a variable
simple_hits = search_title("kic_apothekenumschau_simple_speech_articles", "Tilidin")
hits = search_title("kic_apothekenumschau_articles", "Tilidin")

for i, hit in enumerate(simple_hits, start=1):
    globals()[f"simple_hit{i}"] = hit
    print(f"Hit {i} saved to variable 'simple_hit{i}'")

for i, hit in enumerate(hits, start=1):
    globals()[f"hit{i}"] = hit
    print(f"Hit {i} saved to variable 'hit{i}'")

# calculate the BERT score
article = [hit2]
simple_article = [simple_hit1]

# calculate the BERT score
P, R, F1 = score(article, simple_article, lang='de')

print(f'Precision: {P.mean().item():.4f}')
print(f'Recall: {R.mean().item():.4f}')
print(f'F1 Score: {F1.mean().item():.4f}')

# calculate BLEU score
bleu = calculate_bleu(article, simple_article)
print(f'BLEU Score: {bleu:.4f}')
print()