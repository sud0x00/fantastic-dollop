# Install sentence transformers with !pip install sentence-transformers

from sentence_transformers import SentenceTransformer, util
import pandas as pd

df = pd.read_csv("youtube-transcripts.csv")
print(len(df)) 

embedder = SentenceTransformer('all-MiniLM-L6-v2') # https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models/

# Concatenate titles and description to create the corpus
corpus = df["title"] + ". " + df["desc"]


# Create embeddings
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

query = 'Half life 3 release date'
top_k = 10


# Find the closest top_k sentences of the corpus based on cosine similarity
query_embedding = embedder.encode(query, convert_to_tensor=True)
hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
hits = hits[0] # Get the hits for the first query

print(f"\nTop {top_k} most similar sentences in corpus:")
for hit in hits:
    hit_id = hit['corpus_id']
    video_data = df.iloc[hit_id]
    title = video_data["title"]
    print("-", title, "(Score: {:.4f})".format(hit['score']))
