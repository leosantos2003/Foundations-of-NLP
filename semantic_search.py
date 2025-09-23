from sentence_transformers import SentenceTransformer, util
import torch

# 1. Defines a list of sentences for analysis
sentences = [
    "A UFRGS é referência em Processamento de Linguagem Natural.",
    "Pesquisas em PLN na universidade gaúcha têm destaque.",
    "O chunking semântico é uma técnica avançada em RAG.",
    "Grandes modelos de linguagem precisam de bases de conhecimento externas.",
    "O futebol é um esporte popular no Brasil.",
    "A avaliação de LLMs é um campo de estudo crítico."
]

# 2. Loads a pre-trained embedding model
print("Loading model 'all-MiniLM-L6-v2'...\n")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model successfully loaded.\n")

# 3. Generates the embedding vectors for each sentence
# 'encode' converts text into numeric vectors (tensors)
print("Generating embeddings for sentences...\n")
embeddings = model.encode(sentences, convert_to_tensor=True)

# Prints the format of embeddings for verification
# Each sentence became a 384-dimensional vector
print(f"Embeddings tensor format: {embeddings.shape}")
print("Embeddings gerados.\n")

# 4. Calculates the cosine similarity matrix between all sentences
# 'util.cos_sim' calculates the similarity between all pairs of embeddings
print("Calculating the cosine similarity matrix...\n")
cosine_scores = util.cos_sim(embeddings, embeddings)

# 5. Prints results clearly
print("--- Cosine Similarity Matrix ---\n")
for i in range(len(sentences)):
    for j in range(len(sentences)):
        # Prints the similarity value formatted to 4 decimal places
        print(f"{cosine_scores[i][j]}", end="\t")
    print()

print("\n--- Analysis of Most Similar Pairs ---\n")
# Iterates through all pairs of sentences to find the most similar ones
for i in range(len(sentences)):
    for j in range(len(sentences)):
        score = cosine_scores[i][j]
        # Shows only pairs with high similarity to focus on relevant results
        if score > 0.5:
            print(f"Similarity between '{sentences[i]}' e '{sentences[j]}': {score:.4f}\n")