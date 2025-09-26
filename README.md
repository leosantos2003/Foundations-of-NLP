# Foundations of Natural Language Processing (NLP)

## About

This repository documents the learnings on some of the foundations of Natural Language Processing.

The focuses were on (1) understanding the theory behind semantic search and implement a practical example, (2) mapping and understanding each step of the Retrieval-Augmented Generation (RAG) pipeline and (3) setting up the base of project developments environment and exploring the abstractions of the LangChain framework.

## Semantic Search

Traditional information search works in a relatively simple way, by looking for exact keyword matches. **Semantic search**, on the other hand, searches for meaning. The central concept behind it are the Vector Space Model and the Embeddings.

1. **Vector Space Model**: Approach that represents text as vectors in a multidimensional space, where proximity indicates semantic similarity. The main idea is that texts with similar meanings will be positioned close to each other on this map.

2. **Vector Embeddings**: Numerical representations of text that capture its meaning. Pre-trained language models, such as those in the BERT family, are excellent at generating these embeddings.

3. **Cosine Similarity**: Metric used to measure the angle between two vectors, determining their similarity.

   * If two vectors point in the same direction, the angle between them is 0°, and the cosine of 0° is 1. This means that the texts are very similar.
   * If the vectors are orthogonal (90° angle), the cosine is 0, indicating low or no similarity.
   * If they point in opposite directions (180° angle), the cosine is -1, indicating that they are opposite in meaning.
   * The mathematical formula is $$\text{similarity} = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$, where A and B are the embedding vectors. The programming libraries do this calculation.

## RAG Architecture

## Enviroment and Development

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact

Leonardo Santos - <leorsantos2003@gmail.com>
