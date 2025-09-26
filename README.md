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

**Retrieval-Augmented Generation (RAG)** is a technique that enhances Large Language Models (LLMs) by connecting them to external, up-to-date knowledge sources. This allows LLMs to generate more accurate, detailed, and less "hallucinatory" responses because they don't rely solely on the data they were originally trained on.

1. **Ingestion (Load)**: Loading raw data (e.g., PDFs).

2. **Chunking (Split)**: Dividing long documents into smaller pieces called "chunks".

3. **Embedding**: Converting each chunk of text into a numeric vector called "embedding".

4. **Indexing (Store)**: Storage of embeddings in a vector database optimized for fast searches.

5. **Retrieve**: Searches the vector database for the most relevant chunks for a given question.

6. **Generate**: Sending the original question and the retrieved context to an LLM to formulate the answer.

## Enviroment and Development

To solidify the concepts learned through practical activities, the following Python scripts were created.

1. `load_model.py`: This script focuses on the first steps to working with Hugging Face models. Download the pre-trained model `pierreguillou/bert-base-cased-squad-v1.1-portuguese`, which specializes in Question and Answer (QA) tasks in Portuguese.
    * Uses `AutoModelForQuestionAnswering` to load the model.
    * Using `AutoTokenizer` to load the corresponding tokenizer.
    * A test function (`test_tokenizer`) demonstrates how text is converted to numeric IDs so the model can process it.
    * ```console
      Starting tokenizer download for the model: 'pierreguillou/bert-base-cased-squad-v1.1-portuguese'...

      Tokenizer successfully loaded.

      Starting model download: 'pierreguillou/bert-base-cased-squad-v1.1-portuguese'...

      Model successfully loaded.

      Environment set and loading test finished.

      --- Testing the Tokenizer ---

      Original sentence: 'A UFRGS fica em Porto Alegre.'

      Token IDs (input_ids): [101, 177, 9549, 22322, 17146, 1968, 173, 2268, 4844, 119, 102]

      These are the numberes that the model really sees.
      ```

2. `explore_langchain.py`: This script introduces two of the main abstractions of the LangChain library, essential for building RAG pipelines. Loads a PDF document and splits it into smaller, overlapping pieces.
    * Loader: Uses PyPDFLoader to load the pages of a PDF file as Document objects.
    * Splitter: Uses the `RecursiveCharacterTextSplitter` to break documents into chunks of a defined size, maintaining an overlap to avoid losing context.
    * ```bash
      rgstdhf
      ```

3. `semantic_search.py`: This script demonstrates the concept of searching by similarity of meaning, rather than just keywords. Converts a list of sentences into numeric vectors (embeddings) and calculates the cosine similarity between all pairs of sentences.
    * Using the `sentence-transformers` library to load an embeddings model (`all-MiniLM-L6-v2`).
    * Generating embedding tensors from text with the `model.encode()` method.
    * Calculation and analysis of a similarity matrix to find the most semantically related sentences.

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact

Leonardo Santos - <leorsantos2003@gmail.com>
