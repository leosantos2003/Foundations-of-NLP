# Foundations of Natural Language Processing (NLP)

## About

This repository documents the learnings on some of the foundations of Natural Language Processing.

The focuses were on (1) understanding the theory behind semantic search and implement a practical example, (2) mapping and understanding each step of the Retrieval-Augmented Generation (RAG) pipeline and (3) setting up the base of project development environments and exploring the abstractions of the LangChain framework.

## Semantic Search

Traditional information search works in a relatively simple way, by looking for exact keyword matches. **Semantic Search**, on the other hand, searches for meaning. The central concept behind it are the Vector Space Model and the Embeddings.

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

## Environment and Development

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
    * ```console
      --- Phase 1: Using a 'Loader' ---

      Loading the PDF document from: test_text.pdf

      Document loaded. Number of pages ('Document' objects): 36

      First page excerpt: 'From Wikipedia, the free encyclopedia
      For other uses, see Aristotle (disambiguation).
      Aristotle
      Ἀριστοτέλης
      Roman copy (in marble) of a Greek bronze bust of Aristotle by Lysippos (c. 330 BC), with mod...'

      --- Phase 2: Using a 'Splitter' ---
      Document split into chunks. Total number of generated chunks: 98

      Below, the first generated chunk as example:
      --------------------
      From Wikipedia, the free encyclopedia
      For other uses, see Aristotle (disambiguation).
      Aristotle
      Ἀριστοτέλης
      Roman copy (in marble) of a Greek bronze bust of Aristotle by Lysippos (c. 330 BC), with modern
      alabaster mantle
      Born 384 BC
      Stagira, Chalcidian League
      Died 322 BC (aged 61–62)
      Chalcis, Euboea, Macedonian Empire
      Education
      Education Platonic Academy
      Philosophical work
      Era Ancient Greek philosophy
      Region Western philosophy
      School Peripatetic school
      Notable students Alexander the Great, Theophrastus, Aristoxenus
      Main interests
       Logic
       Natural philosophy
       Metaphysics
       Ethics
       Politics
       Rhetoric
       Poetics
      Notable works  Organon
       Physics
      --------------------
      ```

3. `semantic_search.py`: This script demonstrates the concept of searching by similarity of meaning, rather than just keywords. Converts a list of sentences into numeric vectors (embeddings) and calculates the cosine similarity between all pairs of sentences.
    * Using the `sentence-transformers` library to load an embeddings model (`all-MiniLM-L6-v2`).
    * Generating embedding tensors from text with the `model.encode()` method.
    * Calculation and analysis of a similarity matrix to find the most semantically related sentences.
    * ```console
      Model successfully loaded.

      Generating embeddings for sentences...

      Embeddings tensor format: torch.Size([6, 384])
      Embeddings gerados.

      Calculating the cosine similarity matrix...

      --- Cosine Similarity Matrix ---

      1.0000001192092896      0.28566789627075195     0.48072513937950134     0.704100489616394       0.2552812099456787      0.3252101540565491
      0.28566789627075195     1.0000001192092896      0.41025662422180176     0.32074612379074097     0.42910951375961304     0.48023542761802673
      0.48072513937950134     0.41025662422180176     1.0000001192092896      0.39394882321357727     0.3833175301551819      0.40890002250671387
      0.704100489616394       0.32074612379074097     0.39394882321357727     1.000000238418579       0.28459349274635315     0.261221706867218
      0.2552812099456787      0.42910951375961304     0.3833175301551819      0.28459349274635315     0.9999998807907104      0.5031449794769287
      0.3252101540565491      0.48023542761802673     0.40890002250671387     0.261221706867218       0.5031449794769287      1.0000001192092896

      --- Analysis of Most Similar Pairs ---

      Similarity between 'A UFRGS é referência em Processamento de Linguagem Natural.' e 'A UFRGS é referência em Processamento de Linguagem Natural.': 1.0000

      Similarity between 'A UFRGS é referência em Processamento de Linguagem Natural.' e 'Grandes modelos de linguagem precisam de bases de conhecimento externas.': 0.7041

      Similarity between 'Pesquisas em PLN na universidade gaúcha têm destaque.' e 'Pesquisas em PLN na universidade gaúcha têm destaque.': 1.0000

      Similarity between 'O chunking semântico é uma técnica avançada em RAG.' e 'O chunking semântico é uma técnica avançada em RAG.': 1.0000

      Similarity between 'Grandes modelos de linguagem precisam de bases de conhecimento externas.' e 'A UFRGS é referência em Processamento de Linguagem Natural.': 0.7041

      Similarity between 'Grandes modelos de linguagem precisam de bases de conhecimento externas.' e 'Grandes modelos de linguagem precisam de bases de conhecimento externas.': 1.0000

      Similarity between 'O futebol é um esporte popular no Brasil.' e 'O futebol é um esporte popular no Brasil.': 1.0000

      Similarity between 'O futebol é um esporte popular no Brasil.' e 'A avaliação de LLMs é um campo de estudo crítico.': 0.5031  

      Similarity between 'A avaliação de LLMs é um campo de estudo crítico.' e 'O futebol é um esporte popular no Brasil.': 0.5031  

      Similarity between 'A avaliação de LLMs é um campo de estudo crítico.' e 'A avaliação de LLMs é um campo de estudo crítico.': 1.0000
      ```
      * **Intersting detail about the results**: the similarity between the sentences `"A UFRGS é referência em Processamento de Linguagem Natural."` and `"Pesquisas em PLN na universidade gaúcha têm destaque."`           was very low. That is, firstly, because the `all-MiniLM-L6-v2` model was primarily trained in English, having low understanding capacity of Portuguese. And secondly, the two senteces use very different             words that could represent more distant vectors in the vectorial space, even though they have similar meanings to the human comprehension.

## How to run the scripts

1. Clone the repository:

```bash
git clone https://github.com/leosantos2003/Foundations-of-NLP
```

2. Create and activate a virtual environment:

```bash
# Linux
python3 -m venv .venv
source ven/bin/activate

# Windows
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Run the scripts:

```bash
python load_model.py
python explore_langchain.py
python semantic_search.py
```

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact

Leonardo Santos - <leorsantos2003@gmail.com>
