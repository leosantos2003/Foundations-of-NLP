from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def explore_abstractions_langchain(pdf_file_path):
    """
    This function demonstrates the use of the Loader and Splitter Longchain abstractinos
    """
    print(f"--- Phase 1: Using a 'Loader' ---\n")
    print(f"Loading the PDF document from: {pdf_file_path}\n")

    # 1. Loader: Instantiates PyPDFLoader(pdf_file_path)
    loader = PyPDFLoader(pdf_file_path)

    # .load() method performz the loading and returns a 'Documents' list of objects
    # Usually, each PDF page becomes a 'Document' object
    pages = loader.load()

    print(f"Document loaded. Number of pages ('Document' objects): {len(pages)}\n")
    # Shows a first page content excerpt for verification
    print(f"First page excerpt: '{pages[0].page_content[:200]}...'\n")

    print(f"--- Phase 2: Using a 'Splitter' ---")

    # 2. Splitter: Instantiates RecursiveCharcterTextSplitter
    #   - chunk_size: defines the maximun size of each chunck (in characters)
    #   - chunk_overlap: defines character overlap between chunks to maintain the context continuity
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    # .split_documents() methods gets a 'Document' list and splits it
    chunks = text_splitter.split_documents(pages)

    print(f"Document split into chunks. Total number of generated chunks: {len(chunks)}\n")
    print(f"Below, the first generated chunk as example:")
    print("-" * 20)
    print(chunks[0].page_content)
    print("-" * 20)

if __name__ == "__main__":
    # 'test_text.pdf" must be in the same directory
    pdf_test_path = "test_text.pdf"
    explore_abstractions_langchain(pdf_test_path)