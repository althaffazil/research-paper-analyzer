from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    # Preservation of page numbers in metadata is automatic here
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True
    )
    return splitter.split_documents(docs)