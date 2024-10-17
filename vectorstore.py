import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

index_name = os.environ.get('INDEX_NAME')
# 1. Load
loader = PyMuPDFLoader('./vectorstore자료.pdf')
docs = loader.load()
# 2. Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)
# 3. Embed Setting
embeddings = OpenAIEmbeddings()
# 4. Add Data To Index
PineconeVectorStore.from_documents(
    documents=split_docs,
    embedding=embeddings,
    index_name=index_name,
)
