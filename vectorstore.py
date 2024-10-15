from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings

# .env에 PINECONE_API_KEY 설정 필요
from langchain_pinecone import PineconeVectorStore

# Datasource에 따라 1회 진행

# Pinecone 에서 생성한 index 이름
INDEX_NAME = 'bitebook'

# 현재 파일의 위치
CURRENT = Path(__file__).resolve().parent

FILES = [
        'pdfs/타짜.pdf', 
        # 추가할 데이터 소스(pdf) 위치 추가
    ]

for filename in FILES:
    loader = PyMuPDFLoader(CURRENT / filename)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )

    split_docs = text_splitter.split_documents(docs)

    # 사용하려면 .env 에 PINECONE_API_KEY 설정해야함
    PineconeVectorStore.from_documents(
        documents=split_docs,
        embedding=OpenAIEmbeddings(),
        index_name=INDEX_NAME
    )