from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain import hub  # 프롬프트 가져올곳 / 직접 생성도 가능
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

structured_llm_grader = llm.with_structured_output(GradeDocuments)

vectorstore = PineconeVectorStore.from_existing_index(
    index_name="bitebook", embedding=OpenAIEmbeddings()
)

rag_system = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:"""

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            rag_system,
        ),
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

system = """당신은 입력된 단어나 구를 해당 주제와 관련된 책을 추천해 달라는 질문으로 변환하는 질문 재작성자입니다.
입력이 이미 질문이라면, 웹 검색에 최적화된 형태로 개선하세요.
결과로 나온 질문은 항상 입력과 관련된 책을 추천하도록 해야 합니다.
입력의 의미를 변경하지 마세요."""

re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "입력: {question}\n\n재작성된 질문:",
        ),
    ]
)

# 아래 항목들이 graph.py 에서 사용하는 요소들
retriever = vectorstore.as_retriever()
rag_chain = rag_prompt | llm | StrOutputParser()
question_rewriter = re_write_prompt | llm | StrOutputParser()
web_search_tool = TavilySearchResults(max_results=3)
retrieval_grader = grade_prompt | structured_llm_grader