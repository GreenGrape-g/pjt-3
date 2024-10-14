import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START
from typing import List
from typing_extensions import TypedDict

# 환경 변수 로드
load_dotenv()

# 그래프 상태 정의
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[str]

# PDF 문서 로드 및 분할
loader = PyMuPDFLoader('path_to_your_pdf.pdf')
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
doc_splits = text_splitter.split_documents(docs)

# 벡터 저장소 및 리트리버 생성
vectorstore = FAISS.from_documents(documents=doc_splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# LLM 및 도구 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
web_search_tool = TavilySearchResults(max_results=3)

# 문서 관련성 평가 모델
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="yes 또는 no로 문서의 관련성을 평가합니다.")

structured_llm_grader = llm.with_structured_output(GradeDocuments)

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "문서가 질문과 의미적으로 관련이 있는지 평가해주세요. 관련 있으면 'yes', 없으면 'no'로 답해주세요."),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
])

retrieval_grader = grade_prompt | structured_llm_grader

# 질문 재작성 프롬프트
re_write_prompt = ChatPromptTemplate.from_messages([
    ("system", "입력된 질문을 웹 검색에 최적화된 형태로 변환하세요."),
    ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
])
question_rewriter = re_write_prompt | llm | StrOutputParser()

# RAG 답변 생성 프롬프트
text = '''
    넌 질문-답변을 도와주는 AI 도서전문가야.
    아래 제공되는 Context를 통해서 사용자 Question에 대해 답을 해줘야해.

    Context에는 직접적으로 없어도, 추론하거나 계산할 수 있는 답변은 최대한 만들어 봐.
    대답할 수 없는 답변의 경우, 웹 페이지 검색을 통해 찾았으면 좋겠어.
    답변을 할 때에는 다음과 같은 구성으로 보여줬으면 좋겠어.
    
    이미지
    제목
    저자
    출판사
    추천 이유
    
    답은 적절히 \n를 통해 문단을 나눠줘 한국어로 만들어 줘. 
    # Question:
    {question}

    # Context:
    {context}
'''
prompt = ChatPromptTemplate.from_template(text)
rag_chain = prompt | llm | StrOutputParser()

# 헬퍼 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 워크플로우 함수
def retrieve(state):
    # 질문에 대한 문서 검색
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    # 검색된 문서를 바탕으로 답변 생성
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": format_docs(documents), "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    # 문서가 질문과 관련이 있는지 평가
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        if score.binary_score == "yes":
            filtered_docs.append(d)
        else:
            web_search = "Yes"
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def transform_query(state):
    # 질문을 재작성하여 더 나은 검색 쿼리 생성
    better_question = question_rewriter.invoke({"question": state["question"]})
    state["question"] = better_question
    return state

def web_search(state):
    # 웹 검색 수행
    question = state["question"]
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results_doc = Document(page_content=web_results)
    state["documents"].append(web_results_doc)
    return state

def decide_to_generate(state):
    # 생성 작업 수행 여부 결정
    if state["web_search"] == "Yes":
        return "transform_query"
    return "generate"

# 워크플로우 그래프 빌드
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_node", web_search)

# 노드 간의 엣지 정의
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_to_generate, {
    "transform_query": "transform_query",
    "generate": "generate"
})
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

# 워크플로우 컴파일 및 실행
app = workflow.compile()
question = '사용자가 질문하고자 하는 내용'
answer = app.invoke({'question': question})
print(answer['generation'])