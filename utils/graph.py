# graph.py

from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START
import re

from .elems import (
    retrieval_grader,
    question_rewriter,
    web_search_tool
)

from .optimization import Optimization

class GraphState(TypedDict):
    """
    그래프의 상태를 나타냅니다.

    속성:
        question: 원본 질문
        generation: 최종 생성된 응답
        web_search: 웹 검색 수행 여부 ('Yes' 또는 'No')
        documents: 검색된 문서 리스트
        better_question: 재작성된 질문
        author: 추출된 저자 이름
    """
    question: str
    generation: str
    web_search: str
    documents: List[Document]
    better_question: str
    author: str

# 저자 추출 함수
def extract_author(question):
    """
    질문에서 저자 이름을 추출합니다.
    """
    # 저자 이름 뒤에 '작가', '저자', '의' 등이 올 수 있도록 패턴 수정
    match = re.search(r'(?P<author>[가-힣]{2,5})\s?(?:작가|저자|의)', question)
    if match:
        return match.group('author')
    return None

# 사용자의 question을 웹 검색하기 적합한 질문으로 변환
def transform_query(state):
    """
    질문을 재작성하여 더 나은 형태로 변환합니다.

    Args:
        state (dict): 현재 그래프 상태

    Returns:
        dict: 재작성된 질문과 추출된 저자를 상태에 추가
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]

    # 질문 재작성
    better_question = question_rewriter.invoke({"question": question})
    print(f"Rewritten Question: {better_question}")  # 디버깅 로그

    # 저자 추출
    author = extract_author(better_question)
    print(f"Extracted Author: {author}")  # 디버깅 로그

    return {
        "better_question": better_question,
        "author": author,
        "question": better_question,
        "documents": state.get("documents", [])
    }

# Tavily API로 질문에 대한 답을 Web에서 검색
def web_search_node(state):
    """
    재작성된 질문을 기반으로 웹 검색을 수행합니다.

    Args:
        state (dict): 현재 그래프 상태

    Returns:
        dict: 웹 검색 결과를 문서 리스트에 추가
    """
    print("---WEB SEARCH NODE---")
    better_question = state.get("better_question", "")
    author = state.get("author", "")
    documents = state.get("documents", [])

    if not author:
        print("Author not found. Skipping web search.")
        return {"documents": documents, "question": better_question}

    # 웹 검색 수행 (Tavily 도구 사용)
    docs = web_search_tool.invoke({"query": better_question})
    print("Docs:", docs)  # 검색 결과 출력

    # 각 검색 결과가 딕셔너리인지 확인하고 Document 생성
    if isinstance(docs, list) and all(isinstance(d, dict) for d in docs):
        new_documents = [Document(page_content=d.get('content', '')) for d in docs]
    elif isinstance(docs, list) and all(isinstance(d, str) for d in docs):
        new_documents = [Document(page_content=d) for d in docs]
    else:
        print("Unexpected docs format.")
        new_documents = []

    # 기존 문서에 웹 검색 결과 추가
    updated_documents = documents + new_documents

    return {"documents": updated_documents, "question": better_question}

# 최적화된 응답을 생성하는 함수
def optimize(state):
    """
    생성된 응답을 원하는 톤과 스타일로 최적화합니다.

    Args:
        state (dict): 현재 그래프 상태

    Returns:
        dict: 최적화된 응답을 상태에 추가
    """
    print("---OPTIMIZE RESPONSE---")
    better_question = state.get("better_question", "")
    author = state.get("author", "")
    documents = state.get("documents", [])

    if not better_question:
        return {"generation": "죄송하지만, 질문을 이해할 수 없습니다. 다시 한번 말씀해 주세요."}

    # Optimization 단계에서 톤앤매너 및 스타일 적용하여 응답 생성
    optimizer = Optimization(
        tone="친절한",
        style="설득력 있는",
        additional_instructions="응답이 친근하고 환영하는 느낌이 들도록 해주세요."
    )
    optimized_response = optimizer.optimize_response(better_question)
    print(f"Optimized Response: {optimized_response}")  # 디버깅 로그

    return {"generation": optimized_response, "documents": documents, "question": better_question}

# 워크플로우 설정
workflow = StateGraph(GraphState)

# Node 연결
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_node", web_search_node)
workflow.add_node("optimize", optimize)

# 그래프 구축
workflow.add_edge(START, "transform_query")
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "optimize")
workflow.add_edge("optimize", END)

# 그래프 컴파일
lg_app = workflow.compile()

def websearch_rag(question):
    """
    질문을 받아 최적화된 답변을 생성합니다.

    :param question: 사용자의 질문
    :return: 최적화된 답변
    """
    ans = lg_app.invoke({'question': question})
    return ans['generation']
