import re
from kiwipiepy import Kiwi
from collections import Counter
from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START

from .elems import (
    question_rewriter,
    web_search_tool
)

from .optimization import Optimization

class GraphState(TypedDict):
    """그래프의 상태를 나타냅니다.

    속성:
        question: 원본 질문
        generation: 최종 생성된 응답
        documents: 검색된 문서 리스트
        better_question: 재작성된 질문
        top_words: 추출된 상위 단어 리스트
    """
    question: str
    generation: str
    documents: List[Document]
    better_question: str
    top_words: List[tuple]

def preprocess_korean_text(text):
    """
    한국어 텍스트를 전처리하고 명사를 추출합니다.
    
    Args:
        text (str): 입력 텍스트.
        
    Returns:
        list: 추출된 명사의 리스트.
    """
    # 특수 문자, 숫자, 공백 제거
    text = re.sub(r'[^가-힣\s]', '', text)
    
    # 형태소 분석기 초기화
    kiwi = Kiwi()
    
    # 명사 추출
    tokens = kiwi.tokenize(text)  # pos_tagger 인수 제거
    
    # 토큰 구조 확인 (디버깅용)
    print("Sample Tokens:", tokens[:5])  # 첫 5개 토큰 출력
    
    # 명사 추출
    try:
        nouns = [word for word, pos in tokens if pos.startswith('N')]
    except ValueError:
        # 만약 토큰이 두 개 이상일 경우
        nouns = [word for token in tokens if token[1].startswith('N') for word in [token[0]]]
    
    return nouns

def get_most_common_words(nouns, n=5):
    """
    명사 리스트에서 가장 빈도가 높은 단어 n개를 추출합니다.
    
    Args:
        nouns (list): 명사의 리스트.
        n (int): 추출할 단어의 개수.
        
    Returns:
        list of tuples: (단어, 빈도) 형태의 리스트.
    """
    count = Counter(nouns)
    top_n = count.most_common(n)
    return top_n

def extract_top_korean_words(text, n=5):
    """
    주어진 한국어 텍스트에서 가장 빈도가 높은 단어 n개를 추출합니다.
    
    Args:
        text (str): 입력 텍스트.
        n (int): 추출할 단어의 개수. 기본값은 5.
        
    Returns:
        list of tuples: (단어, 빈도) 형태의 리스트.
    """
    # 텍스트 전처리 및 형태소 분석
    nouns = preprocess_korean_text(text)
    
    # 가장 빈도가 높은 단어 추출
    top_words = get_most_common_words(nouns, n)
    
    return top_words

def transform_query(state):
    """질문을 재작성하여 더 나은 형태로 변환합니다."""
    print("---TRANSFORM QUERY---")
    question = state["question"]

    # 질문 재작성
    better_question = question_rewriter.invoke({"question": question})
    print(f"Rewritten Question: {better_question}")

    return {
        "better_question": better_question,
        "question": better_question,
        "documents": state.get("documents", [])
    }

def web_search_node(state):
    """재작성된 질문을 기반으로 웹 검색을 수행합니다."""
    print("---WEB SEARCH NODE---")
    better_question = state.get("better_question", "")
    documents = state.get("documents", [])

    # 웹 검색 수행
    docs = web_search_tool.invoke({"query": better_question})
    print("Docs:", docs)

    # 검색 결과 처리
    if isinstance(docs, list) and all(isinstance(d, dict) for d in docs):
        new_documents = [Document(page_content=d.get('content', '')) for d in docs]
    elif isinstance(docs, list) and all(isinstance(d, str) for d in docs):
        new_documents = [Document(page_content=d) for d in docs]
    else:
        print("Unexpected docs format.")
        new_documents = []

    updated_documents = documents + new_documents

    return {"documents": updated_documents, "question": better_question}

def extract_top_words(state):
    """검색된 문서들에서 빈도가 높은 단어를 추출합니다."""
    print("---EXTRACT TOP WORDS---")
    documents = state.get("documents", [])
    
    if not documents:
        print("No documents to process for top words.")
        return {"top_words": []}
    
    # 모든 문서의 내용을 하나의 텍스트로 결합
    combined_text = " ".join([doc.page_content for doc in documents])
    
    # 상위 5개 단어 추출
    top_words = extract_top_korean_words(combined_text, n=5)
    print(f"Top Words: {top_words}")
    
    return {"top_words": top_words}

def optimize(state):
    """생성된 응답을 원하는 톤과 스타일로 최적화합니다."""
    print("---OPTIMIZE RESPONSE---")
    better_question = state.get("better_question", "")
    documents = state.get("documents", [])
    top_words = state.get("top_words", [])
    
    if not better_question:
        return {"generation": "죄송하지만, 질문을 이해할 수 없습니다. 다시 한번 말씀해 주세요."}
    
    # 상위 단어를 추가적인 컨텍스트로 활용
    top_words_str = ", ".join([word for word, count in top_words])
    
    # 응답 최적화
    optimizer = Optimization(
        tone="친절한",
        style="설득력 있는",
        additional_instructions=f"상위 단어: {top_words_str}. 응답이 친근하고 환영하는 느낌이 들도록 해주세요."
    )
    optimized_response = optimizer.optimize_response(better_question)
    print(f"Optimized Response: {optimized_response}")
    
    return {"generation": optimized_response, "documents": documents, "question": better_question}

# 워크플로우 설정
workflow = StateGraph(GraphState)

# Node 연결
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_node", web_search_node)
workflow.add_node("extract_top_words", extract_top_words)  # 새로운 노드 추가
workflow.add_node("optimize", optimize)

# 그래프 구축
workflow.add_edge(START, "transform_query")
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "extract_top_words")  # 새로운 연결 추가
workflow.add_edge("extract_top_words", "optimize")        # 새로운 연결 추가
workflow.add_edge("optimize", END)

# 그래프 컴파일
lg_app = workflow.compile()

def websearch_rag(question):
    """질문을 받아 최적화된 답변을 생성합니다."""
    ans = lg_app.invoke({'question': question})
    return ans['generation']

# 테스트 예시
if __name__ == "__main__":
    test_question = "한국어 문서에서 가장 많이 나오는 단어를 추출하여 답변을 최적화하는 방법을 알고 싶어요."
    optimized_answer = websearch_rag(test_question)
    print("Optimized Answer:", optimized_answer)
