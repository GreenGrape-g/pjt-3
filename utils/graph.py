from typing import List, Dict
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START

from .custom_types import State
from .chatbot_system import chatbot
from .judgement import is_about_books, decide_next_node
from .elems import question_rewriter, web_search_tool
from .optimization import Optimization
from .nlp_utils import extract_top_korean_words

class GraphState(TypedDict):
    """Graph의 상태를 나타냅니다."""
    question: str
    response: str
    is_book_question: bool
    generation: str
    documents: List[Dict]
    better_question: str
    top_words: List[tuple]
    messages: List[Dict]

def judgement_node(state: GraphState) -> GraphState:
    """챗봇의 응답을 기반으로 책 질문인지 판단합니다."""
    response = state["response"]
    is_book_question = is_about_books(response)
    state["is_book_question"] = is_book_question
    return state

def transform_query_node(state: GraphState) -> GraphState:
    """질문을 재작성하여 더 나은 형태로 변환합니다."""
    print("---TRANSFORM QUERY---")
    question = state["question"]

    # 질문 재작성
    better_question = question_rewriter.invoke({"question": question})
    print(f"Rewritten Question: {better_question}")

    state["better_question"] = better_question
    state["question"] = better_question
    return state

def web_search_node(state: GraphState) -> GraphState:
    """재작성된 질문을 기반으로 웹 검색을 수행합니다."""
    print("---WEB SEARCH NODE---")
    better_question = state.get("better_question", "")
    documents = state.get("documents", [])

    # 웹 검색 수행
    docs = web_search_tool.invoke(better_question)
    print("Docs:", docs)

    # 검색 결과 처리
    for doc in docs:
        documents.append(doc)
    
    state["documents"] = documents
    return state

def extract_top_words_node(state: GraphState) -> GraphState:
    """검색된 문서에서 상위 단어를 추출합니다."""
    print("---EXTRACT TOP WORDS---")
    text = " ".join([doc["content"] for doc in state["documents"]])
    top_words = extract_top_korean_words(text, n=5)
    state["top_words"] = top_words
    print(f"Top Words: {top_words}")
    return state

def optimize_node(state: GraphState) -> GraphState:
    """생성된 응답을 원하는 톤과 스타일로 최적화합니다."""
    print("---OPTIMIZE RESPONSE---")
    better_question = state.get("better_question", "")
    top_words = state.get("top_words", [])

    if not better_question:
        state["generation"] = "죄송하지만, 질문을 이해할 수 없습니다. 다시 한번 말씀해 주세요."
        return state

    # 상위 단어를 추가적인 컨텍스트로 활용
    top_words_str = ", ".join([word for word, count in top_words])

    # 응답 최적화
    optimizer = Optimization(
        tone="친절한",
        style="설득력 있는",
        num_books=1,  # 항상 1권 추천
        additional_instructions=f"상위 단어: {top_words_str}. 응답이 친근하고 환영하는 느낌이 들도록 해주세요.",
        conversation_history=state.get('messages', [])
    )
    optimized_response = optimizer.optimize_response(better_question)
    print(f"Optimized Response: {optimized_response}")

    state["generation"] = optimized_response
    return state

def graph_main(state: State) -> Dict:
    """그래프를 실행하여 최종 응답을 생성합니다."""
    # 그래프 상태 초기화
    graph_state = {
        "question": state["messages"][-1]["content"],
        "response": "",
        "is_book_question": False,
        "generation": "",
        "documents": [],
        "better_question": "",
        "top_words": [],
        "messages": state["messages"]
    }

    workflow = StateGraph(GraphState)

    # 노드 정의
    workflow.add_node("chatbot", chatbot)
    workflow.add_node("judgement", judgement_node)
    workflow.add_node("transform_query", transform_query_node)
    workflow.add_node("web_search_node", web_search_node)
    workflow.add_node("extract_top_words", extract_top_words_node)
    workflow.add_node("optimize", optimize_node)

    # 에지 정의
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", "judgement")

    # 조건부 엣지 추가
    workflow.add_conditional_edges(
        "judgement",
        decide_next_node, 
        {
            "transform_query": "transform_query",
            "end": END,
        },
    )

    # transform_query 이후의 노드 연결
    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "extract_top_words")
    workflow.add_edge("extract_top_words", "optimize")
    workflow.add_edge("optimize", END)

    # 그래프 컴파일
    lg_app = workflow.compile()

    # 그래프 실행
    ans = lg_app.invoke(graph_state)

    # 최종 응답 결정
    # 책 관련 질문이면 'generation'에 최종 응답이 저장되고,
    # 일반 질문이면 'response'에 챗봇의 응답이 저장됩니다.
    final_response = ans.get('generation') or ans.get('response', '죄송하지만, 답변을 생성할 수 없습니다.')

    # 응답 반환
    return {'generation': final_response}

