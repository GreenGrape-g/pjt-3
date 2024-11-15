from typing import List, Dict
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from .custom_types import State
from .chatbot_system import chatbot
from .judgement import decide_next_node, is_about_author, is_about_books, is_about_negative
from .optimization import Optimization

# GraphState 클래스 정의
class GraphState(TypedDict):
    # 사용자 대화 내역
    messages: List[Dict[str, str]]
    question: str
    response: str
    generation: str
    is_author_question: bool
    is_book_question: bool
    is_negative: bool
    documents: List[str]
    
def judgement_node(state: GraphState) -> GraphState:
    """챗봇의 응답을 기반으로 책 질문인지, 작가 질문인지 및 부정적인 단어 포함 여부를 판단합니다."""
    print("---JUDGEMENT NODE---")
    response = state["response"]
    state["is_negative"] = is_about_negative(response)
    state["is_book_question"] = is_about_books(response)
    state["is_author_question"] = is_about_author(response)
    return state

def optimize_node(state: GraphState) -> GraphState:
    """생성된 응답을 원하는 톤과 스타일로 최적화합니다."""
    print("---OPTIMIZE RESPONSE---")
    try:
        initial_response = state.get("response", "")
        num_books = 2 if state.get("is_author_question", False) else 1
        optimizer = Optimization(
            tone="친절한",
            style="설득력 있는",
            additional_instructions="응답이 친근하고 환영하는 느낌이 들도록 해주세요.",
            conversation_history=state.get("messages", []),
        )
        state["generation"] = optimizer.optimize_response(initial_response, num_books=num_books)
    except Exception as e:
        print(f"Optimization failed: {e}")
        state["generation"] = "죄송하지만, 응답을 최적화할 수 없습니다."
    return state

def graph_main(state: State) -> Dict:
    """그래프를 실행하여 최종 응답을 생성합니다."""
    # 초기 그래프 상태 설정
    class State(TypedDict):
    # 사용자 대화 내역
        messages: List[Dict[str, str]]
        question: str
        response: str
        generation: str
        is_author_question: bool
        is_book_question: bool
        is_negative: bool
        documents: List[str]
    
    # 그래프 정의 및 실행
    workflow = StateGraph(GraphState)
    workflow.add_node("chatbot", chatbot)
    workflow.add_node("judgement", judgement_node)
    workflow.add_node("optimize", optimize_node)
    # 그래프 연결 설정
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", "judgement")
    workflow.add_conditional_edges(
        "judgement",
        decide_next_node, 
        {"optimize": "optimize", "end": END},
    )
    workflow.add_edge("optimize", END)
    # 그래프 컴파일 및 실행
    lg_app = workflow.compile()
    ans = lg_app.invoke(state)
    final_response = ans.get("generation") or ans.get("response", "죄송하지만, 답변을 생성할 수 없습니다.")
    return {"generation": final_response}
