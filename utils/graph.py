from typing import List, Dict
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from .custom_types import State
from langchain.schema import Document
from .chatbot_system import chatbot
from .judgement import decide_next_node, is_about_books, is_about_author, is_about_negative
from .optimization import Optimization
from .elems import web_search_tool

from .judgement import chatbot, create_response, human_node  # 'chatbot' 함수만 임포트 (순환 참조 방지)

# ToolMessage 클래스 정의
class ToolMessage(BaseMessage):
    def __init__(self, content: str, tool_call_id: Optional[str] = None):
        super().__init__(content=content)
        self.tool_call_id = tool_call_id

# create_response 함수 정의
def create_response(response: str, ai_message: AIMessage) -> ToolMessage:
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"],
    )

# GraphState 클래스 정의
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
        ask_human: 인간에게 도움을 요청하는 플래그
    """
    question: str
    response: str
    documents: List[Dict]
    generation: str
    messages: List[Dict]
    is_author_question: bool  # 작가 질문 여부
    is_book_question: bool     # 책 질문 여부
    is_negative: bool          # 부정적인 단어 포함 여부

def web_search_node(state: GraphState) -> GraphState:
    """웹 검색을 수행하고 결과를 문서로 추가합니다."""
    print("---WEB SEARCH---")
    try:
        question = state["response"]
        docs = web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        state["documents"].append(Document(page_content=web_results))
    except Exception as e:
        print(f"Web search failed: {e}")
        state["documents"].append(Document(page_content="검색 결과를 찾을 수 없습니다."))
    return state

def web_search_node_author(state: GraphState) -> GraphState:
    """작가 관련 웹 검색을 수행하고 결과를 문서로 추가합니다."""
    print("---WEB SEARCH AUTHOR---")
    try:
        question = state["response"]
        docs = web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        state["documents"].append(Document(page_content=web_results))
    except Exception as e:
        print(f"Web search failed: {e}")
        state["documents"].append(Document(page_content="검색 결과를 찾을 수 없습니다."))
    return state

def judgement_node(state: GraphState) -> GraphState:
    """챗봇의 응답을 기반으로 책 질문인지, 작가 질문인지 및 부정적인 단어 포함 여부를 판단합니다."""
    print("---JUDGEMENT NODE---")
    response = state["response"]
    state["is_book_question"] = is_about_books(response)
    state["is_author_question"] = is_about_author(response)
    state["is_negative"] = is_about_negative(response)
    return state

def optimize_node(state: GraphState) -> GraphState:
    """생성된 응답을 원하는 톤과 스타일로 최적화합니다."""
    print("---OPTIMIZE RESPONSE---")
    try:
        better_question = state.get("response", "")
        if state.get("is_author_question", False):
            num_books = 2
        else:
            num_books = 1
        optimizer = Optimization(
            tone="친절한",
            style="설득력 있는",
            additional_instructions="응답이 친근하고 환영하는 느낌이 들도록 해주세요.",
            conversation_history=state.get("messages", []),
        )
        state["generation"] = optimizer.optimize_response(better_question, num_books=num_books)
    except Exception as e:
        print(f"Optimization failed: {e}")
        state["generation"] = "죄송하지만, 응답을 최적화할 수 없습니다."
    return state

def graph_main(state: State) -> Dict:
    """그래프를 실행하여 최종 응답을 생성합니다."""
    # 초기 그래프 상태 설정
    graph_state: GraphState = {
        "question": state["messages"][-1]["content"],
        "response": "",  # 챗봇 노드에서 설정될 응답
        "documents": [],
        "generation": "",
        "messages": state["messages"],
        "is_author_question": False,
        "is_book_question": False,
        "is_negative": False
    }
    # 그래프 정의 및 실행
    workflow = StateGraph(GraphState)
    workflow.add_node("chatbot", chatbot)
    workflow.add_node("judgement", judgement_node)
    workflow.add_node("optimize", optimize_node)
    workflow.add_node("web_search_node", web_search_node)
    workflow.add_node("web_search_node_author", web_search_node_author)
    # 그래프 연결 설정
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", "judgement")
    workflow.add_conditional_edges(
        "judgement",
        decide_next_node,
        {
            "web_search_node_author": "web_search_node_author",
            "web_search_node": "web_search_node",
            "end": END
        },
    )
    workflow.add_edge("web_search_node_author", "optimize")
    workflow.add_edge("web_search_node", "optimize")
    workflow.add_edge("optimize", END)
    lg_app = workflow.compile()
    ans = lg_app.invoke(graph_state)
    final_response = ans.get("generation") or ans.get("response", "죄송하지만, 답변을 생성할 수 없습니다.")
    return {"generation": final_response}
