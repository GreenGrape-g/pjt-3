from typing import List, Dict
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langchain.schema import Document
from .custom_types import State
from .chatbot_system import chatbot
from .judgement import is_about_books, decide_next_node, is_about_negative
from .elems import question_rewriter, web_search_tool
from .optimization import Optimization

class GraphState(TypedDict):
    """Graph의 상태를 나타냅니다."""
    question: str
    response: str
    is_book_question: bool
    is_negative: bool
    generation: str
    documents: List[Dict]
    better_question: str
    top_words: List[tuple]
    messages: List[Dict]

def judgement_node(state: GraphState) -> GraphState:
    """챗봇의 응답을 기반으로 책 질문인지 및 부정적인 단어 포함 여부를 판단합니다."""
    response = state["response"]
    state["is_book_question"] = is_about_books(response)
    state["is_negative"] = is_about_negative(response)
    return state

def transform_query_node(state: GraphState) -> GraphState:
    """질문을 재작성하여 더 나은 형태로 변환합니다."""
    print("---TRANSFORM QUERY---")
    question = state["response"]
    state["better_question"] = question_rewriter.invoke({"question": question})
    return state

def web_search_node(state: GraphState) -> GraphState:
    """웹 검색을 수행하고 결과를 문서로 추가합니다."""
    print("---WEB SEARCH---")
    try:
        question = state["better_question"]
        docs = web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        state["documents"].append(Document(page_content=web_results))
    except Exception as e:
        print(f"Web search failed: {e}")
        state["documents"].append(Document(page_content="검색 결과를 찾을 수 없습니다."))
    return state

def optimize_node(state: GraphState) -> GraphState:
    """생성된 응답을 원하는 톤과 스타일로 최적화합니다."""
    print("---OPTIMIZE RESPONSE---")
    try:
        better_question = state.get("better_question", "")

        optimizer = Optimization(
            tone="친절한",
            style="설득력 있는",
            num_books=1,
            additional_instructions=f"응답이 친근하고 환영하는 느낌이 들도록 해주세요.",
            conversation_history=state.get("messages", []),
        )
        state["generation"] = optimizer.optimize_response(better_question)
    except Exception as e:
        print(f"Optimization failed: {e}")
        state["generation"] = "죄송하지만, 응답을 최적화할 수 없습니다."
    return state

def graph_main(state: State) -> Dict:
    """그래프를 실행하여 최종 응답을 생성합니다."""
    # 초기 그래프 상태 설정
    graph_state: GraphState = {
        "question": state["messages"][-1]["content"],
        "response": "",
        "is_book_question": False,
        "is_negative": False,
        "generation": "",
        "documents": [],
        "better_question": "",
        "messages": state["messages"],
    }

    # 그래프 정의 및 실행
    workflow = StateGraph(GraphState)
    workflow.add_node("chatbot", chatbot)
    workflow.add_node("judgement", judgement_node)
    workflow.add_node("transform_query", transform_query_node)
    workflow.add_node("web_search_node", web_search_node)
    workflow.add_node("optimize", optimize_node)

    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", "judgement")
    workflow.add_conditional_edges(
        "judgement",
        decide_next_node, 
        {"transform_query": "transform_query", "end": END},
    )
    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "optimize")
    workflow.add_edge("optimize", END)

    lg_app = workflow.compile()
    ans = lg_app.invoke(graph_state)

    final_response = ans.get("generation") or ans.get("response", "죄송하지만, 답변을 생성할 수 없습니다.")
    return {"generation": final_response}
