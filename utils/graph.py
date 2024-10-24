# utils/graph.py

from typing import List, Optional
from typing_extensions import TypedDict
from langchain.schema import AIMessage, Document, BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
import re

from .elems import (
    question_rewriter,
    web_search_tool
)

from .optimization import Optimization

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
    generation: str
    web_search: str
    documents: List[Document]
    better_question: str
    author: str
    ask_human: bool

# 저자 추출 함수
def extract_author(question: str) -> Optional[str]:
    """
    질문에서 저자 이름을 추출합니다.
    """
    match = re.search(r'(?P<author>[가-힣]{2,5})\s?(?:작가|저자|의)', question)
    if match:
        return match.group('author')
    return None

# 질문 재작성 함수
def transform_query(state: GraphState) -> dict:
    """
    질문을 재작성하여 더 나은 형태로 변환합니다.

    Args:
        state (GraphState): 현재 그래프 상태

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

# 웹 검색 노드 정의
def web_search_node(state: GraphState) -> dict:
    """
    재작성된 질문을 기반으로 웹 검색을 수행합니다.

    Args:
        state (GraphState): 현재 그래프 상태

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

# 최적화 노드 정의
def optimize(state: GraphState) -> dict:
    """
    생성된 응답을 원하는 톤과 스타일로 최적화합니다.

    Args:
        state (GraphState): 현재 그래프 상태

    Returns:
        dict: 최적화된 응답을 상태에 추가
    """
    print("---OPTIMIZE RESPONSE---")
    better_question = state.get("better_question", "")
    documents = state.get("documents", [])

    if not better_question:
        return {"generation": "죄송하지만, 질문을 이해할 수 없습니다. 다시 한번 말씀해 주세요."}

    # 톤과 스타일을 적용하여 응답 생성
    optimizer = Optimization(
        tone="친절한",
        style="설득력 있는",
        additional_instructions="응답이 친근하고 환영하는 느낌이 들도록 해주세요."
    )
    optimized_response = optimizer.optimize_response(better_question)
    print(f"Optimized Response: {optimized_response}")  # 디버깅 로그

    return {"generation": optimized_response, "documents": documents, "question": better_question}

# 인간 노드 정의
def human_node(state: GraphState) -> dict:
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        # 일반적으로 사용자는 인터럽트 동안 상태를 업데이트했을 것입니다.
        # 그렇지 않은 경우 LLM이 계속 진행할 수 있도록 자리 표시자 ToolMessage를 포함합니다.
        new_messages.append(
            create_response("No response from human.", state["messages"][-1])
        )
    return {
        "messages": new_messages,
        "ask_human": False,
    }
    
    

# 워크플로우 설정
workflow = StateGraph(GraphState)

# 챗봇 노드 정의
workflow.add_node("chatbot", chatbot)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_node", web_search_node)
workflow.add_node("optimize", optimize)
workflow.add_node("human", human_node)
workflow.add_node("tools", ToolNode(tools=[web_search_tool]))

# 다음 노드를 선택하는 함수 정의
def select_next_node(state: GraphState) -> str:
    if state["ask_human"]:
        return "human"
    # 그렇지 않으면 tools 노드로 라우팅
    return "tools"

# 조건부 엣지 추가
workflow.add_conditional_edges(
    "chatbot",
    select_next_node,
    {"human": "human", "tools": "tools", END: "transform_query"},
)

# 엣지 추가
workflow.add_edge("tools", "chatbot")
workflow.add_edge("human", "chatbot")
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", "transform_query")
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "optimize")
workflow.add_edge("optimize", END)

store = {}

def get_session_history(session_id):
    print(f'[대화ID]: {session_id}를 불러옵니다...')
    # 채팅 보관소에 없다면
    if session_id not in store:
        # 새로 만든다
        store[session_id] = ChatMessageHistory()
    
    # 있으면 그대로 주고, 없으면 방금 위에서 만든걸 준다.
    return store[session_id]

def my_runnable(inputs):
    # 여기에 실제로 실행할 코드를 작성합니다
    return {"output": f"Processed question: {inputs['question']}"}

runnable = RunnablePassthrough(my_runnable)

rag_with_history = RunnableWithMessageHistory(
    runnable=runnable,
    get_session_history=get_session_history,
    input_messages_key='question',
    history_messages_key='chat_history'
)
rag_with_history.invoke(
    # 우리가 실제로 물어볼 질문
    {'question':'question'},
    # 이 대화가 어떤 대화기록 id (session_id)에 소속되는지 작성되었는지
    config={'configurable': {'session_id': 'abc'}}
)

compiled_graph = workflow.compile(
    checkpointer=rag_with_history,
    interrupt_before=["human"],
)

# websearch_rag 함수 정의
def websearch_rag(question: str) -> str:
    """
    질문을 받아 최적화된 답변을 생성합니다.

    :param question: 사용자의 질문
    :return: 최적화된 답변
    """
    ans = compiled_graph.invoke({'question': question})
    return ans['generation']