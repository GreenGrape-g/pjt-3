# utils/chatbot_system.py

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType


# LLM 초기화
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 에이전트 생성
agent_executor = initialize_agent(
    tools=[],  # 도구가 필요하지 않다면 빈 리스트로 설정
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={
        "system_message": "당신은 사용자에게 친절하고 직접적으로 답변하는 비서입니다. 책 관련 질문이면 그 질문을 따라합니다. 일상 질문이면 친절하게 답변해주고, 모호한 단어면 다시 질문해주세요."
    }
)

# 챗봇 로직
def chatbot(state):
    """그래프에서 사용 가능한 챗봇 함수."""
    last_message = state["messages"][-1]["content"]

    # 대화 기록 추출
    chat_history = [(msg["role"], msg["content"]) for msg in state["messages"][:-1]]

    # 에이전트를 사용하여 응답 생성
    response = agent_executor({"input": last_message, "chat_history": chat_history})['output']

    # 응답을 메시지 리스트에 추가
    state["messages"].append({"role": "assistant", "content": response})

    # 상태 업데이트
    state['response'] = response

    return state
