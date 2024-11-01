from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

load_dotenv()

class ChatbotSystem:
    """Chatbot 시스템을 초기화하고 메시지를 통해 응답을 생성하는 클래스입니다."""

    def __init__(self):
        # LLM 초기화
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)

        # 에이전트 초기화
        self.agent_executor = initialize_agent(
            tools=[],  # 도구가 필요하지 않다면 빈 리스트로 설정
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            agent_kwargs={
                "system_message": "당신은 사용자에게 친절하고 직접적으로 답변하는 비서입니다. "
                                  "책은 1권만 추천하고, 일상 질문이면 친절하게 답변해주고, 모호한 단어면 다시 질문해주세요. "
                                  "**모든 응답은 한국어로 작성합니다.**"
            }
        )

    def generate_response(self, state):
        """주어진 상태로부터 응답을 생성하고 상태를 업데이트합니다."""
        last_message = state["messages"][-1]["content"]

        # 대화 기록 추출 (마지막 메시지 제외)
        chat_history = [(msg["role"], msg["content"]) for msg in state["messages"][:-1]]

        try:
            # 에이전트를 사용하여 응답 생성
            response = self.agent_executor({"input": last_message, "chat_history": chat_history})['output']

            # 응답을 메시지 리스트에 추가
            state["messages"].append({"role": "assistant", "content": response})

            # 상태 업데이트
            state["response"] = response
        except Exception as e:
            # 오류 발생 시 기본 메시지 설정
            print(f"Chatbot generation error: {e}")
            response = "죄송합니다, 현재 요청을 처리할 수 없습니다. 다시 시도해주세요."
            state["messages"].append({"role": "assistant", "content": response})
            state["response"] = response

        return state

# 챗봇 인스턴스 생성
chatbot_system = ChatbotSystem()

# 챗봇 함수 (그래프에서 사용 가능)
def chatbot(state):
    return chatbot_system.generate_response(state)
