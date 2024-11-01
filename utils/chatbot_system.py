from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool  # TavilyTool이 존재한다고 가정

load_dotenv()

def tavily_function(input_text):
    # 여기에 tavily와 연결되는 API 호출이나 원하는 기능을 정의합니다.
    # 예를 들어, 특정 API로 요청을 보내거나 특정 작업을 수행하도록 설정
    response = f"tavily 기능으로 처리된 입력: {input_text}"  # 예시 응답
    return response

# Tool 클래스를 사용하여 커스텀 도구 생성
tavily_tool = Tool(
    name="TavilyTool",
    func=tavily_function,
    description="사용자의 요청을 tavily 기능으로 처리하는 도구입니다."
)

class ChatbotSystem:
    """Chatbot 시스템을 초기화하고 메시지를 통해 응답을 생성하는 클래스입니다."""

    def __init__(self):
        # LLM 초기화
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)

        # tavily 도구를 포함한 도구 리스트 초기화
        tools = [tavily_tool]  # tavily_tool을 추가한 도구 리스트

        # 에이전트 초기화
        self.agent_executor = initialize_agent(
            tools=tools,
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
