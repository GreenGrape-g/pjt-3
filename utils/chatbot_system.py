from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools.tavily_search import TavilySearchResults

# 환경 변수 로드
load_dotenv()

# Tavily 도구 초기화
tool = TavilySearchResults(max_results=5)
tools = [tool]

# 언어 모델 초기화
llm = ChatOpenAI(model="chatgpt-4o-latest", temperature=1)

# 시스템 메시지 설정
system_message = """당신은 사용자에게 모든 질문에 대해 자연스럽고 친절하게 답변할 수 있는 비서입니다.

**책 추천 시:**
- 사용자의 구체적인 선호를 파악하기 위해 필요한 경우 추가 질문을 합니다.
- 사용자의 요구가 명확할 경우, 추가 질문 없이 사용자가 제공한 장르나 작가를 포함하여 간결하게 책 제목을 나열하여 추천을 제공합니다.
- 특정 책 제목이 포함된 질문에 대해서는 해당 책을 책 제목, 저자, 출판사, 추천 이유 순으로 나열하여 답변합니다.

**예시:**
- 사용자: "너무 한낮의 연애를 읽고 싶어."
- 비서:
    ```
    <br>책 제목: '너무 한낮의 연애'<br>작가: '김금희'<br>출판사: '문학동네'<br>추천 이유: 이 책은 일상 속에서 느낄 수 있는 따뜻한 감정을 섬세하게 그려내어 독자들에게 큰 공감을 불러일으킵니다.
    ```
- 사용자: "김영하 작가의 책 추천해줘."
- 비서:
    ```
    두 권을 추천드립니다. 두 권 중 더 마음에 드는 책을 고르시면 됩니다.<br>

    <br>책 제목: '살인자의 기억법'<br>작가: '김영하'<br>출판사: '문학동네'<br>추천 이유: 이 소설은 독특한 구성과 깊이 있는 캐릭터 분석으로 독자들에게 강렬한 인상을 남깁니다.<br>

    <br>책 제목: '오직 두 사람'<br>작가: '김영하'<br>출판사: '문학동네'<br>추천 이유: 사랑과 인간관계에 대한 섬세한 통찰을 제공하며, 감동적인 이야기가 돋보입니다.
    ```
- 사용자의 질문에 작가 이름이 포함된 경우, 2권을 추천합니다.

**일상적인 질문 시:**
- 친절하고 자연스럽게 답변합니다.

**모호한 질문 시:**
- 친절하고 자연스럽게 간단한 추가 질문을 합니다.

- 사용자: "잠자기 전에 읽을만한 책 추천해줘."
- 비서:
    ```
    어떤 분야의 책을 선호하시나요?
    ```

**응답 형식:**
- 모든 응답은 반드시 한국어로 작성됩니다.
- 책 제목은 작은 따옴표로 감싸주세요.
- enter 대신에 <br>로 구분해주세요.
- \n은 사용하지 않습니다.
- 책 제목, 작가, 출판사, 추천 이유는 각각 새로운 줄에 작성해주세요.
- 여러 권의 책을 추천할 경우, 각 책의 정보를 구분하여 작성해주세요.
- 추천할 수 없는 경우에는 자연스럽게 다른 도움을 제안합니다.
- 줄 바꿈이 발생할 경우 <br>를 붙여주세요.
"""

# 에이전트 초기화
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={
        "system_message": system_message
    }
)

class ChatbotSystem:
    """
    Chatbot 시스템을 초기화하고 메시지를 통해 응답을 생성하는 클래스입니다.
    """
    def __init__(self):
        self.agent_executor = agent_executor

    def generate_response(self, state):
        """
        주어진 상태로부터 응답을 생성하고 상태를 업데이트합니다.

        Args:
            state (dict): 현재 대화 상태를 담은 딕셔너리

        Returns:
            dict: 업데이트된 상태를 반환합니다.
        """
        last_message = state["messages"][-1]["content"]
        # 대화 기록 추출 (마지막 메시지 제외)
        chat_history = [(msg["role"], msg["content"]) for msg in state["messages"][:-1]]
        try:
            # 에이전트를 사용하여 응답 생성
            response = self.agent_executor({
                "input": last_message,
                "chat_history": chat_history
            })['output']
            # 응답 내 줄바꿈을 '<br>'로 변환
            response = response.replace("\n", "<br>")
            # 응답을 메시지 리스트에 추가
            state["messages"].append({
                "role": "assistant",
                "content": response
            })
            # 상태 업데이트
            state["response"] = response
        except Exception as e:
            # 오류 발생 시 기본 메시지 설정
            print(f"Chatbot generation error: {e}")
            error_message = "죄송합니다, 현재 요청을 처리할 수 없습니다. 다시 시도해주세요."
            state["messages"].append({
                "role": "assistant",
                "content": error_message
            })
            state["response"] = error_message
        return state

# 챗봇 인스턴스 생성
chatbot_system = ChatbotSystem()

    
# 챗봇 함수 (그래프에서 사용 가능)
def chatbot(state):
    """
    주어진 상태를 사용하여 챗봇 응답을 생성합니다.

    Args:
        state (dict): 현재 대화 상태를 담은 딕셔너리

    Returns:
        dict: 업데이트된 상태를 반환합니다.
    """
    return chatbot_system.generate_response(state)
