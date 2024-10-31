from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# LLM 초기화
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 질문 재작성 시스템 프롬프트
rewrite_system = """# 당신은 도서 전문가입니다. 입력 받은 문장에서 책과 저자를 바탕으로 질문 혹은 대답을 재창조합니다.

**모든 응답은 한국어로 작성합니다.** 영어일 경우 한국어로 해석합니다.
구체적인 책이 나올 경우에는 밑에 나오는 **예시**처럼 책 이름과 저자 순으로 말합니다.

예시
Q 한강작가의 흰을 읽고 싶어
A 책 제목 흰, 저자 한강

"""

re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rewrite_system),
        (
            "human",
            "{question}"
        ),
    ]
)

# 질문 재작성 객체 생성
question_rewriter = re_write_prompt | llm | StrOutputParser()

# 웹 검색 도구 초기화
web_search_tool = TavilySearchResults(
    max_results=5,
    lang='ko',        # 언어 설정 (예: 'ko' for Korean)
    mkt='ko-KR'       # 시장 설정 (예: 'ko-KR' for South Korea)
)

