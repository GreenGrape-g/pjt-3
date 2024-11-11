from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# LLM 초기화
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 질문 재작성 시스템 프롬프트
rewrite_system = """# 당신은 도서 전문가입니다. 입력 받은 ai의 답변에서 책과 저자를 바탕으로 질문 혹은 답변을 재창조합니다.

**모든 응답은 한국어로 작성합니다.** 영어일 경우 한국어로 해석합니다.

구체적인 책이 나올 경우에는 밑에 나오는 **예시**처럼 책 이름과 저자 순으로 답변합니다.

**예시**

question: 가을에 어울리는 로맨스 책 추천해줘

response: 가을에 어울리는 로맨스 책으로는 '노르웨이의 숲'을 추천드립니다. 하루키 무라카미의 작품으로, 청춘의 아픔과 사랑을 그린 작품입니다.

better_question : 노르웨이 숲, 무라카미 하루키

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
    mkt='ko-KR'        # 시장 설정 (예: 'ko-KR' for South Korea)
)
