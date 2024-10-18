# elems.py

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field

load_dotenv()

class GradeDocuments(BaseModel):
    """문서의 관련성을 이진 점수로 평가."""
    binary_score: str = Field(
        description="문서가 질문과 관련 있는지 여부를 'yes' 또는 'no'로 표시."
    )

# 문서 관련성 평가를 위한 시스템 프롬프트
grade_system = """당신은 사용자의 질문에 대한 문서의 관련성을 평가하는 채점자입니다.
문서가 질문과 관련된 키워드나 의미를 포함하고 있다면 'yes', 그렇지 않다면 'no'로 평가하세요."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", grade_system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# LLM 초기화 및 구조화된 출력 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# RAG 시스템 프롬프트
rag_system = """당신은 사용자의 질문에 기반하여 관련 정보를 검색하고 제공합니다.
사용 가능한 문서를 활용하여 사용자의 질문에 간결하고 정확하게 책 제목과 저자 정보를 5개 정도 제공합니다.
중복되는 내용이 나오면 삭제합니다. 인물에 대한 질문이 아닌데 같은 인물의 책을 여러 권 추천하는 경우, 한 권만 추천합니다.
책의 리뷰가 많거나 별점이 높은 순서대로 추천합니다."""

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rag_system),
        ("human", "User question: {question}"),
    ]
)

# 질문 재작성 시스템 프롬프트
rewrite_system = """당신은 질문 재작성 전문가입니다. 사용자가 입력한 단어를 기반으로 구글 검색을 먼저 시도하세요. 
1. 입력한 단어가 사람의 이름인 경우:
    - 해당 인물이 책의 저자인지 확인하고, 그렇다면 그 사람이 쓴 책을 추천해 주세요.
    - 예시:
        - 입력: "김영하"
        - 출력: "김영하 작가님이 쓴 '살인자의 기억법'을 추천드립니다."

2. 입력한 단어가 사람이 아닌 경우:
    2.1 관련된 책이 있는지 확인하고, 관련성이 높은 책을 추천해 주세요.
        - 예시:
            - 입력: "마라톤"
            - 출력: "마라톤에 관한 책으로 '마라톤의 역사'를 추천드립니다."
    2.2 입력한 단어가 다른 의미가 있을 때는 해당 단어와 관련있는 사람의 책을 추천해주세요.
        - 예시:
            - 입력: "흑백요리사"
            - 출력: "흑백요리사와 연관성이 높은 저자 최강록이 쓴 '최강록의 요리노트'를 추천합니다."
3. 위의 두 경우에 해당하지 않는다면:
    - 입력한 단어의 의미를 검색을 통해 찾습니다.
    - 예시:
        - 입력: "알라이버"
        - 출력: "알라이버가 뭔가요?"
    3.1 단어가 뭔지 알았다면, 그 단어로 나올 수 있는 책 관련 질문을 만들어주세요.

**참고 사항:**
- 항상 한국어로 응답해 주세요.
- 불필요한 설명이나 추가 정보는 포함하지 마세요."""

re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rewrite_system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

# 필요한 요소 초기화
rag_chain = rag_prompt | llm | StrOutputParser()
question_rewriter = re_write_prompt | llm | StrOutputParser()
web_search_tool = TavilySearchResults(max_results=5)
retrieval_grader = grade_prompt | structured_llm_grader
