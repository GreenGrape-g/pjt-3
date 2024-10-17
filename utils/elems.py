from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

structured_llm_grader = llm.with_structured_output(GradeDocuments)

vectorstore = PineconeVectorStore.from_existing_index(
    index_name="bitebook", embedding=OpenAIEmbeddings()
)

rag_system = """You are an assistant that helps retrieve relevant information based on the user's query.
사용 가능한 문서를 활용하여 사용자의 질문에 간결하고 정확하게 책 제목과 저자 정보를 5개 정도 제공합니다.
중복되는 내용이 나오면 삭제합니다. 인물에 대한 질문이 아닌데 같은 인물의 책을 여러 권 추천하는 경우, 한 권만 추천합니다. 
책의 리뷰가 많거나 별점이 높은 순서대로 추천합니다."""

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rag_system),
        ("human", "User question: {question}"),
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

system = """당신은 질문 재작성 전문가입니다. 사용자가 입력한 단어를 기반으로 구글 검색을 먼저 시도하세요. 
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
        - 입력: "흑백요리사"
        - 출력: "흑백요리사와 연관성이 높은 저자 최강록이 쓴 '최강록의 요리노트'를 추천합니다.
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
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

# 아래 항목들이 graph.py 에서 사용하는 요소들
retriever = vectorstore.as_retriever()
rag_chain = rag_prompt | llm | StrOutputParser()
question_rewriter = re_write_prompt | llm | StrOutputParser()
web_search_tool = TavilySearchResults(max_results=5)
retrieval_grader = grade_prompt | structured_llm_grader