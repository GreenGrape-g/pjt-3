# optimization.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import requests
import os
from dotenv import load_dotenv
import re

# 환경 변수 로드
load_dotenv()

class OptimizationPrompt(BaseModel):
    """
    최적화된 답변을 생성하기 위한 프롬프트 클래스.

    속성:
        optimized_response: LLM이 생성한 정제된 답변
    """
    optimized_response: str = Field(description="톤과 스타일을 조정한 후의 최적화된 답변.")

# 언어 모델 초기화
tone_style_llm = ChatOpenAI(model="gpt-4", temperature=0.7)

class Optimization:
    def __init__(self, tone, style, num_books=1, additional_instructions=None):
        """
        원하는 톤, 스타일 및 추가 지시사항으로 최적화 설정 초기화.

        :param tone: 답변의 원하는 톤 (예: "격식있는", "캐주얼한", "친근한").
        :param style: 답변에 사용할 글쓰기 스타일 (예: "정보전달형", "설득력 있는", "간결한").
        :param num_books: 추천할 책의 수.
        :param additional_instructions: 응답 생성에 포함할 추가 지시사항 (옵션).
        """
        self.tone = tone
        self.style = style
        self.num_books = num_books
        self.additional_instructions = additional_instructions
        self.naver_client_id = os.getenv("NAVER_CLIENT_ID")
        self.naver_client_secret = os.getenv("NAVER_CLIENT_SECRET")

        # 톤과 스타일 최적화를 위한 시스템 프롬프트 정의
        self.optimization_system = f"""당신은 사용자의 질문에 대해 친근하고 친구 같은 느낌으로 답변하는 책 추천 도우미입니다.
아래의 구조를 따라 답변을 구성하세요:

구조:
각 책마다 다음 정보를 포함하여 총 {self.num_books}권의 책을 추천하세요.
1. 책 이미지
2. 책 제목 (책 제목은 큰따옴표로 감싸세요: "책 제목")
3. 작가
4. 출판사
5. 추천 이유
6. 구매 링크 (네이버 책 정보 페이지 링크)

주의사항:
- 예시는 참고용이며 실제 답변에 사용하지 마세요.
- 생성된 답변에 반드시 위의 구조를 따르세요.
- 각 항목이 끝날 때마다 줄바꿈을 하지 마세요.
- 책 제목을 추출하기 쉽게 큰따옴표로 감싸주세요.
- 책을 제외한 다른 미디어는 추천하지 못합니다.
- 각 답변을 완료한 뒤에는 엔터로 구분해주세요.
- 답변은 한국어로 해주세요.
- 적절한 답변을 찾을 수 없는 경우에는 "질문한 내용과 관련된 책을 찾을 수 없습니다."로 해주세요.

예시 (참고용):
"안녕하세요! 감동적인 소설을 찾고 계시나요? 제가 추천해드릴 책은 다음과 같습니다:
1. 책 이미지: [이미지 링크] 책 제목: "예시 책 제목1" 작가: 작가명1 출판사: 출판사명1 추천 이유: 이 책은... 구매 링크: [네이버 책 정보 페이지 링크]
2. 책 이미지: [이미지 링크] 책 제목: "예시 책 제목2" 작가: 작가명2 출판사: 출판사명2 추천 이유: 이 책은... 구매 링크: [네이버 책 정보 페이지 링크]
..."
"""

        # 최적화를 위한 채팅 프롬프트 템플릿 정의
        self.optimization_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.optimization_system),
                (
                    "human",
                    "생성된 답변: {response}\n\n추가 지시사항: {additional_instructions}"
                ),
            ]
        )

        # 최적화를 위한 구조화된 LLM 생성
        self.structured_optimizer = tone_style_llm.with_structured_output(OptimizationPrompt)

    def optimize_response(self, generated_response):
        """
        톤, 스타일 및 추가 지시사항에 따라 최적화된 응답 생성.

        :param generated_response: 이전 단계에서 생성된 콘텐츠.
        :return: 원하는 톤과 스타일에 맞는 최적화된 응답.
        """
        # 프롬프트에 필요한 변수를 전달
        prompt_data = self.optimization_prompt.format_prompt(
            response=generated_response,
            additional_instructions=self.additional_instructions or "없음"
        )

        # 최적화된 응답 생성
        optimized_response = self.structured_optimizer.invoke(prompt_data.to_messages())
        optimized_text = optimized_response.optimized_response

        # 최적화된 응답에서 책 제목들 추출
        book_titles = self.extract_book_titles(optimized_text)

        # 네이버 API를 사용하여 책 정보 가져오기
        if book_titles:
            book_info_list = []
            for title in book_titles:
                search_results = self.search_book_info(title)
                if search_results:
                    book_info_list.append(search_results[0])  # 첫 번째 결과 사용

            # 책 정보를 최적화된 응답에 통합
            final_response = self.insert_book_info(optimized_text, book_info_list)
            return final_response
        else:
            return optimized_text

    def extract_book_titles(self, text):
        """
        주어진 텍스트에서 책 제목들 추출.
        """
        # 책 제목이 큰따옴표로 감싸져 있다고 가정
        titles = re.findall(r'"([^"]+)"', text)
        return titles

    def insert_book_info(self, text, book_info_list):
        """
        책 정보를 텍스트에 삽입.
        """
        # 텍스트에서 기존 책 정보 제거
        keys_to_remove = ['책 이미지', '책 제목', '작가', '출판사', '추천 이유', '구매 링크']
        for key in keys_to_remove:
            pattern = f"{key}:.*?"
            text = re.sub(pattern, '', text)

        # 텍스트를 공백으로 분리하여 다시 연결
        updated_text = ' '.join(text.split())

        # 각 책 정보 삽입
        book_details_list = []
        for idx, book_info in enumerate(book_info_list, 1):
            book_details = f"{idx}. 책 이미지: {book_info['image']} 책 제목: \"{book_info['title']}\" 작가: {book_info['author']} 출판사: {book_info['publisher']} 추천 이유: {book_info['description']} 구매 링크: {book_info['link']}"
            book_details_list.append(book_details)

        book_details_text = ' '.join(book_details_list)
        return f"{updated_text} {book_details_text}"

    def search_book_info(self, query):
        """
        네이버 검색 API를 사용하여 책 정보 가져오기.

        :param query: 책에 대한 검색 쿼리.
        :return: 네이버의 검색 결과.
        """
        headers = {
            "X-Naver-Client-Id": self.naver_client_id,
            "X-Naver-Client-Secret": self.naver_client_secret,
        }
        params = {
            "query": query,
            "display": 1  # 각 책 제목마다 1개의 결과만 가져옴
        }
        response = requests.get("https://openapi.naver.com/v1/search/book.json", headers=headers, params=params)

        if response.status_code == 200:
            results = response.json()["items"]
            book_info_list = []
            for item in results:
                book_info = {
                    "title": item["title"].replace("<b>", "").replace("</b>", ""),
                    "author": item["author"],
                    "publisher": item["publisher"],
                    "link": item["link"],
                    "image": item["image"],
                    "description": item["description"]
                }
                book_info_list.append(book_info)
            return book_info_list
        else:
            return []

# 예제 사용법
if __name__ == "__main__":
    tone = "친근함"
    style = "설득력 있는"
    num_books = 3  # 추천할 책의 수
    additional_instructions = "따뜻한 인사를 추가하세요."

    optimizer = Optimization(tone, style, num_books, additional_instructions)
    generated_response = "최근에 감동적인 소설을 3권 추천해줘."
    optimized_response = optimizer.optimize_response(generated_response)
    print(optimized_response)
