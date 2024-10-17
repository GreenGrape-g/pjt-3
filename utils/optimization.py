from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import requests
import os
from dotenv import load_dotenv
import re
import time

# 환경 변수 로드
load_dotenv()

class OptimizationPrompt(BaseModel):
    """
    최적화된 답변을 생성하기 위한 프롬프트 클래스.
    """
    optimized_response: str = Field(description="톤과 스타일을 조정한 후의 최적화된 답변.")

# 언어 모델 초기화
tone_style_llm = ChatOpenAI(model="gpt-4", temperature=0.7)

class Optimization:
    def __init__(self, tone, style, num_books=1, additional_instructions=None):
        """
        원하는 톤, 스타일 및 추가 지시사항으로 최적화 설정 초기화.
        """
        self.max_books = 5
        self.requested_num_books = num_books
        self.num_books = min(num_books, self.max_books)
        self.tone = tone
        self.style = style
        self.additional_instructions = additional_instructions
        self.naver_client_id = os.getenv("NAVER_CLIENT_ID")
        self.naver_client_secret = os.getenv("NAVER_CLIENT_SECRET")

        # 톤과 스타일 최적화를 위한 시스템 프롬프트 정의
        self.optimization_system = f"""당신은 사용자의 질문에 따라 실제로 존재하는 책의 제목만을 추천하는 도우미입니다.
반드시 실제로 존재하는 책만 추천하세요. 존재하지 않는 책을 만들어내지 마세요.

아래의 구조를 따라 답변을 구성하세요:

구조:
각 책마다 다음 정보를 포함하여 총 {self.num_books}권의 책 제목만 추천하세요.
1. 책 제목 (큰따옴표로 감싸세요: "책 제목")

주의사항:
- 실제로 존재하는 책만 추천하세요.
- 추천 이유, 작가, 출판사 등의 정보는 포함하지 마세요.
- 추천할 책의 수는 {self.max_books}권을 초과하지 마세요.
- 추천할 책의 수가 {self.max_books}권을 초과하면, 최대 {self.max_books}권만 추천하고, 다음과 같은 완곡한 거절 표현을 사용하세요:
  "좋은 책 {self.max_books}권만 추천해줄게. 더 원하면, 그때 가서 더 추천해줄게."

예시 (참고용):
"안녕하세요! 다음 책들을 추천해드릴게요:
1. "요리의 과학"
2. "한강의 문학 세계"
3. "흑백요리사"
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

        # 책 제목의 수를 최대 self.num_books로 제한
        book_titles = book_titles[:self.num_books]

        # 네이버 API를 사용하여 책 정보 가져오기
        if book_titles:
            book_info_list = []
            for title in book_titles:
                time.sleep(0.5)  # 요청 간 딜레이 추가
                search_results = self.search_book_info(title, search_type='title')
                if search_results:
                    book_info = search_results[0]  # 첫 번째 결과 사용

                    # 평론가 리뷰 가져오기 (추후 구현)
                    professional_review = self.get_professional_review(title)
                    if professional_review:
                        book_info['description'] = professional_review  # 추천 이유를 평론가 리뷰로 대체

                    book_info_list.append(book_info)
                else:
                    # 연관성 있는 책을 추천하도록 시도
                    related_books = self.search_related_books(title)
                    if related_books:
                        # 최대 추천 수를 초과하지 않도록 추가
                        remaining = self.max_books - len(book_info_list)
                        book_info_list.extend(related_books[:remaining])
                    else:
                        print(f"책 '{title}'을(를) 찾을 수 없습니다.")

            if not book_info_list:
                # 모든 책이 검색되지 않았을 경우
                return "죄송하지만, 관련된 책을 찾을 수 없습니다."

            # 책 정보를 최적화된 응답에 통합
            final_response = self.insert_book_info(optimized_text, book_info_list)
            return final_response
        else:
            # 책 제목을 추출하지 못한 경우
            return "죄송하지만, 관련된 책을 찾을 수 없습니다."

    def extract_book_titles(self, text):
        """
        주어진 텍스트에서 책 제목들 추출.
        """
        # 큰따옴표로 감싸진 책 제목 추출
        titles = re.findall(r'"([^"]+)"', text)
        return titles

    def clean_html(self, text):
        """
        텍스트에서 HTML 태그 제거.
        """
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', text)
        return cleantext

    def clean_title(self, title):
        """
        책 제목에서 소제목(부제)를 제거합니다.
        """
        # HTML 태그 제거
        title = self.clean_html(title)
        # 괄호 및 그 안의 내용 제거
        title = re.sub(r'\(.*?\)', '', title)
        # 콜론(:) 뒤의 내용 제거
        title = title.split(':')[0]
        # 하이픈(-) 뒤의 내용 제거
        title = title.split('-')[0]
        # 양쪽 공백 제거
        title = title.strip()
        return title

    def extract_bid(self, link):
        """
        네이버 책 링크에서 catalogId 값을 추출합니다.
        """
        match = re.search(r'/book/catalog/(\d+)', link)
        if match:
            return match.group(1)
        else:
            return None

    def insert_book_info(self, text, book_info_list):
        """
        책 정보를 텍스트에 삽입.
        """
        # 텍스트에서 기존 책 정보 제거
        keys_to_remove = ['책 이미지', '책 제목', '작가', '출판사', '추천 이유', '구매 링크']
        for key in keys_to_remove:
            pattern = f"{key}:.*?(?=(책 이미지|$))"
            text = re.sub(pattern, '', text, flags=re.DOTALL)

        # 텍스트를 공백으로 분리하여 다시 연결
        updated_text = ' '.join(text.split())

        # 각 책 정보 삽입
        book_details_list = []
        for idx, book_info in enumerate(book_info_list, 1):
            # 이미지 링크 확인 및 처리
            image_link = book_info['image'] if book_info['image'] else "이미지 링크 없음"
            if image_link.startswith('http://'):
                image_link = image_link.replace('http://', 'https://')

            # 네이버 책 상세 페이지 링크 생성
            bid = self.extract_bid(book_info['link'])
            if bid:
                naver_book_detail_link = f"https://search.shopping.naver.com/book/catalog/{bid}"
            else:
                # bid를 추출하지 못한 경우 검색 링크 사용
                encoded_title = requests.utils.quote(book_info['title'])
                naver_book_detail_link = f"https://search.shopping.naver.com/search/all?query={encoded_title}&catId=50005542"

            # 정보가 없을 경우 기본 값 설정
            author = book_info['author'] if book_info['author'] else "작가 정보 없음"
            publisher = book_info['publisher'] if book_info['publisher'] else "출판사 정보 없음"
            description = book_info['description'] if book_info['description'] else "추천 이유 없음"

            book_details = f"""{idx}. 책 이미지: {image_link}
책 제목: "{book_info['title']}"
작가: {author}
출판사: {publisher}
추천 이유: {description}
구매 링크: {naver_book_detail_link}"""
            book_details_list.append(book_details)

        book_details_text = '\n'.join(book_details_list)
        return f"{updated_text}\n{book_details_text}"

    def search_book_info(self, query, search_type='title'):
        """
        네이버 검색 API를 사용하여 책 정보 가져오기.
        search_type: 'title', 'author', 'keyword'
        """
        headers = {
            "X-Naver-Client-Id": self.naver_client_id,
            "X-Naver-Client-Secret": self.naver_client_secret,
        }
        params = {
            "display": self.num_books,
            "sort": "sim"
        }
        if search_type == 'title':
            params["d_titl"] = query  # 제목 검색
        elif search_type == 'author':
            params["d_auth"] = query  # 작가명 검색
        else:
            params["query"] = query  # 전체 검색

        try:
            response = requests.get("https://openapi.naver.com/v1/search/book_adv.json", headers=headers, params=params)
            response.raise_for_status()
            results = response.json().get("items", [])
            if not results:
                print(f"검색 결과가 없습니다. Query: {query}, Search Type: {search_type}")
                return []
        except requests.exceptions.HTTPError as e:
            print(f"HTTPError: {e}")
            return []
        except Exception as e:
            print(f"Exception: {e}")
            return []

        book_info_list = []
        for item in results:
            title = self.clean_title(item.get("title", ""))
            book_info = {
                "title": title,
                "author": self.clean_html(item.get("author", "")),
                "publisher": item.get("publisher", ""),
                "link": item.get("link", ""),
                "image": item.get("image", ""),
                "description": self.clean_html(item.get("description", "")),
                "isbn": item.get("isbn", "")
            }
            book_info_list.append(book_info)
        return book_info_list

    def search_related_books(self, query):
        """
        관련성이 높은 책을 검색하여 추천합니다.
        """
        # 예시: 제목 기반 전체 검색
        related_results = self.search_book_info(query, search_type='keyword')
        return related_results

    def get_professional_review(self, title):
        """
        벡터스토어에서 책 제목과 관련된 평론가 리뷰 가져오기.
        """
        # 현재는 빈 문자열을 반환하지만, 추후 벡터스토어 연동 시 구현 예정
        return ""

# 예제 사용법
if __name__ == "__main__":
    tone = "친근함"
    style = "전문적"
    num_books = 5  # 추천할 책의 수 (최대 5개)
    additional_instructions = "추천 이유를 전문적이고 깊이 있게 작성하세요."

    optimizer = Optimization(tone, style, num_books, additional_instructions)
    generated_response = "흑백요리사와 관련된 책을 추천해줘."
    optimized_response = optimizer.optimize_response(generated_response)
    print(optimized_response)
