# optimization.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
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
    def __init__(self, tone, style, num_books=1, additional_instructions=None, conversation_history=None):
        """
        원하는 톤, 스타일 및 추가 지시사항으로 최적화 설정 초기화.

        :param tone: 답변의 원하는 톤 (예: "격식있는", "캐주얼한", "친근한").
        :param style: 답변에 사용할 글쓰기 스타일 (예: "정보전달형", "설득력 있는", "간결한").
        :param num_books: 추천할 책의 수.
        :param additional_instructions: 응답 생성에 포함할 추가 지시사항 (옵션).
        :param conversation_history: 이전 대화 기록을 담은 리스트 (옵션).
        """
        self.tone = tone
        self.style = style
        self.num_books = num_books
        self.additional_instructions = additional_instructions
        self.conversation_history = conversation_history or []
        self.naver_client_id = os.getenv("NAVER_CLIENT_ID")
        self.naver_client_secret = os.getenv("NAVER_CLIENT_SECRET")

        # 톤과 스타일 최적화를 위한 시스템 프롬프트 정의
        self.optimization_system = f"""당신은 사용자의 질문에 대해 친절하지만 전문적으로 답변하는 책 추천 도우미입니다.
아래의 구조를 따라 답변을 구성하세요:

구조:
각 책마다 다음 정보를 포함하여 총 {self.num_books}권의 책을 추천하세요.
1. 책 이미지
2. 책 제목 (책 제목은 큰따옴표로 감싸세요: "책 제목")
3. 작가
4. 출판사
5. 필요에 따라 후속 질문이나 제안을 추가하세요.

주의사항:
- 반드시 실제로 존재하는 책만 추천하세요.
- 추천할 책은 네이버 API를 사용하여 검색 결과가 있는 책으로 한정하세요.
- 이전 대화 내용을 참고하여 답변하세요.
- 예시는 참고용이며 실제 답변에 사용하지 마세요.
- 생성된 답변에 반드시 위의 구조를 따르세요.
- 각 항목이 끝날 때마다 줄바꿈을 하지 마세요.
- 책 제목을 추출하기 쉽게 큰따옴표로 감싸주세요.
- 책을 제외한 다른 미디어는 추천하지 못합니다.
- 각 답변을 완료한 뒤에는 엔터로 구분해주세요.
- 답변은 한국어로 해주세요.
- 적절한 답변을 찾을 수 없는 경우에는 "죄송하지만 관련된 책을 찾을 수 없습니다."로 해주세요.

예시 (참고용):
"안녕하세요! 흑백요리사와 어울리는 책을 찾고 계시나요? 제가 추천해드릴 책은 다음과 같습니다:
1. 책 이미지: ![책 이미지](이미지 링크) 책 제목: "예시 책 제목1" 작가: 작가명1 출판사: 출판사명1 추천 이유: 이 책은... 필요하시면 다른 추천도 드릴까요? 구매 링크: [교보문고](교보문고 링크), [알라딘](알라딘 링크), [영풍문고](영풍문고 링크)
..."
"""

        # 최적화를 위한 채팅 프롬프트 템플릿 정의
        self.optimization_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.optimization_system),
                *self.conversation_history,  # 이전 대화 내용을 포함
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

        # 대화 기록 업데이트
        self.conversation_history.append(("human", generated_response))

        # 최적화된 응답 생성
        optimized_response = self.structured_optimizer.invoke(prompt_data.to_messages())
        optimized_text = optimized_response.optimized_response

        # 최적화된 응답에서 책 제목들 추출
        book_titles = self.extract_book_titles(optimized_text)

        # 네이버 API를 사용하여 책 정보 가져오기
        if book_titles:
            book_info_list = []
            valid_titles = []
            for title in book_titles:
                search_results = self.search_book_info(title)
                if search_results:
                    book_info_list.append(search_results[0])  # 첫 번째 결과 사용
                    valid_titles.append(title)
                else:
                    print(f"'{title}'에 대한 검색 결과가 없습니다.")

            if not book_info_list:
                return "죄송하지만 관련된 책을 찾을 수 없었습니다. 질문을 더 구체적으로 만들어주실 수 있으신가요?"

            # 존재하는 책들로 최적화된 응답을 수정
            optimized_text = self.rewrite_response(optimized_text, valid_titles)

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

    def rewrite_response(self, text, valid_titles):
        """
        존재하는 책 제목들로만 응답을 재작성.
        """
        # 원본 응답에서 존재하지 않는 책 정보를 제거
        lines = text.split('\n')
        new_lines = []
        for line in lines:
            title_in_line = re.findall(r'"([^"]+)"', line)
            if not title_in_line or title_in_line[0] in valid_titles:
                new_lines.append(line)
        return '\n'.join(new_lines)

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
            title = book_info['title']
            # 책 제목에서 소제목 제거
            main_title = title.split('(')[0].strip()

            # 추천 이유를 네이버 책 소개로부터 가져와서 3문장으로 요약
            description = book_info['description']
            summary = self.summarize_text(description, 3)

            # 구매 링크 생성 (서점의 검색 결과 페이지 링크 사용)
            encoded_title = requests.utils.quote(main_title)
            kyobo_link = f"https://search.kyobobook.co.kr/search?keyword={encoded_title}"
            aladin_link = f"https://www.aladin.co.kr/search/wsearchresult.aspx?SearchTarget=All&SearchWord={encoded_title}"
            yes24_link = f"https://www.yes24.com/Product/Search?query={encoded_title}"

            # 서점 이름에 하이퍼링크 적용 (Markdown 형식)
            purchase_links = f"[교보문고]({kyobo_link}), [알라딘]({aladin_link}), [영풍문고]({yes24_link})"

            # 책 이미지에 Markdown 형식의 이미지 링크 적용
            book_image = f"![책 이미지]({book_info['image']})"

            book_details = f"{idx}. 책 이미지: {book_image} 책 제목: \"{main_title}\" 작가: {book_info['author']} 출판사: {book_info['publisher']} 추천 이유: {summary} 구매 링크: {purchase_links}"
            book_details_list.append(book_details)

        # 필요에 따라 후속 질문이나 제안을 추가
        follow_up = "더 궁금한 점이 있으시면 말씀해주세요!"

        book_details_text = '\n'.join(book_details_list)
        return f"{updated_text}\n{book_details_text}\n{follow_up}"

    def summarize_text(self, text, num_sentences):
        """
        텍스트를 주어진 문장 수로 요약.

        :param text: 요약할 텍스트
        :param num_sentences: 원하는 문장 수
        :return: 요약된 텍스트
        """
        # 텍스트가 없으면 빈 문자열 반환
        if not text:
            return "상세한 내용은 링크를 참고해주세요."

        # 마침표 기준으로 문장 분리
        sentences = re.split(r'(?<=[.!?]) +', text)
        short_description = ' '.join(sentences[:num_sentences])
        return short_description

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
            results = response.json().get("items")
            if results:
                book_info_list = []
                for item in results:
                    title = item.get("title", "").replace("<b>", "").replace("</b>", "")
                    author = item.get("author", "")
                    publisher = item.get("publisher", "")
                    image = item.get("image", "")
                    description = item.get("description", "")
                    link = item.get("link", "")
                    isbn = item.get("isbn", "")  # ISBN-10과 ISBN-13이 함께 제공됨

                    book_info = {
                        "title": title,
                        "author": author,
                        "publisher": publisher,
                        "image": image,
                        "description": description,
                        "link": link,
                        "isbn": isbn
                    }
                    book_info_list.append(book_info)
                return book_info_list
            else:
                return []
        else:
            print(f"네이버 API 요청 실패: {response.status_code}")
            return []
