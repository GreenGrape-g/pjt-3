import os
import re
import logging
import requests
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)

# 환경 변수 로드
load_dotenv()


class Optimization:
    """
    사용자의 질문에 대해 최적화된 응답을 생성하는 클래스입니다.
    """

    def __init__(
        self,
        tone: str,
        style: str,
        additional_instructions: str = None,
        conversation_history: list = None
    ):
        """
        초기화 메서드로, 필요한 설정과 언어 모델을 준비합니다.

        Args:
            tone (str): 응답의 어조
            style (str): 응답의 스타일
            additional_instructions (str, optional): 추가 지침
            conversation_history (list, optional): 대화 기록
        """
        self.tone = tone
        self.style = style
        self.additional_instructions = additional_instructions or "한국어로만 답변해주세요."
        self.conversation_history = conversation_history or []

        # 네이버 API 자격 증명 로드
        self.naver_client_id = os.getenv('NAVER_CLIENT_ID')
        self.naver_client_secret = os.getenv('NAVER_CLIENT_SECRET')
        if not self.naver_client_id or not self.naver_client_secret:
            raise ValueError("NAVER_CLIENT_ID 및 NAVER_CLIENT_SECRET 환경 변수를 설정해주세요.")

        # 시스템 프롬프트 설정
        self.optimization_system = """당신은 사용자의 질문에 대해 전문적으로 친절하게 답변하는 도서 전문가입니다.

**언어 관련 필수 지침:**
* 모든 상황에서 반드시 한국어로만 답변하세요.
* 외국 도서나 작가에 대해 이야기할 때도 반드시 한국어로 답변하세요.
* 책 제목은 한국어로 번역된 제목을 우선적으로 사용하되, 원제목을 괄호 안에 표기하세요.
* 작가 이름은 한국어 표기를 먼저 쓰고 원어 이름을 괄호 안에 표기하세요.
* 영어나 다른 외국어로 된 답변은 절대 금지입니다.

**책 추천 시 지침:**
* **특정 책 제목 언급 시:**
  - 사용자가 특정 책 제목을 언급하면, 해당 책을 **단독으로 직접 추천**하고 간단한 응원의 메시지를 덧붙이세요.
  - 예시:
    ```
    책 제목: [너무 한낮의 연애]
    작가: 박완서
    출판사: 문학동네
    추천 이유: 이 책은 일상 속에서 느낄 수 있는 따뜻한 감정을 섬세하게 그려내어 독자들에게 큰 공감을 불러일으킵니다. 즐거운 독서 되세요!
    ```

* **특정 작가 언급 시:**
  - 사용자가 특정 작가를 언급하면, 해당 작가의 **주요 책 제목을 간결하게 나열**하고 각 책에 대해 위와 같은 형식으로 상세히 작성하세요.
  - 예시:
    ```
    책 제목: [살인자의 기억법]
    작가: 김영하
    출판사: 문학동네
    추천 이유: 이 소설은 독특한 구성과 깊이 있는 캐릭터 분석으로 독자들에게 강렬한 인상을 남깁니다. 추천드립니다!

    책 제목: [오직 두 사람]
    작가: 김영하
    출판사: 문학동네
    추천 이유: 사랑과 인간관계에 대한 섬세한 통찰을 제공하며, 감동적인 이야기가 돋보입니다. 즐겁게 읽어보세요!
    ```

* **일반적인 책 추천 요청 시:**
  - 사용자의 요청에 따라 **1권**의 유일한 책을 **간결하게** 작성하고, 책 제목, 저자, 출판사, 추천 이유 순으로 상세히 작성하세요.
  - 예시:
    ```
    책 제목: [사랑의 온도]
    작가: 김민지
    출판사: 시공사
    추천 이유: 이 소설은 사랑의 다양한 모습을 섬세하게 그려내어 독자들에게 깊은 감동을 줍니다.
    ```

**모호하거나 불명확한 질문에 대한 대응:**
* 사용자의 요구가 불명확할 경우, **간단한 추가 질문**을 통해 명확히 이해하려고 노력하세요.
  - 예시: "어떤 장르의 책을 원하시나요?" 또는 "특정 작가를 선호하시나요?"

**응답 형식:**
* 모든 답변은 반드시 한국어로 작성하세요.
* 절대로 영어로 답변하지 마세요.
* 반드시 실제로 존재하는 책만 추천하세요.
* 추천할 책은 네이버 API를 사용하여 검색 결과가 있는 책으로 한정하세요.
* 동일한 책을 여러 번 추천하지 마세요.
* 이전 대화 내용을 참고하여 답변하세요.
* 책 제목을 추출하기 쉽게 대괄호로 감싸주세요.
* 각 항목을 명확하게 구분하여 작성하세요.
* 각 답변을 완료한 뒤에는 줄바꿈을 하세요.
* 책을 제외한 다른 미디어는 추천하지 못합니다.
* 적절한 답변을 찾을 수 없는 경우에는 "죄송하지만 관련된 책을 찾을 수 없습니다."라고 답변하세요.
* 추천 이유는 3문장 이내로 해주세요.
* 줄거리는 제외해주세요.

**추가 사항:**
* 책과 관련 없는 질문이 들어올 경우, 자연스럽고 친절하게 대화를 이어가며 응답하세요.
  - 예시: "책 추천 외에 다른 도움이 필요하신가요?" 또는 "다른 주제에 대해 이야기해볼까요?"
"""

        # 메시지 템플릿 리스트 생성
        self.messages = [("system", self.optimization_system)]
        self.add_conversation_history()

        # 언어 모델 초기화
        self.structured_optimizer = ChatOpenAI(
            model="chatgpt-4o-latest",
            temperature=0.7
        )

    def add_conversation_history(self):
        """대화 기록을 메시지 리스트에 추가합니다."""
        for msg in self.conversation_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                self.messages.append(("human", content))
            elif role == "assistant":
                self.messages.append(("ai", content))

    def optimize_response(self, question: str, num_books: int = 1) -> str:
        """
        사용자의 질문에 최적화된 응답을 생성합니다.

        Args:
            question (str): 사용자의 질문
            num_books (int, optional): 추천할 책의 수

        Returns:
            str: 최적화된 응답
        """
        logging.debug(f"Optimizing response for question: {question} with num_books={num_books}")

        # 메시지 리스트에 새로운 질문 추가
        prompt_messages = self.messages + [
            ("human", f"{question}\n\n{self.additional_instructions}")
        ]

        # 새로운 프롬프트 템플릿 생성
        prompt = ChatPromptTemplate.from_messages(prompt_messages)

        # 프롬프트 포맷팅
        prompt_data = prompt.format_prompt(num_books=num_books)
        logging.debug(f"Formatted prompt: {prompt_data}")

        # 최적화된 응답 생성
        optimized_response = self.structured_optimizer(
            prompt_data.to_messages()
        ).content.strip()
        logging.debug(f"Optimized response from LLM: {optimized_response}")

        # 최적화된 응답에서 책 제목 추출
        book_titles = self.extract_book_titles(optimized_response)
        logging.debug(f"Extracted book titles: {book_titles}")

        # 중복된 책 제목 제거
        unique_book_titles = list(set(book_titles))
        logging.debug(f"Unique book titles: {unique_book_titles}")

        # 네이버 API를 사용하여 책 정보 가져오기
        if unique_book_titles:
            book_info_list, valid_titles = self.get_valid_book_info(unique_book_titles, num_books)
            if book_info_list:
                # 존재하는 책들로 응답을 재작성
                optimized_text = self.rewrite_response(optimized_response, valid_titles)
                logging.debug(f"Rewritten optimized text: {optimized_text}")

                # 책 정보를 응답에 통합
                final_response = self.insert_book_info(optimized_text, book_info_list)
                logging.debug(f"Final response after inserting book info: {final_response}")
            else:
                logging.warning("관련된 책을 찾을 수 없었습니다.")
                final_response = "죄송하지만 관련된 책을 찾을 수 없었습니다. 질문을 더 구체적으로 만들어주실 수 있으신가요?"
        else:
            logging.debug("No book titles extracted; returning optimized response as is.")
            final_response = optimized_response

        logging.debug(f"Final response to return: {final_response}")
        return final_response

    def extract_book_titles(self, text: str) -> list:
        """
        주어진 텍스트에서 책 제목들을 추출합니다.

        Args:
            text (str): 텍스트

        Returns:
            list: 추출된 책 제목 리스트
        """
        titles = re.findall(r"\[([^\]]+)\]", text)
        unique_titles = list(set(titles))
        logging.debug(f"Extracted unique titles from text: {unique_titles}")
        return unique_titles

    def get_valid_book_info(self, titles: list, num_books: int) -> tuple:
        """
        유효한 책 정보를 가져옵니다.

        Args:
            titles (list): 책 제목 리스트
            num_books (int): 추천할 책의 수

        Returns:
            tuple: 책 정보 리스트와 유효한 책 제목 리스트
        """
        book_info_list = []
        valid_titles = []
        for title in titles[:num_books]:
            search_results = self.search_book_info(title)
            if search_results:
                if search_results[0]['title'] not in [book['title'] for book in book_info_list]:
                    book_info_list.append(search_results[0])
                    valid_titles.append(title)
            else:
                logging.warning(f"'{title}'에 대한 검색 결과가 없습니다.")
        return book_info_list, valid_titles

    def rewrite_response(self, text: str, valid_titles: list) -> str:
        """
        존재하는 책 제목들로만 응답을 재작성합니다.

        Args:
            text (str): 원본 응답 텍스트
            valid_titles (list): 유효한 책 제목 리스트

        Returns:
            str: 재작성된 응답 텍스트
        """
        lines = text.split('\n')
        new_lines = []
        for line in lines:
            title_in_line = re.findall(r"\[([^\]]+)\]", line)
            if not title_in_line or title_in_line[0] in valid_titles:
                new_lines.append(line)
        rewritten_text = '\n'.join(new_lines)
        logging.debug(f"Rewritten response: {rewritten_text}")
        return rewritten_text

    def insert_book_info(self, text: str, book_info_list: list) -> str:
        """
        책 정보를 텍스트에 삽입합니다.

        Args:
            text (str): 원본 텍스트
            book_info_list (list): 책 정보 리스트

        Returns:
            str: 책 정보가 삽입된 최종 응답 텍스트
        """
        # 기존 응답에서 불필요한 항목 제거
        keys_to_remove = ['책 제목', '작가', '출판사', '추천 이유']
        for key in keys_to_remove:
            pattern = f"^{key}:.*$"
            text = re.sub(pattern, '', text, flags=re.MULTILINE)

        # 책 세부사항 생성
        book_details_list = []
        for book_info in book_info_list:
            title = re.sub('<[^<]+?>', '', book_info['title']).split('(')[0].strip()
            author = self.format_author_names(book_info['author'])
            publisher = book_info.get("publisher", "출판사 정보 없음")
            description = book_info.get("description", "상세 설명을 찾을 수 없습니다.")
            summary = self.summarize_text(description, 3)

            book_details = (
                f"책 제목: {title}\n"
                f"작가: {author}\n"
                f"출판사: {publisher}\n"
                f"추천 이유: {summary}"
            )
            book_details_list.append(book_details)

        # 최종 응답 생성
        book_details_text = '\n\n'.join(book_details_list)
        follow_up = "\n\n더 궁금한 점이 있으시면 말씀해주세요!"
        final_response = f"{book_details_text}{follow_up}"
        logging.debug(f"Final response constructed: {final_response}")
        return final_response

    def format_author_names(self, author_str: str) -> str:
        """
        작가 이름을 포맷팅합니다.

        Args:
            author_str (str): 작가 이름 문자열

        Returns:
            str: 포맷팅된 작가 이름
        """
        authors = [a.strip() for a in author_str.split(',')]
        formatted_authors = []
        for author_name in authors:
            formatted_authors.append(author_name)
        return ', '.join(formatted_authors)

    def summarize_text(self, text: str, num_sentences: int) -> str:
        """
        텍스트를 주어진 문장 수로 요약합니다.

        Args:
            text (str): 요약할 텍스트
            num_sentences (int): 원하는 문장 수

        Returns:
            str: 요약된 텍스트
        """
        if not text:
            return "상세한 내용은 링크를 참고해주세요."
        sentences = re.split(r'(?<=[.!?]) +', text)
        short_description = ' '.join(sentences[:num_sentences])
        logging.debug(f"Summarized text: {short_description}")
        return short_description

    def search_book_info(self, query: str) -> list:
        """
        네이버 검색 API를 사용하여 책 정보를 가져옵니다.

        Args:
            query (str): 검색할 책 제목

        Returns:
            list: 책 정보 리스트
        """
        headers = {
            "X-Naver-Client-Id": self.naver_client_id,
            "X-Naver-Client-Secret": self.naver_client_secret,
        }

        def get_search_results(search_query: str, display_count: int = 10) -> list:
            params = {
                "query": search_query,
                "display": display_count,
                "sort": "sim"
            }
            try:
                response = requests.get(
                    "https://openapi.naver.com/v1/search/book.json",
                    headers=headers,
                    params=params
                )
                response.raise_for_status()
                return response.json().get("items", [])
            except requests.exceptions.RequestException as e:
                logging.error(f"네이버 API 요청 실패: {e}")
                return []

        # 한글 제목으로 검색
        korean_title = query.split("(")[0].strip() if "(" in query else query
        results = get_search_results(korean_title)

        # 결과 필터링 및 정렬
        filtered_results = self.filter_and_sort_results(results, korean_title)

        # 결과가 없을 경우 빈 리스트 반환
        if not filtered_results:
            logging.debug("한글 도서 검색 결과가 없습니다.")
            return []

        return filtered_results[:1]  # 가장 적절한 결과 하나만 반환

    def filter_and_sort_results(self, results: list, query: str) -> list:
        """
        검색 결과를 필터링하고 정렬합니다.

        Args:
            results (list): 검색 결과 리스트
            query (str): 검색어

        Returns:
            list: 필터링되고 정렬된 결과 리스트
        """
        filtered_results = []
        for item in results:
            title = item.get("title", "").replace("<b>", "").replace("</b>", "")
            author = item.get("author", "")
            publisher = item.get("publisher", "")
            description = item.get("description", "")
            score = 0

            # 제목에 한글이 포함된 경우 가산점
            if self.contains_korean(title):
                score += 3
            # 출판사가 한글인 경우 가산점
            if self.contains_korean(publisher):
                score += 2
            # 검색어와 제목의 유사도 점수 추가
            if query.lower() in title.lower():
                score += 2
            # 작가 이름이 검색어에 포함되면 추가 점수
            if query.lower() in author.lower():
                score += 2

            filtered_results.append({
                "title": title,
                "author": author,
                "publisher": publisher,
                "description": description,
                "score": score
            })

        # 점수에 따라 정렬
        filtered_results.sort(key=lambda x: x["score"], reverse=True)
        return filtered_results

    @staticmethod
    def contains_korean(text: str) -> bool:
        """
        텍스트에 한글이 포함되어 있는지 확인합니다.

        Args:
            text (str): 확인할 텍스트

        Returns:
            bool: 한글 포함 여부
        """
        return any('\u3131' <= c <= '\uD7A3' for c in text)


# 사용 예시 (테스트용)
if __name__ == "__main__":
    # 환경 변수 설정 예시 (실제 값으로 대체해야 함)
    os.environ['NAVER_CLIENT_ID'] = 'your_naver_client_id'
    os.environ['NAVER_CLIENT_SECRET'] = 'your_naver_client_secret'

    # 최적화 인스턴스 생성
    optimizer = Optimization(
        tone="친절한",
        style="설득력 있는",
        additional_instructions="응답이 친근하고 환영하는 느낌이 들도록 해주세요.",
        conversation_history=[
            {"role": "user", "content": "로맨스 소설 추천해줘."},
            {"role": "assistant", "content": "[사랑의 온도], [별의 계절], [마지막 편지] 등이 있습니다. 어떤 책이 궁금하신가요?"}
        ]
    )

    # 질문 최적화 예시
    question = "너무 한낮의 연애를 읽고 싶어."
    response = optimizer.optimize_response(question, num_books=1)
    print(response)
