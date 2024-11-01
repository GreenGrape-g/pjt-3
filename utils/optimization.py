# utils/optimization.py
import os
import requests
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import re
import logging

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)

# 환경 변수 로드
load_dotenv()

# 시스템 프롬프트 설정
system_prompt = """당신은 도움이 되는 도서 추천 전문가입니다. 
외국 도서를 포함한 모든 상황에서 반드시 한국어로만 답변해주세요.
영어나 다른 외국어로 된 답변은 절대 하지 마세요."""

# 언어 모델 초기화
tone_style_llm = ChatOpenAI(model="gpt-4", temperature=0)

class Optimization:
    def __init__(self, tone, style, num_books=1, additional_instructions=None, conversation_history=None):
        self.tone = tone
        self.style = style
        self.num_books = num_books
        self.additional_instructions = additional_instructions or "한국어로만 답변해주세요"
        self.conversation_history = conversation_history or []
        
        # 네이버 API 자격 증명 로드
        self.naver_client_id = os.getenv('NAVER_CLIENT_ID')
        self.naver_client_secret = os.getenv('NAVER_CLIENT_SECRET')
        
        if not self.naver_client_id or not self.naver_client_secret:
            raise ValueError("NAVER_CLIENT_ID 및 NAVER_CLIENT_SECRET 환경 변수를 설정해주세요.")
        
        self.optimization_system = f"""당신은 사용자의 질문에 대해 친절하지만 전문적으로 답변하는 도서 전문가입니다. 

언어 관련 필수 지침:
- 모든 상황에서 반드시 한국어로만 답변하세요.
- 외국 도서나 작가에 대해 이야기할 때도 반드시 한국어로 답변하세요.
- 책 제목은 한국어로 번역된 제목을 우선적으로 사용하되, 원제목을 괄호 안에 표기하세요.
- 작가 이름은 한국어 표기를 먼저 쓰고 원어 이름을 괄호 안에 표기하세요.
- 영어나 다른 외국어로 된 답변은 절대 금지입니다.

사용자의 다양한 질문에 유연하게 대응하되, 책을 추천해달라는 요청이 들어오면 아래의 구조를 따라 총 {self.num_books}권의 유일한 책을 추천하세요:

주의사항:
- 모든 답변은 반드시 한국어로만 작성하세요.
- 절대로 영어로 답변하지 마세요.
- 반드시 실제로 존재하는 책만 추천하세요.
- 추천할 책은 네이버 API를 사용하여 검색 결과가 있는 책으로 한정하세요.
- 동일한 책을 여러 번 추천하지 마세요.
- 이전 대화 내용을 참고하여 답변하세요.
- 각 항목이 끝날 때마다 줄바꿈을 하세요.
- 책 제목을 추출하기 쉽게 큰따옴표로 감싸주세요.
- 책을 제외한 다른 미디어는 추천하지 못합니다.
- 각 답변을 완료한 뒤에는 엔터로 구분해주세요.
- 답변은 한국어로 해주세요.
- 적절한 답변을 찾을 수 없는 경우에는 "죄송하지만 관련된 책을 찾을 수 없습니다."로 해주세요.

추가 사항:
- 책과 관련 없는 질문이 들어올 경우, 자연스럽고 친절하게 대화를 이어가며 응답하세요.
"""

        # 메시지 템플릿 리스트 생성
        messages = [
            ("system", self.optimization_system)
        ]

        # 대화 기록을 메시지 리스트에 추가
        for msg in self.conversation_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                messages.append(("human", content))
            elif role == "assistant":
                messages.append(("ai", content))

        # 마지막에 추가 지시사항과 질문을 포함한 human 메시지 추가
        messages.append(
            (
                "human",
                "생성된 질문: {question}\n\n추가 지시사항: {additional_instructions}"
            )
        )

        # 최적화를 위한 채팅 프롬프트 템플릿 정의
        self.optimization_prompt = ChatPromptTemplate.from_messages(messages)

        # 구조화된 LLM 생성
        self.structured_optimizer = tone_style_llm

    def optimize_response(self, question):
        logging.debug(f"Optimizing response for question: {question}")
        # 프롬프트에 필요한 변수를 전달
        prompt_data = self.optimization_prompt.format_prompt(
            question=question,
            additional_instructions=self.additional_instructions or "없음"
        )
        logging.debug(f"Formatted prompt: {prompt_data}")

        # 최적화된 응답 생성
        optimized_response = self.structured_optimizer(prompt_data.to_messages()).content.strip()
        logging.debug(f"Optimized response from LLM: {optimized_response}")

        # 최적화된 응답에서 책 제목들 추출
        book_titles = self.extract_book_titles(optimized_response)
        logging.debug(f"Extracted book titles: {book_titles}")

        # 중복된 책 제목 제거
        unique_book_titles = list(set(book_titles))
        logging.debug(f"Unique book titles: {unique_book_titles}")

        # 네이버 API를 사용하여 책 정보 가져오기
        if unique_book_titles:
            book_info_list = []
            valid_titles = []
            for title in unique_book_titles:
                search_results = self.search_book_info(title)
                if search_results:
                    # 이미 추가된 책 정보인지 확인
                    if search_results[0]['title'] not in [book['title'] for book in book_info_list]:
                        book_info_list.append(search_results[0])  # 첫 번째 결과 사용
                        valid_titles.append(title)
                else:
                    logging.warning(f"'{title}'에 대한 검색 결과가 없습니다.")

            if book_info_list:
                # 존재하는 책들로 최적화된 응답을 수정
                optimized_text = self.rewrite_response(optimized_response, valid_titles)
                logging.debug(f"Rewritten optimized text: {optimized_text}")

                # 책 정보를 최적화된 응답에 통합
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

    def extract_book_titles(self, text):
        """
        주어진 텍스트에서 책 제목들 추출.
        """
        # 책 제목이 큰따옴표로 감싸져 있다고 가정
        titles = re.findall(r'"([^"]+)"', text)
        # 중복 제거
        unique_titles = list(set(titles))
        logging.debug(f"Extracted unique titles from text: {unique_titles}")
        return unique_titles

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
        rewritten_text = '\n'.join(new_lines)
        logging.debug(f"Rewritten response: {rewritten_text}")
        return rewritten_text

    def insert_book_info(self, text, book_info_list):
        """
        책 정보를 텍스트에 삽입.
        필수 구성 요소: 책 이미지, 책 제목, 작가, 출판사, 추천 이유, 구매 링크.
        """
        # 텍스트에서 기존 책 정보 제거
        keys_to_remove = ['책 이미지', '책 제목', '작가', '출판사', '추천 이유', '구매 링크']
        for key in keys_to_remove:
            pattern = f"^{key}:.*$"
            text = re.sub(pattern, '', text, flags=re.MULTILINE)

        updated_text = ' '.join(text.split())

        # 이미 책 이미지가 삽입된 응답인 경우 바로 반환
        if re.search(r'책 이미지:', updated_text):
            return updated_text

        # 고유 책 정보 필터링
        seen_titles = set()
        unique_book_info = []
        for book_info in book_info_list:
            title = book_info['title']
            if title not in seen_titles:
                unique_book_info.append(book_info)
                seen_titles.add(title)

        # 필수 정보만 포함한 책 세부사항 생성
        book_details_list = []
        for book_info in unique_book_info:
            # HTML 태그 제거 및 소제목 제거
            title = re.sub('<[^<]+?>', '', book_info['title']).split('(')[0].strip()
            author = book_info['author']
            description = book_info.get("description", "상세 설명을 찾을 수 없습니다.")
            summary = self.summarize_text(description, 2)

            # 구매 링크 생성
            encoded_title = requests.utils.quote(title)
            kyobo_link = f"https://search.kyobobook.co.kr/search?keyword={encoded_title}"
            aladin_link = f"https://www.aladin.co.kr/search/wsearchresult.aspx?SearchTarget=All&SearchWord={encoded_title}"
            yes24_link = f"https://www.yes24.com/Product/Search?query={encoded_title}"

            purchase_links = f"[교보문고]({kyobo_link}), [알라딘]({aladin_link}), [예스24]({yes24_link})"

            # 작가 이름 처리
            authors = [a.strip() for a in author.split(',')]
            formatted_authors = []
            for author_name in authors:
                # 번역자 정보가 있는 경우 별도 처리
                if '옮긴이' in author_name or '역' in author_name:
                    formatted_authors.append(author_name)
                else:
                    # 한글 이름이 있는 경우와 없는 경우 처리
                    if any('\u3131' <= c <= '\u318F' or '\uAC00' <= c <= '\uD7A3' for c in author_name):
                        formatted_authors.append(author_name)
                    else:
                        # 외국 작가의 경우 원어 이름을 괄호 안에 표시
                        # 원어 이름을 별도로 제공하지 않으므로 그대로 사용
                        formatted_authors.append(author_name)

            author_text = ', '.join(formatted_authors)

            # 책 세부정보 템플릿
            book_details = (
                f"책 이미지: ![책 이미지]({book_info.get('image', '')})\n"
                f"책 제목: \"{title}\"\n"  # 큰 따옴표로 책 제목 감싸기
                f"작가: '{author_text}'\n"    # 작은 따옴표로 작가 이름 감싸기
                f"출판사: {book_info.get('publisher', '출판사 정보 없음')}\n"
                f"추천 이유: {summary}\n"
                f"구매 링크: {purchase_links}"
            )
            book_details_list.append(book_details)

        # 최종 응답 생성
        book_details_text = '\n\n'.join(book_details_list)
        follow_up = "\n\n더 궁금한 점이 있으시면 말씀해주세요!"
        final_response = f"{book_details_text}{follow_up}"
        logging.debug(f"Final response constructed: {final_response}")
        return final_response

    def summarize_text(self, text, num_sentences):
        """
        텍스트를 주어진 문장 수로 요약합니다.

        :param text: 요약할 텍스트
        :param num_sentences: 원하는 문장 수
        :return: 요약된 텍스트
        """
        # 텍스트가 없으면 기본 메시지 반환
        if not text:
            return "상세한 내용은 링크를 참고해주세요."

        # 마침표 기준으로 문장 분리
        sentences = re.split(r'(?<=[.!?]) +', text)
        short_description = ' '.join(sentences[:num_sentences])
        logging.debug(f"Summarized text: {short_description}")
        return short_description

    def search_book_info(self, query):
        """
        네이버 검색 API를 사용하여 책 정보 가져오기.
        한글 도서를 우선적으로 검색하고, 결과가 없을 경우 원어 검색을 시도합니다.
        """
        headers = {
            "X-Naver-Client-Id": self.naver_client_id,
            "X-Naver-Client-Secret": self.naver_client_secret,
        }

        def get_search_results(search_query, display_count=10):
            params = {
                "query": search_query,
                "display": display_count,  # 더 많은 결과를 가져와서 필터링
                "sort": "sim"  # 정확도 순 정렬
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

        # 1. 한글 제목으로 우선 검색
        korean_title = query
        if "(" in query:  # 원제가 괄호 안에 있는 경우
            korean_title = query.split("(")[0].strip()

        results = get_search_results(korean_title)

        # 결과 필터링 및 정렬
        filtered_results = []
        for item in results:
            title = item.get("title", "").replace("<b>", "").replace("</b>", "")
            author = item.get("author", "")
            publisher = item.get("publisher", "")
            image = item.get("image", "")
            description = item.get("description", "")
            link = item.get("link", "")
            isbn = item.get("isbn", "")

            score = 0
            # 제목에 한글이 포함된 경우 높은 점수
            if any('\u3131' <= c <= '\u318F' or '\uAC00' <= c <= '\uD7A3' for c in title):
                score += 3
            # 출판사가 한글인 경우 가산점
            if any('\u3131' <= c <= '\u318F' or '\uAC00' <= c <= '\uD7A3' for c in publisher):
                score += 2

            # 검색어와 제목의 유사도 점수 추가
            if korean_title.lower() in title.lower():
                score += 2

            # 작가 이름이 검색어에 포함되면 추가 점수
            if korean_title.lower() in author.lower():
                score += 2

            filtered_results.append({
                "title": title,
                "author": author,
                "publisher": publisher,
                "image": image,
                "description": description,
                "link": link,
                "isbn": isbn,
                "score": score
            })

        # 점수에 따라 정렬
        filtered_results.sort(key=lambda x: x["score"], reverse=True)

        # 결과가 없거나 점수가 너무 낮은 경우 다른 검색 시도
        if not filtered_results or filtered_results[0]["score"] < 2:
            logging.debug("한글 도서 검색 결과가 없거나 부적절하여 추가 검색 시도")
            # 여기서 필요한 경우 다른 검색 전략을 시도할 수 있습니다
            return []

        # score 필드 제거 후 반환
        for result in filtered_results:
            del result["score"]

        return filtered_results[:1]  # 가장 적절한 결과 하나만 반환
