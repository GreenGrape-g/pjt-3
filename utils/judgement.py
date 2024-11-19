from typing import Dict
import re

def is_about_books(response: str) -> bool:
    """
    챗봇의 응답이 책과 관련된지 여부를 판단합니다.

    Args:
        response (str): 챗봇의 응답 문자열

    Returns:
        bool: 책과 관련된 경우 True, 그렇지 않으면 False
    """
    book_keywords = [
        '책', '소설', '문학', '작가', '읽다', '독서',
        '출판사', '장르', '챕터', '이야기', '도서관', '베스트셀러',
        '추천', '저자', '서적', '도서'
    ]
    return any(keyword in response for keyword in book_keywords)

def is_about_author(response: str) -> bool:
    """
    챗봇의 응답이 작가와 관련된지 여부를 판단합니다.

    Args:
        response (str): 챗봇의 응답 문자열

    Returns:
        bool: 작가와 관련된 경우 True, 그렇지 않으면 False
    """
    # 작가 관련 질문을 더 잘 감지하기 위한 정규식 패턴 개선
    author_patterns = [
        r'작가\s*(는|가)?\s*누구',
        r'저자\s*(는|가)?\s*누구',
        r'누가\s*(썼|저술했)',
        r'글쓴이\s*(는|가)?\s*누구'
    ]
    for pattern in author_patterns:
        if re.search(pattern, response):
            return True
    return False

def is_about_negative(response: str) -> bool:
    """
    챗봇의 응답에 부정적인 단어가 포함되어 있는지 여부를 판단합니다.

    Args:
        response (str): 챗봇의 응답 문자열

    Returns:
        bool: 부정적인 단어가 포함된 경우 True, 그렇지 않으면 False
    """
    negative_keywords = [
        '불가능', '받고', '등', '중에서', '몇', '두 권을', '두 권', '네,', '아니요,', '두 가지', '세 가지', '또는,'
    ]
    return any(keyword in response for keyword in negative_keywords)

def decide_next_node(state: Dict) -> str:
    """
    현재 상태를 기반으로 다음 노드를 결정하는 함수입니다.

    Args:
        state (Dict): 현재 상태 정보를 담은 딕셔너리

    Returns:
        str: 다음 노드의 이름 ('optimization', 'end' 중 하나)
    """
    # 부정적인 응답을 먼저 확인
    if  state.get("is_negative", False):
        return "end"
    # 책 관련 질문을 확인
    elif state.get("is_book_question", False):
        return "optimization"
    # 부정적인 응답을 마지막에 확인
    # 작가 관련 질문을 마지막에 확인
    elif state.get("is_author_question", False):
        return "optimization"
    else:
        return "end"

# 예시로 상태 딕셔너리를 생성하여 함수 동작을 확인합니다.
if __name__ == "__main__":
    # 테스트 응답 문자열
    response = "이 책의 작가는 누구인가요?"

    # 상태 딕셔너리 생성
    state = {
        "is_book_question": is_about_books(response),
        "is_author_question": is_about_author(response),
        "is_negative": is_about_negative(response)
    }

    # 다음 노드 결정
    next_node = decide_next_node(state)
    print(f"다음 노드: {next_node}")
