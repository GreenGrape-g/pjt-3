# utils/nlp_utils.py

from kiwipiepy import Kiwi
from collections import Counter

def extract_top_korean_words(text: str, n: int = 5) -> list:
    """
    주어진 한국어 텍스트에서 상위 N개의 단어를 추출합니다.

    Args:
        text (str): 분석할 한국어 텍스트.
        n (int): 추출할 상위 단어의 수.

    Returns:
        list: (단어, 빈도수) 튜플의 리스트.
    """
    kiwi = Kiwi()
    # 형태소 분석을 수행하여 명사만 추출
    tokens = [token.form for token in kiwi.tokenize(text) if token.tag.startswith('N')]
    # 빈도수를 계산하여 상위 N개 단어를 추출
    counter = Counter(tokens)
    top_words = counter.most_common(n)
    return top_words
