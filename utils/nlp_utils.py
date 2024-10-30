# utils/nlp_utils.py

import re
from typing import List, Tuple
from kiwipiepy import Kiwi
from collections import Counter

def extract_top_korean_phrases(text: str, n: int = 5) -> List[Tuple[str, int]]:
    """
    주어진 한국어 텍스트에서 상위 N개의 중요한 명사구를 추출합니다.
    강조된 내용 (예: '' 또는 <>)을 우선으로 추출합니다.

    Args:
        text (str): 분석할 한국어 텍스트.
        n (int): 추출할 상위 명사구의 수.

    Returns:
        List[Tuple[str, int]]: (명사구, 빈도수) 튜플의 리스트.
    """
    kiwi = Kiwi()
    tokens = kiwi.tokenize(text)

    # 1. 문장 내 강조된 구 추출
    # 추출할 기호: '', "", <>, `` 등
    # regex 패턴: '([^']+)'|"([^"]+)"|<([^>]+)>
    pattern = re.compile(r"'([^']+)'|\"([^\"]+)\"|<([^>]+)>")
    highlighted_phrases = re.findall(pattern, text)

    # Process highlighted phrases
    # 각 매치는 하나의 그룹만 채워짐
    special_phrases = []
    for match in highlighted_phrases:
        phrase = next(filter(None, match))
        phrase = phrase.strip()
        if phrase:
            special_phrases.append(phrase)

    # Initialize counter
    counter = Counter()

    # Add special phrases with higher counts
    for phrase in special_phrases:
        # 전체 구문을 하나의 단위로 처리
        counter[phrase] += 10  # 높은 우선순위 부여

    # 2. 텍스트 전체에서 명사구 추출
    noun_phrases = []
    current_phrase = []

    for token in tokens:
        word = token.form
        pos = token.tag

        if pos.startswith('NN'):  # 일반 명사(NNG), 고유 명사(NNP) 등
            if current_phrase and pos.startswith('NN'):
                current_phrase.append(word)
            else:
                current_phrase = [word]
        elif pos.startswith('JK'):  # 조사(JKO, JKC 등)
            if current_phrase:
                current_phrase.append(word)
        else:
            if len(current_phrase) >= 2:  # 최소 2개 이상의 단어로 구성된 명사구
                phrase = ''.join(current_phrase)
                noun_phrases.append(phrase)
                counter[phrase] += 1
            current_phrase = []

    # 마지막에 남은 명사구 처리
    if len(current_phrase) >= 2:
        phrase = ''.join(current_phrase)
        noun_phrases.append(phrase)
        counter[phrase] += 1

    # 3. 빈도수를 기준으로 상위 N개 문장 선택
    top_phrases = counter.most_common(n)

    return top_phrases

