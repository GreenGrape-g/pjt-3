from konlpy.tag import Okt
from collections import Counter

def extract_top_korean_words(text, n=5):
    # 텍스트 전처리 및 형태소 분석
    nouns = preprocess_korean_text(text)
    
    # 가장 빈도가 높은 단어 추출
    top_words = get_most_common_words(nouns, n)
    
    return top_words

# 사용 예시
text = "한국어 문서에서 가장 많이 나오는 단어를 추출하는 예제입니다. 한국어 처리는 영어와 다른 방식으로 접근해야 합니다."
top_5_words = extract_top_korean_words(text)
print(top_5_words)