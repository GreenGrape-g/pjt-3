# 책 추천 챗봇 웹 애플리케이션
이 프로젝트는 Flask와 OpenAI의 GPT-4 모델을 사용하여 책 추천 챗봇을 구현한 웹 애플리케이션입니다. 사용자의 질문에 따라 적절한 책을 추천하고, 네이버 API를 통해 실제 책 정보를 제공합니다.

## 주요 기능
책 추천 챗봇: 자연어 대화를 통해 사용자에게 맞는 책을 추천합니다.
네이버 API 연동: 추천된 책의 상세 정보를 제공합니다.
대화 이력 관리: 사용자의 대화 기록을 기반으로 더욱 정확한 추천을 제공합니다.
웹 인터페이스 제공: 사용자 친화적인 웹 기반 UI를 제공합니다.
## 설치 및 실행 방법

1. 리포지토리 클론
```
git clone https://github.com/yourusername/yourproject.git
cd yourproject 
```

2. 가상 환경 생성 및 활성화

```
# 가상 환경 생성
python -m venv venv
# 가상 환경 활성화
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate 
```

3. 의존성 설치
``` 
pip install -r requirements.txt 
```

4. 환경 변수 설정
프로젝트 루트 디렉토리에 .env 파일을 생성하고 다음 내용을 추가하세요:

``` 
NAVER_CLIENT_ID=your_naver_client_id
NAVER_CLIENT_SECRET=your_naver_client_secret
OPENAI_API_KEY=your_openai_api_key 
```
주의: 실제 API 키 값으로 대체해야 합니다.

5. 애플리케이션 실행
```
flask run
``` 
웹 브라우저에서 http://localhost:5000에 접속하여 애플리케이션을 사용합니다.

사용 방법
웹 브라우저를 통해 애플리케이션에 접속합니다.
챗봇 인터페이스에서 질문을 입력하여 책 추천을 받습니다.
추천된 책의 정보를 확인하고, 추가 질문을 통해 더 많은 정보를 얻을 수 있습니다.
기여 방법
프로젝트에 기여하고 싶으시다면 다음 절차를 따라주세요:

1. 이 리포지토리를 포크합니다.

2. 새로운 브랜치를 생성합니다:

``` 
git checkout -b feature/your-feature-name
```
3. 변경 사항을 커밋합니다:

```
git commit -m 'Add new feature'
```
4. 브랜치에 푸시합니다:

```
git push origin feature/your-feature-name
```
5. Pull Request를 생성합니다.

라이선스
이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 LICENSE 파일을 참고하세요.

문의
프로젝트 관련 문의는 greengrgr1102@gmail.com으로 연락해주세요.
