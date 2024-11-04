# session_manager.py

class ChatbotSession:
    """대화 상태를 관리하는 클래스"""
    
    def __init__(self):
        self.state = {
            "messages": [],  # 대화 내역 저장
            "context": {}    # 추가 정보를 저장할 수 있는 공간
        }
    
    def add_message(self, role, content):
        """새로운 메시지를 추가하여 대화 기록을 업데이트합니다."""
        self.state["messages"].append({"role": role, "content": content})
    
    def get_last_user_message(self):
        """사용자의 마지막 메시지를 반환합니다."""
        user_messages = [msg for msg in self.state["messages"] if msg["role"] == "user"]
        return user_messages[-1]["content"] if user_messages else None
    
    def reset_session(self):
        """세션 초기화 (새로운 대화를 위해 기록 삭제)"""
        self.state = {"messages": [], "context": {}}
