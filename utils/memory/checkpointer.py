import json
import os

class SimpleCheckpointer:
    def __init__(self, checkpoint_file="checkpoint.json"):
        self.checkpoint_file = checkpoint_file

    def save_state(self, state: dict):
        """
        상태를 파일로 저장하는 함수.
        :param state: 저장할 상태 (딕셔너리 형태)
        """
        with open(self.checkpoint_file, 'w') as f:
            json.dump(state, f)
        print(f"State saved to {self.checkpoint_file}")

    def load_state(self) -> dict:
        """
        저장된 상태를 불러오는 함수.
        :return: 불러온 상태 (딕셔너리 형태)
        """
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                state = json.load(f)
            print(f"State loaded from {self.checkpoint_file}")
            return state
        else:
            print("No checkpoint found. Starting fresh.")
            return {}

    def clear_checkpoint(self):
        """
        체크포인트 파일을 삭제하는 함수 (필요시 호출).
        """
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            print(f"Checkpoint {self.checkpoint_file} cleared.")

# 사용 예시
if __name__ == "__main__":
    checkpointer = SimpleCheckpointer()

    # 예: 작업 상태
    current_state = {
        'step': 3,
        'progress': '50%',
        'data': {'name': 'example', 'value': 42}
    }

    # 상태 저장
    checkpointer.save_state(current_state)

    # 상태 불러오기
    loaded_state = checkpointer.load_state()
    print("Loaded state:", loaded_state)
