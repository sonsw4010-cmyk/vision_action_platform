from pathlib import Path
from typing import List
import joblib
import json

class ModelLoader:
    def __init__(self):
        self.model = None
        self.feature_columns: List[str] = []
        self.classes: List[str] = []
        self.sequence_steps: int = 30
        self.is_loaded: bool = False

    def load_model(self) -> None:
        model_dir = Path(__file__).resolve().parent.parent / "models"
        
        # 우리가 직접 학습한 17개 동작 모션캡처 모델
        model_path = model_dir / "rf_action_motion_capture_full.joblib"
        feature_cols_path = model_dir / "rf_feature_columns.json"

        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

        print(f"[ModelLoader] 17개 동작 모델 로드: {model_path.name}")
        self.model = joblib.load(model_path)
        self.classes = list(getattr(self.model, "classes_", []))

        if feature_cols_path.exists():
            with open(feature_cols_path, "r", encoding="utf-8") as f:
                self.feature_columns = json.load(f)

        self.sequence_steps = 30
        self.is_loaded = True
        print(f"[ModelLoader] 모델 로드 완료: {model_path.name}")
        print(f"[ModelLoader] 17개 클래스: {self.classes}")
        print(f"[ModelLoader] Feature 컬럼: {len(self.feature_columns)}개 검증 완료")

model_loader = ModelLoader()
