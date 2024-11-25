import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Union
import torch

class TimeSeriesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, stride: int = 1, inference: bool = False) -> None:
        """
        Args:
            df: 입력 데이터프레임
            stride: 윈도우 스트라이드
            inference: 추론 모드 여부
        """
        self.inference = inference
        self.column_names = df.filter(regex='^P\\d+$').columns.tolist()
        self.file_ids = df['file_id'].values if 'file_id' in df.columns else None
        
        if inference:
            self.values = df[self.column_names].values.astype(np.float32)
            self._prepare_inference_data()
        else:
            self._prepare_training_data(df, stride)
            
    def _normalize_columns(self, data: np.ndarray) -> np.ndarray:
        """벡터화된 열 정규화"""
        mins = data.min(axis=0, keepdims=True)
        maxs = data.max(axis=0, keepdims=True)
        
        # mins와 maxs가 같으면 전체를 0으로 반환
        is_constant = (maxs == mins)
        if np.any(is_constant):
            normalized_data = np.zeros_like(data)
            normalized_data[:, is_constant.squeeze()] = 0
            return normalized_data
        
        # 정규화 수행
        return (data - mins) / (maxs - mins)
    
    def _prepare_inference_data(self) -> None:
        """추론 데이터 준비 - 단일 시퀀스"""
        self.normalized_values = self._normalize_columns(self.values)

    def _prepare_training_data(self, df: pd.DataFrame, stride: int, WINDOW_GIVEN:int = 10080) -> None:
        """학습 데이터 준비 - 윈도우 단위"""
        self.values = df[self.column_names].values.astype(np.float32)
        
        # 시작 인덱스 계산 (stride 적용)
        potential_starts = np.arange(0, len(df) - WINDOW_GIVEN, stride)
        
        # 각 윈도우의 마지막 다음 지점(window_size + 1)이 사고가 없는(0) 경우만 필터링
        accident_labels = df['anomaly'].values
        valid_starts = [
            idx for idx in potential_starts 
            if idx + WINDOW_GIVEN < len(df) and  # 범위 체크
            accident_labels[idx + WINDOW_GIVEN] == 0  # 윈도우 다음 지점 체크
        ]
        self.start_idx = np.array(valid_starts)
        
        # 유효한 윈도우들만 추출하여 정규화
        windows = np.array([
            self.values[i:i + WINDOW_GIVEN] 
            for i in self.start_idx
        ])
        
        # (윈도우 수, 윈도우 크기, 특성 수)로 한번에 정규화
        self.input_data = np.stack([
            self._normalize_columns(window) for window in windows
        ])
        
    def __len__(self) -> int:
        if self.inference:
            return len(self.column_names)
        return len(self.start_idx) * len(self.column_names)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        if self.inference:
            col_idx = idx
            col_name = self.column_names[col_idx]
            col_data = self.normalized_values[:, col_idx]
            file_id = self.file_ids[idx] if self.file_ids is not None else None
            return {
                "column_name": col_name,
                "input": torch.from_numpy(col_data).unsqueeze(-1),  # (time_steps, 1)
                "file_id": file_id
            }
        
        window_idx = idx // len(self.column_names)
        col_idx = idx % len(self.column_names)
        
        return {
            "column_name": self.column_names[col_idx],
            "input": torch.from_numpy(self.input_data[window_idx, :, col_idx]).unsqueeze(-1)
        }