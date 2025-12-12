"""
Turkish News Dataset Preparation Module
"""

import pandas as pd
import json
import random
import os
from typing import List, Dict


class NewsDatasetPreparator:
    def __init__(self, min_samples: int = 500, max_samples: int = 2000):
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.instruction = "Verilen haber metnine uygun 8-12 kelimelik profesyonel bir Turkce haber basligi yaz."
    
    def is_valid_sample(self, content: str, headline: str) -> bool:
        words = headline.split()
        if len(words) < 6 or len(words) > 14:
            return False
        if len(content) < 200 or len(content) > 1200:
            return False
        if headline.count("!") > 1 or headline.count("?") > 2:
            return False
        return True
    
    def load_and_filter_data(self, xls_path: str) -> pd.DataFrame:
        print("Loading data from Excel file...")
        df = pd.read_excel(xls_path)
        df = df.dropna(subset=["content", "headline"])
        print(f"Raw samples: {len(df)}")
        
        df_filtered = df[df.apply(
            lambda row: self.is_valid_sample(
                str(row["content"]).strip(),
                str(row["headline"]).strip()
            ), axis=1
        )]
        print(f"Filtered samples: {len(df_filtered)}")
        
        if len(df_filtered) > self.max_samples:
            df_filtered = df_filtered.sample(self.max_samples, random_state=42)
            print(f"Sampled to: {len(df_filtered)}")
        
        return df_filtered
    
    def convert_to_sft_format(self, df: pd.DataFrame) -> List[Dict]:
        print("Converting to SFT format...")
        sft_data = []
        for _, row in df.iterrows():
            sft_data.append({
                "instruction": self.instruction,
                "input": str(row["content"]).strip(),
                "output": str(row["headline"]).strip()
            })
        random.shuffle(sft_data)
        print(f"Total samples: {len(sft_data)}")
        return sft_data
    
    def save_dataset(self, data: List[Dict], output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Dataset saved: {output_path}")
    
    def prepare_full_pipeline(self, xls_path: str, output_path: str):
        print("="*70)
        print("DATA PREPARATION STARTING")
        print("="*70)
        df = self.load_and_filter_data(xls_path)
        sft_data = self.convert_to_sft_format(df)
        self.save_dataset(sft_data, output_path)
        print("DATA PREPARATION COMPLETED")
        return sft_data
