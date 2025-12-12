"""
Model Evaluation Module
"""

from unsloth import FastLanguageModel
from typing import List, Dict


class ModelEvaluator:
    def __init__(self):
        self.alpaca_prompt = """### Instruction:
{}

### Input:
{}

### Response:
"""
        self.instruction = "Verilen haber metnine uygun 8-12 kelimelik profesyonel bir Turkce haber basligi yaz."
    
    def load_base_model(self):
        print("Loading base model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Llama-3.2-1B-Instruct",
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    
    def load_finetuned_model(self, model_path: str):
        print(f"Loading fine-tuned model from: {model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    
    def generate_headline(self, model, tokenizer, article: str) -> str:
        inputs = tokenizer([self.alpaca_prompt.format(self.instruction, article)], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=24, do_sample=False)
        text = tokenizer.batch_decode(outputs)[0]
        
        if "### Response:" in text:
            text = text.split("### Response:")[-1].strip()
        for marker in ["<|end_of_text|>", "<|eot_id|>"]:
            if marker in text:
                text = text.split(marker)[0].strip()
        text = text.split("\n")[0].strip()
        
        return text
    
    def compare_models(self, base_model, base_tokenizer, ft_model, ft_tokenizer, test_articles: List[str]) -> List[Dict]:
        print("="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        results = []
        for i, article in enumerate(test_articles, 1):
            print(f"\nTest {i}/{len(test_articles)}")
            print(f"Article: {article[:80]}...")
            
            base_headline = self.generate_headline(base_model, base_tokenizer, article)
            ft_headline = self.generate_headline(ft_model, ft_tokenizer, article)
            
            print(f"BASE: {base_headline}")
            print(f"FINE-TUNED: {ft_headline}")
            
            results.append({
                "article": article,
                "base_headline": base_headline,
                "finetuned_headline": ft_headline
            })
        
        return results
