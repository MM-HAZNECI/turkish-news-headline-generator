# Turkish News Headline Generator

Fine-tuned LLM for generating professional Turkish news headlines from article content.

## Overview

This project implements Supervised Fine-Tuning (SFT) on Llama 3.2 1B model to generate professional Turkish news headlines. Using LoRA (Low-Rank Adaptation), the model learns to create concise, informative headlines from news article content.

## Features

- Specialized for Turkish language news headlines
- Fine-tuned on 1,500+ curated news samples
- Efficient training using LoRA/PEFT
- Interactive demo for headline generation
- Comprehensive evaluation framework

## Project Structure
```
turkish-news-headline-generator/
├── data/                  # Dataset files
├── models/                # Saved models
├── outputs/               # Training outputs
├── src/                   # Source code
│   ├── data_preparation.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── inference.py
└── main.py               # Main execution script
```

## Installation
```bash
pip install torch transformers datasets accelerate bitsandbytes trl peft
pip install unsloth @ git+https://github.com/unslothai/unsloth.git
pip install kagglehub pandas openpyxl
```

## Usage

### Training
```bash
python main.py
```

### Inference
```python
from src.inference import HeadlineGenerator

generator = HeadlineGenerator("models/llama32_turkish_news_ft")
headline = generator.generate("Your news article here...")
print(headline)
```

## Model Details

- **Base Model:** Llama 3.2 1B Instruct
- **Fine-Tuning Method:** LoRA (PEFT)
- **Dataset:** Turkish News Dataset (Kaggle)
- **Training Steps:** 800
- **LoRA Config:** r=16, alpha=16

## Results

The fine-tuned model shows significant improvement over the base model in generating relevant, concise Turkish headlines that match journalistic standards.
