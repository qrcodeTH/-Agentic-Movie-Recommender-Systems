# config.py
# Handles model loading, data processing, and global constants.

import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import List

print("--- Initializing Configuration ---")

# --- A) Main Generative LLM (Qwen) ---
print("🔧 Initializing the Qwen3 4B Language Model...")
model_id = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    load_in_4bit=True,
)
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=4096,
    return_full_text=False
)
print("✅ Qwen LLM Initialized.")


# --- B) Movie Data & Knowledge Base ---
print("🎬 Loading and Processing Movie Dataset...")
MOVIE_DATA_PATH = '/kaggle/input/tmdb-movies-dataset-2023-930k-movies/TMDB_movie_dataset_v11.csv'

def parse_comma_separated_string(data_str: str) -> List[str]:
    """Parses a comma-separated string into a list of items."""
    if isinstance(data_str, str) and data_str:
        return [item.strip() for item in data_str.split(',')]
    return []

  df = pd.read_csv(MOVIE_DATA_PATH)
  print("Parsing genres and keywords...")
  df['genres_list'] = df['genres'].apply(parse_comma_separated_string)
  df['keywords_list'] = df['keywords'].apply(parse_comma_separated_string)
  df['overview'] = df['overview'].fillna('')
  df = df.dropna(subset=['title', 'vote_average'])
  print("✅ Movie Dataset Loaded.")

# Create the master list of all genres for the LLM prompt
if not df.empty:
    all_genres = set()
    for genres_list in df['genres_list']:
        for genre in genres_list:
            all_genres.add(genre.lower())
    MOVIE_GENRES = all_genres
    print(f"Found {len(df)} movies and {len(MOVIE_GENRES)} unique genres.")
else:
    MOVIE_GENRES = set()
    print("Skipping genre processing due to missing data file.")

print("--- Configuration Loaded ---")
