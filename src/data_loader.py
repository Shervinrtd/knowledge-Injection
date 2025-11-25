import pandas as pd
from datasets import load_dataset
import os

def download_and_process_data():
    print("  Downloading PubMedQA dataset...")
    
    # We load the "pqa_labeled" subset (human-labeled, high quality)
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    
    # We want to extract: Question, Context (The text), and Long Answer
    data = []
    for item in dataset:
        row = {
            "id": item["pubid"],
            "question": item["question"],
            "context": "".join(item["context"]["contexts"]), # Combine context snippets
            "answer": item["long_answer"]
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save to the processed folder
    output_path = "data/processed/pubmed_qa_clean.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f" Data saved to {output_path}")
    print(f" Total samples: {len(df)}")
    print("Here is a preview:")
    print(df.head(2))

if __name__ == "__main__":
    download_and_process_data()
