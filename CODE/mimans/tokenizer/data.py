from datasets import load_dataset
import os
import time

OUTPUT_DIR = "code_tokenizer_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Languages ordered by popularity
LANGUAGES = [
    ("python", "python"),
    ("javascript", "javascript"),
    ("typescript", "typescript"),
    ("java", "java"),
    ("cpp", "cpp"),
    ("csharp", "c-sharp"),
    ("rust", "rust"),
]

MAX_FILES = 25000   # Safe number for 500M model

for name, folder in LANGUAGES:
    print(f"\n=== Downloading {name.upper()} ===")
    
    try:
        ds = load_dataset(
            "bigcode/starcoderdata",
            data_dir=folder,
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        
        count = 0
        output_file = os.path.join(OUTPUT_DIR, f"{name}.txt")
        
        with open(output_file, "w", encoding="utf-8", errors="ignore") as f:
            for sample in ds:
                try:
                    # Try different possible content keys
                    content = (
                        sample.get("content") or 
                        sample.get("text") or 
                        sample.get("code") or 
                        sample.get("source")
                    )
                    
                    if content and isinstance(content, str) and len(content.strip()) > 80:
                        f.write(content.strip() + "\n\n" + "="*80 + "\n\n")
                        count += 1
                        
                        if count % 2000 == 0:
                            print(f"  → {name}: {count} files written")
                        
                        if count >= MAX_FILES:
                            break
                            
                except Exception as e:
                    # Skip bad samples silently
                    continue
        
        print(f"Finished {name}: {count} files saved → {output_file}")
        
    except Exception as e:
        print(f"Failed to download {name}: {e}")
        print("Skipping this language and continuing...\n")
        time.sleep(2)  # small delay before next language
        continue

print("\n=== All downloads finished ===")