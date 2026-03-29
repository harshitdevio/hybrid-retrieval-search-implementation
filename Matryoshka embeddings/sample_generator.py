import wikipedia
import os

# Create folder
os.makedirs("rag_docs", exist_ok=True)

topics = [
    "Artificial Intelligence", "Machine Learning", "Physics", "India",
    "History", "Biology", "Chemistry", "Mathematics", "Space",
    "Computer Science", "Philosophy", "Economics"
]

def chunk_text(text, chunk_size=300):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

doc_id = 0

for topic in topics:
    try:
        page = wikipedia.page(topic)
        chunks = list(chunk_text(page.content))

        for chunk in chunks:
            if doc_id >= 1000:
                break

            filename = f"rag_docs/doc_{doc_id}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(chunk)

            doc_id += 1

    except Exception as e:
        print(f"Skipping {topic}: {e}")

    if doc_id >= 1000:
        break

print(f"Created {doc_id} documents!")   