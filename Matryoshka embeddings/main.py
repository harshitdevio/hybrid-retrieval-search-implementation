from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import os

client = QdrantClient("http://localhost:6333")
model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)

client.create_collection(
    collection_name="matryoshka_test",
    vectors_config={
        "matryoshka-64dim": models.VectorParams(size=64, distance=models.Distance.COSINE),
        "matryoshka-128dim": models.VectorParams(size=128, distance=models.Distance.COSINE),
        "matryoshka-256dim": models.VectorParams(size=256, distance=models.Distance.COSINE),
    }
    
)

rag_docs_path =  "./rag_docs"
points = []

for idx, filename in enumerate(os.listdir(rag_docs_path)):
    if filename.endswith('.txt'):
        filepath = os.path.join(rag_docs_path, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        embedding = model.encode(text)
        
        points.append(
            models.PointStruct(
                id=idx,
                vector={
                    "matryoshka-64dim": embedding[:64].tolist(),
                    "matryoshka-128dim": embedding[:128].tolist(),
                    "matryoshka-256dim": embedding[:256].tolist(),
                },
                payload={"filename": filename, "text": text[:500]} 
            )
        )

print(f"Uploading {len(points)} documents...")
client.upsert(collection_name="matryoshka_test", points=points)

query = "your search query here"
query_embedding = model.encode(query)

results = client.query_points(
    collection_name="matryoshka_test",
    prefetch=[
        models.Prefetch(
            prefetch=[
                models.Prefetch(
                    query=query_embedding[:64].tolist(),
                    using="matryoshka-64dim",
                    limit=100,
                ),
            ],
            query=query_embedding[:128].tolist(),
            using="matryoshka-128dim",
            limit=50,
        )
    ],
    query=query_embedding[:256].tolist(),
    using="matryoshka-256dim",
    limit=10,
)

for point in results.points:
    print(f"\nScore: {point.score:.4f}")
    print(f"File: {point.payload['filename']}")
    print(f"Text: {point.payload['text'][:200]}...")