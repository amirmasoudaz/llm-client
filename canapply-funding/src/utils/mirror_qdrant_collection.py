import time
from src.config import settings
from tqdm import tqdm
from qdrant_client import QdrantClient, models

# ---- config ----
SOURCE_URL = os.getenv("SOURCE_URL", settings.QDRANT_URL)
TARGET_URL = os.getenv("TARGET_URL", "http://127.0.0.1:6333")
COLLECTION = os.getenv("COLLECTION", "tag_indexer_tagger_logs")
SOURCE_API_KEY = os.getenv("SOURCE_API_KEY") or settings.QDRANT_API_KEY
TARGET_API_KEY = os.getenv("TARGET_API_KEY")

BATCH = int(os.getenv("BATCH", "1000"))
RETRIES = int(os.getenv("RETRIES", "5"))
SLEEP = float(os.getenv("SLEEP", "0.5"))

# ---- helpers ----
def ensure_target_collection(dst: QdrantClient, size: int = 1):
    if dst.collection_exists(COLLECTION):
        dst.delete_collection(COLLECTION)
    dst.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=size, distance=models.Distance.DOT),
        on_disk_payload=True,
    )

def count_points(client: QdrantClient) -> int:
    try:
        return client.count(COLLECTION, exact=True).count
    except Exception:
        return 0

# ---- main ----
def main():
    src = QdrantClient(url=SOURCE_URL, api_key=SOURCE_API_KEY, timeout=120)
    dst = QdrantClient(url=TARGET_URL, api_key=TARGET_API_KEY, timeout=120)

    src_count = count_points(src)
    print(f"[SOURCE] {COLLECTION}: {src_count} points")
    print("[TARGET] Recreating target collection ...")
    ensure_target_collection(dst, size=1)

    offset = None
    total = 0
    pbar = tqdm(total=src_count, desc="Syncing", unit="pts")

    while True:
        records, offset = src.scroll(
            collection_name=COLLECTION,
            with_payload=True,
            with_vectors=True,
            limit=BATCH,
            offset=offset,
        )
        if not records:
            break

        points = [models.PointStruct(id=r.id, vector=r.vector, payload=r.payload) for r in records]

        # Retry logic
        for attempt in range(RETRIES):
            try:
                dst.upsert(collection_name=COLLECTION, points=points, wait=True)
                break
            except Exception as e:
                if attempt == RETRIES - 1:
                    raise
                print(f"⚠️  Retry {attempt+1}/{RETRIES} after error: {e}")
                time.sleep(SLEEP * (attempt + 1))

        total += len(points)
        pbar.update(len(points))

    pbar.close()
    dst_count = count_points(dst)
    print(f"\n✅ Completed! Target now has {dst_count} points (source had {src_count})")

if __name__ == "__main__":
    main()
