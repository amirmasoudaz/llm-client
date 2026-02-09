from pathlib import Path
import asyncio
import json

from tqdm import tqdm

from src.db.session import DB
import pandas as pd


CPD = Path(__file__).resolve().parent.parent.parent
MAPPING_PATH = CPD / "data" / "tags_profs_mapping_to_map_stage.csv"


async def main():
    query = "SELECT * FROM funding_professors"
    rows = await DB.fetch_all(query)
    df = pd.read_csv(MAPPING_PATH)

    hash_to_id = {row["prof_hash"].hex(): row["id"] for row in rows}
    
    id_to_categories = {row["id"]: row.get("categories", "") for row in rows}
    
    matches = df[df["prof_id"].isin(hash_to_id.keys())]
    id_to_tags = {}
    for _, row in matches.iterrows():
        prof_id = row["prof_id"]
        db_id = hash_to_id[prof_id]
        tag = row["tag"]
        
        if db_id not in id_to_tags:
            id_to_tags[db_id] = []
        id_to_tags[db_id].append(tag)
    
    print(f"Found {len(matches)} matching tag-professor pairs")
    print(f"Found {len(id_to_tags)} unique professors with tags")

    updated_count = 0
    skipped_count = 0
    
    for prof_id, tags in tqdm(id_to_tags.items(), total=len(id_to_tags)):
        categories_str = id_to_categories.get(prof_id, "")
        if not categories_str:
            skipped_count += 1
            continue
        
        if categories_str == '["professor"]':
            tags_json = json.dumps(tags)
            
            update_query = "UPDATE funding_professors SET categories = %s WHERE id = %s"
            await DB.execute(update_query, (tags_json, prof_id))
            updated_count += 1
        else:
            skipped_count += 1

    print(f"\nSummary: Updated {updated_count} professors, skipped {skipped_count} professors")

    await DB.close()
    return id_to_tags

if __name__ == "__main__":
    asyncio.run(main())