import asyncio
import argparse
import httpx

BASE_URL_TEMPLATE = "http://127.0.0.1:4003/api/v1/funding/{}/{}"


async def fetch(client: httpx.AsyncClient, id_: int, action: str):
    url = BASE_URL_TEMPLATE.format(id_, action)
    try:
        resp = await client.get(url, timeout=20.0)
        print(f"ID {id_} -> {resp.status_code}")
        return resp.json()
    except Exception as e:
        print(f"ID {id_} failed: {e}")
        return None


async def process_batches(ids: list[int], batch_size: int, action: str):
    async with httpx.AsyncClient() as client:
        for i in range(0, len(ids), batch_size):
            batch = ids[i:i+batch_size]
            print(f"\nProcessing batch: {batch}")
            results = await asyncio.gather(
                *(fetch(client, id_, action) for id_ in batch)
            )
            for r in results:
                if r:
                    print(r)


def parse_ids(ids_str: str) -> list[int]:
    """Parse comma-separated or space-separated IDs."""
    # Try comma-separated first
    if ',' in ids_str:
        return [int(id_.strip()) for id_ in ids_str.split(',')]
    # Otherwise, split by spaces
    return [int(id_.strip()) for id_ in ids_str.split()]


def main():
    parser = argparse.ArgumentParser(
        description="Batch process funding IDs with send or review action"
    )
    parser.add_argument(
        'ids',
        type=str,
        help='Comma-separated or space-separated list of IDs (e.g., "6260,12205,12059" or "6260 12205 12059")'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Number of IDs to process in parallel per batch (default: 1)'
    )
    parser.add_argument(
        '--action',
        type=str,
        choices=['send', 'review'],
        default='review',
        help='Action to perform: "send" or "review" (default: review)'
    )
    
    args = parser.parse_args()
    
    try:
        ids = parse_ids(args.ids)
    except ValueError as e:
        print(f"Error parsing IDs: {e}")
        print("IDs must be integers separated by commas or spaces")
        return
    
    if not ids:
        print("Error: No valid IDs provided")
        return
    
    print(f"Processing {len(ids)} IDs with batch size {args.batch_size} and action '{args.action}'")
    asyncio.run(process_batches(ids, args.batch_size, args.action))


if __name__ == "__main__":
    main()
