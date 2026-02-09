from blake3 import blake3


def get_hash(content: bytes | str, length: int = 16) -> str:
    if not length:
        length = None
    if isinstance(content, bytes):
        return blake3(content).hexdigest()[:length]
    return blake3(content.encode("utf-8")).hexdigest()[:length]