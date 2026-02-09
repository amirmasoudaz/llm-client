# src/recommender/tag_constants.py

from __future__ import annotations

# Shared lexical helpers for tagging / recommendation logic.

ALIAS_MAP: dict[str, list[str]] = {
    # AI / ML / Data
    "ai": ["artificial intelligence", "computational intelligence"],
    "ml": ["machine learning", "statistical learning", "probabilistic modeling"],
    "dl": ["deep learning", "neural networks", "deep neural networks"],
    "rl": ["reinforcement learning", "deep reinforcement learning"],
    "cv": ["computer vision", "machine vision", "image processing", "image analysis"],
    "nlp": ["natural language processing", "computational linguistics", "text mining", "speech processing", "speech recognition"],
    "ir": ["information retrieval", "web search", "search engines"],
    "dm": ["data mining", "knowledge discovery"],
    "ds": ["data science", "data analytics", "big data"],
    "comm": ["recommender systems", "recommendation systems"],
    
    # Human-Centric
    "hci": ["human computer interaction", "human-computer interaction", "user interface", "user experience", "social computing"],
    "vis": ["data visualization", "information visualization", "scientific visualization", "visual analytics"],
    "ar": ["augmented reality"],
    "vr": ["virtual reality"],
    "xr": ["extended reality", "mixed reality"],

    # Systems & Networking
    "sys": ["computer systems", "distributed systems", "operating systems", "storage systems", "cloud computing"],
    "os": ["operating systems"],
    "net": ["computer networking", "network protocols", "computer networks", "wireless networks", "mobile computing"],
    "hpc": ["high performance computing", "supercomputing", "parallel computing"],
    "iot": ["internet of things", "cyber-physical systems", "sensor networks"],
    "db": ["databases", "database systems", "data management", "transaction processing"],

    # Security & Crypto
    "sec": ["cybersecurity", "computer security", "network security", "information security", "privacy"],
    "crypto": ["cryptography", "cryptology", "crypto"],

    # Software & Theory
    "se": ["software engineering", "software testing", "software verification", "formal methods"],
    "pl": ["programming languages", "compilers", "program analysis", "type theory"],
    "alg": ["algorithms", "algorithm design", "computational complexity", "theory of computation"],
    "log": ["logic in computer science", "computational logic"],

    # Hardware & Architecture
    "arch": ["computer architecture", "processor design", "hardware accelerators"],
    "cad": ["computer aided design", "electronic design automation", "vlsi"],

    # Interdisciplinary / Applied
    "rob": ["robotics", "autonomous systems", "control systems"],
    "bio": ["bioinformatics", "computational biology", "genomics"],
    "med": ["medical informatics", "health informatics", "medical imaging"],
    "fin": ["computational finance", "fintech"],
    "edu": ["computer science education", "educational technology"],
    "cg": ["computer graphics", "rendering", "animation", "computational photography"],
    "qc": ["quantum computing", "quantum information", "quantum algorithms"],
}

ACRONYM_STOPWORDS = {
    # Common articles/conjunctions/prepositions
    "and", "of", "for", "in", "to", "on", "the", "a", "an", "with",
    "at", "by", "from", "up", "about", "into", "over", "after", "under", "above",
    "is", "are", "was", "were", "be", "been", "being",
    "or", "but", "nor", "so", "yet", "as",
    
    # Pronouns/determiners
    "this", "that", "these", "those", "which", "what", "whose", "who", "whom", 
    "my", "your", "his", "her", "its", "our", "their",
    
    # Common functional words in vague academic tags
    "based", "using", "via", "towards", "through", "during", "within", "without",
    "introduction", "principles", "foundations", "advanced", "applied", "theoretical",
    "study", "analysis", "methods", "systems", "technologies", "applications",
    "approach", "approaches", "perspective", "perspectives",
}
