# src/agents/memory/engine.py
"""Memory Agent - Manages user preferences and long-term memory with semantic search."""

from __future__ import annotations

import json
import struct
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, Tuple
from dataclasses import dataclass

from llm_client import OpenAIClient, GPT5Mini, TextEmbedding3Small

from src.services.db import DatabaseService
from src.config import get_settings
from src.tools.get_hash import get_hash


@dataclass
class Memory:
    """Memory item."""
    id: int
    student_id: int
    memory_type: str
    content: str
    content_hash: str
    source: str
    confidence: float
    is_active: bool
    created_at: datetime
    embedding: Optional[bytes] = None


class MemoryAgent:
    """
    Agent for managing user memories and preferences.
    
    Provides:
    - Memory storage and retrieval
    - Semantic search via embeddings
    - Memory lifecycle management (expiration, deactivation)
    - Memory extraction from conversations
    """
    
    # Memory type definitions
    MEMORY_TYPES = [
        "tone",        # Email/letter tone preferences
        "do_dont",     # Things to include/exclude
        "preference",  # General preferences
        "goal",        # Academic goals
        "bio",         # Background information
        "instruction", # User instructions
        "guardrail",   # Constraints/limitations
        "other",       # Miscellaneous
    ]
    
    # Default TTLs for memory types (in days)
    DEFAULT_TTLS = {
        "tone": None,        # Never expire
        "do_dont": None,     # Never expire
        "preference": None,  # Never expire
        "goal": 365,         # 1 year
        "bio": None,         # Never expire
        "instruction": 90,   # 3 months
        "guardrail": None,   # Never expire
        "other": 30,         # 1 month
    }
    
    EMBEDDING_DIM = TextEmbedding3Small.output_dimensions
    
    def __init__(self, db: DatabaseService):
        self.db = db
        self.llm = OpenAIClient(
            GPT5Mini,
            cache_backend="pg_redis",
            cache_collection="memory_agent",
        )
    
    # =========================================================================
    # Core Memory Operations
    # =========================================================================
    
    async def push(
        self,
        student_id: int,
        memory_type: str,
        content: str,
        source: Literal["user", "system", "inferred"] = "inferred",
        confidence: float = 0.7,
        ttl_days: Optional[int] = None,
        embed: bool = True,
    ) -> Memory:
        """
        Store a new memory.
        
        Args:
            student_id: The student this memory belongs to
            memory_type: Type of memory (tone, do_dont, preference, etc.)
            content: The memory content
            source: How the memory was obtained
            confidence: Confidence level (0.0 to 1.0)
            ttl_days: Time to live in days (None for no expiration)
            embed: Whether to compute embedding for semantic search
        """
        # Validate memory type
        if memory_type not in self.MEMORY_TYPES:
            memory_type = "other"
        
        # Compute expiration
        if ttl_days is None:
            ttl_days = self.DEFAULT_TTLS.get(memory_type)
        
        expires_at = None
        if ttl_days is not None:
            expires_at = datetime.now(datetime.UTC) + timedelta(days=ttl_days)
        
        # Calculate content hash
        content_hash = get_hash(content)
        
        # Check for duplicates
        existing = await self._find_similar(student_id, content, memory_type, content_hash=content_hash)
        if existing:
            # Update existing memory instead of creating duplicate
            return await self._update_memory(existing, content, confidence, source, content_hash=content_hash)
        
        # Create memory record
        mem = await self.db.create_memory(
            student_id=student_id,
            memory_type=memory_type,
            content=content,
            content_hash=content_hash,
            source=source,
            confidence=confidence,
            expires_at=expires_at,
        )
        
        # Compute and store embedding
        if embed:
            embedding = await self._embed(content)
            if embedding:
                await self._store_embedding(mem.id, embedding)
        
        return Memory(
            id=int(mem.id),
            student_id=student_id,
            memory_type=memory_type,
            content=content,
            content_hash=content_hash,
            source=source,
            confidence=confidence,
            is_active=True,
            created_at=mem.created_at,
        )
    
    async def pull(
        self,
        student_id: int,
        memory_type: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 20,
        min_confidence: float = 0.0,
    ) -> List[Memory]:
        """
        Retrieve memories for a student.
        
        Args:
            student_id: The student to get memories for
            memory_type: Filter by type (optional)
            query: Semantic search query (optional)
            limit: Maximum memories to return
            min_confidence: Minimum confidence threshold
        """
        if query:
            # Semantic search
            return await self._semantic_search(
                student_id=student_id,
                query=query,
                memory_type=memory_type,
                limit=limit,
                min_confidence=min_confidence,
            )
        else:
            # Direct retrieval
            memories = await self.db.get_student_memories(
                student_id=student_id,
                memory_type=memory_type,
                active_only=True,
            )
            
            # Filter by confidence
            memories = [m for m in memories if float(m.confidence) >= min_confidence]
            
            return [
                Memory(
                    id=int(m.id),
                    student_id=int(m.student_id),
                    memory_type=m.memory_type,
                    content=m.content,
                    content_hash=m.content_hash or "",
                    source=m.source,
                    confidence=float(m.confidence),
                    is_active=m.is_active,
                    created_at=m.created_at,
                    embedding=m.embedding,
                )
                for m in memories[:limit]
            ]
    
    async def forget(self, memory_id: int) -> bool:
        """Deactivate a memory (soft delete)."""
        try:
            await self.db.deactivate_memory(memory_id)
            return True
        except Exception:
            return False
    
    async def update(
        self,
        memory_id: int,
        content: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> Optional[Memory]:
        """Update an existing memory."""
        mem = await self.db.client.aimemory.find_unique(where={"id": memory_id})
        if not mem:
            return None
        
        updates = {}
        if content is not None:
            updates["content"] = content
            updates["content_hash"] = get_hash(content)
        if confidence is not None:
            updates["confidence"] = confidence
        
        if updates:
            mem = await self.db.client.aimemory.update(
                where={"id": memory_id},
                data=updates,
            )
            
            # Update embedding if content changed
            if content is not None:
                embedding = await self._embed(content)
                if embedding:
                    await self._store_embedding(memory_id, embedding)
        
        return Memory(
            id=int(mem.id),
            student_id=int(mem.student_id),
            memory_type=mem.memory_type,
            content=mem.content,
            content_hash=mem.content_hash or "",
            source=mem.source,
            confidence=float(mem.confidence),
            is_active=mem.is_active,
            created_at=mem.created_at,
        )
    
    # =========================================================================
    # Memory Extraction
    # =========================================================================
    
    async def extract_from_conversation(
        self,
        student_id: int,
        messages: List[Dict[str, str]],
        auto_store: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Extract memories from a conversation.
        
        Analyzes conversation to identify:
        - Tone preferences
        - Do's and don'ts
        - Goals and aspirations
        - Background facts
        - Instructions
        """
        # Build conversation text
        conversation = "\n".join([
            f"{m.get('role', 'unknown').upper()}: {m.get('content', '')}"
            for m in messages
        ])
        
        prompt = f"""Analyze this conversation to identify user preferences, instructions, and facts that should be remembered for future interactions.

Conversation:
{conversation}

For each memory found, identify:
1. type: one of "tone", "do_dont", "preference", "goal", "bio", "instruction", "guardrail"
2. content: the specific thing to remember (be concise)
3. confidence: how confident you are this should be stored (0.0 to 1.0)
4. source: "user" if explicitly stated, "inferred" if deduced

Only extract clear, explicit items. Do not over-infer.

Respond with a JSON object containing a "memories" array."""

        try:
            response = await self.llm.get_response(
                messages=[{"role": "user", "content": prompt}],
                response_format="json_object",
                temperature=0,
            )
            
            output = response.get("output", {})
            if isinstance(output, str):
                output = json.loads(output)
            
            memories = output.get("memories", [])
            
            if auto_store:
                for mem in memories:
                    if mem.get("confidence", 0) >= 0.6:
                        await self.push(
                            student_id=student_id,
                            memory_type=mem.get("type", "other"),
                            content=mem.get("content", ""),
                            source=mem.get("source", "inferred"),
                            confidence=mem.get("confidence", 0.7),
                        )
            
            return memories
            
        except Exception:
            return []
    
    # =========================================================================
    # Semantic Search
    # =========================================================================
    
    async def _embed(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text."""
        try:
            response = await self.llm.get_response(
                messages=[{"role": "user", "content": text}],
                model_override="text-embedding-3-small",
                response_format="embedding",
            )
            return response.get("embedding", [])
        except Exception:
            return None
    
    async def _store_embedding(self, memory_id: int, embedding: List[float]) -> None:
        """Store embedding as binary blob."""
        # Pack floats into bytes
        blob = struct.pack(f"{len(embedding)}f", *embedding)
        
        await self.db.client.aimemory.update(
            where={"id": memory_id},
            data={"embedding": blob},
        )
    
    def _unpack_embedding(self, blob: bytes) -> List[float]:
        """Unpack embedding from binary blob."""
        n_floats = len(blob) // 4
        return list(struct.unpack(f"{n_floats}f", blob))
    
    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    
    async def _semantic_search(
        self,
        student_id: int,
        query: str,
        memory_type: Optional[str] = None,
        limit: int = 10,
        min_confidence: float = 0.0,
        min_similarity: float = 0.5,
    ) -> List[Memory]:
        """Perform semantic search over memories."""
        # Get query embedding
        query_embedding = await self._embed(query)
        if not query_embedding:
            # Fall back to keyword search
            return await self.pull(
                student_id=student_id,
                memory_type=memory_type,
                limit=limit,
                min_confidence=min_confidence,
            )
        
        # Get all memories with embeddings
        where: Dict[str, Any] = {
            "student_id": student_id,
            "is_active": True,
            "embedding": {"not": None},
        }
        if memory_type:
            where["memory_type"] = memory_type
        
        memories = await self.db.client.aimemory.find_many(where=where)
        
        # Compute similarities
        scored: List[Tuple[float, Any]] = []
        for mem in memories:
            if not mem.embedding:
                continue
            if float(mem.confidence) < min_confidence:
                continue
            
            mem_embedding = self._unpack_embedding(mem.embedding)
            similarity = self._cosine_similarity(query_embedding, mem_embedding)
            
            if similarity >= min_similarity:
                scored.append((similarity, mem))
        
        # Sort by similarity
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [
            Memory(
                id=int(mem.id),
                student_id=int(mem.student_id),
                memory_type=mem.memory_type,
                content=mem.content,
                content_hash=mem.content_hash or "",
                source=mem.source,
                confidence=float(mem.confidence),
                is_active=mem.is_active,
                created_at=mem.created_at,
            )
            for _, mem in scored[:limit]
        ]
    
    # =========================================================================
    # Helpers
    # =========================================================================
    
    async def _find_similar(
        self,
        student_id: int,
        content: str,
        memory_type: str,
        content_hash: Optional[str] = None,
        threshold: float = 0.9,
    ) -> Optional[int]:
        """Find a similar existing memory (for deduplication)."""
        # 1. Try exact hash match first (fastest)
        if content_hash:
            existing_exact = await self.db.client.aimemory.find_first(
                where={
                    "student_id": student_id,
                    "memory_type": memory_type,
                    "content_hash": content_hash,
                    "is_active": True,
                }
            )
            if existing_exact:
                return int(existing_exact.id)

        # 2. Simple approach: check for high text overlap
        existing = await self.db.get_student_memories(
            student_id=student_id,
            memory_type=memory_type,
            active_only=True,
        )
        
        content_lower = content.lower().strip()
        
        for mem in existing:
            existing_lower = mem.content.lower().strip()
            
            # Exact match
            if content_lower == existing_lower:
                return int(mem.id)
            
            # High overlap (Jaccard similarity)
            content_words = set(content_lower.split())
            existing_words = set(existing_lower.split())
            
            if content_words and existing_words:
                intersection = len(content_words & existing_words)
                union = len(content_words | existing_words)
                if union > 0 and intersection / union >= threshold:
                    return int(mem.id)
        
        return None
    
    async def _update_memory(
        self,
        memory_id: int,
        content: str,
        confidence: float,
        source: str,
        content_hash: Optional[str] = None,
    ) -> Memory:
        """Update existing memory with new content."""
        if content_hash is None:
            content_hash = get_hash(content)
            
        mem = await self.db.client.aimemory.update(
            where={"id": memory_id},
            data={
                "content": content,
                "content_hash": content_hash,
                "confidence": confidence,
                "source": source,
            },
        )
        
        # Update embedding
        embedding = await self._embed(content)
        if embedding:
            await self._store_embedding(memory_id, embedding)
        
        return Memory(
            id=int(mem.id),
            student_id=int(mem.student_id),
            memory_type=mem.memory_type,
            content=mem.content,
            content_hash=mem.content_hash or "",
            source=source,
            confidence=confidence,
            is_active=mem.is_active,
            created_at=mem.created_at,
        )
    
    async def cleanup_expired(self, student_id: Optional[int] = None) -> int:
        """Deactivate expired memories."""
        now = datetime.utcnow()
        
        where: Dict[str, Any] = {
            "is_active": True,
            "expires_at": {"lt": now},
        }
        if student_id:
            where["student_id"] = student_id
        
        result = await self.db.client.aimemory.update_many(
            where=where,
            data={"is_active": False},
        )
        
        return result.count if hasattr(result, "count") else 0
    
    async def get_context_memories(
        self,
        student_id: int,
        context: str,
        limit_per_type: int = 3,
    ) -> Dict[str, List[Memory]]:
        """
        Get memories relevant to a specific context.
        
        Returns memories grouped by type, prioritizing relevance.
        """
        result: Dict[str, List[Memory]] = {}
        
        for memory_type in self.MEMORY_TYPES:
            memories = await self._semantic_search(
                student_id=student_id,
                query=context,
                memory_type=memory_type,
                limit=limit_per_type,
                min_similarity=0.3,
            )
            if memories:
                result[memory_type] = memories
        
        return result





