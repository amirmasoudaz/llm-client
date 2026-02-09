# src/recommender/logic.py

from __future__ import annotations

import asyncio, json, re, time, unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz as rf_fuzz, process as rf_process
from tqdm import tqdm
from unidecode import unidecode

try:
    import marisa_trie  # type: ignore
    _MARISA_AVAILABLE = True
except Exception:
    marisa_trie = None  # type: ignore
    _MARISA_AVAILABLE = False

from src.config import settings
from src.db.session import DB
from src.db.queries import SELECT_ALL_PROFESSORS
from src.agents.tag_generation.agent import TagGenerationAgent
from src.recommender.domains import DOMAINS
from src.recommender.tag_constants import ALIAS_MAP, ACRONYM_STOPWORDS
from src.recommender.vector_store import TagPoint, TagVectorStore, make_point_id, make_subfield_id


class EmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: Iterable[str], *, batch_size: int = 128) -> np.ndarray:
        data = [t if isinstance(t, str) else str(t) for t in texts]
        if not data:
            return np.zeros((0, self._model.get_sentence_embedding_dimension()), dtype=np.float32)

        vecs = self._model.encode(
            data,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vecs, dtype=np.float32)


class RecommenderLogic:
    ENABLE_SEMANTIC = settings.ENABLE_SEMANTIC
    DEFAULT_TOP_K = settings.DEFAULT_TOP_K
    RRF_K = settings.RRF_K
    FUZZY_THRESHOLD = settings.FUZZY_THRESHOLD
    MAX_CANDIDATES_PER_LIST = settings.MAX_CANDIDATES_PER_LIST

    TAG_MATCH_TOPK = settings.TAG_MATCH_TOPK
    RELATED_TOPK = settings.RELATED_TOPK
    NEIGHBOR_TOPK = settings.NEIGHBOR_TOPK
    MAX_NEIGHBOR_GRAPH_ROWS = settings.MAX_NEIGHBOR_GRAPH_ROWS

    W_TAG_MATCH = settings.W_TAG_MATCH
    W_RELATED = settings.W_RELATED

    SHORT_QUERY_LEN = settings.SHORT_QUERY_LEN
    FUZZY_SHORT_THRESHOLD = settings.FUZZY_SHORT_THRESHOLD

    ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    _non_alnum = re.compile(r"[^a-z0-9\s\-+&/]")
    _multi_space = re.compile(r"\s+")

    def __init__(self, cpd: Optional[Path] = None, *, lazy_init: bool = True):
        self.cpd = cpd or Path(__file__).resolve().parent.parent.parent

        self.catalog: pd.DataFrame = pd.DataFrame()
        self.n_docs: int = 0
        self.tagidx_to_profs: Dict[int, List[Tuple[str, float]]] = {}
        self.prof_to_inst_norm: dict[str, str] = {}
        self.prof_to_name: dict[str, str] = {}
        self.prof_to_country_norm: dict[str, str] = {}
        self.inst_norm_to_display: dict[str, str] = {}
        self.inst_to_prof_ids: dict[str, set[str]] = {}
        self.country_norm_to_prof_ids: dict[str, set[str]] = {}

        self._alias_to_ids: Dict[str, List[int]] = {}
        self._acronyms = set()
        self._tokenized_docs: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._tag_list: List[str] = []
        self._tag_norm_list: List[str] = []
        self._tag_norm_to_ids: Dict[str, List[int]] = {}
        self._acronym_to_ids: Dict[str, List[int]] = {}
        self._embeddings: Optional[np.ndarray] = None
        self._all_prof_sorted_by_name: List[str] = []
        self._inst_sorted: dict[str, List[str]] = {}
        self._token_to_doc_ids: Dict[str, List[int]] = {}
        self._first_char_index: Dict[str, List[int]] = {}
        self._tag_trie = None
        self._token_trie = None
        self._related_neighbors: Dict[int, List[Tuple[int, float]]] = {}

        self._normalized_alias_map = {
            alias.lower(): [self.normalize(phrase) for phrase in phrases]
            for alias, phrases in ALIAS_MAP.items()
        }

        self._embedding_backend = EmbeddingModel(settings.RECOMMENDER_LOCAL_EMBED_MODEL)

        collection_name = settings.FUNDING_TAG_COLLECTION
        self._vector_store = TagVectorStore(collection=collection_name)
        self._tag_indexer = TagIndexer()

        if not lazy_init:
            self.prep_data(force_rebuild=False)

    def clear_variables(self) -> None:
        self.catalog = pd.DataFrame()
        self.n_docs = 0
        self.tagidx_to_profs = {}
        self.prof_to_inst_norm = {}
        self.prof_to_name = {}
        self.prof_to_country_norm = {}
        self.inst_norm_to_display = {}
        self.inst_to_prof_ids = {}
        self.country_norm_to_prof_ids = {}

        self._alias_to_ids = {}
        self._acronyms = set()
        self._tokenized_docs = []
        self._bm25 = None
        self._tag_list = []
        self._tag_norm_list = []
        self._tag_norm_to_ids = {}
        self._acronym_to_ids = {}
        self._embeddings = None
        self._all_prof_sorted_by_name = []
        self._inst_sorted = {}
        self._token_to_doc_ids = {}
        self._first_char_index = {}
        self._tag_trie = None
        self._token_trie = None
        self._related_neighbors = {}

    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        text = text.lower()
        text = self._non_alnum.sub(" ", text)
        text = self._multi_space.sub(" ", text).strip()
        return text

    def _make_acronym(self, norm: str) -> str:
        tokens = [tok for tok in norm.split() if tok and tok not in ACRONYM_STOPWORDS]
        ac = "".join(tok[0] for tok in tokens if tok)
        return ac if len(ac) >= 3 else ""

    def _build_fast_lookup_structures(self) -> None:
        token_to_docs: Dict[str, set[int]] = defaultdict(set)
        first_char_index: Dict[str, set[int]] = defaultdict(set)

        for doc_id, tokens in enumerate(self._tokenized_docs):
            unique_tokens = set(tokens)
            for token in unique_tokens:
                if not token:
                    continue
                token_to_docs[token].add(doc_id)
                first_char_index[token[0]].add(doc_id)

        for doc_id, norm in enumerate(self._tag_norm_list):
            if norm:
                first_char_index[norm[0]].add(doc_id)

        self._token_to_doc_ids = {token: sorted(doc_ids) for token, doc_ids in token_to_docs.items()}
        self._first_char_index = {ch: sorted(doc_ids) for ch, doc_ids in first_char_index.items()}

        if _MARISA_AVAILABLE and self._tag_norm_list:
            self._tag_trie = marisa_trie.Trie(self._tag_norm_list)
            self._token_trie = marisa_trie.Trie(self._token_to_doc_ids.keys())
        else:
            self._tag_trie = None
            self._token_trie = None

    def _candidate_ids_for_fuzzy(self, q_norm: str, limit: int) -> List[int]:
        candidates: set[int] = set()
        tokens = [tok for tok in q_norm.split() if tok]

        if self._token_trie:
            for token in tokens or [q_norm]:
                for key in self._token_trie.iterkeys(token):
                    for doc_id in self._token_to_doc_ids.get(key, []):
                        candidates.add(doc_id)
                        if len(candidates) >= limit * 3:
                            break
                if len(candidates) >= limit * 3:
                    break
        else:
            for token, doc_ids in self._token_to_doc_ids.items():
                if token.startswith(q_norm) or any(token.startswith(t) for t in tokens):
                    candidates.update(doc_ids)
                    if len(candidates) >= limit * 3:
                        break

        if not candidates and q_norm:
            candidates.update(self._first_char_index.get(q_norm[0], []))

        if len(candidates) < max(limit, 20):
            bm25_boost = self._bm25_candidates(q_norm, max(limit * 2, 20))
            candidates.update(bm25_boost)

        if not candidates:
            candidates = set(range(min(self.n_docs, self.MAX_CANDIDATES_PER_LIST)))

        return list(candidates)[: self.MAX_CANDIDATES_PER_LIST * 3]

    def _bm25_candidates(self, q_norm: str, limit: int) -> List[int]:
        if self._bm25 is None:
            return []
        tokens = q_norm.split() or [q_norm]
        scores = self._bm25.get_scores(tokens)
        idx = np.argpartition(scores, -limit)[-limit:]
        idx_sorted = idx[np.argsort(scores[idx])[::-1]]
        return idx_sorted.tolist()

    def _bm25_candidates_with_scores(self, q_norm: str, limit: int) -> List[Tuple[int, float]]:
        if self._bm25 is None:
            return []
        tokens = q_norm.split() or [q_norm]
        scores = self._bm25.get_scores(tokens)
        if limit >= len(scores):
            idx_sorted = np.argsort(scores)[::-1]
        else:
            idx = np.argpartition(scores, -limit)[-limit:]
            idx_sorted = idx[np.argsort(scores[idx])[::-1]]
        return [(int(i), float(scores[i])) for i in idx_sorted[:limit]]

    def _semantic_candidates(self, query: str, limit: int) -> List[int]:
        if not self.ENABLE_SEMANTIC or self._embeddings is None or self._embeddings.size == 0:
            return []
        query_vec = self._embedding_backend.embed([query], batch_size=1)
        if query_vec.size == 0:
            return []
        qv = query_vec[0]
        scores = self._embeddings @ qv
        idx = np.argpartition(scores, -limit)[-limit:]
        idx_sorted = idx[np.argsort(scores[idx])[::-1]]
        return idx_sorted.tolist()

    def _compute_neighbors(self, embeddings: np.ndarray) -> Dict[int, List[Tuple[int, float]]]:
        if embeddings is None or embeddings.size == 0:
            return {}
        n_rows = embeddings.shape[0]
        if n_rows > self.MAX_NEIGHBOR_GRAPH_ROWS:
            # Building an all-pairs similarity matrix explodes to O(n^2) memory; skip when too large.
            print(
                f"Skipping neighbor graph ({n_rows} rows > MAX_NEIGHBOR_GRAPH_ROWS={self.MAX_NEIGHBOR_GRAPH_ROWS}); "
                "falling back to BM25 for related tags."
            )
            return {}
        normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
        sims = normalized @ normalized.T
        neighbors: Dict[int, List[Tuple[int, float]]] = {}
        k = min(self.NEIGHBOR_TOPK + 1, normalized.shape[0])
        for idx in range(normalized.shape[0]):
            row = sims[idx]
            top_idx = np.argpartition(row, -k)[-k:]
            top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
            entries: List[Tuple[int, float]] = []
            for nid in top_idx:
                if nid == idx:
                    continue
                entries.append((int(nid), float(row[nid])))
                if len(entries) >= self.NEIGHBOR_TOPK:
                    break
            if entries:
                neighbors[idx] = entries
        return neighbors

    def load_from_store(self) -> bool:
        """
        Fast read-only load from Qdrant vector store.
        
        This method only fetches existing indexed data from Qdrant without triggering
        any indexing operations. Use this for fast startup when you want to serve
        requests immediately with existing data while reindexing runs in background.
        
        Returns:
            True if data was loaded successfully, False if no data exists.
        """
        try:
            records = self._vector_store.fetch_all(with_vectors=True)
        except Exception as e:
            print(f"Failed to load from vector store: {e}")
            return False

        if not records:
            print("No existing index data found in vector store.")
            return False

        self._populate_from_records(records)
        
        # Load professor metadata from database (this is fast - just a DB query)
        self._tag_indexer._update_prof_metadata_from_db_sync()
        metadata = self._tag_indexer.get_prof_metadata()
        self.prof_to_inst_norm = metadata.get("prof_to_inst_norm", {})
        self.prof_to_name = metadata.get("prof_to_name", {})
        self.prof_to_country_norm = metadata.get("prof_to_country_norm", {})
        self.inst_norm_to_display = metadata.get("inst_norm_to_display", {})
        self.inst_to_prof_ids = metadata.get("inst_to_prof_ids", {})
        self.country_norm_to_prof_ids = metadata.get("country_norm_to_prof_ids", {})

        self._all_prof_sorted_by_name = sorted(self.prof_to_inst_norm.keys(), key=self._sort_key)
        self._inst_sorted = {
            inst_norm: sorted(prof_ids, key=self._sort_key)
            for inst_norm, prof_ids in self.inst_to_prof_ids.items()
        }
        
        print(f"Loaded {self.n_docs} documents from vector store.")
        return True

    def _populate_from_records(self, records: list) -> None:
        """
        Populate internal data structures from Qdrant records.
        Shared logic between load_from_store() and prep_data().
        """
        self.clear_variables()
        if not records:
            return

        rows: List[Dict[str, Any]] = []
        vectors: List[np.ndarray] = []
        alias_to_ids: Dict[str, set[int]] = defaultdict(set)
        tag_norm_to_ids: Dict[str, set[int]] = defaultdict(set)
        acronym_to_ids: Dict[str, set[int]] = defaultdict(set)

        for idx, record in enumerate(records):
            payload = record.payload or {}
            vector = np.asarray(record.vector, dtype=np.float32)
            vectors.append(vector)

            tag = payload.get("tag", "")
            domain = payload.get("domain", "")
            subfield = payload.get("subfield", "")

            tag_norm = payload.get("tag_norm") or self.normalize(tag)
            domain_norm = payload.get("domain_norm") or self.normalize(domain)
            subfield_norm = payload.get("subfield_norm") or self.normalize(subfield)
            text = payload.get("text") or " ".join(filter(None, [tag_norm, domain_norm, subfield_norm]))
            semantic_text = payload.get("semantic_text") or f"{tag} | {domain} > {subfield}"
            aliases = payload.get("aliases") or []
            acronym = payload.get("acronym") or self._make_acronym(tag_norm)

            rows.append(
                {
                    "cat_idx": idx,
                    "point_id": record.id,
                    "tag": tag,
                    "domain": domain,
                    "subfield": subfield,
                    "tag_norm": tag_norm,
                    "domain_norm": domain_norm,
                    "subfield_norm": subfield_norm,
                    "text": text,
                    "semantic_text": semantic_text,
                    "aliases": aliases,
                    "acronym": acronym,
                }
            )

            for alias in aliases:
                alias_to_ids[alias.lower()].add(idx)
            if acronym:
                acronym_to_ids[acronym].add(idx)
            tag_norm_to_ids[tag_norm].add(idx)

            profs = payload.get("professors") or []
            self.tagidx_to_profs[idx] = [
                (str(prof.get("id")), float(prof.get("weight", 1.0)))
                for prof in profs
                if prof.get("id") is not None
            ]

        catalog = pd.DataFrame(rows)
        self.catalog = catalog
        self.n_docs = len(catalog)

        self._tag_list = catalog["tag"].tolist()
        self._tag_norm_list = catalog["tag_norm"].tolist()
        self._tokenized_docs = [row.split() for row in catalog["text"].tolist()]
        self._bm25 = BM25Okapi(self._tokenized_docs) if self._tokenized_docs else None

        # augment alias map with canonical rules
        for alias, phrases in self._normalized_alias_map.items():
            for idx, tag_norm in enumerate(self._tag_norm_list):
                if any(phrase and phrase in tag_norm for phrase in phrases):
                    alias_to_ids[alias].add(idx)

        self._alias_to_ids = {alias: sorted(ids) for alias, ids in alias_to_ids.items()}
        self._tag_norm_to_ids = {norm: sorted(ids) for norm, ids in tag_norm_to_ids.items()}
        self._acronym_to_ids = {ac: sorted(ids) for ac, ids in acronym_to_ids.items()}
        self._acronyms = set(self._acronym_to_ids.keys())

        self._embeddings = np.vstack(vectors) if vectors else None
        if self._embeddings is not None and self._embeddings.size:
            self._related_neighbors = self._compute_neighbors(self._embeddings)

        self._build_fast_lookup_structures()

    def prep_data(self, force_rebuild: bool = False) -> None:
        """
        Full indexing + load: runs TagIndexer to update Qdrant, then loads into memory.
        
        This is the slow path that should be run in background after startup.
        For fast startup, use load_from_store() instead.
        """
        self._tag_indexer.index_data_sync(force_rebuild=force_rebuild)
        records = self._vector_store.fetch_all(with_vectors=True)

        self._populate_from_records(records)

        metadata = self._tag_indexer.get_prof_metadata()
        self.prof_to_inst_norm = metadata.get("prof_to_inst_norm", {})
        self.prof_to_name = metadata.get("prof_to_name", {})
        self.prof_to_country_norm = metadata.get("prof_to_country_norm", {})
        self.inst_norm_to_display = metadata.get("inst_norm_to_display", {})
        self.inst_to_prof_ids = metadata.get("inst_to_prof_ids", {})
        self.country_norm_to_prof_ids = metadata.get("country_norm_to_prof_ids", {})

        self._all_prof_sorted_by_name = sorted(self.prof_to_inst_norm.keys(), key=self._sort_key)
        self._inst_sorted = {
            inst_norm: sorted(prof_ids, key=self._sort_key)
            for inst_norm, prof_ids in self.inst_to_prof_ids.items()
        }

    def _sort_key(self, prof_id: str) -> Tuple[str, str, str]:
        name = (self.prof_to_name.get(prof_id) or "").strip().lower()
        inst = (self.prof_to_inst_norm.get(prof_id) or "")
        return name if name else "~", inst, prof_id

    def _prefix_candidates(self, q_norm: str, limit: int) -> List[int]:
        matches: List[int] = []
        seen: set[int] = set()

        # if len(q_norm) <= 2 and q_norm in self._alias_to_ids: 
        if q_norm in self._alias_to_ids:  # TODO: check if this is needed
            for doc_id in self._alias_to_ids[q_norm]:
                if doc_id not in seen:
                    matches.append(doc_id)
                    seen.add(doc_id)
                    if len(matches) >= limit:
                        return matches[:limit]

        if len(q_norm) >= 3:
            for ac, doc_ids in self._acronym_to_ids.items():
                if ac.startswith(q_norm):
                    for doc_id in doc_ids:
                        if doc_id not in seen:
                            matches.append(doc_id)
                            seen.add(doc_id)
                            if len(matches) >= limit:
                                return matches[:limit]

        if self._tag_trie is not None:
            for key in self._tag_trie.iterkeys(q_norm):
                for doc_id in self._tag_norm_to_ids.get(key, []):
                    if doc_id not in seen:
                        matches.append(doc_id)
                        seen.add(doc_id)
                        if len(matches) >= limit:
                            return matches[:limit]
        else:
            for doc_id, tag_norm in enumerate(self._tag_norm_list):
                if tag_norm.startswith(q_norm) or any(tok.startswith(q_norm) for tok in tag_norm.split()):
                    if doc_id not in seen:
                        matches.append(doc_id)
                        seen.add(doc_id)
                        if len(matches) >= limit:
                            break

        tokens = q_norm.split()
        if tokens:
            last = tokens[-1]
            for doc_id in self._token_to_doc_ids.get(last, []):
                if doc_id not in seen:
                    matches.append(doc_id)
                    seen.add(doc_id)
                    if len(matches) >= limit:
                        break

        return matches[:limit]

    def _retrieve_tag_candidates(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        query = (query or "").strip()
        if len(query) < 2:
            return []

        q_norm = self.normalize(query)
        limit = max(self.MAX_CANDIDATES_PER_LIST, top_k * 5)

        prefix_idx = self._prefix_candidates(q_norm, limit)

        if len(q_norm) <= 2:
            fuzzy_idx: List[int] = []
        else:
            cutoff = self.FUZZY_SHORT_THRESHOLD if len(q_norm) <= self.SHORT_QUERY_LEN else self.FUZZY_THRESHOLD
            candidate_ids = self._candidate_ids_for_fuzzy(q_norm, limit)
            if candidate_ids:
                choices = {doc_id: self._tag_list[doc_id] for doc_id in candidate_ids}
                fuzzy_raw = rf_process.extract(query, choices, scorer=rf_fuzz.WRatio, limit=limit, score_cutoff=cutoff)
                fuzzy_idx = [int(idx) for (_choice, _score, idx) in fuzzy_raw]
            else:
                fuzzy_idx = []

        bm25_idx = self._bm25_candidates(q_norm, limit)
        sem_idx = self._semantic_candidates(query, limit)

        def as_rank_map(items: List[int]) -> Dict[int, int]:
            return {doc_id: rank for rank, doc_id in enumerate(items, start=1)}

        scores: Dict[int, float] = {}
        ranks: List[Tuple[Dict[int, int], float]] = []
        short2 = len(q_norm) <= 2
        short3 = len(q_norm) == 3

        if prefix_idx:
            ranks.append((as_rank_map(prefix_idx), 2.6 if short2 else (2.2 if short3 else 1.2)))
        if fuzzy_idx:
            ranks.append((as_rank_map(fuzzy_idx), 0.0 if short2 else (1.6 if short3 else 1.1)))
        if bm25_idx:
            ranks.append((as_rank_map(bm25_idx), 1.5 if short2 else 1.3))
        if sem_idx:
            ranks.append((as_rank_map(sem_idx), 1.2 if short2 else 1.0))

        for rank_map, weight in ranks:
            if weight <= 0:
                continue
            for doc_id, rank in rank_map.items():
                scores[doc_id] = scores.get(doc_id, 0.0) + weight * (1.0 / (self.RRF_K + rank))

        if not scores:
            return []

        top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return top

    def _resolve_country_pool(self, country: str) -> Tuple[set[str], List[dict[str, Any]]]:
        c_norm = self.normalize(country)
        explanation = {"filter": "country", "value": country, "matches": []}

        total_candidates = sum(len(ids) for ids in self.country_norm_to_prof_ids.values())
        pool = set(self.country_norm_to_prof_ids.get(c_norm, set()))
        if pool:
            explanation["matches"].append({"country_norm": c_norm, "mode": "exact"})

        explanation["pre_filter_candidates"] = total_candidates
        explanation["post_filter_candidates"] = len(pool)
        return pool, [explanation]

    def _resolve_institute_pool(self, institute_name: str) -> Tuple[set[str], List[dict[str, Any]]]:
        in_norm = self.normalize(institute_name)
        explanation = {"filter": "institute", "value": institute_name, "matches": []}

        pool = set(self.inst_to_prof_ids.get(in_norm, set()))
        if pool:
            explanation["matches"].append(
                {"inst_norm": in_norm, "display": self.inst_norm_to_display.get(in_norm, ""), "mode": "exact"}
            )

        if not pool and in_norm:
            for inst_norm, ids in self.inst_to_prof_ids.items():
                if in_norm in inst_norm:
                    pool |= ids
                    explanation["matches"].append(
                        {"inst_norm": inst_norm, "display": self.inst_norm_to_display.get(inst_norm, ""), "mode": "substring"}
                    )

        if not pool and in_norm:
            keys = list(self.inst_to_prof_ids.keys())
            fuzzy_hits = rf_process.extract(in_norm, keys, scorer=rf_fuzz.WRatio, limit=3, score_cutoff=70)
            for inst_norm, score, _ in fuzzy_hits:
                ids = self.inst_to_prof_ids.get(inst_norm, set())
                if ids:
                    pool |= ids
                    explanation["matches"].append(
                        {
                            "inst_norm": inst_norm,
                            "display": self.inst_norm_to_display.get(inst_norm, ""),
                            "mode": "fuzzy",
                            "score": float(score),
                        }
                    )

        explanation["pre_filter_candidates"] = sum(len(ids) for ids in self.inst_to_prof_ids.values())
        explanation["post_filter_candidates"] = len(pool)
        return pool, [explanation]

    def _apply_prof_name_query(
        self,
        scores: Dict[str, float],
        prof_query: str,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Hybrid filter + boost:
        - If any names pass the threshold, keep only those and boost by similarity.
        - Otherwise, leave the original scores untouched.
        """
        pq = (prof_query or "").strip()
        expl: Dict[str, Any] = {
            "filter": "prof_q",
            "value": prof_query,
            "pre_filter_candidates": len(scores),
        }
        if len(pq) < 2 or not scores:
            expl["post_filter_candidates"] = len(scores)
            expl["note"] = "skipped (short query or empty candidate set)"
            return scores, expl

        name_sims: Dict[str, float] = {}
        for pid in scores.keys():
            name = self.prof_to_name.get(pid, "")
            if not name:
                name_sims[pid] = 0.0
                continue
            name_sims[pid] = float(rf_fuzz.WRatio(pq, name))

        if not name_sims:
            expl["post_filter_candidates"] = len(scores)
            expl["note"] = "no names available for scoring"
            return scores, expl

        THRESHOLD = 50.0
        matched = {pid for pid, sim in name_sims.items() if sim >= THRESHOLD}
        if matched:
            max_sim = max(name_sims[pid] for pid in matched) or 1.0
            adjusted: Dict[str, float] = {}
            for pid in matched:
                base = scores.get(pid, 0.0)
                sim = name_sims.get(pid, 0.0)
                adjusted[pid] = base * (1.0 + 0.5 * (sim / max_sim))
            expl["post_filter_candidates"] = len(adjusted)
            expl["mode"] = "filtered_and_boosted"
            expl["top_matches"] = sorted(
                [
                    {
                        "prof_id": pid,
                        "name": self.prof_to_name.get(pid, ""),
                        "name_score": round(name_sims.get(pid, 0.0), 2),
                    }
                    for pid in matched
                ],
                key=lambda item: item["name_score"],
                reverse=True,
            )[:5]
            expl["threshold"] = THRESHOLD
            return adjusted, expl

        expl["post_filter_candidates"] = len(scores)
        expl["mode"] = "no_match_threshold"
        expl["threshold"] = THRESHOLD
        expl["note"] = "no name matches above threshold; base ranking preserved"
        return scores, expl

    def rank_all(
        self,
        tags: List[str],
        institute_name: Optional[str] = None,
        country: Optional[str] = None,
        professor_name: Optional[str] = None,
        expand_related: bool = True,
        domain_filters: Optional[List[str]] = None,
        subfield_filters: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        agg_scores: Dict[str, float] = {}
        explanations: List[Dict[str, Any]] = []

        domain_norm_filters = {self.normalize(d) for d in (domain_filters or []) if d}
        subfield_norm_filters = {self.normalize(s) for s in (subfield_filters or []) if s}

        if not tags:
            pool: set[str] = set(self._all_prof_sorted_by_name)

            if institute_name:
                inst_pool, inst_expl = self._resolve_institute_pool(institute_name)
                explanations.extend(inst_expl)
                pool &= inst_pool

            if country:
                country_pool, country_expl = self._resolve_country_pool(country)
                explanations.extend(country_expl)
                pool &= country_pool

            if not pool:
                explanations.append({"note": "no-tags fallback produced empty pool"})
                return {"professor_ids": [], "explanations": explanations}

            base_scores = {pid: 1.0 for pid in pool}
            if professor_name:
                base_scores, profq_expl = self._apply_prof_name_query(base_scores, professor_name)
                explanations.append(profq_expl)
                ordered = [pid for pid, _ in sorted(base_scores.items(), key=lambda item: (-item[1], item[0]))]
            else:
                if institute_name:
                    ordered = []
                    for inst_norm, ids in self._inst_sorted.items():
                        overlap = [pid for pid in ids if pid in pool]
                        if overlap:
                            ordered.extend(overlap)
                else:
                    ordered = [pid for pid in self._all_prof_sorted_by_name if pid in pool]

            explanations.append({"note": "no-tags fallback", "count": len(ordered)})
            return {"professor_ids": ordered, "explanations": explanations}

        for tag_query in tags:
            candidates = self._retrieve_tag_candidates(tag_query, self.TAG_MATCH_TOPK)
            filtered_candidates: List[Tuple[int, float]] = []
            for doc_id, fused_score in candidates:
                row = self.catalog.iloc[doc_id]
                if domain_norm_filters and row["domain_norm"] not in domain_norm_filters:
                    continue
                if subfield_norm_filters and row["subfield_norm"] not in subfield_norm_filters:
                    continue
                filtered_candidates.append((doc_id, fused_score))

            max_fused = filtered_candidates[0][1] if filtered_candidates else 1.0

            for doc_id, fused_score in filtered_candidates:
                row = self.catalog.iloc[doc_id]
                normalized_score = float(fused_score) / max_fused if max_fused else 0.0
                for prof_id, weight in self.tagidx_to_profs.get(doc_id, []):
                    agg_scores[prof_id] = agg_scores.get(prof_id, 0.0) + self.W_TAG_MATCH * normalized_score * weight

                explanations.append(
                    {
                        "input": tag_query,
                        "matched": row["tag"],
                        "mode": "fused",
                        "fused_score": round(float(fused_score), 6),
                        "normalized": round(float(normalized_score), 6),
                    }
                )

            if expand_related and filtered_candidates:
                doc_id = filtered_candidates[0][0]
                related_entries: List[Tuple[int, float]] = []
                if self._related_neighbors:
                    for neighbor_id, score in self._related_neighbors.get(doc_id, []):
                        row_rel = self.catalog.iloc[neighbor_id]
                        if domain_norm_filters and row_rel["domain_norm"] not in domain_norm_filters:
                            continue
                        if subfield_norm_filters and row_rel["subfield_norm"] not in subfield_norm_filters:
                            continue
                        related_entries.append((neighbor_id, score))
                        if len(related_entries) >= self.RELATED_TOPK:
                            break

                if not related_entries:
                    related_entries = [
                        (doc, score)
                        for doc, score in self._bm25_candidates_with_scores(self.normalize(tag_query), self.RELATED_TOPK)
                        if (
                            (not domain_norm_filters or self.catalog.iloc[doc]["domain_norm"] in domain_norm_filters)
                            and (
                                not subfield_norm_filters
                                or self.catalog.iloc[doc]["subfield_norm"] in subfield_norm_filters
                            )
                        )
                    ]

                if related_entries:
                    max_rel = max(score for _, score in related_entries) or 1.0
                    related_notes = []
                    for rel_doc_id, rel_score in related_entries[: self.RELATED_TOPK]:
                        rel_norm = (rel_score / max_rel) if max_rel else 0.0
                        for prof_id, weight in self.tagidx_to_profs.get(rel_doc_id, []):
                            agg_scores[prof_id] = agg_scores.get(prof_id, 0.0) + self.W_RELATED * rel_norm * weight
                        related_notes.append(
                            {
                                "tag": self.catalog.iloc[rel_doc_id]["tag"],
                                "related_norm": round(rel_norm, 6),
                            }
                        )
                    explanations.append(
                        {
                            "input": tag_query,
                            "related": related_notes,
                            "mode": "semantic_related" if self._related_neighbors else "bm25_related",
                        }
                    )

        if domain_filters:
            explanations.append(
                {"filter": "domains", "value": sorted(domain_filters), "normalized": sorted(domain_norm_filters)}
            )
        if subfield_filters:
            explanations.append(
                {"filter": "subfields", "value": sorted(subfield_filters), "normalized": sorted(subfield_norm_filters)}
            )

        if institute_name:
            inst_norm = self.normalize(institute_name)
            allow: set[str] = set()
            allow |= self.inst_to_prof_ids.get(inst_norm, set())
            if not allow:
                for key, ids in self.inst_to_prof_ids.items():
                    if inst_norm in key:
                        allow |= ids

            pre = len(agg_scores)
            if allow:
                agg_scores = {pid: score for pid, score in agg_scores.items() if pid in allow}
            else:
                agg_scores = {}
            explanations.append(
                {
                    "filter": "institute",
                    "value": institute_name,
                    "matched_prof_count": len(allow),
                    "pre_filter_candidates": pre,
                    "post_filter_candidates": len(agg_scores),
                }
            )

        if country:
            c_norm = self.normalize(country)
            allow_country: set[str] = set(self.country_norm_to_prof_ids.get(c_norm, set()))
            pre_country = len(agg_scores)
            if allow_country:
                agg_scores = {pid: score for pid, score in agg_scores.items() if pid in allow_country}
            else:
                agg_scores = {}
            explanations.append(
                {
                    "filter": "country",
                    "value": country,
                    "matched_prof_count": len(allow_country),
                    "pre_filter_candidates": pre_country,
                    "post_filter_candidates": len(agg_scores),
                }
            )

        if professor_name:
            agg_scores, profq_expl = self._apply_prof_name_query(agg_scores, professor_name)
            explanations.append(profq_expl)

        if not agg_scores:
            return {"professor_ids": [], "explanations": explanations}

        ranked = sorted(agg_scores.items(), key=lambda item: (-item[1], item[0]))
        professor_ids = [pid for pid, _ in ranked]
        return {"professor_ids": professor_ids, "explanations": explanations}

    def suggest(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        candidates = self._retrieve_tag_candidates(query, top_k)
        if not candidates:
            return []

        suggestions: List[Dict[str, Any]] = []
        for doc_id, fused_score in candidates:
            row = self.catalog.iloc[doc_id]
            profs = {prof_id for prof_id, _ in self.tagidx_to_profs.get(doc_id, [])}
            suggestions.append(
                {
                    "tag": row["tag"],
                    "domain": row["domain"],
                    "subfield": row["subfield"],
                    "score": round(float(fused_score), 6),
                    "prof_count": len(profs),
                }
            )
        return suggestions


class TagIndexer:
    TAG_ASSIGN_TOPK = 3
    TAG_SIM_THRESHOLD = 0.0

    ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    _PARENTS_SET: Optional[set[bytes]] = None
    _PARENTS_LOCK = asyncio.Lock()

    _re_ws = re.compile(r"\s+")
    _re_punc = re.compile(r"[^\w\s\-]")

    def __init__(self):
        self._embedding_backend = EmbeddingModel(settings.RECOMMENDER_LOCAL_EMBED_MODEL)
        self.vector_store = TagVectorStore(collection=settings.FUNDING_TAG_COLLECTION)
        self.subfield_store = TagVectorStore(collection=settings.FUNDING_SUBFIELD_COLLECTION)

        self._normalized_alias_map = {
            alias.lower(): [self._normalize_tag(phrase) for phrase in phrases]
            for alias, phrases in ALIAS_MAP.items()
        }

        self._prof_metadata: Dict[str, Dict[str, Any]] = {
            "prof_to_inst_norm": {},
            "prof_to_name": {},
            "inst_norm_to_display": {},
            "inst_to_prof_ids": {},
            "prof_to_country_norm": {},
            "country_norm_to_prof_ids": {},
        }

    def get_prof_metadata(self) -> Dict[str, Dict[str, Any]]:
        return {
            key: value.copy() if isinstance(value, dict) else value
            for key, value in self._prof_metadata.items()
        }

    def _normalize_tag(self, s: str) -> str:
        if not s:
            return ""
        s = unidecode(s.lower().strip())
        s = self._re_punc.sub(" ", s).replace("-", " ")
        s = self._re_ws.sub(" ", s).strip()
        return s

    @staticmethod
    def _make_acronym(norm: str) -> str:
        tokens = [tok for tok in norm.split() if tok and tok not in ACRONYM_STOPWORDS]
        ac = "".join(tok[0] for tok in tokens if tok)
        return ac if len(ac) >= 3 else ""

    def _alias_tokens(self, tag_norm: str) -> List[str]:
        tokens: List[str] = []
        for alias, phrases in self._normalized_alias_map.items():
            if any(phrase and phrase in tag_norm for phrase in phrases):
                tokens.append(alias)
        return tokens

    @staticmethod
    async def _fetch_all_profs() -> list[dict[str, Any]]:
        rows = await DB.fetch_all(SELECT_ALL_PROFESSORS)
        for r in rows:
            for key in ("research_areas", "area_of_expertise"):
                val = r.get(key)
                if val and not isinstance(val, (list, dict)):
                    try:
                        r[key] = json.loads(val)
                    except Exception:
                        r[key] = []
        return rows

    def _load_tagged_prof_ids_from_store(self) -> set[bytes]:
        try:
            records = self.vector_store.fetch_all(with_vectors=False)
        except Exception:
            return set()

        prof_ids: set[bytes] = set()
        for record in records:
            payload = record.payload or {}
            for prof in payload.get("professors") or []:
                pid = prof.get("id")
                if isinstance(pid, str):
                    try:
                        prof_ids.add(bytes.fromhex(pid))
                    except ValueError:
                        continue
        return prof_ids

    @staticmethod
    async def _load_parent_hashes() -> set[bytes]:
        rows = await DB.fetch_all("SELECT prof_hash FROM funding_professors")
        parents: set[bytes] = set()
        for row in rows or []:
            ph = row.get("prof_hash")
            if isinstance(ph, (bytes, bytearray, memoryview)):
                parents.add(bytes(ph))
        return parents

    async def _get_parent_set(self) -> set[bytes]:
        if self._PARENTS_SET is None:
            async with self._PARENTS_LOCK:
                if self._PARENTS_SET is None:
                    self._PARENTS_SET = await self._load_parent_hashes()
        return self._PARENTS_SET

    async def _embed_texts(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        items = [t if isinstance(t, str) else str(t) for t in texts if t]
        if not items:
            return np.zeros((0, 0), dtype=np.float32)

        def _run() -> np.ndarray:
            return self._embedding_backend.embed(items, batch_size=batch_size)

        return await asyncio.to_thread(_run)

    @staticmethod
    def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.size == 0 or b.size == 0:
            return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        return a_norm @ b_norm.T

    @staticmethod
    async def _generate_tags_for_profs(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        already_tagged, untagged = [], []
        for row in all_rows:
            row["others"] = json.loads(row.get("others", "{}") or {})
            if not row["others"]:
                untagged.append(row)
            else:
                already_tagged.append(row)

        already_tagged_out = [
            {
                "primary_topics": row["others"].get("primary_topics", []),
                "secondary_topics": row["others"].get("secondary_topics", []),
                "confidence": row["others"].get("tagging_confidence", 0.0),
                "sources_used": row["others"].get("tagging_sources_used", []),
                "notes": row["others"].get("tagging_notes", ""),
                "prof_hash": bytes(row["prof_hash"])
                    if isinstance(row["prof_hash"], (bytearray, memoryview))
                    else row["prof_hash"],
                "full_name": row["full_name"],
                "institute": row["institute"],
            }
            for row in already_tagged
        ]

        untagged_out = await TagGenerationAgent().batch_generate(untagged, regen=False)

        return already_tagged_out + untagged_out

    @staticmethod
    def _topic_weight(*, is_primary: bool, expertise: str) -> float:
        expertise = (expertise or "").lower()
        base = {"advanced": 1.3, "intermediate": 1.1, "basic": 0.9}.get(expertise, 0.9)
        multiplier = 1.0 if is_primary else 0.65
        return round(base * multiplier, 4)

    def _sorted_topics(self, tag_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        level = {"advanced": 3, "intermediate": 2, "basic": 1}
        topics: Dict[str, Dict[str, Any]] = {}

        def ingest(entries: Iterable[Dict[str, Any]], is_primary: bool) -> None:
            for entry in entries or []:
                if not isinstance(entry, dict):
                    continue
                raw_tag = " ".join(str(entry.get("tag", "")).strip().split())
                if not raw_tag:
                    continue
                expertise = str(entry.get("expertise") or "").lower()
                weight = self._topic_weight(is_primary=is_primary, expertise=expertise)
                key = self._normalize_tag(raw_tag)

                existing = topics.get(key)
                score_tuple = (level.get(expertise, 0), 1 if is_primary else 0, weight)
                if existing:
                    existing_score = (
                        level.get(existing["expertise"], 0),
                        1 if existing["source"] == "primary" else 0,
                        existing["weight"],
                    )
                    if score_tuple <= existing_score:
                        continue

                topics[key] = {
                    "tag": raw_tag,
                    "normalized": key,
                    "expertise": expertise or "unknown",
                    "source": "primary" if is_primary else "secondary",
                    "weight": weight,
                    "score_tuple": score_tuple,
                }

        ingest(tag_json.get("primary_topics"), True)
        ingest(tag_json.get("secondary_topics"), False)

        ordered = sorted(
            topics.values(),
            key=lambda item: (item["score_tuple"][0], item["score_tuple"][1], item["score_tuple"][2]),
            reverse=True,
        )
        return ordered

    @staticmethod
    def _flatten_ontology(domains: dict[str, dict[str, str]]) -> pd.DataFrame:
        rows: list[dict[str, str]] = []
        for domain, subs in domains.items():
            for subfield, description in subs.items():
                embed_text = f"{subfield}. {description}".strip()
                rows.append(
                    {
                        "domain": domain,
                        "subfield": subfield,
                        "description": description,
                        "embed_text": embed_text,
                    }
                )
        return pd.DataFrame(rows)

    def _assign_tags_to_subfields(
        self,
        tag_vecs: np.ndarray,
        sub_vecs: np.ndarray,
        sub_df: pd.DataFrame,
        tags: list[str],
    ) -> pd.DataFrame:
        sims = (
            self._cosine_rows(tag_vecs, sub_vecs)
            if tag_vecs.size and sub_vecs.size
            else np.zeros((len(tags), len(sub_df)), dtype=np.float32)
        )
        out_rows: list[dict[str, Any]] = []
        for idx, tag in enumerate(tags):
            if sims.shape[1] == 0:
                out_rows.append(
                    {
                        "tag": tag,
                        "assigned": False,
                        "domain": None,
                        "subfield": None,
                        "similarity": None,
                        "alternatives": json.dumps([]),
                    }
                )
                continue

            choose = min(self.TAG_ASSIGN_TOPK, sims.shape[1] - 1)
            idx_candidates = np.argpartition(-sims[idx], choose)[: self.TAG_ASSIGN_TOPK]
            idx_candidates = idx_candidates[np.argsort(-sims[idx, idx_candidates])]
            kept = [(j, float(sims[idx, j])) for j in idx_candidates if sims[idx, j] >= self.TAG_SIM_THRESHOLD]

            if not kept:
                out_rows.append(
                    {
                        "tag": tag,
                        "assigned": False,
                        "domain": None,
                        "subfield": None,
                        "similarity": None,
                        "alternatives": json.dumps([]),
                    }
                )
                continue

            best_idx, best_sim = kept[0]
            alternatives = [
                {
                    "domain": sub_df.iloc[j]["domain"],
                    "subfield": sub_df.iloc[j]["subfield"],
                    "similarity": float(sims[idx, j]),
                }
                for j, _ in kept[1:]
            ]
            out_rows.append(
                {
                    "tag": tag,
                    "assigned": True,
                    "domain": sub_df.iloc[best_idx]["domain"],
                    "subfield": sub_df.iloc[best_idx]["subfield"],
                    "similarity": best_sim,
                    "alternatives": json.dumps(alternatives, ensure_ascii=False),
                }
            )
        return pd.DataFrame(out_rows)

    def _update_prof_metadata(self, prof_rows: List[Dict[str, Any]]) -> None:
        if not prof_rows:
            self._prof_metadata = {
                "prof_to_inst_norm": {},
                "prof_to_name": {},
                "inst_norm_to_display": {},
                "inst_to_prof_ids": {},
                "prof_to_country_norm": {},
                "country_norm_to_prof_ids": {},
            }
            return

        df = pd.DataFrame(prof_rows)
        df["prof_id"] = df["prof_hash"].apply(
            lambda val: (bytes(val) if isinstance(val, (bytes, bytearray, memoryview)) else bytes.fromhex(val)).hex()
        )
        df["institute"] = df["institute"].fillna("").astype(str)
        df["full_name"] = df["full_name"].fillna("").astype(str)
        if "country" not in df.columns:
            df["country"] = ""
        df["country"] = df["country"].fillna("").astype(str)
        df["inst_norm"] = df["institute"].map(self._normalize_tag)
        df["country_norm"] = df["country"].map(self._normalize_tag)

        prof_to_inst_norm = dict(zip(df["prof_id"], df["inst_norm"]))
        prof_to_name = dict(zip(df["prof_id"], df["full_name"]))
        prof_to_country_norm = dict(zip(df["prof_id"], df["country_norm"]))
        inst_norm_to_display = df.groupby("inst_norm")["institute"].agg(lambda s: max(s, key=len)).to_dict()
        inst_to_prof_ids = df.groupby("inst_norm")["prof_id"].apply(lambda s: set(s.astype(str))).to_dict()
        country_norm_to_prof_ids = df.groupby("country_norm")["prof_id"].apply(lambda s: set(s.astype(str))).to_dict()

        self._prof_metadata = {
            "prof_to_inst_norm": prof_to_inst_norm,
            "prof_to_name": prof_to_name,
            "inst_norm_to_display": inst_norm_to_display,
            "inst_to_prof_ids": inst_to_prof_ids,
            "prof_to_country_norm": prof_to_country_norm,
            "country_norm_to_prof_ids": country_norm_to_prof_ids,
        }

    def _update_prof_metadata_from_db_sync(self) -> None:
        """
        Fetch professors from database and update metadata.
        
        This is a sync wrapper for fast startup - it runs the async fetch
        in a new event loop and updates professor metadata without running
        the full indexing pipeline.
        """
        async def _fetch_and_update():
            all_rows = await self._fetch_all_profs()
            for row in all_rows:
                ph = row.get("prof_hash")
                if isinstance(ph, (bytes, bytearray, memoryview)):
                    prof_hash = bytes(ph)
                else:
                    prof_hash = bytes.fromhex(str(ph))
                row["prof_hash"] = prof_hash
            self._update_prof_metadata(all_rows)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're already in an async context, need to run in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _fetch_and_update())
                future.result()
        else:
            asyncio.run(_fetch_and_update())

    def _build_tag_points(
        self,
        tag_rows: pd.DataFrame,
        tag_vecs: np.ndarray,
        tag2idx: Dict[str, int],
    ) -> Tuple[List[TagPoint], List[int]]:
        if tag_rows.empty:
            return [], []

        grouped = tag_rows.groupby(["tag", "domain", "subfield"], dropna=False)
        point_ids = [make_point_id(self._normalize_tag(tag), self._normalize_tag(domain or ""), self._normalize_tag(subfield or "")) for tag, domain, subfield in grouped.groups.keys()]

        existing_records = self.vector_store.retrieve_by_ids(point_ids, with_vectors=False)
        existing_map = {record.id: record.payload or {} for record in existing_records}

        points: List[TagPoint] = []
        deletions: List[int] = []
        point_iter = zip(grouped.groups.keys(), point_ids)

        for (tag, domain, subfield), point_id in tqdm(point_iter, desc="Building tag points", total=len(point_ids)):
            group = grouped.get_group((tag, domain, subfield))
            tag_norm = self._normalize_tag(tag)
            domain_norm = self._normalize_tag(domain or "")
            subfield_norm = self._normalize_tag(subfield or "")
            acronym = self._make_acronym(tag_norm)
            aliases = self._alias_tokens(tag_norm)
            text = " ".join(filter(None, [tag_norm, domain_norm, subfield_norm, acronym, " ".join(aliases)]))
            semantic_text = " | ".join(filter(None, [tag, domain, subfield, " ".join(aliases), acronym]))

            professors: Dict[str, float] = {}
            expertise_counts: Dict[str, int] = {}
            sources: set[str] = set()

            for row in group.itertuples():
                pid = str(row.prof_id)
                weight = float(getattr(row, "weight", 1.0) or 1.0)
                if pid in professors:
                    if weight > professors[pid]:
                        professors[pid] = weight
                else:
                    professors[pid] = weight

                expertise = str(getattr(row, "expertise", "") or "unknown").lower()
                expertise_counts[expertise] = expertise_counts.get(expertise, 0) + 1

                source = str(getattr(row, "source", "") or "primary").lower()
                sources.add(source)

            existing = existing_map.get(point_id, {})
            for prof in existing.get("professors", []) or []:
                pid = str(prof.get("id"))
                weight = float(prof.get("weight", 1.0))
                if pid:
                    professors[pid] = max(professors.get(pid, 0.0), weight)
            for key, val in (existing.get("expertise_counts") or {}).items():
                expertise_counts[key] = expertise_counts.get(key, 0) + int(val)
            for src in existing.get("sources") or []:
                if src:
                    sources.add(str(src))

            if not professors:
                deletions.append(point_id)
                continue

            prof_list = [
                {"id": pid, "weight": round(weight, 6)}
                for pid, weight in sorted(professors.items(), key=lambda item: (-item[1], item[0]))
            ]

            vector_idx = tag2idx.get(tag)
            if vector_idx is None or tag_vecs.shape[0] == 0:
                vector = np.zeros((tag_vecs.shape[1] if tag_vecs.size else 0,), dtype=np.float32)
            else:
                vector = tag_vecs[vector_idx]

            payload = {
                "tag": tag,
                "tag_norm": tag_norm,
                "domain": domain,
                "domain_norm": domain_norm,
                "subfield": subfield,
                "subfield_norm": subfield_norm,
                "acronym": acronym,
                "aliases": aliases,
                "text": text,
                "semantic_text": semantic_text,
                "professors": prof_list,
                "expertise_counts": expertise_counts,
                "sources": sorted(sources),
                "vector_model": self._embedding_backend.model_name,
                "updated_at": time.time(),
            }
            points.append(TagPoint(point_id=point_id, vector=vector, payload=payload))

        return points, deletions

    async def _load_or_create_subfields(self, force_rebuild: bool) -> Tuple[pd.DataFrame, np.ndarray]:
        if not force_rebuild:
            try:
                records = self.subfield_store.fetch_all(with_vectors=True)
            except Exception:
                records = []
            if records:
                rows: List[Dict[str, Any]] = []
                vectors: List[np.ndarray] = []
                for record in records:
                    payload = record.payload or {}
                    vectors.append(np.asarray(record.vector, dtype=np.float32))
                    rows.append(
                        {
                            "domain": payload.get("domain", ""),
                            "subfield": payload.get("subfield", ""),
                            "description": payload.get("description", ""),
                            "embed_text": payload.get("embed_text", ""),
                        }
                    )
                if rows and vectors:
                    return pd.DataFrame(rows), np.vstack(vectors)

        sub_df = self._flatten_ontology(DOMAINS)
        sub_vecs = await self._embed_texts(sub_df["embed_text"].tolist())
        vector_dim = sub_vecs.shape[1] if sub_vecs.size else 0
        if vector_dim == 0:
            raise RuntimeError("Subfield embeddings produced zero-dimensional vectors.")

        self.subfield_store.ensure_collection(vector_dim)
        points: List[TagPoint] = []
        for row, vector in zip(sub_df.itertuples(index=False), sub_vecs):
            pid = make_subfield_id(row.domain, row.subfield)
            payload = {
                "domain": row.domain,
                "subfield": row.subfield,
                "description": row.description,
                "embed_text": row.embed_text,
            }
            points.append(TagPoint(point_id=pid, vector=vector, payload=payload))
        if points:
            self.subfield_store.upsert_points(points)
        return sub_df, sub_vecs

    async def index_data(self, force_rebuild: bool = False) -> Dict[str, int]:
        sub_df, sub_vecs = await self._load_or_create_subfields(force_rebuild=force_rebuild)

        all_rows = await self._fetch_all_profs()
        for row in all_rows:
            ph = row.get("prof_hash")
            if isinstance(ph, (bytes, bytearray, memoryview)):
                prof_hash = bytes(ph)
            else:
                prof_hash = bytes.fromhex(str(ph))
            row["prof_hash"] = prof_hash

        self._update_prof_metadata(all_rows)
        tagged_professors = self._load_tagged_prof_ids_from_store()

        if force_rebuild:
            missing_rows = all_rows
        else:
            missing_rows = []
            for row in all_rows:
                if row["prof_hash"] not in tagged_professors:
                    missing_rows.append(row)

        if not missing_rows and not force_rebuild:
            return {"processed_professors": 0, "generated_tags": 0, "upserted_vectors": 0}

        tag_results = await self._generate_tags_for_profs(missing_rows)

        parents = await self._get_parent_set()
        to_map: list[dict[str, Any]] = []
        for tag_json in tag_results:
            if tag_json["prof_hash"] not in parents:
                continue

            topics = self._sorted_topics(tag_json)
            if topics:
                to_map.append({"topics": topics, "tag_json": tag_json})

        if not to_map:
            return {
                "processed_professors": len(missing_rows),
                "generated_tags": 0,
                "upserted_vectors": 0,
            }

        unique_tags = sorted({topic["tag"] for payload in to_map for topic in payload["topics"]})
        tag2idx = {tag: idx for idx, tag in enumerate(unique_tags)}
        tag_vecs_full = await self._embed_texts(unique_tags, batch_size=500)

        new_rows: list[pd.DataFrame] = []
        for payload in tqdm(to_map, desc="Preparing tag vectors"):
            topics = payload["topics"]
            tag_json = payload["tag_json"]
            idxs = [tag2idx[topic["tag"]] for topic in topics if topic["tag"] in tag2idx]
            if not idxs:
                continue

            selected_vecs = tag_vecs_full[idxs] if tag_vecs_full.size else np.zeros((len(topics), 0), dtype=np.float32)
            rows_df = self._assign_tags_to_subfields(selected_vecs, sub_vecs, sub_df, [topic["tag"] for topic in topics])
            rows_df["prof_id"] = tag_json["prof_hash"].hex()
            rows_df["full_name"] = tag_json["full_name"]
            rows_df["institute"] = tag_json["institute"]
            rows_df["weight"] = [topic["weight"] for topic in topics]
            rows_df["expertise"] = [topic["expertise"] for topic in topics]
            rows_df["source"] = [topic["source"] for topic in topics]
            new_rows.append(rows_df)

        if not new_rows:
            return {
                "processed_professors": len(missing_rows),
                "generated_tags": 0,
                "upserted_vectors": 0
            }

        tag_rows = pd.concat(new_rows, ignore_index=True)
        vector_dim = tag_vecs_full.shape[1] if tag_vecs_full.size else 0
        if vector_dim == 0:
            return {
                "processed_professors": len(missing_rows),
                "generated_tags": len(tag_rows),
                "upserted_vectors": 0,
            }

        self.vector_store.ensure_collection(vector_dim)
        points, deletions = self._build_tag_points(tag_rows, tag_vecs_full, tag2idx)

        if points:
            self.vector_store.upsert_points(points)
            print("Upserted", len(points), "tag vectors.")
        if deletions:
            self.vector_store.delete_points(deletions)
            print("Deleted", len(deletions), "empty tag vectors.")

        return {
            "processed_professors": len(missing_rows),
            "generated_tags": len(tag_rows),
            "upserted_vectors": len(points),
        }

    def index_data_sync(self, force_rebuild: bool = False) -> Dict[str, int]:
        return asyncio.run(self.index_data(force_rebuild=force_rebuild))
