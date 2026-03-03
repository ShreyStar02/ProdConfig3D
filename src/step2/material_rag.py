"""
Material RAG pipeline for curating MDL materials based on NIM output.
Scans local MDL roots, builds a local index, and ranks materials by text similarity
with PBR-constraint filtering.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib
import json
import logging
import re
import time
import asyncio
import warnings
from collections import defaultdict

import numpy as np

try:
    import clip
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    try:
        from sklearn.exceptions import InconsistentVersionWarning
    except Exception:
        InconsistentVersionWarning = Warning
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    InconsistentVersionWarning = Warning

try:
    from scipy import sparse as scipy_sparse
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


_CLIP_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}


def _select_torch_device() -> str:
    if CLIP_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_clip_model(model_name: str = "ViT-B/32") -> Tuple[Any, str]:
    if not CLIP_AVAILABLE:
        raise RuntimeError("CLIP backend unavailable")

    device = _select_torch_device()
    cache_key = (model_name, device)
    cached = _CLIP_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached, device

    model, _ = clip.load(model_name, device=device)
    model.eval()
    _CLIP_MODEL_CACHE[cache_key] = model
    return model, device


PRODUCT_CATEGORY_PRIORS: Dict[str, List[str]] = {
    "shoe": ["fabric", "leather", "rubber", "plastic", "metal", "foam"],
    "bottle": ["plastic", "glass", "metal", "rubber"],
}

TEXTURE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png"}
DEFAULT_TEXTURE_TOP_K = 3


@dataclass
class MaterialDocument:
    doc_id: str
    name: str
    category: str
    source_root: str
    source_path: str
    rel_path: str
    tags: List[str]
    text: str
    pbr: Dict[str, Optional[float]]


@dataclass
class MaterialIndex:
    backend: str
    documents: List[MaterialDocument]
    embeddings: Any
    vectorizer: Any
    index_hash: str
    built_at: str

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "backend": self.backend,
            "index_hash": self.index_hash,
            "built_at": self.built_at,
            "documents": [doc.__dict__ for doc in self.documents],
        }
        metadata_path = index_dir / "index_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        if self.backend == "clip":
            embeddings_path = index_dir / "embeddings.npy"
            np.save(embeddings_path, self.embeddings)
        elif self.backend == "tfidf":
            if not SCIPY_AVAILABLE:
                raise RuntimeError("SciPy required for TF-IDF index persistence")
            matrix_path = index_dir / "embeddings.npz"
            scipy_sparse.save_npz(matrix_path, self.embeddings)
            if self.vectorizer is not None:
                if not JOBLIB_AVAILABLE:
                    raise RuntimeError("joblib required for TF-IDF vectorizer persistence")
                vectorizer_path = index_dir / "vectorizer.joblib"
                joblib.dump(self.vectorizer, vectorizer_path)
        else:
            raise RuntimeError(f"Unsupported index backend: {self.backend}")

    @staticmethod
    def load(index_dir: Path) -> Optional["MaterialIndex"]:
        metadata_path = index_dir / "index_metadata.json"
        if not metadata_path.exists():
            return None

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        documents = [MaterialDocument(**doc) for doc in metadata.get("documents", [])]
        backend = metadata.get("backend")

        if backend == "clip":
            embeddings_path = index_dir / "embeddings.npy"
            embeddings = np.load(embeddings_path) if embeddings_path.exists() else None
            return MaterialIndex(
                backend=backend,
                documents=documents,
                embeddings=embeddings,
                vectorizer=None,
                index_hash=metadata.get("index_hash", ""),
                built_at=metadata.get("built_at", ""),
            )

        if backend == "tfidf":
            if not SCIPY_AVAILABLE:
                return None
            matrix_path = index_dir / "embeddings.npz"
            embeddings = scipy_sparse.load_npz(matrix_path) if matrix_path.exists() else None
            vectorizer = None
            vectorizer_path = index_dir / "vectorizer.joblib"
            if vectorizer_path.exists() and JOBLIB_AVAILABLE:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
                    vectorizer = joblib.load(vectorizer_path)
            return MaterialIndex(
                backend=backend,
                documents=documents,
                embeddings=embeddings,
                vectorizer=vectorizer,
                index_hash=metadata.get("index_hash", ""),
                built_at=metadata.get("built_at", ""),
            )

        return None


class MaterialIndexer:
    def __init__(self, material_roots: List[Path], index_dir: Path):
        self.material_roots = [Path(root) for root in material_roots]
        self.index_dir = Path(index_dir)

    def build_or_load(self, force_rebuild: bool = False) -> MaterialIndex:
        index_hash = self._compute_index_hash()
        preferred_backend = self._select_backend()
        logger.debug(
            "Material RAG index: build_or_load start (force_rebuild=%s, index_hash=%s, preferred_backend=%s)",
            force_rebuild,
            index_hash,
            preferred_backend,
        )
        if not force_rebuild:
            metadata_path = self.index_dir / "index_metadata.json"
            cached_backend = None
            cached_hash = None
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    cached_backend = metadata.get("backend")
                    cached_hash = metadata.get("index_hash")
                except Exception:
                    cached_backend = None
                    cached_hash = None

            if cached_backend and cached_hash == index_hash:
                if preferred_backend == "clip" and cached_backend != "clip":
                    logger.info(
                        "Material RAG index cache backend '%s' is older than preferred backend 'clip'; rebuilding index",
                        cached_backend,
                    )
                elif cached_backend == "clip" and not CLIP_AVAILABLE:
                    logger.warning(
                        "Material RAG index cache uses CLIP backend but CLIP is unavailable; rebuilding index"
                    )
                elif cached_backend == "tfidf" and not SKLEARN_AVAILABLE:
                    logger.warning(
                        "Material RAG index cache uses TF-IDF backend but scikit-learn is unavailable; rebuilding index"
                    )
                else:
                    cached = MaterialIndex.load(self.index_dir)
                    if cached is not None:
                        logger.debug(
                            "Material RAG index: cache hit (backend=%s, docs=%s)",
                            cached.backend,
                            len(cached.documents),
                        )
                        return cached

        documents = self._scan_materials()
        if not documents:
            logger.debug("Material RAG index: no materials found")
            return MaterialIndex(
                backend="none",
                documents=[],
                embeddings=None,
                vectorizer=None,
                index_hash=index_hash,
                built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )

        backend = preferred_backend
        logger.debug("Material RAG index: building backend=%s docs=%s", backend, len(documents))
        if backend == "clip":
            embeddings = self._embed_clip([doc.text for doc in documents])
            index = MaterialIndex(
                backend="clip",
                documents=documents,
                embeddings=embeddings,
                vectorizer=None,
                index_hash=index_hash,
                built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
        else:
            if not SKLEARN_AVAILABLE:
                raise RuntimeError("scikit-learn required for TF-IDF indexing")
            vectorizer = TfidfVectorizer()
            embeddings = vectorizer.fit_transform([doc.text for doc in documents])
            index = MaterialIndex(
                backend="tfidf",
                documents=documents,
                embeddings=embeddings,
                vectorizer=vectorizer,
                index_hash=index_hash,
                built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )

        index.save(self.index_dir)
        logger.debug("Material RAG index: saved to %s", self.index_dir)
        return index

    def _select_backend(self) -> str:
        if CLIP_AVAILABLE:
            return "clip"
        return "tfidf"

    def _scan_materials(self) -> List[MaterialDocument]:
        documents: List[MaterialDocument] = []
        for root in self.material_roots:
            if not root.exists():
                logger.warning("Material root not found: %s", root)
                continue
            root_count = 0
            for path in root.rglob("*.mdl"):
                doc = self._create_document(root, path)
                if doc:
                    documents.append(doc)
                    root_count += 1
            logger.debug("Material RAG index: scanned %s (%s docs)", root, root_count)
        logger.debug("Material RAG index: scanned total docs=%s", len(documents))
        return documents

    def _create_document(self, root: Path, path: Path) -> Optional[MaterialDocument]:
        try:
            rel_path = str(path.relative_to(root))
        except ValueError:
            rel_path = str(path)

        name = path.stem
        category = path.parent.name
        tags = [part.lower() for part in path.parent.parts if part]
        text_meta, pbr = self._extract_mdl_metadata(path)
        text = " ".join(filter(None, [name, category, rel_path, text_meta, " ".join(tags)]))

        doc_id = hashlib.sha256(f"{root}|{rel_path}".encode("utf-8")).hexdigest()
        return MaterialDocument(
            doc_id=doc_id,
            name=name,
            category=category,
            source_root=str(root),
            source_path=str(path),
            rel_path=rel_path,
            tags=tags,
            text=text,
            pbr=pbr,
        )

    def _extract_mdl_metadata(self, path: Path) -> Tuple[str, Dict[str, Optional[float]]]:
        try:
            content = path.read_text(errors="ignore")
        except Exception:
            return "", {"roughness": None, "metallic": None, "opacity": None}

        display = _match_first(r"display_name\(\"([^\"]+)\"\)", content)
        description = _match_first(r"description\(\"([^\"]+)\"\)", content)
        text_meta = " ".join(filter(None, [display, description]))

        roughness = _match_float(r"\broughness\b\s*[:=]\s*([0-9]*\.?[0-9]+)", content)
        metallic = _match_float(r"\bmetallic\b\s*[:=]\s*([0-9]*\.?[0-9]+)", content)
        opacity = _match_float(r"\bopacity\b\s*[:=]\s*([0-9]*\.?[0-9]+)", content)

        return text_meta, {"roughness": roughness, "metallic": metallic, "opacity": opacity}

    def _embed_clip(self, texts: List[str]) -> np.ndarray:
        logger.debug("Material RAG index: embedding %s docs with CLIP", len(texts))
        model, device = _get_clip_model("ViT-B/32")
        embeddings = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                tokens = clip.tokenize(batch, truncate=True).to(device)
                features = model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                embeddings.append(features.cpu().numpy())
                logger.debug(
                    "Material RAG index: embedded batch %s-%s",
                    i,
                    min(i + batch_size, len(texts)),
                )
        return np.vstack(embeddings)

    def _compute_index_hash(self) -> str:
        hasher = hashlib.sha256()
        for root in sorted(self.material_roots):
            hasher.update(str(root).encode("utf-8"))
            if not root.exists():
                continue
            for path in sorted(root.rglob("*.mdl")):
                try:
                    stat = path.stat()
                    hasher.update(str(path).encode("utf-8"))
                    hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
                    hasher.update(str(stat.st_size).encode("utf-8"))
                except OSError:
                    continue
        return hasher.hexdigest()


class MaterialRAGCurator:
    def __init__(self, material_roots: List[Path], index_dir: Path):
        self.indexer = MaterialIndexer(material_roots, index_dir)

    async def curate(
        self,
        segments: List[Any],
        ai_materials: Dict[str, Any],
        product_name: str,
        top_k: int,
        candidate_pool_size: int,
        constraints: Dict[str, float],
        force_rebuild: bool = False,
        similarity_threshold: Optional[float] = None,
        allowlist_strict: Optional[bool] = None,
        allowlist_policy: Optional[str] = None,
        product_context: Optional[Dict[str, Any]] = None,
        use_product_name_in_query: Optional[bool] = None,
        nim_client: Optional[Any] = None,
        nim_rerank_enabled: bool = False,
        nim_rerank_temperature: Optional[float] = None,
        nim_rerank_max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        logger.debug(
            "Material RAG curate: segments=%s top_k=%s pool=%s",
            len(segments),
            top_k,
            candidate_pool_size,
        )
        texture_catalog = _build_texture_catalog(self.indexer.material_roots)
        index = self.indexer.build_or_load(force_rebuild=force_rebuild)
        if index.backend == "none" or not index.documents:
            return {
                "version": "1.0",
                "product_name": product_name,
                "index_info": {
                    "backend": index.backend,
                    "index_hash": index.index_hash,
                    "built_at": index.built_at,
                    "material_roots": [str(root) for root in self.indexer.material_roots],
                    "total_materials": 0,
                },
                "segments": [],
                "warnings": ["No MDL materials found in configured roots."],
            }

        curated_segments = []
        rerank_requests: List[Tuple[int, str, str, str, Dict[str, Any], List[Dict[str, Any]], int]] = []
        effective_allowlist_policy = _resolve_allowlist_policy(allowlist_policy, allowlist_strict)
        product_category = str((product_context or {}).get("product_category") or "").strip().lower() or None
        query_uses_raw_product_name = bool(use_product_name_in_query) if use_product_name_in_query is not None else False
        for segment in segments:
            seg_label, seg_id, part_type, path_context = _segment_fields(segment)
            ai_rec = ai_materials.get(seg_label, {}) if ai_materials else {}
            allowed_categories = _extract_allowed_categories(ai_rec)
            category_source = "ai"
            if not allowed_categories and product_category:
                allowed_categories = _infer_part_allowed_categories(seg_label, part_type, product_category)
                if allowed_categories:
                    category_source = "heuristic"

            segment_allowlist_policy = effective_allowlist_policy
            if category_source == "heuristic" and segment_allowlist_policy == "off":
                segment_allowlist_policy = "soft"

            query_text, desired_pbr, texture_recs = _build_query(
                product_name,
                seg_label,
                part_type,
                ai_rec,
                path_context,
                product_context=product_context,
                use_product_name_in_query=query_uses_raw_product_name,
            )
            logger.debug(
                "Material RAG curate: segment=%s allowed_categories=%s",
                seg_label,
                ",".join(allowed_categories) if allowed_categories else "<none>",
            )
            candidates, retrieval_stats = self._rank_candidates(
                index,
                query_text,
                desired_pbr,
                constraints,
                similarity_threshold,
                allowed_categories,
                segment_allowlist_policy,
                product_category,
            )
            top_candidates = candidates[:candidate_pool_size]
            self._attach_texture_recommendations(
                top_candidates,
                texture_hints=texture_recs,
                texture_catalog=texture_catalog,
                texture_top_k=DEFAULT_TEXTURE_TOP_K,
            )
            final_candidates = top_candidates[:top_k]
            rerank_info = None

            logger.debug(
                "Material RAG curate: segment=%s candidates=%s top_pool=%s",
                seg_label,
                len(candidates),
                len(top_candidates),
            )

            curated_segments.append({
                "label": seg_label,
                "segment_id": seg_id,
                "query": {
                    "text": query_text,
                    "textures": texture_recs,
                    "desired_pbr": desired_pbr,
                },
                "candidates": final_candidates,
                "rerank": rerank_info,
                "retrieval_stats": retrieval_stats,
                "category_source": category_source,
            })

            if nim_rerank_enabled and nim_client is not None and top_candidates:
                rerank_requests.append(
                    (
                        len(curated_segments) - 1,
                        seg_label,
                        part_type,
                        query_text,
                        ai_rec,
                        top_candidates,
                        top_k,
                    )
                )

        if nim_rerank_enabled and nim_client is not None and rerank_requests:
            max_concurrency = _resolve_nim_max_concurrency(nim_client)
            semaphore = asyncio.Semaphore(max_concurrency)

            async def _rerank_one(
                req: Tuple[int, str, str, str, Dict[str, Any], List[Dict[str, Any]], int]
            ) -> Tuple[int, List[Dict[str, Any]]]:
                idx, seg_label, part_type, query_text, ai_rec, top_candidates, top_k = req
                async with semaphore:
                    reranked = await _nim_rerank_candidates(
                        nim_client,
                        product_name,
                        seg_label,
                        part_type,
                        query_text,
                        ai_rec,
                        top_candidates,
                        top_k,
                        nim_rerank_temperature,
                        nim_rerank_max_tokens,
                    )
                return idx, reranked

            rerank_results = await asyncio.gather(*[_rerank_one(req) for req in rerank_requests])
            for idx, reranked in rerank_results:
                if not reranked:
                    continue
                curated_segments[idx]["candidates"] = reranked
                curated_segments[idx]["rerank"] = {
                    "enabled": True,
                    "source": "nim",
                    "returned": len(reranked),
                }

        return {
            "version": "1.0",
            "product_name": product_name,
            "index_info": {
                "backend": index.backend,
                "index_hash": index.index_hash,
                "built_at": index.built_at,
                "material_roots": [str(root) for root in self.indexer.material_roots],
                "total_materials": len(index.documents),
            },
            "segments": curated_segments,
            "warnings": [],
        }

    def _attach_texture_recommendations(
        self,
        candidates: List[Dict[str, Any]],
        texture_hints: List[str],
        texture_catalog: Dict[str, Any],
        texture_top_k: int,
    ) -> None:
        for candidate in candidates:
            texture_paths = _resolve_candidate_texture_paths(candidate, texture_catalog)
            recommended_paths, reason = _rank_texture_paths(
                texture_paths=texture_paths,
                texture_hints=texture_hints,
                candidate_name=str(candidate.get("name") or ""),
                candidate_category=str(candidate.get("category") or ""),
                top_k=texture_top_k,
            )

            candidate["recommended_texture_paths"] = recommended_paths
            candidate["recommended_texture_path"] = recommended_paths[0] if recommended_paths else None
            candidate["texture_match_reason"] = reason

    def _rank_candidates(
        self,
        index: MaterialIndex,
        query_text: str,
        desired_pbr: Dict[str, Optional[float]],
        constraints: Dict[str, float],
        similarity_threshold: float,
        allowed_categories: List[str],
        allowlist_policy: str,
        product_category: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        similarity_scores = _compute_similarity(index, query_text)
        candidates = []
        allowed_set = {cat.lower() for cat in allowed_categories}
        prior_categories = set(PRODUCT_CATEGORY_PRIORS.get(product_category or "", []))
        stats = {
            "pre_filter_count": len(index.documents),
            "dropped_by_similarity": 0,
            "dropped_by_allowlist": 0,
            "dropped_by_pbr": 0,
            "allowlist_policy": allowlist_policy,
            "product_category": product_category,
        }
        for doc, similarity in zip(index.documents, similarity_scores):
            if similarity_threshold is not None and similarity < similarity_threshold:
                stats["dropped_by_similarity"] += 1
                continue
            doc_category = (doc.category or "").lower()
            if allowed_set and allowlist_policy == "strict" and doc_category not in allowed_set:
                stats["dropped_by_allowlist"] += 1
                continue
            hard_pass = _passes_hard_constraints(doc.pbr, desired_pbr, constraints)
            if not hard_pass:
                stats["dropped_by_pbr"] += 1
                continue

            adjusted_similarity = float(similarity)
            if allowed_set and allowlist_policy == "soft":
                if doc_category in allowed_set:
                    adjusted_similarity += 0.15
                else:
                    adjusted_similarity -= 0.20

            # Product-category priors are applied in ranking logic, not prompt text.
            if prior_categories:
                if doc_category in prior_categories:
                    adjusted_similarity += 0.04
                else:
                    adjusted_similarity -= 0.08

            candidates.append({
                "doc_id": doc.doc_id,
                "name": doc.name,
                "category": doc.category,
                "source_root": doc.source_root,
                "source_path": doc.source_path,
                "rel_path": doc.rel_path,
                "tags": doc.tags,
                "score": round(adjusted_similarity, 6),
                "similarity": round(float(similarity), 6),
                "pbr_values": doc.pbr,
            })

        candidates.sort(key=lambda item: item["score"], reverse=True)
        stats["post_filter_count"] = len(candidates)
        logger.debug(
            "Material RAG rank: backend=%s query_len=%s candidates=%s",
            index.backend,
            len(query_text),
            len(candidates),
        )
        return candidates, stats


def _compute_similarity(index: MaterialIndex, query_text: str) -> List[float]:
    logger.debug(
        "Material RAG similarity: backend=%s docs=%s",
        index.backend,
        len(index.documents),
    )
    if index.backend == "clip":
        if not CLIP_AVAILABLE:
            raise RuntimeError("CLIP backend unavailable for similarity computation")
        model, device = _get_clip_model("ViT-B/32")
        with torch.no_grad():
            tokens = clip.tokenize([query_text], truncate=True).to(device)
            query_vec = model.encode_text(tokens)
            query_vec = query_vec / query_vec.norm(dim=-1, keepdim=True)
            query_vec = query_vec.cpu().numpy()
        scores = (index.embeddings @ query_vec.T).squeeze(axis=1)
        return scores.tolist()

    if index.backend == "tfidf":
        if index.vectorizer is None or index.embeddings is None:
            return [0.0] * len(index.documents)
        query_vec = index.vectorizer.transform([query_text])
        scores = cosine_similarity(index.embeddings, query_vec).reshape(-1)
        return scores.tolist()

    return [0.0] * len(index.documents)


def _build_query(
    product_name: str,
    segment_label: str,
    part_type: str,
    ai_rec: Dict[str, Any],
    path_context: Optional[str] = None,
    product_context: Optional[Dict[str, Any]] = None,
    use_product_name_in_query: bool = False,
) -> Tuple[str, Dict[str, Optional[float]], List[str]]:
    primary = ai_rec.get("primary_material", {})
    alternatives = ai_rec.get("alternative_materials", [])
    texture_recs = ai_rec.get("texture_recommendations", [])

    tokens = [segment_label, part_type]
    if use_product_name_in_query and product_name:
        tokens.insert(0, product_name)

    if product_context:
        category = str(product_context.get("product_category") or "").strip()
        if category:
            tokens.insert(0, category)
        product_tokens = product_context.get("product_tokens") or []
        if isinstance(product_tokens, list):
            tokens.extend(str(tok) for tok in product_tokens if str(tok).strip())
    if path_context:
        tokens.append(path_context)
    if primary.get("name"):
        tokens.append(primary.get("name"))
    if primary.get("category"):
        tokens.append(primary.get("category"))
    for alt in alternatives:
        if alt.get("name"):
            tokens.append(alt.get("name"))
        if alt.get("category"):
            tokens.append(alt.get("category"))
    tokens.extend(texture_recs)

    desired_pbr = {}
    pbr = primary.get("pbr_properties", {})
    for key in ["roughness", "metallic", "opacity"]:
        if key in pbr:
            desired_pbr[key] = pbr.get(key)

    return " ".join([tok for tok in tokens if tok]), desired_pbr, texture_recs


def _extract_allowed_categories(ai_rec: Dict[str, Any]) -> List[str]:
    categories = []
    primary = ai_rec.get("primary_material", {})
    if primary.get("category"):
        categories.append(str(primary.get("category")))
    for alt in ai_rec.get("alternative_materials", []):
        if alt.get("category"):
            categories.append(str(alt.get("category")))
    return [cat.strip() for cat in categories if cat and str(cat).strip()]


def _resolve_allowlist_policy(
    allowlist_policy: Optional[str],
    allowlist_strict: Optional[bool],
) -> str:
    if allowlist_policy:
        policy = str(allowlist_policy).strip().lower()
        if policy in {"off", "soft", "strict"}:
            return policy

    if allowlist_strict is True:
        return "strict"
    if allowlist_strict is False:
        return "off"

    # Default: soft guidance to reduce odd out-of-domain picks without over-filtering.
    return "soft"


def _resolve_nim_max_concurrency(nim_client: Any) -> int:
    config = getattr(nim_client, "config", None)
    value = getattr(config, "max_concurrency", 3) if config is not None else 3
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 3
    return max(1, parsed)


def _infer_part_allowed_categories(
    segment_label: str,
    part_type: str,
    product_category: str,
) -> List[str]:
    if product_category != "shoe":
        return []

    tokens = f"{segment_label} {part_type}".lower()

    if any(term in tokens for term in ["sole", "outsole", "midsole", "bottom"]):
        return ["rubber", "foam", "plastic"]
    if any(term in tokens for term in ["eyelet", "lace_hole", "grommet"]):
        return ["metal", "plastic", "rubber"]
    if any(term in tokens for term in ["lace", "tongue", "sock", "lining", "upper", "body", "patch", "loop", "tag"]):
        return ["fabric", "leather", "plastic", "rubber"]

    return ["fabric", "leather", "rubber", "plastic", "foam"]


async def _nim_rerank_candidates(
    nim_client: Any,
    product_name: str,
    segment_label: str,
    part_type: str,
    query_text: str,
    ai_rec: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    top_k: int,
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> List[Dict[str, Any]]:
    request = {
        "product": product_name,
        "segment": {
            "label": segment_label,
            "part_type": part_type,
            "query_text": query_text,
            "ai_materials": ai_rec
        },
        "candidates": [
            {
                "doc_id": item.get("doc_id"),
                "name": item.get("name"),
                "category": item.get("category"),
                "rel_path": item.get("rel_path")
            }
            for item in candidates
        ],
        "top_k": top_k
    }

    prompt = (
        "Select the best materials for the product segment using ONLY the provided candidates. "
        "Return strict JSON with this schema:\n"
        "{\n"
        "  \"selected\": [\n"
        "    {\"doc_id\": \"...\", \"reason\": \"...\"}\n"
        "  ]\n"
        "}\n\n"
        "Use the segment context and AI material hints. Do not invent candidates."\
        f"\n\nPayload:\n{json.dumps(request, indent=2)}"
    )

    response = None
    logger.debug(
        "Material RAG NIM rerank: candidates=%s top_k=%s",
        len(candidates),
        top_k,
    )
    if temperature is not None and max_tokens is not None:
        response = await nim_client.chat_completion(prompt, temperature=temperature, max_tokens=max_tokens)
    elif temperature is not None:
        response = await nim_client.chat_completion(prompt, temperature=temperature)
    elif max_tokens is not None:
        response = await nim_client.chat_completion(prompt, max_tokens=max_tokens)
    else:
        response = await nim_client.chat_completion(prompt)

    if not response or not response.success:
        logger.debug("Material RAG NIM rerank: no response or failed")
        return []

    content = response.data
    json_start = content.find("{")
    json_end = content.rfind("}") + 1
    if json_start < 0 or json_end <= json_start:
        logger.debug("Material RAG NIM rerank: no JSON payload in response")
        return []

    try:
        parsed = json.loads(content[json_start:json_end])
    except json.JSONDecodeError:
        logger.debug("Material RAG NIM rerank: JSON parse failed")
        return []

    selected = parsed.get("selected", [])
    if not isinstance(selected, list):
        logger.debug("Material RAG NIM rerank: selected is not a list")
        return []

    candidate_map = {item.get("doc_id"): item for item in candidates}
    reranked = []
    for item in selected:
        doc_id = item.get("doc_id")
        if not doc_id or doc_id not in candidate_map:
            continue
        candidate = dict(candidate_map[doc_id])
        candidate["rerank_reason"] = item.get("reason")
        reranked.append(candidate)

    logger.debug("Material RAG NIM rerank: returned=%s", len(reranked))

    return reranked


def _segment_fields(segment: Any) -> Tuple[str, Optional[int], str, Optional[str]]:
    if hasattr(segment, "label"):
        label = segment.label
        seg_id = getattr(segment, "segment_id", None)
        part_type = segment.properties.get("part_type", "unknown") if hasattr(segment, "properties") else "unknown"
        path_context = None
        if hasattr(segment, "properties"):
            mesh_paths = segment.properties.get("mesh_paths")
            if isinstance(mesh_paths, list) and mesh_paths:
                path_context = " | ".join(str(path) for path in mesh_paths)
        return label, seg_id, part_type, path_context
    if isinstance(segment, dict):
        path_context = None
        mesh_paths = segment.get("mesh_paths")
        if isinstance(mesh_paths, list) and mesh_paths:
            path_context = " | ".join(str(path) for path in mesh_paths)
        return segment.get("label", "unknown"), segment.get("id"), segment.get("part_type", "unknown"), path_context
    return "unknown", None, "unknown", None


def _passes_hard_constraints(
    pbr_values: Dict[str, Optional[float]],
    desired: Dict[str, Optional[float]],
    constraints: Dict[str, float]
) -> bool:
    for key in ["roughness", "metallic", "opacity"]:
        target = desired.get(key)
        if target is None:
            continue
        doc_val = pbr_values.get(key)
        if doc_val is None:
            continue
        tolerance = constraints.get(f"{key}_tolerance")
        if tolerance is None:
            continue
        if abs(doc_val - target) > tolerance:
            return False
    return True


def _match_first(pattern: str, text: str) -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def _match_float(pattern: str, text: str) -> Optional[float]:
    match = re.search(pattern, text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _normalize_dir_key(path: Path) -> str:
    return str(path.resolve()).lower()


def _tokenize_text(value: str) -> Set[str]:
    return {tok for tok in re.split(r"[^a-z0-9]+", (value or "").lower()) if tok}


def _build_texture_catalog(material_roots: List[Path]) -> Dict[str, Any]:
    textures_by_dir: Dict[str, List[Path]] = defaultdict(list)
    textures_by_root: Dict[str, List[Path]] = defaultdict(list)

    for root in material_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue

        root_key = _normalize_dir_key(root_path)
        for file_path in root_path.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in TEXTURE_EXTENSIONS:
                continue

            resolved = file_path.resolve()
            textures_by_dir[_normalize_dir_key(resolved.parent)].append(resolved)
            textures_by_root[root_key].append(resolved)

    for key in list(textures_by_dir.keys()):
        textures_by_dir[key] = sorted(textures_by_dir[key], key=lambda p: str(p).lower())
    for key in list(textures_by_root.keys()):
        textures_by_root[key] = sorted(textures_by_root[key], key=lambda p: str(p).lower())

    return {
        "by_dir": textures_by_dir,
        "by_root": textures_by_root,
    }


def _resolve_candidate_texture_paths(candidate: Dict[str, Any], texture_catalog: Dict[str, Any]) -> List[Path]:
    by_dir: Dict[str, List[Path]] = texture_catalog.get("by_dir", {})
    by_root: Dict[str, List[Path]] = texture_catalog.get("by_root", {})

    source_path_raw = str(candidate.get("source_path") or "").strip()
    source_root_raw = str(candidate.get("source_root") or "").strip()
    rel_path_raw = str(candidate.get("rel_path") or "").strip()
    category = str(candidate.get("category") or "").strip()

    source_path = Path(source_path_raw) if source_path_raw else None
    source_root = Path(source_root_raw) if source_root_raw else None
    rel_path = Path(rel_path_raw) if rel_path_raw else None

    search_dirs: List[Path] = []
    if source_path:
        search_dirs.extend([
            source_path.parent / "textures",
            source_path.parent,
        ])

    if source_root and rel_path and rel_path.parent != Path("."):
        rel_parent = source_root / rel_path.parent
        search_dirs.extend([
            rel_parent / "textures",
            rel_parent,
        ])

    if source_root and category:
        search_dirs.extend([
            source_root / category / "textures",
            source_root / category,
            source_root / "textures" / category,
            source_root / "textures",
        ])

    resolved_candidates: List[Path] = []
    seen_paths: Set[str] = set()
    for folder in search_dirs:
        key = _normalize_dir_key(folder)
        for tex_path in by_dir.get(key, []):
            tex_key = str(tex_path).lower()
            if tex_key in seen_paths:
                continue
            seen_paths.add(tex_key)
            resolved_candidates.append(tex_path)

    # Fallback: use any texture found under the same root if no direct folder match was found.
    if not resolved_candidates and source_root:
        root_key = _normalize_dir_key(source_root)
        for tex_path in by_root.get(root_key, []):
            tex_key = str(tex_path).lower()
            if tex_key in seen_paths:
                continue
            seen_paths.add(tex_key)
            resolved_candidates.append(tex_path)

    return resolved_candidates


def _rank_texture_paths(
    texture_paths: List[Path],
    texture_hints: List[str],
    candidate_name: str,
    candidate_category: str,
    top_k: int,
) -> Tuple[List[str], str]:
    if not texture_paths:
        return [], "no_texture_found"

    hint_tokens = _tokenize_text(" ".join(texture_hints or []))
    name_tokens = _tokenize_text(candidate_name)
    category_tokens = _tokenize_text(candidate_category)

    scored: List[Tuple[float, str, Path]] = []
    for tex_path in texture_paths:
        stem_tokens = _tokenize_text(tex_path.stem)
        path_tokens = _tokenize_text(str(tex_path.parent))

        hint_hits = len(hint_tokens.intersection(stem_tokens.union(path_tokens)))
        name_hits = len(name_tokens.intersection(stem_tokens))
        category_hits = len(category_tokens.intersection(path_tokens))

        score = (hint_hits * 3.0) + (name_hits * 1.5) + (category_hits * 1.0)
        if "texture" in path_tokens:
            score += 0.1

        # Secondary deterministic tie breaker is full path string.
        scored.append((score, str(tex_path).lower(), tex_path))

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [str(item[2]) for item in scored[:max(1, top_k)]]

    top_score = scored[0][0]
    if top_score <= 0:
        return selected, "fallback_any"

    top_path = scored[0][2]
    top_tokens = _tokenize_text(top_path.stem).union(_tokenize_text(str(top_path.parent)))
    if hint_tokens.intersection(top_tokens):
        return selected, "query_hint_match"
    if name_tokens.intersection(_tokenize_text(top_path.stem)):
        return selected, "mdl_name_match"
    if category_tokens.intersection(_tokenize_text(str(top_path.parent))):
        return selected, "category_path_match"

    return selected, "heuristic_match"
