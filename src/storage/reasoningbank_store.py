from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol

from src.config.load_config import AppConfig
from src.reasoningbank.rbmem_claims_v1 import (
    RBMemClaimsV1ValidationError,
    claim_text_projection,
    contains_kb_alias,
    is_rbmem_claims_v1,
    parse_rbmem_claims_v1,
    validate_rbmem_claims_v1,
)


class ReasoningBankError(RuntimeError):
    pass


class ReasoningBankConfigError(ReasoningBankError):
    """Misconfiguration or incompatible on-disk state (actionable by users)."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class ReasoningBankDependencyError(ReasoningBankError):
    def __init__(self, message: str, *, missing: list[str] | None = None) -> None:
        super().__init__(message)
        self.missing = missing or []


RBRole = Literal["global", "orchestrator", "mof_expert", "tio2_expert"]
RBStatus = Literal["active", "archived"]
RBType = Literal["reasoningbank_item", "manual_note"]


@dataclass(frozen=True)
class MemoryItem:
    mem_id: str
    status: RBStatus
    role: RBRole
    type: RBType
    content: str
    source_run_id: str | None
    created_at: float
    updated_at: float
    schema_version: int
    extra: dict[str, Any]


class EmbeddingFunction(Protocol):
    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002 (Chroma signature)
        ...


def _utc_ts() -> float:
    return time.time()


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _get_env_any(names: list[str], default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        if value.strip() == "":
            continue
        return value.strip()
    return default


def _get_env_bool_any(names: list[str], default: bool) -> bool:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _resolve_embedding_base_url(default: str) -> str:
    # Embeddings and chat can be routed to different OpenAI-compatible gateways.
    # Prefer the dedicated embeddings vars, and fall back to generic OpenAI vars.
    return _get_env_any(
        [
            "C2XC_EMBEDDING_API_BASE",
            "EMBEDDING_API_BASE",
            "OPENAI_API_BASE",
            "OPENAI_BASE_URL",
        ],
        default,
    )


def _resolve_embedding_api_key() -> str:
    return _get_env_any(
        [
            "C2XC_EMBEDDING_API_KEY",
            "EMBEDDING_API_KEY",
            "OPENAI_API_KEY",
        ],
        "",
    )


def _resolve_embedding_model(default: str) -> str:
    return _get_env_any(
        [
            "C2XC_EMBEDDING_MODEL",
            "EMBEDDING_MODEL",
        ],
        default,
    )


def _resolve_embedding_dim_raw() -> str:
    return _get_env_any(
        [
            "C2XC_EMBEDDING_DIM",
            "EMBEDDING_DIM",
        ],
        "",
    )


def _resolve_embedding_send_dimensions(default: bool) -> bool:
    return _get_env_bool_any(
        [
            "C2XC_EMBEDDING_SEND_DIMENSIONS",
            "EMBEDDING_SEND_DIMENSIONS",
        ],
        default,
    )


_CHROMA_OP_LOCK = threading.RLock()

_DOC_TYPE_ITEM = "item"
_DOC_TYPE_CLAIM = "claim"


def default_chroma_dir() -> Path:
    return Path(_get_env("C2XC_RB_CHROMA_DIR", "data/chroma")).expanduser().resolve()


def default_collection_name() -> str:
    return _get_env("C2XC_RB_COLLECTION", "reasoningbank")


class _HashEmbeddingFunction:
    """Deterministic embedding function for tests/dry-run.

    DO NOT use for scientific deployment. This exists to allow end-to-end pipeline tests
    without requiring network access or real embedding models.
    """

    def __init__(self, *, dim: int = 32) -> None:
        self._dim = int(dim)
        if self._dim < 8:
            self._dim = 8

    @staticmethod
    def name() -> str:
        # Chroma expects embedding functions to implement a stable name().
        return "c2xc_hash_embedding_v1"

    def embed_query(self, *, input: list[str]) -> list[list[float]]:  # noqa: A002
        # Newer Chroma versions call embed_query() for query embeddings.
        return self.__call__(input)

    def get_config(self) -> dict[str, Any]:
        # Used by Chroma to persist embedding function configuration for a collection.
        return {"dim": int(self._dim)}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "_HashEmbeddingFunction":
        dim = config.get("dim")
        try:
            dim_i = int(dim)
        except Exception:
            dim_i = 32
        return _HashEmbeddingFunction(dim=dim_i)

    def is_legacy(self) -> bool:
        # Provide the method Chroma expects for avoiding "legacy" warnings.
        return False

    def default_space(self) -> str:
        # Align with Chroma defaults; hash vectors are arbitrary but cosine is a reasonable default.
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        out: list[list[float]] = []
        for text in input:
            data = (text or "").encode("utf-8")
            digest = hashlib.sha256(data).digest()
            # Expand digest to desired dimension deterministically.
            buf = bytearray()
            seed = digest
            while len(buf) < self._dim:
                seed = hashlib.sha256(seed).digest()
                buf.extend(seed)
            vec = [((b / 255.0) * 2.0 - 1.0) for b in buf[: self._dim]]
            out.append(vec)
        return out


class _OpenAICompatibleEmbeddingFunction:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float | None = None,
        dimensions: int | None = None,
        send_dimensions: bool = False,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise ReasoningBankDependencyError(
                "Missing dependency: openai. Install it in the runtime environment.",
                missing=["python:openai"],
            ) from e

        if not api_key.strip():
            raise ReasoningBankDependencyError(
                "Missing embeddings API key for ReasoningBank (set C2XC_EMBEDDING_API_KEY or OPENAI_API_KEY).",
                missing=["C2XC_EMBEDDING_API_KEY", "OPENAI_API_KEY"],
            )

        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        self._model = model
        self._dimensions = int(dimensions) if dimensions is not None else None
        self._send_dimensions = bool(send_dimensions)
        self._base_url = base_url

    @staticmethod
    def name() -> str:
        return "c2xc_openai_compatible_embedding_v1"

    def embed_query(self, *, input: list[str]) -> list[list[float]]:  # noqa: A002
        return self.__call__(input)

    def get_config(self) -> dict[str, Any]:
        return {
            "base_url": str(self._base_url),
            "model": str(self._model),
            "dimensions": int(self._dimensions) if self._dimensions is not None else None,
            "send_dimensions": bool(self._send_dimensions),
        }

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "_OpenAICompatibleEmbeddingFunction":
        # NOTE: OPENAI_API_KEY is intentionally read from env; we do not persist secrets into Chroma config.
        base_url = _resolve_embedding_base_url(
            str(config.get("base_url") or "https://api.openai.com/v1")
        )
        model = str(config.get("model") or _resolve_embedding_model("text-embedding-3-small"))
        dimensions = config.get("dimensions")
        send_dimensions = bool(config.get("send_dimensions") or False)

        api_key = _resolve_embedding_api_key()

        dim_i: int | None
        try:
            dim_i = int(dimensions) if dimensions is not None else None
        except Exception:
            dim_i = None

        return _OpenAICompatibleEmbeddingFunction(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_s=None,
            dimensions=dim_i,
            send_dimensions=send_dimensions,
        )

    def is_legacy(self) -> bool:
        return False

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        payload: dict[str, Any] = {"model": self._model, "input": input}
        if self._send_dimensions and self._dimensions is not None:
            payload["dimensions"] = int(self._dimensions)
        resp = self._client.embeddings.create(**payload)
        # Keep ordering consistent with input.
        return [list(d.embedding) for d in resp.data]


def _build_embedding_function(*, embedding_mode: str, hash_embedding_dim: int) -> EmbeddingFunction:
    mode = (embedding_mode or "openai").strip().lower()
    if mode == "hash":
        return _HashEmbeddingFunction(dim=int(hash_embedding_dim))
    if mode != "openai":
        raise ReasoningBankError(f"Invalid C2XC_RB_EMBEDDING_MODE: {mode!r} (expected 'openai' or 'hash').")

    base_url = _resolve_embedding_base_url("https://api.openai.com/v1")
    api_key = _resolve_embedding_api_key()
    model = _resolve_embedding_model("text-embedding-3-small")

    dimensions: int | None = None
    raw_dim = _resolve_embedding_dim_raw()
    if raw_dim:
        try:
            dimensions = int(raw_dim)
        except Exception:
            dimensions = None

    send_dimensions = _resolve_embedding_send_dimensions(False)
    return _OpenAICompatibleEmbeddingFunction(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_s=None,
        dimensions=dimensions,
        send_dimensions=send_dimensions,
    )


def _parse_role(value: Any) -> RBRole:
    s = str(value or "").strip()
    if s not in {"global", "orchestrator", "mof_expert", "tio2_expert"}:
        raise ReasoningBankError(f"Invalid RB role: {s!r}")
    return s  # type: ignore[return-value]


def _parse_status(value: Any) -> RBStatus:
    s = str(value or "").strip()
    if s not in {"active", "archived"}:
        raise ReasoningBankError(f"Invalid RB status: {s!r}")
    return s  # type: ignore[return-value]


def _parse_type(value: Any) -> RBType:
    s = str(value or "").strip()
    if s not in {"reasoningbank_item", "manual_note"}:
        raise ReasoningBankError(f"Invalid RB type: {s!r}")
    return s  # type: ignore[return-value]


def _metadata_to_item(*, mem_id: str, doc: str, meta: dict[str, Any] | None) -> MemoryItem:
    meta = meta or {}
    extra: dict[str, Any]
    try:
        extra = json.loads(str(meta.get("extra_json") or "{}"))
        if not isinstance(extra, dict):
            extra = {}
    except Exception:
        extra = {}

    source_run_id: str | None = None
    raw_source = (meta.get("source_run_id") or "").strip() if isinstance(meta.get("source_run_id"), str) else meta.get("source_run_id")
    if isinstance(raw_source, str):
        source_run_id = raw_source.strip() or None
    elif raw_source is not None:
        source_run_id = str(raw_source)

    return MemoryItem(
        mem_id=str(mem_id),
        status=_parse_status(meta.get("status") or "active"),
        role=_parse_role(meta.get("role") or "global"),
        type=_parse_type(meta.get("type") or "manual_note"),
        content=str(doc or ""),
        source_run_id=source_run_id,
        created_at=float(meta.get("created_at") or 0.0),
        updated_at=float(meta.get("updated_at") or 0.0),
        schema_version=int(meta.get("schema_version") or 1),
        extra=extra,
    )


def _item_to_metadata(item: MemoryItem) -> dict[str, Any]:
    return {
        "doc_type": _DOC_TYPE_ITEM,
        "status": item.status,
        "role": item.role,
        "type": item.type,
        "source_run_id": item.source_run_id or "",
        "created_at": float(item.created_at),
        "updated_at": float(item.updated_at),
        "schema_version": int(item.schema_version),
        "extra_json": json.dumps(item.extra or {}, ensure_ascii=False, separators=(",", ":")),
    }


def _claim_doc_id(*, parent_mem_id: str, claim_id: str) -> str:
    return f"{parent_mem_id}::claim::{claim_id}"


def _claim_to_metadata(*, parent: MemoryItem, claim_id: str, claim_status: str | None) -> dict[str, Any]:
    # Mirror item filters onto claim docs so query(where=...) keeps working.
    meta: dict[str, Any] = {
        "doc_type": _DOC_TYPE_CLAIM,
        "parent_mem_id": str(parent.mem_id),
        "claim_id": str(claim_id),
        "status": str(parent.status),
        "role": str(parent.role),
        "type": str(parent.type),
        "source_run_id": str(parent.source_run_id or ""),
        # Keep these aligned with the parent item to make maintenance/debugging easier.
        "created_at": float(parent.created_at),
        "updated_at": float(parent.updated_at),
        "schema_version": int(parent.schema_version),
    }
    if claim_status:
        meta["claim_status"] = str(claim_status)
    return meta


def _build_where(
    *,
    role: list[str] | None,
    status: list[str] | None,
    type: list[str] | None,  # noqa: A002
) -> dict[str, Any] | None:
    """Build a Chroma `where` filter.

    Chroma's filter validation evolves across versions. Recent versions require the top-level `where`
    object to contain exactly one operator (e.g. {"$and": [...]}). We normalize to:
      - None (no filtering)
      - single-clause dict (one field)
      - {"$and": [clause1, clause2, ...]} when multiple fields are provided
    """
    clauses: list[dict[str, Any]] = []
    if role:
        clauses.append({"role": {"$in": [str(r) for r in role]}})
    if status:
        clauses.append({"status": {"$in": [str(s) for s in status]}})
    if type:
        clauses.append({"type": {"$in": [str(t) for t in type]}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class ReasoningBankStore:
    def __init__(
        self,
        *,
        chroma_dir: Path,
        collection_name: str,
        embedding_mode: str,
        hash_embedding_dim: int,
    ) -> None:
        self._chroma_dir = chroma_dir.expanduser().resolve()
        self._collection_name = (collection_name or "").strip() or default_collection_name()
        self._chroma_dir.mkdir(parents=True, exist_ok=True)

        try:
            import chromadb  # type: ignore
        except Exception as e:
            raise ReasoningBankDependencyError(
                "Missing dependency: chromadb. Install it in the runtime environment.",
                missing=["python:chromadb"],
            ) from e

        self._embedding_fn = _build_embedding_function(
            embedding_mode=embedding_mode,
            hash_embedding_dim=int(hash_embedding_dim),
        )
        # PersistentClient stores in a local directory; single-user/single-instance assumed.
        self._client = chromadb.PersistentClient(path=str(self._chroma_dir))
        try:
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                embedding_function=self._embedding_fn,  # type: ignore[arg-type]
            )
        except Exception as e:
            embed_cfg: dict[str, Any] = {}
            get_cfg = getattr(self._embedding_fn, "get_config", None)
            if callable(get_cfg):
                try:
                    embed_cfg = dict(get_cfg() or {})
                except Exception:
                    embed_cfg = {}

            hint = (
                "Failed to open ReasoningBank Chroma collection. This is usually caused by an existing collection "
                "created with a different embedding function/config (e.g. switching from hash to openai embeddings, "
                "changing embedding model/base_url, or upgrading Chroma).\n\n"
                "To fix (choose ONE):\n"
                "1) Create a new collection: set C2XC_RB_COLLECTION to a new name.\n"
                "2) Reset local Chroma storage: delete the directory at C2XC_RB_CHROMA_DIR.\n\n"
                "NOTE: resetting or changing collection name will make existing stored memories unavailable unless you migrate them."
            )
            details = {
                "chroma_dir": str(self._chroma_dir),
                "collection_name": str(self._collection_name),
                "embedding_mode": str(embedding_mode),
                "embedding_config": embed_cfg,
                "error": str(e),
            }
            raise ReasoningBankConfigError(hint, details=details) from e

    @classmethod
    def from_config(cls, cfg: AppConfig) -> "ReasoningBankStore":
        chroma_dir = Path(os.getenv("C2XC_RB_CHROMA_DIR", "") or cfg.reasoningbank.chroma_dir).expanduser().resolve()
        collection_name = os.getenv("C2XC_RB_COLLECTION", "") or cfg.reasoningbank.collection_name
        embedding_mode = os.getenv("C2XC_RB_EMBEDDING_MODE", "") or cfg.reasoningbank.embedding_mode
        raw_dim = os.getenv("C2XC_RB_HASH_EMBEDDING_DIM", "")
        hash_dim = int(raw_dim) if raw_dim.strip().isdigit() else int(cfg.reasoningbank.hash_embedding_dim)
        return cls(
            chroma_dir=chroma_dir,
            collection_name=collection_name,
            embedding_mode=embedding_mode,
            hash_embedding_dim=hash_dim,
        )

    @property
    def chroma_dir(self) -> Path:
        return self._chroma_dir

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def get(self, *, mem_id: str) -> MemoryItem | None:
        mid = (mem_id or "").strip()
        if not mid:
            return None
        if "::claim::" in mid:
            # Claim docs are internal index artifacts; only item ids are valid for external reads.
            return None
        with _CHROMA_OP_LOCK:
            res = self._collection.get(ids=[mid], include=["metadatas", "documents"])
        ids = res.get("ids") or []
        if not ids:
            return None
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        doc = str(docs[0] or "")
        meta = metas[0] if metas else None
        return _metadata_to_item(mem_id=mid, doc=doc, meta=meta)

    def get_many(self, *, mem_ids: list[str], include_content: bool = True) -> list[MemoryItem]:
        """Batch fetch memory items by mem_id (best-effort).

        Returns items in the same order as `mem_ids`, skipping missing ids.
        """
        want = [str(mid).strip() for mid in (mem_ids or []) if str(mid).strip()]
        if not want:
            return []

        include = ["metadatas"]
        if include_content:
            include.append("documents")

        with _CHROMA_OP_LOCK:
            res = self._collection.get(ids=want, include=include)

        ids = res.get("ids") or []
        metas = res.get("metadatas") or []
        docs = res.get("documents") or []

        by_id: dict[str, MemoryItem] = {}
        for i, mem_id in enumerate(ids):
            mid = str(mem_id)
            meta = metas[i] if i < len(metas) else None
            doc = str(docs[i] or "") if include_content and i < len(docs) else ""
            by_id[mid] = _metadata_to_item(mem_id=mid, doc=doc, meta=meta)

        out: list[MemoryItem] = []
        for mid in want:
            it = by_id.get(mid)
            if it is not None:
                out.append(it)
        return out

    def _iter_all_items(
        self, *, where: dict[str, Any] | None, include_documents: bool, batch_size: int = 200
    ) -> Iterable[MemoryItem]:
        offset = 0
        while True:
            include = ["metadatas"]
            if include_documents:
                include.append("documents")

            with _CHROMA_OP_LOCK:
                res = self._collection.get(where=where, include=include, limit=batch_size, offset=offset)
            ids = res.get("ids") or []
            if not ids:
                return
            metas = res.get("metadatas") or []
            docs = res.get("documents") or []

            for i, mem_id in enumerate(ids):
                doc = str(docs[i] or "") if include_documents and i < len(docs) else ""
                meta = metas[i] if i < len(metas) else None
                # Claim docs are internal index artifacts; keep list_all() item-only.
                if str(mem_id).find("::claim::") != -1:
                    continue
                if isinstance(meta, dict) and str(meta.get("doc_type") or "") == _DOC_TYPE_CLAIM:
                    continue
                yield _metadata_to_item(mem_id=str(mem_id), doc=doc, meta=meta)

            offset += len(ids)
            if len(ids) < batch_size:
                return

    def list_all(
        self,
        *,
        role: list[str] | None = None,
        status: list[str] | None = None,
        type: list[str] | None = None,  # noqa: A002
        include_content: bool = True,
    ) -> list[MemoryItem]:
        return list(
            self._iter_all_items(
                where=_build_where(role=role, status=status, type=type),
                include_documents=bool(include_content),
            )
        )

    def query(
        self,
        *,
        query: str,
        n_results: int,
        role: list[str] | None = None,
        status: list[str] | None = None,
        type: list[str] | None = None,  # noqa: A002
    ) -> list[dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []

        # Claim-level retrieval: query claim docs and aggregate to item docs.
        where_clauses: list[dict[str, Any]] = []
        base = _build_where(role=role, status=status, type=type)
        if base is not None:
            if "$and" in base and isinstance(base.get("$and"), list):
                where_clauses.extend(list(base["$and"]))
            else:
                where_clauses.append(base)
        where_clauses.append({"doc_type": _DOC_TYPE_CLAIM})
        where_or_none: dict[str, Any] | None
        if not where_clauses:
            where_or_none = None
        elif len(where_clauses) == 1:
            where_or_none = where_clauses[0]
        else:
            where_or_none = {"$and": where_clauses}

        # We need enough claim hits to produce n_results distinct items.
        # Use a small multiplier to reduce the risk of multiple claims from the same parent dominating.
        claim_k = max(int(n_results) * 4, int(n_results))

        with _CHROMA_OP_LOCK:
            res = self._collection.query(
                query_texts=[q],
                n_results=int(claim_k),
                where=where_or_none,
                include=["metadatas", "documents", "distances"],
            )

        ids = (res.get("ids") or [[]])[0] or []
        metas = (res.get("metadatas") or [[]])[0] or []
        docs = (res.get("documents") or [[]])[0] or []
        distances = (res.get("distances") or [[]])[0] or []

        by_parent: dict[str, dict[str, Any]] = {}
        for i, doc_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else None
            if not isinstance(meta, dict):
                continue
            if str(meta.get("doc_type") or "") != _DOC_TYPE_CLAIM:
                continue
            parent_id = str(meta.get("parent_mem_id") or "").strip()
            if not parent_id:
                continue
            dist = float(distances[i]) if i < len(distances) and distances[i] is not None else None
            hit = {
                "claim_id": str(meta.get("claim_id") or "").strip() or None,
                "claim_status": str(meta.get("claim_status") or "").strip() or None,
                "distance": dist,
                "text": str(docs[i] or "") if i < len(docs) else "",
            }
            cur = by_parent.get(parent_id)
            if cur is None:
                by_parent[parent_id] = {
                    "distance": dist,
                    "matched_claims": [hit],
                }
                continue
            # Keep best distance for ranking.
            best = cur.get("distance")
            if best is None or (dist is not None and dist < float(best)):
                cur["distance"] = dist
            cur["matched_claims"].append(hit)

        # Sort parents by best (min) distance.
        ranked = sorted(
            by_parent.items(),
            key=lambda kv: float(kv[1]["distance"]) if kv[1].get("distance") is not None else 1e9,
        )
        top_parent_ids = [pid for pid, _ in ranked[: int(n_results)]]
        if not top_parent_ids:
            return []

        items = self.get_many(mem_ids=top_parent_ids, include_content=True)
        by_id = {it.mem_id: it for it in items}

        out: list[dict[str, Any]] = []
        for parent_id, meta in ranked[: int(n_results)]:
            item = by_id.get(parent_id)
            if item is None:
                continue
            out.append(
                {
                    "item": item,
                    "distance": meta.get("distance"),
                    "matched_claims": meta.get("matched_claims") or [],
                }
            )
        return out

    def _delete_claim_docs(self, *, parent_mem_id: str) -> None:
        parent = (parent_mem_id or "").strip()
        if not parent:
            return
        try:
            self._collection.delete(where={"$and": [{"doc_type": _DOC_TYPE_CLAIM}, {"parent_mem_id": parent}]})
            return
        except Exception:
            # Fallback: list ids then delete explicitly.
            try:
                res = self._collection.get(
                    where={"$and": [{"doc_type": _DOC_TYPE_CLAIM}, {"parent_mem_id": parent}]},
                    include=[],
                    limit=5000,
                )
                ids = [str(x) for x in (res.get("ids") or []) if str(x).strip()]
                if ids:
                    self._collection.delete(ids=ids)
            except Exception:
                return

    def _build_claim_docs_for_item(self, item: MemoryItem) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        """Build derived claim docs for a given item (best-effort)."""
        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict[str, Any]] = []

        if item.type == "reasoningbank_item" and is_rbmem_claims_v1(item.content):
            try:
                parsed = parse_rbmem_claims_v1(item.content)
                for claim in parsed.claims:
                    cid = _claim_doc_id(parent_mem_id=item.mem_id, claim_id=claim.claim_id)
                    ids.append(cid)
                    docs.append(claim_text_projection(claim))
                    metas.append(_claim_to_metadata(parent=item, claim_id=claim.claim_id, claim_status=claim.status))
                return ids, docs, metas
            except Exception:
                # Fall through to legacy projection.
                pass

        # Legacy/manual note: index the full content as a single claim doc for retrieval.
        cid = _claim_doc_id(parent_mem_id=item.mem_id, claim_id="legacy")
        ids.append(cid)
        docs.append(str(item.content or "")[:1000])
        metas.append(_claim_to_metadata(parent=item, claim_id="legacy", claim_status=None))
        return ids, docs, metas

    def rebuild_claim_index(self, *, include_archived: bool = True) -> int:
        """Rebuild derived claim docs for all items in the collection.

        Returns the number of items processed.
        """
        processed = 0
        status = None if include_archived else ["active"]
        items = self.list_all(status=status, include_content=True)
        with _CHROMA_OP_LOCK:
            for it in items:
                self._delete_claim_docs(parent_mem_id=it.mem_id)
                ids, docs, metas = self._build_claim_docs_for_item(it)
                if ids:
                    self._collection.upsert(ids=ids, documents=docs, metadatas=metas)
                processed += 1
        return processed

    def upsert(
        self,
        *,
        mem_id: str | None,
        status: str,
        role: str,
        type: str,  # noqa: A002
        content: str,
        source_run_id: str | None,
        schema_version: int = 1,
        extra: dict[str, Any] | None = None,
        now_ts: float | None = None,
        preserve_created_at: bool = True,
        validate: bool = True,
    ) -> MemoryItem:
        now = float(now_ts) if now_ts is not None else _utc_ts()
        mid = (mem_id or "").strip() or _new_uuid()

        existing = self.get(mem_id=mid)
        created_at = now
        if preserve_created_at and existing is not None:
            created_at = float(existing.created_at)

        item = MemoryItem(
            mem_id=mid,
            status=_parse_status(status),
            role=_parse_role(role),
            type=_parse_type(type),
            content=str(content or "").strip(),
            source_run_id=(str(source_run_id).strip() if source_run_id is not None else None),
            created_at=created_at,
            updated_at=now,
            schema_version=int(schema_version),
            extra=extra or {},
        )

        if not item.content:
            raise ReasoningBankError("content cannot be empty.")

        # Hard gate: RB items must be structured; manual notes are free-text.
        #
        # NOTE: For strict rollback / migrations we may need to restore legacy snapshots that
        # were created before RBMEM_CLAIMS_V1 existed. Callers can set validate=False for that
        # very specific purpose (never for normal writes).
        if validate and item.type == "reasoningbank_item":
            try:
                validate_rbmem_claims_v1(item.content, max_claims=10, forbid_kb_alias=True)
            except RBMemClaimsV1ValidationError as e:
                # Provide a concise error with actionable details (issues list).
                details = "; ".join(e.issues[:6])
                raise ReasoningBankError(f"Invalid RBMEM_CLAIMS_V1 content. {details}".strip()) from e

        with _CHROMA_OP_LOCK:
            # 1) Upsert item doc (truth).
            self._collection.upsert(
                ids=[item.mem_id],
                documents=[item.content],
                metadatas=[_item_to_metadata(item)],
            )
            # 2) Rebuild derived claim docs (index).
            self._delete_claim_docs(parent_mem_id=item.mem_id)
            ids, docs, metas = self._build_claim_docs_for_item(item)
            if ids:
                self._collection.upsert(ids=ids, documents=docs, metadatas=metas)
        return item

    def archive(self, *, mem_id: str, now_ts: float | None = None) -> MemoryItem:
        existing = self.get(mem_id=mem_id)
        if existing is None:
            raise ReasoningBankError("Memory not found.")
        if existing.status == "archived":
            return existing
        return self.upsert(
            mem_id=existing.mem_id,
            status="archived",
            role=existing.role,
            type=existing.type,
            content=existing.content,
            source_run_id=existing.source_run_id,
            schema_version=existing.schema_version,
            extra=existing.extra,
            now_ts=now_ts,
            preserve_created_at=True,
            validate=False,
        )
