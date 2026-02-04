from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    pass


def _as_int(value: Any, *, key: str) -> int:
    try:
        return int(value)
    except Exception as e:
        raise ConfigError(f"Invalid int for {key}: {value!r}") from e


def _as_str(value: Any, *, key: str) -> str:
    if value is None:
        raise ConfigError(f"Missing required config key: {key}")
    return str(value)

def _as_float(value: Any, *, key: str) -> float:
    try:
        return float(value)
    except Exception as e:
        raise ConfigError(f"Invalid float for {key}: {value!r}") from e


def _require_upper_alpha(value: Any, *, key: str) -> str:
    """Parse an uppercase A-Z string (used for citation alias prefixes)."""
    s = _as_str(value, key=key).strip()
    if not s:
        raise ConfigError(f"Invalid {key}: empty string")
    if not s.isalpha() or s.upper() != s:
        raise ConfigError(f"Invalid {key}: must be uppercase letters A-Z, got {s!r}")
    return s

def _resolve_path(
    value: Any, *, key: str, base_dir: Path, fallback_base_dirs: list[Path] | None = None
) -> Path:
    raw = _as_str(value, key=key)
    p = Path(raw).expanduser()
    if not p.is_absolute():
        candidates = [base_dir]
        if fallback_base_dirs:
            candidates.extend(fallback_base_dirs)
        # As a last resort, interpret relative paths against current working directory.
        candidates.append(Path.cwd())

        resolved: Path | None = None
        for b in candidates:
            candidate = (b / p).resolve()
            if candidate.exists():
                resolved = candidate
                break
        p = resolved or (base_dir / p).resolve()
    else:
        p = p.resolve()
    return p


@dataclass(frozen=True)
class LimitsConfig:
    n_runs_max: int
    recipes_per_run_max: int


@dataclass(frozen=True)
class RecapConfig:
    max_rounds: int
    max_depth: int
    max_steps: int
    acceptance_max_repairs: int


@dataclass(frozen=True)
class KBConfig:
    default_mode: str
    default_top_k: int


@dataclass(frozen=True)
class CitationConfig:
    alias_prefix: str


@dataclass(frozen=True)
class EvidenceConfig:
    max_full_chunks_in_generate_recipes: int
    kb_list_default_limit: int
    kb_list_max_limit: int


@dataclass(frozen=True)
class ReasoningBankConfig:
    chroma_dir: str
    collection_name: str
    embedding_mode: str
    hash_embedding_dim: int
    k_role: int
    k_global: int
    max_full_memories_in_generate_recipes: int
    mem_list_default_limit: int
    mem_list_max_limit: int
    near_duplicate_threshold: float
    strategy_version: str
    context_template: str
    extract_prompt_template: str
    merge_prompt_template: str
    # RB learn dereference (B-scheme): allow extractor to open factual originals (no LLM logs by default).
    learn_deref_max_calls_total: int
    learn_deref_max_full_calls: int
    learn_deref_max_chars_total: int
    learn_deref_excerpt_chars: int
    learn_deref_full_chars: int
    learn_deref_list_events_default_limit: int
    learn_deref_list_events_max_limit: int
    # RB learn: candidate memory selection for claim-level state updates.
    learn_candidate_max_items: int
    learn_candidate_semantic_top_k: int


@dataclass(frozen=True)
class RolesConfig:
    """Role -> instruction string injected into ReCAP prompts."""

    by_role: dict[str, str]

    def get(self, role: str, default: str = "") -> str:
        return self.by_role.get(role, default)


@dataclass(frozen=True)
class PriorsConfig:
    """Authoritative / strong priors that are always injected into the system prompt."""

    system_description_path: str
    microenvironment_tio2_path: str
    microenvironment_mof_path: str

    system_description_md: str
    microenvironment_tio2_md: str
    microenvironment_mof_md: str


@dataclass(frozen=True)
class PromptConfig:
    system_base: str
    down_prompt_template: str
    action_taken_prompt_template: str
    up_prompt_template: str
    generate_recipes_prompt_template: str


@dataclass(frozen=True)
class AppConfig:
    limits: LimitsConfig
    recap: RecapConfig
    kb: KBConfig
    citations: CitationConfig
    evidence: EvidenceConfig
    reasoningbank: ReasoningBankConfig
    roles: RolesConfig
    priors: PriorsConfig
    prompts: PromptConfig


def default_config_path() -> Path:
    return Path(os.getenv("C2XC_CONFIG_PATH", "config/default.toml")).expanduser().resolve()


def load_app_config(path: Path | None = None) -> AppConfig:
    cfg_path = path or default_config_path()
    if not cfg_path.exists():
        raise ConfigError(f"Config file not found: {cfg_path}")

    try:
        import tomllib  # py3.11+
    except Exception as e:
        raise ConfigError("tomllib is required (Python 3.11+).") from e

    raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))

    limits = raw.get("limits", {})
    recap = raw.get("recap", {})
    kb = raw.get("kb", {})
    citations = raw.get("citations", {})
    evidence = raw.get("evidence", {})
    reasoningbank = raw.get("reasoningbank", {})
    roles = raw.get("roles", {})
    priors = raw.get("priors", {})
    prompts = raw.get("prompts", {})

    base_dir = cfg_path.parent
    # Common case: config lives in `<repo>/config/default.toml`, but prior files live in `<repo>/docs/...`.
    fallback_base_dirs = [base_dir.parent]
    system_description_path = _resolve_path(
        priors.get("system_description_path"),
        key="priors.system_description_path",
        base_dir=base_dir,
        fallback_base_dirs=fallback_base_dirs,
    )
    microenvironment_tio2_path = _resolve_path(
        priors.get("microenvironment_tio2_path"),
        key="priors.microenvironment_tio2_path",
        base_dir=base_dir,
        fallback_base_dirs=fallback_base_dirs,
    )
    microenvironment_mof_path = _resolve_path(
        priors.get("microenvironment_mof_path"),
        key="priors.microenvironment_mof_path",
        base_dir=base_dir,
        fallback_base_dirs=fallback_base_dirs,
    )

    for p, key in (
        (system_description_path, "priors.system_description_path"),
        (microenvironment_tio2_path, "priors.microenvironment_tio2_path"),
        (microenvironment_mof_path, "priors.microenvironment_mof_path"),
    ):
        if not p.exists():
            raise ConfigError(f"Prior file not found for {key}: {p}")

    return AppConfig(
        limits=LimitsConfig(
            n_runs_max=_as_int(limits.get("n_runs_max"), key="limits.n_runs_max"),
            recipes_per_run_max=_as_int(
                limits.get("recipes_per_run_max"), key="limits.recipes_per_run_max"
            ),
        ),
        recap=RecapConfig(
            max_rounds=_as_int(recap.get("max_rounds"), key="recap.max_rounds"),
            max_depth=_as_int(recap.get("max_depth"), key="recap.max_depth"),
            max_steps=_as_int(recap.get("max_steps"), key="recap.max_steps"),
            acceptance_max_repairs=_as_int(
                recap.get("acceptance_max_repairs", 3), key="recap.acceptance_max_repairs"
            ),
        ),
        kb=KBConfig(
            default_mode=_as_str(kb.get("default_mode"), key="kb.default_mode"),
            default_top_k=_as_int(kb.get("default_top_k"), key="kb.default_top_k"),
        ),
        citations=CitationConfig(
            alias_prefix=_require_upper_alpha(
                citations.get("alias_prefix"), key="citations.alias_prefix"
            ),
        ),
        evidence=EvidenceConfig(
            max_full_chunks_in_generate_recipes=_as_int(
                evidence.get("max_full_chunks_in_generate_recipes"),
                key="evidence.max_full_chunks_in_generate_recipes",
            ),
            kb_list_default_limit=_as_int(
                evidence.get("kb_list_default_limit"), key="evidence.kb_list_default_limit"
            ),
            kb_list_max_limit=_as_int(
                evidence.get("kb_list_max_limit"), key="evidence.kb_list_max_limit"
            ),
        ),
        reasoningbank=ReasoningBankConfig(
            chroma_dir=str(
                _resolve_path(
                    reasoningbank.get("chroma_dir"),
                    key="reasoningbank.chroma_dir",
                    base_dir=base_dir,
                    fallback_base_dirs=fallback_base_dirs,
                )
            ),
            collection_name=_as_str(reasoningbank.get("collection_name"), key="reasoningbank.collection_name"),
            embedding_mode=_as_str(reasoningbank.get("embedding_mode"), key="reasoningbank.embedding_mode"),
            hash_embedding_dim=_as_int(
                reasoningbank.get("hash_embedding_dim"), key="reasoningbank.hash_embedding_dim"
            ),
            k_role=_as_int(reasoningbank.get("k_role"), key="reasoningbank.k_role"),
            k_global=_as_int(reasoningbank.get("k_global"), key="reasoningbank.k_global"),
            max_full_memories_in_generate_recipes=_as_int(
                reasoningbank.get("max_full_memories_in_generate_recipes"),
                key="reasoningbank.max_full_memories_in_generate_recipes",
            ),
            mem_list_default_limit=_as_int(
                reasoningbank.get("mem_list_default_limit"),
                key="reasoningbank.mem_list_default_limit",
            ),
            mem_list_max_limit=_as_int(
                reasoningbank.get("mem_list_max_limit"),
                key="reasoningbank.mem_list_max_limit",
            ),
            near_duplicate_threshold=_as_float(
                reasoningbank.get("near_duplicate_threshold"),
                key="reasoningbank.near_duplicate_threshold",
            ),
            strategy_version=_as_str(
                reasoningbank.get("strategy_version"), key="reasoningbank.strategy_version"
            ),
            context_template=_as_str(
                reasoningbank.get("context_template"), key="reasoningbank.context_template"
            ),
            extract_prompt_template=_as_str(
                reasoningbank.get("extract_prompt_template"),
                key="reasoningbank.extract_prompt_template",
            ),
            merge_prompt_template=_as_str(
                reasoningbank.get("merge_prompt_template"), key="reasoningbank.merge_prompt_template"
            ),
            learn_deref_max_calls_total=_as_int(
                reasoningbank.get("learn_deref_max_calls_total"),
                key="reasoningbank.learn_deref_max_calls_total",
            ),
            learn_deref_max_full_calls=_as_int(
                reasoningbank.get("learn_deref_max_full_calls"),
                key="reasoningbank.learn_deref_max_full_calls",
            ),
            learn_deref_max_chars_total=_as_int(
                reasoningbank.get("learn_deref_max_chars_total"),
                key="reasoningbank.learn_deref_max_chars_total",
            ),
            learn_deref_excerpt_chars=_as_int(
                reasoningbank.get("learn_deref_excerpt_chars"),
                key="reasoningbank.learn_deref_excerpt_chars",
            ),
            learn_deref_full_chars=_as_int(
                reasoningbank.get("learn_deref_full_chars"),
                key="reasoningbank.learn_deref_full_chars",
            ),
            learn_deref_list_events_default_limit=_as_int(
                reasoningbank.get("learn_deref_list_events_default_limit"),
                key="reasoningbank.learn_deref_list_events_default_limit",
            ),
            learn_deref_list_events_max_limit=_as_int(
                reasoningbank.get("learn_deref_list_events_max_limit"),
                key="reasoningbank.learn_deref_list_events_max_limit",
            ),
            learn_candidate_max_items=_as_int(
                reasoningbank.get("learn_candidate_max_items"),
                key="reasoningbank.learn_candidate_max_items",
            ),
            learn_candidate_semantic_top_k=_as_int(
                reasoningbank.get("learn_candidate_semantic_top_k"),
                key="reasoningbank.learn_candidate_semantic_top_k",
            ),
        ),
        roles=RolesConfig(
            by_role={
                "orchestrator": _as_str(roles.get("orchestrator"), key="roles.orchestrator"),
                "mof_expert": _as_str(roles.get("mof_expert"), key="roles.mof_expert"),
                "tio2_expert": _as_str(roles.get("tio2_expert"), key="roles.tio2_expert"),
            }
        ),
        priors=PriorsConfig(
            system_description_path=str(system_description_path),
            microenvironment_tio2_path=str(microenvironment_tio2_path),
            microenvironment_mof_path=str(microenvironment_mof_path),
            system_description_md=system_description_path.read_text(encoding="utf-8"),
            microenvironment_tio2_md=microenvironment_tio2_path.read_text(encoding="utf-8"),
            microenvironment_mof_md=microenvironment_mof_path.read_text(encoding="utf-8"),
        ),
        prompts=PromptConfig(
            system_base=_as_str(prompts.get("system_base"), key="prompts.system_base"),
            down_prompt_template=_as_str(
                prompts.get("down_prompt_template"), key="prompts.down_prompt_template"
            ),
            action_taken_prompt_template=_as_str(
                prompts.get("action_taken_prompt_template"),
                key="prompts.action_taken_prompt_template",
            ),
            up_prompt_template=_as_str(
                prompts.get("up_prompt_template"), key="prompts.up_prompt_template"
            ),
            generate_recipes_prompt_template=_as_str(
                prompts.get("generate_recipes_prompt_template"),
                key="prompts.generate_recipes_prompt_template",
            ),
        ),
    )
