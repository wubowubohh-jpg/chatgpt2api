from __future__ import annotations

import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import RLock
from typing import Any

from services.account_service import AccountService, account_service
from services.openai_backend_api import OpenAIBackendAPI
from utils.log import logger


@dataclass(frozen=True)
class ModelRoute:
    account_types: frozenset[str]
    allow_anonymous: bool = False
    resolved_model: str = ""


class ModelUnavailableError(RuntimeError):
    pass


_GPT_VERSION_RE = re.compile(r"^(gpt-\d+)\.(\d+)(?=$|-)", re.IGNORECASE)
_GPT_VERSION_CANONICAL_RE = re.compile(r"^(gpt-\d+)-(\d+)(?=$|-)", re.IGNORECASE)
_MODEL_ROUTE_FAMILY_HINTS = {
    "gpt-5-6-luna": ("gpt-5-6",),
}


def normalize_model_identifier(model: object) -> str:
    """Normalize GPT version punctuation without changing the model family."""
    value = str(model or "").strip().lower()
    return _GPT_VERSION_RE.sub(r"\1-\2", value, count=1)


def _dotted_model_identifier(model: str) -> str:
    return _GPT_VERSION_CANONICAL_RE.sub(r"\1.\2", model, count=1)


def resolve_model_identifier(model: object, available_model_ids: list[str] | tuple[str, ...] | set[str]) -> str:
    """Return the real catalog id matching a request spelling, or an empty string."""
    requested = str(model or "").strip()
    if not requested:
        return ""
    exact_key = requested.lower()
    normalized_ids: dict[str, str] = {}
    exact_ids: dict[str, str] = {}
    for model_id in available_model_ids:
        candidate = str(model_id or "").strip()
        if not candidate:
            continue
        exact_ids.setdefault(candidate.lower(), candidate)
        normalized_ids.setdefault(normalize_model_identifier(candidate), candidate)
    if exact_key in exact_ids:
        return exact_ids[exact_key]
    return normalized_ids.get(normalize_model_identifier(requested), "")


def model_compatibility_entries(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build compatible public names without changing their upstream model slug."""
    by_id = {
        str(item.get("id") or "").strip(): item
        for item in models
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    entries: list[dict[str, Any]] = []
    seen = set(by_id)

    def append_entry(model_id: str, source: dict[str, Any]) -> None:
        if model_id in seen:
            return
        entry = dict(source)
        entry["id"] = model_id
        entry["root"] = model_id
        entry["parent"] = None
        entry["owned_by"] = "chatgpt2api"
        entries.append(entry)
        seen.add(model_id)

    for model_id, item in by_id.items():
        normalized = normalize_model_identifier(model_id)
        if not normalized.startswith("gpt-") or normalized != model_id.lower():
            continue
        alias = _dotted_model_identifier(normalized)
        if alias != model_id.lower():
            append_entry(alias, item)

    base_model = resolve_model_identifier("gpt-5-6", set(by_id))
    luna_model = resolve_model_identifier("gpt-5-6-luna", set(by_id))
    if base_model and not luna_model:
        append_entry("gpt-5-6-luna", by_id[base_model])
        append_entry("gpt-5.6-luna", by_id[base_model])
    return sorted(entries, key=lambda item: str(item.get("id") or ""))


class ModelCatalogService:
    """Caches the model catalogs advertised to each active account type."""

    def __init__(
        self,
        accounts: AccountService,
        *,
        backend_factory: Callable[..., Any] = OpenAIBackendAPI,
        cache_ttl_seconds: float = 300,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._accounts = accounts
        self._backend_factory = backend_factory
        self._cache_ttl_seconds = max(1.0, float(cache_ttl_seconds))
        self._clock = clock
        self._lock = RLock()
        self._expires_at = 0.0
        self._account_signature: tuple[tuple[str, int], ...] = ()
        self._anonymous_models: dict[str, dict[str, Any]] = {}
        self._models_by_account_type: dict[str, dict[str, dict[str, Any]]] = {}

    @staticmethod
    def _model_map(result: object) -> dict[str, dict[str, Any]]:
        if not isinstance(result, dict) or not isinstance(result.get("data"), list):
            raise TypeError("upstream model response has no data list")
        models: dict[str, dict[str, Any]] = {}
        for item in result["data"]:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id") or "").strip()
            if model_id and model_id not in models:
                models[model_id] = dict(item)
        return models

    def _active_accounts_by_type(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for account in self._accounts.list_accounts():
            if not isinstance(account, dict) or account.get("status") in {"禁用", "异常"}:
                continue
            access_token = str(account.get("access_token") or "").strip()
            account_type = self._accounts._normalize_account_type(account.get("type"))
            if access_token and account_type:
                groups.setdefault(account_type, []).append(access_token)
        return groups

    @staticmethod
    def _signature(groups: dict[str, list[str]]) -> tuple[tuple[str, int], ...]:
        return tuple(
            (account_type, len(tokens))
            for account_type, tokens in sorted(groups.items())
        )

    def _fetch_models(self, access_token: str = "") -> dict[str, dict[str, Any]]:
        backend = self._backend_factory(access_token=access_token)
        try:
            return self._model_map(backend.list_models())
        finally:
            backend.close()

    def _fetch_account_type_models(
        self,
        account_type: str,
        access_tokens: list[str],
    ) -> dict[str, dict[str, Any]] | None:
        attempted_tokens: set[str] = set()
        last_error: Exception | None = None
        for access_token in access_tokens:
            try:
                resolved_token = self._accounts.refresh_access_token(
                    access_token,
                    event="list_models",
                ) or access_token
                if resolved_token in attempted_tokens:
                    continue
                attempted_tokens.add(resolved_token)
                return self._fetch_models(resolved_token)
            except Exception as exc:  # noqa: BLE001 - try the next account for any upstream failure
                last_error = exc
        if last_error is not None:
            logger.warning({
                "event": "model_catalog_account_type_failed",
                "account_type": account_type,
                "error_type": type(last_error).__name__,
            })
        return None

    def _refresh(self, groups: dict[str, list[str]], signature: tuple[tuple[str, int], ...]) -> None:
        models_by_account_type: dict[str, dict[str, dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=min(4, len(groups) + 1)) as executor:
            anonymous_future = executor.submit(self._fetch_models)
            account_futures = {
                account_type: executor.submit(
                    self._fetch_account_type_models,
                    account_type,
                    access_tokens,
                )
                for account_type, access_tokens in groups.items()
            }
            try:
                anonymous_models = anonymous_future.result()
            except Exception as exc:  # noqa: BLE001 - retain cached models on upstream failure
                logger.warning({
                    "event": "model_catalog_anonymous_failed",
                    "error_type": type(exc).__name__,
                })
                anonymous_models = self._anonymous_models

            for account_type, future in account_futures.items():
                models = future.result()
                if models is not None:
                    models_by_account_type[account_type] = models
                elif account_type in self._models_by_account_type:
                    models_by_account_type[account_type] = self._models_by_account_type[account_type]

        self._anonymous_models = anonymous_models
        self._models_by_account_type = models_by_account_type
        self._account_signature = signature
        self._expires_at = self._clock() + self._cache_ttl_seconds

    def _ensure_catalog(self) -> None:
        groups = self._active_accounts_by_type()
        signature = self._signature(groups)
        with self._lock:
            if signature == self._account_signature and self._clock() < self._expires_at:
                return
            self._refresh(groups, signature)

    def list_models(self) -> dict[str, Any]:
        self._ensure_catalog()
        with self._lock:
            union: dict[str, dict[str, Any]] = {
                model_id: dict(item)
                for model_id, item in self._anonymous_models.items()
            }
            for account_type in sorted(self._models_by_account_type):
                for model_id, item in self._models_by_account_type[account_type].items():
                    union.setdefault(model_id, dict(item))
        return {
            "object": "list",
            "data": [union[model_id] for model_id in sorted(union)],
        }

    def resolve_model(self, model: str) -> str:
        self._ensure_catalog()
        with self._lock:
            model_ids = set(self._anonymous_models)
            for models in self._models_by_account_type.values():
                model_ids.update(models)
            resolved_model = resolve_model_identifier(model, model_ids)
            if resolved_model:
                return resolved_model
            normalized_model = normalize_model_identifier(model)
            if any(
                resolve_model_identifier(hint, model_ids)
                for hint in _MODEL_ROUTE_FAMILY_HINTS.get(normalized_model, ())
            ):
                return normalized_model
            return ""

    def route_for_model(self, model: str) -> ModelRoute:
        resolved_model = self.resolve_model(model)
        if not resolved_model:
            return ModelRoute(account_types=frozenset(), allow_anonymous=False)
        with self._lock:
            directly_advertised = (
                resolved_model in self._anonymous_models
                or any(resolved_model in models for models in self._models_by_account_type.values())
            )
            route_models = {resolved_model}
            if not directly_advertised:
                route_models.update(_MODEL_ROUTE_FAMILY_HINTS.get(resolved_model, ()))
            account_types = frozenset(
                account_type
                for account_type, models in self._models_by_account_type.items()
                if route_models.intersection(models)
            )
            return ModelRoute(
                account_types=account_types,
                allow_anonymous=bool(route_models.intersection(self._anonymous_models)),
                resolved_model=resolved_model,
            )


model_catalog_service = ModelCatalogService(account_service)
