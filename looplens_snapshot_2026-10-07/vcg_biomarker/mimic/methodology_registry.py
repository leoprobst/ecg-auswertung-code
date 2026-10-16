from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .storage import content_hash, file_sha256


def methodology_source_lock_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "resources"
        / "methodology"
        / "SOURCE.json"
    )


class MethodologyRegistry:
    def __init__(self) -> None:
        self._source_lock_path = methodology_source_lock_path()
        self._source_lock = json.loads(
            self._source_lock_path.read_text(encoding="utf-8")
        )
        self._validate()
        self._source_lock_hash = file_sha256(self._source_lock_path)

    @property
    def source_lock_path(self) -> Path:
        return self._source_lock_path

    @property
    def source_lock_hash(self) -> str:
        return self._source_lock_hash

    @property
    def registry_hash(self) -> str:
        return content_hash(
            {
                "version": self._source_lock["version"],
                "source_lock_sha256": self._source_lock_hash,
                "source_ids": [
                    source["source_id"]
                    for source in self._source_lock["sources"]
                ],
            }
        )

    @property
    def source_lock(self) -> dict:
        return dict(self._source_lock)

    def frame(self) -> pd.DataFrame:
        records = []
        for source in self._source_lock["sources"]:
            records.append(
                {
                    "source_id": source["source_id"],
                    "title": source["title"],
                    "version": source["version"],
                    "doi": source.get("doi", ""),
                    "url": source["url"],
                    "implementation_scope": source[
                        "implementation_scope"
                    ],
                    "platform_safeguards": " | ".join(
                        source["platform_safeguards"]
                    ),
                    "source_lock_version": self._source_lock["version"],
                    "source_lock_sha256": self._source_lock_hash,
                    "runtime_network_access": False,
                }
            )
        return pd.DataFrame(records)

    def _validate(self) -> None:
        required_root = {
            "registry",
            "version",
            "retrieved_at",
            "runtime_network_access",
            "sources",
        }
        missing = required_root - set(self._source_lock)
        if missing:
            raise ValueError(
                f"Methodology source lock is missing fields: {sorted(missing)}"
            )
        if self._source_lock["runtime_network_access"] is not False:
            raise ValueError(
                "Methodology sources must be pinned locally without runtime "
                "network access"
            )
        source_ids = set()
        for source in self._source_lock["sources"]:
            required_source = {
                "source_id",
                "title",
                "version",
                "url",
                "implementation_scope",
                "platform_safeguards",
            }
            missing_source = required_source - set(source)
            if missing_source:
                raise ValueError(
                    "Methodology source entry is incomplete: "
                    f"{sorted(missing_source)}"
                )
            source_id = str(source["source_id"])
            if source_id in source_ids:
                raise ValueError(
                    f"Duplicate methodology source ID: {source_id}"
                )
            source_ids.add(source_id)
            if not str(source["url"]).startswith("https://"):
                raise ValueError(
                    f"Methodology source URL must use HTTPS: {source_id}"
                )
            safeguards = source["platform_safeguards"]
            if not isinstance(safeguards, list) or not safeguards:
                raise ValueError(
                    f"Methodology source needs platform safeguards: {source_id}"
                )
