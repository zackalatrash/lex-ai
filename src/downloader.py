"""Dataset parsing and document download pipeline.

The download implementation is added in Phase 3. Phase 2 covers reading the
Excel source index and normalizing document metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

import time

import pandas as pd
import requests

from src.config import (
    DATASET_PATH,
    DOWNLOAD_REPORT_PATH,
    EXCEL_SHEET_NAME,
    PROCESSED_DOCUMENTS_PATH,
    RAW_DATA_DIR,
    REQUEST_TIMEOUT_SECONDS,
    USER_AGENT,
)
from src.utils import ensure_directory, project_relative, write_json, write_jsonl


REQUIRED_DATASET_COLUMNS = (
    "id",
    "title",
    "url",
    "publisher",
    "date",
    "theme",
    "keywords",
    "bronvermelding",
)


@dataclass(frozen=True)
class DatasetSummary:
    """Small report describing the parsed Excel dataset."""

    dataset_path: str
    sheet_name: str
    rows: int
    urls: int
    missing_urls: int
    duplicate_urls: int
    columns: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        return {
            "dataset_path": self.dataset_path,
            "sheet_name": self.sheet_name,
            "rows": self.rows,
            "urls": self.urls,
            "missing_urls": self.missing_urls,
            "duplicate_urls": self.duplicate_urls,
            "columns": self.columns,
        }


@dataclass(frozen=True)
class DownloadSummary:
    """Report for one document acquisition run."""

    total_records: int
    downloaded: int
    skipped: int
    failed: int
    unsupported: int
    duplicates: int
    missing_urls: int
    failed_urls: list[dict[str, Any]]
    output_dir: str
    report_path: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_records": self.total_records,
            "downloaded": self.downloaded,
            "skipped": self.skipped,
            "failed": self.failed,
            "unsupported": self.unsupported,
            "duplicates": self.duplicates,
            "missing_urls": self.missing_urls,
            "failed_urls": self.failed_urls,
            "output_dir": self.output_dir,
            "report_path": self.report_path,
        }


def load_dataset(
    dataset_path: Path = DATASET_PATH,
    sheet_name: str = EXCEL_SHEET_NAME,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load the configured Excel sheet and optionally keep only the first rows."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {project_relative(dataset_path)}")

    dataframe = pd.read_excel(dataset_path, sheet_name=sheet_name)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be a positive integer")
        dataframe = dataframe.head(limit)
    return dataframe


def inspect_dataset(dataframe: pd.DataFrame, dataset_path: Path = DATASET_PATH) -> DatasetSummary:
    """Build a compact summary for CLI output and documentation."""
    url_series = dataframe["url"] if "url" in dataframe.columns else pd.Series([], dtype=object)
    normalized_urls = url_series.map(clean_optional_string)
    present_urls = normalized_urls.map(bool)

    return DatasetSummary(
        dataset_path=project_relative(dataset_path),
        sheet_name=EXCEL_SHEET_NAME,
        rows=int(len(dataframe)),
        urls=int(present_urls.sum()),
        missing_urls=int((~present_urls).sum()),
        duplicate_urls=int(normalized_urls[present_urls].duplicated().sum()),
        columns=[str(column) for column in dataframe.columns],
    )


def validate_dataset_columns(dataframe: pd.DataFrame) -> None:
    """Ensure the expected assignment columns are available."""
    missing = [column for column in REQUIRED_DATASET_COLUMNS if column not in dataframe.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Dataset is missing required columns: {joined}")


def clean_optional_string(value: Any) -> str:
    """Convert missing spreadsheet values to an empty string and clean text values."""
    if pd.isna(value):
        return ""
    if hasattr(value, "isoformat") and not isinstance(value, str):
        return str(value.date()) if hasattr(value, "date") else value.isoformat()
    return " ".join(str(value).replace("\n", " ").split())


def stable_doc_id(row: pd.Series, fallback_index: int) -> str:
    """Use the dataset id when available, otherwise create a deterministic id."""
    dataset_id = clean_optional_string(row.get("id"))
    if dataset_id:
        return dataset_id
    return f"DOC-{fallback_index + 1:04d}"


def infer_initial_source_type(url: str) -> str:
    """Infer a provisional source type from the URL path.

    Phase 3 will refine this with HTTP content-type checks.
    """
    parsed = urlparse(url)
    path = parsed.path.lower()
    if path.endswith(".pdf"):
        return "pdf"
    if parsed.scheme in {"http", "https"}:
        return "web"
    return "unknown"


def classify_url(url: str, content_type: str | None = None) -> str:
    """Classify a source as PDF, HTML web page, or unsupported/unknown."""
    parsed = urlparse(url)
    path = parsed.path.lower()
    normalized_content_type = (content_type or "").split(";")[0].strip().lower()

    if path.endswith(".pdf"):
        return "pdf"
    if normalized_content_type == "application/pdf":
        return "pdf"
    if normalized_content_type in {"text/html", "application/xhtml+xml"}:
        return "html"
    if parsed.scheme in {"http", "https"} and not normalized_content_type:
        return "web"
    return "unknown"


def raw_file_path(doc_id: str, source_type: str, raw_dir: Path = RAW_DATA_DIR) -> Path:
    """Create a stable safe raw filename for a document."""
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", doc_id).strip("._-") or "document"
    suffix = ".pdf" if source_type == "pdf" else ".html"
    return raw_dir / f"{safe_id}{suffix}"


def normalize_document_record(
    row: pd.Series,
    fallback_index: int,
    duplicate_of: str | None = None,
) -> dict[str, Any]:
    """Convert one dataset row into normalized document metadata."""
    url = clean_optional_string(row.get("url"))
    doc_id = stable_doc_id(row, fallback_index)

    return {
        "doc_id": doc_id,
        "title": clean_optional_string(row.get("title")),
        "url": url,
        "theme": clean_optional_string(row.get("theme")),
        "keywords": clean_optional_string(row.get("keywords")),
        "source_type": infer_initial_source_type(url) if url else "missing",
        "publisher": clean_optional_string(row.get("publisher")),
        "date": clean_optional_string(row.get("date")),
        "citation": clean_optional_string(row.get("bronvermelding")),
        "local_path": None,
        "status": "duplicate" if duplicate_of else ("pending" if url else "missing_url"),
        "error": None if url else "Missing URL in dataset",
        "duplicate_of": duplicate_of,
    }


def extract_document_metadata(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """Normalize all valid spreadsheet rows into document records."""
    validate_dataset_columns(dataframe)

    records: list[dict[str, Any]] = []
    seen_urls: dict[str, str] = {}
    for index, row in dataframe.iterrows():
        url = clean_optional_string(row.get("url"))
        duplicate_of = seen_urls.get(url) if url else None
        record = normalize_document_record(row, int(index), duplicate_of=duplicate_of)
        if url and duplicate_of is None:
            seen_urls[url] = record["doc_id"]
        records.append(record)
    return records


def save_document_metadata(
    records: list[dict[str, Any]],
    output_path: Path = PROCESSED_DOCUMENTS_PATH,
) -> None:
    """Persist normalized metadata as JSONL."""
    write_jsonl(output_path, records)


def parse_dataset(
    dataset_path: Path = DATASET_PATH,
    sheet_name: str = EXCEL_SHEET_NAME,
    output_path: Path = PROCESSED_DOCUMENTS_PATH,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], DatasetSummary]:
    """Load, inspect, normalize, and save dataset metadata."""
    dataframe = load_dataset(dataset_path=dataset_path, sheet_name=sheet_name, limit=limit)
    validate_dataset_columns(dataframe)
    summary = inspect_dataset(dataframe, dataset_path=dataset_path)
    records = extract_document_metadata(dataframe)
    save_document_metadata(records, output_path=output_path)
    return records, summary


def create_http_session() -> requests.Session:
    """Create a session with assignment-friendly request headers."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def get_remote_content_type(session: requests.Session, url: str) -> tuple[str | None, int | None, str | None]:
    """Use HEAD to inspect content type when the URL extension is unclear."""
    try:
        response = session.head(
            url,
            allow_redirects=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400:
            return None, response.status_code, f"HEAD returned HTTP {response.status_code}"
        return response.headers.get("content-type"), response.status_code, None
    except requests.RequestException as error:
        return None, None, f"HEAD failed: {error}"


def download_one_document(
    record: dict[str, Any],
    session: requests.Session,
    raw_dir: Path = RAW_DATA_DIR,
    force: bool = False,
) -> dict[str, Any]:
    """Download one source document and return updated metadata."""
    updated = dict(record)
    url = str(updated.get("url") or "")

    if updated.get("status") == "duplicate":
        return updated
    if not url:
        updated["status"] = "missing_url"
        updated["error"] = "Missing URL in dataset"
        return updated

    source_type = classify_url(url)
    content_type: str | None = None
    head_status: int | None = None
    head_error: str | None = None

    if source_type in {"web", "unknown"}:
        content_type, head_status, head_error = get_remote_content_type(session, url)
        source_type = classify_url(url, content_type)

    if source_type == "web":
        source_type = "html"

    updated["source_type"] = source_type
    updated["content_type"] = content_type
    updated["head_status_code"] = head_status

    if source_type not in {"pdf", "html"}:
        updated["status"] = "unsupported"
        updated["error"] = head_error or f"Unsupported or unknown content type: {content_type or 'unknown'}"
        return updated

    local_path = raw_file_path(str(updated["doc_id"]), source_type, raw_dir=raw_dir)
    updated["local_path"] = project_relative(local_path)

    if local_path.exists() and not force:
        updated["status"] = "skipped_existing"
        updated["error"] = None
        return updated

    ensure_directory(raw_dir)
    try:
        response = None
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = session.get(
                    url,
                    allow_redirects=True,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                if response.status_code != 429:
                    break
                retry_after = float(response.headers.get("Retry-After", 0))
                wait = retry_after if retry_after > 0 else 5 * (2 ** attempt)
                time.sleep(wait)
            except requests.RequestException as error:
                last_error = error
                if attempt < 3:
                    time.sleep(3 * (2 ** attempt))
        if response is None:
            raise last_error or requests.RequestException("All attempts failed")
        updated["status_code"] = response.status_code
        updated["final_url"] = response.url
        response.raise_for_status()

        response_content_type = response.headers.get("content-type")
        updated["content_type"] = response_content_type or content_type
        response_source_type = classify_url(url, response_content_type)
        if response_source_type == "web":
            response_source_type = "html"
        if response_source_type in {"pdf", "html"} and response_source_type != source_type:
            source_type = response_source_type
            updated["source_type"] = source_type
            local_path = raw_file_path(str(updated["doc_id"]), source_type, raw_dir=raw_dir)
            updated["local_path"] = project_relative(local_path)

        if source_type == "unknown":
            updated["status"] = "unsupported"
            updated["error"] = f"Unsupported or unknown content type: {response_content_type or 'unknown'}"
            return updated

        local_path.write_bytes(response.content)
        updated["status"] = "downloaded"
        updated["error"] = None
        updated["bytes"] = local_path.stat().st_size
        return updated
    except requests.RequestException as error:
        updated["status"] = "failed"
        updated["error"] = str(error)
        return updated
    except OSError as error:
        updated["status"] = "failed"
        updated["error"] = f"File write failed: {error}"
        return updated


def build_download_summary(
    records: list[dict[str, Any]],
    raw_dir: Path = RAW_DATA_DIR,
    report_path: Path = DOWNLOAD_REPORT_PATH,
) -> DownloadSummary:
    """Summarize downloader outcomes for logs and `download_report.json`."""
    failed_statuses = {"failed", "unsupported"}
    failed_urls = [
        {
            "doc_id": record.get("doc_id"),
            "title": record.get("title"),
            "url": record.get("url"),
            "status": record.get("status"),
            "status_code": record.get("status_code"),
            "head_status_code": record.get("head_status_code"),
            "error": record.get("error"),
        }
        for record in records
        if record.get("status") in failed_statuses
    ]

    return DownloadSummary(
        total_records=len(records),
        downloaded=sum(1 for record in records if record.get("status") == "downloaded"),
        skipped=sum(1 for record in records if record.get("status") == "skipped_existing"),
        failed=sum(1 for record in records if record.get("status") == "failed"),
        unsupported=sum(1 for record in records if record.get("status") == "unsupported"),
        duplicates=sum(1 for record in records if record.get("status") == "duplicate"),
        missing_urls=sum(1 for record in records if record.get("status") == "missing_url"),
        failed_urls=failed_urls,
        output_dir=project_relative(raw_dir),
        report_path=project_relative(report_path),
    )


def download_documents(
    records: list[dict[str, Any]],
    raw_dir: Path = RAW_DATA_DIR,
    metadata_output_path: Path = PROCESSED_DOCUMENTS_PATH,
    report_path: Path = DOWNLOAD_REPORT_PATH,
    force: bool = False,
) -> tuple[list[dict[str, Any]], DownloadSummary]:
    """Download all document records and persist metadata plus report."""
    ensure_directory(raw_dir)
    session = create_http_session()
    updated_records: list[dict[str, Any]] = []
    for record in records:
        updated_records.append(download_one_document(record, session=session, raw_dir=raw_dir, force=force))
        time.sleep(0.5)
    save_document_metadata(updated_records, output_path=metadata_output_path)
    summary = build_download_summary(updated_records, raw_dir=raw_dir, report_path=report_path)
    write_json(report_path, summary.to_dict())
    return updated_records, summary
