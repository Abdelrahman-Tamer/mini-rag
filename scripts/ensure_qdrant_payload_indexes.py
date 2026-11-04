#!/usr/bin/env python3
"""
NileGuide mini-rag - Qdrant Payload Index Remediation Utility
=============================================================
Safely, idempotently inspects and creates missing payload indexes in Qdrant.

Required by the retrieval pipeline:
- metadata.city        (KEYWORD): used by NLPController.chat city filter
- metadata.is_active   (BOOL)   : used by QdrantDBProvider.search_by_vector (exclude_inactive=True) and rollback
- metadata.category    (KEYWORD): used by NLPController.chat category filter
- metadata.activity_id (INTEGER): used by NLPController.delete_activity
- metadata.asset_id    (KEYWORD): used by NLPController.delete_file_vectors and asset reindexing
- metadata.generation_id (KEYWORD): used by NLPController asset versioning and rollback

Safety Guarantees:
- Default execution is strictly DRY RUN.
- Requires explicit '--apply' flag to perform any mutation.
- Never deletes or recreates collections or existing indexes.
- Validates existing data types before creating indexes.
- Never logs API keys, secret credentials, or unmasked connection URLs.
- Idempotent: repeated runs safely skip already existing indexes.

Usage:
  # Set credentials in environment (recommended):
  $env:QDRANT_DB_URL = "https://<cluster-id>.aws.cloud.qdrant.io:6333"
  $env:QDRANT_DB_API_KEY = "<secret-api-key>"

  # Dry-run inspection (default, non-mutating):
  python scripts/ensure_qdrant_payload_indexes.py

  # Safe production apply (creates only confirmed missing indexes):
  python scripts/ensure_qdrant_payload_indexes.py --apply
"""

import sys
import os
import argparse
import logging
import re
from typing import Dict, Any, Tuple, Optional, List
from urllib.parse import urlparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure local repository modules can be resolved if run from anywhere
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.exceptions import UnexpectedResponse
except ImportError:
    print("ERROR: 'qdrant-client' is required. Install via 'pip install qdrant-client'.", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("qdrant_indexer")


# Confirmed required indexes mapped to their Qdrant schema types and descriptions
REQUIRED_PAYLOAD_INDEXES: Dict[str, Dict[str, Any]] = {
    "metadata.city": {
        "schema": models.PayloadSchemaType.KEYWORD,
        "schema_str": "keyword",
        "description": "Used in NLPController.chat for city-filtered vector search",
        "expected_python_types": (str,),
    },
    "metadata.is_active": {
        "schema": models.PayloadSchemaType.BOOL,
        "schema_str": "bool",
        "description": "Used in QdrantDBProvider.search_by_vector (exclude_inactive=True) and rollback",
        "expected_python_types": (bool,),
    },
    "metadata.category": {
        "schema": models.PayloadSchemaType.KEYWORD,
        "schema_str": "keyword",
        "description": "Used in NLPController.chat when category filter is supplied",
        "expected_python_types": (str,),
    },
    "metadata.activity_id": {
        "schema": models.PayloadSchemaType.INTEGER,
        "schema_str": "integer",
        "description": "Used in NLPController.delete_activity by ID filter",
        "expected_python_types": (int,),
    },
    "metadata.asset_id": {
        "schema": models.PayloadSchemaType.KEYWORD,
        "schema_str": "keyword",
        "description": "Used in NLPController.delete_file_vectors and reindexing",
        "expected_python_types": (str,),
    },
    "metadata.generation_id": {
        "schema": models.PayloadSchemaType.KEYWORD,
        "schema_str": "keyword",
        "description": "Used in NLPController.reindex_asset_safe rollback and pruning",
        "expected_python_types": (str,),
    },
}


def mask_url(url: Optional[str]) -> str:
    """Masks basic auth credentials or secret query tokens from a URL."""
    if not url:
        return "[NOT_CONFIGURED]"
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc
        if "@" in netloc:
            creds, host = netloc.split("@", 1)
            netloc = f"***:***@{host}"
        return f"{parsed.scheme}://{netloc}{parsed.path}"
    except Exception:
        return "[MASKED_URL]"


def sanitize_error_message(err: Any) -> str:
    """Strips API keys, JWT tokens, and credentials from exception strings."""
    msg = str(err)
    # Mask JWT tokens
    msg = re.sub(r'eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]*', '[REDACTED_JWT]', msg)
    # Mask basic auth in URLs
    msg = re.sub(r'://([^:]+):([^@]+)@', r'://***:***@', msg)
    # Mask api-key query parameters or headers
    msg = re.sub(r'(api[-_]?key[\s:=]+)[A-Za-z0-9_-]{8,}', r'\1[REDACTED_KEY]', msg, flags=re.IGNORECASE)
    # Mask Bearer tokens
    msg = re.sub(r'(Bearer\s+)[A-Za-z0-9._-]+', r'\1[REDACTED_TOKEN]', msg, flags=re.IGNORECASE)
    return msg


def resolve_credential_value(
    cli_val: Optional[str],
    env_names: List[str],
    settings_attr: Optional[str] = None
) -> Tuple[Optional[str], str]:
    """
    Resolves a credential or configuration value in strict priority order:
    1. CLI argument (if explicitly provided and non-empty)
    2. Environment variables in env_names order (case-insensitive, non-empty)
    3. Repository settings helpers.config.get_settings (if non-empty)

    Returns:
        Tuple[Optional[str], str]: (resolved_value, source_name)
    """
    # 1. CLI argument
    if cli_val is not None and str(cli_val).strip():
        cleaned = str(cli_val).strip().strip("'\"").strip()
        if cleaned:
            return cleaned, "CLI argument"

    # 2. Environment variables (checking candidate names in priority order)
    for name in env_names:
        val = os.environ.get(name)
        matched_name = name
        if val is None:
            # Case-insensitive scan
            for k, v in os.environ.items():
                if k.upper() == name.upper():
                    val = v
                    matched_name = k
                    break
        if val is not None:
            cleaned = str(val).strip().strip("'\"").strip()
            if cleaned:
                return cleaned, f"environment variable '{matched_name}'"

    # 3. Repository settings fallback
    if settings_attr:
        try:
            from helpers.config import get_settings
            settings = get_settings()
            val = getattr(settings, settings_attr, None)
            if val is not None:
                cleaned = str(val).strip().strip("'\"").strip()
                if cleaned:
                    return cleaned, f"repository settings '{settings_attr}'"
        except Exception:
            pass

    return None, "[NOT_CONFIGURED]"


def get_qdrant_credentials(
    cli_url: Optional[str] = None,
    cli_api_key: Optional[str] = None,
    cli_collection: Optional[str] = None
) -> Tuple[Optional[str], Optional[str], str, str, str, str]:
    """
    Resolves Qdrant URL, API Key, and Collection Name following repo conventions.
    Returns:
        (url, api_key, collection, url_source, key_source, coll_source)
    """
    url, url_source = resolve_credential_value(
        cli_val=cli_url,
        env_names=["QDRANT_DB_URL", "QDRANT_URL"],
        settings_attr="QDRANT_DB_URL"
    )

    api_key, key_source = resolve_credential_value(
        cli_val=cli_api_key,
        env_names=["QDRANT_DB_API_KEY", "QDRANT_API_KEY"],
        settings_attr="QDRANT_DB_API_KEY"
    )

    collection, coll_source = resolve_credential_value(
        cli_val=cli_collection,
        env_names=["VECTOR_DB_COLLECTION_NAME", "QDRANT_COLLECTION_NAME"],
        settings_attr="VECTOR_DB_COLLECTION_NAME"
    )
    if not collection:
        collection = "tourism_knowledge_base"
        coll_source = "default ('tourism_knowledge_base')"

    return url, api_key, collection, url_source, key_source, coll_source


def inspect_collection_schema(
    client: QdrantClient,
    collection_name: str
) -> Dict[str, str]:
    """
    Retrieves the payload_schema from collection info.
    Returns a dictionary mapping field_name to stringified schema type (e.g. 'keyword', 'bool').
    """
    info = client.get_collection(collection_name=collection_name)
    existing_schema: Dict[str, str] = {}
    if hasattr(info, "payload_schema") and info.payload_schema:
        for field, schema_info in info.payload_schema.items():
            dt = schema_info.data_type
            dt_str = dt.value if hasattr(dt, "value") else str(dt).lower()
            existing_schema[field] = dt_str
    return existing_schema


def inspect_sample_payloads(
    client: QdrantClient,
    collection_name: str,
    limit: int = 50
) -> Tuple[Dict[str, set], int]:
    """
    Scrolls a sample of points and determines observed Python data types for nested metadata fields.
    """
    try:
        points, _ = client.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
    except Exception as e:
        logger.warning(f"Could not sample points from collection '{collection_name}': {sanitize_error_message(e)}")
        return {}, 0

    observed_types: Dict[str, set] = {}
    for p in points:
        payload = p.payload or {}
        # Check root fields and nested metadata
        meta = payload.get("metadata", {})
        if isinstance(meta, dict):
            for k, v in meta.items():
                full_path = f"metadata.{k}"
                if full_path not in observed_types:
                    observed_types[full_path] = set()
                if v is not None:
                    observed_types[full_path].add(type(v).__name__)

    return observed_types, len(points)


def run_remediation(
    client: QdrantClient,
    collection_name: str,
    apply: bool = False
) -> int:
    """
    Inspects and conditionally applies missing Qdrant payload indexes.
    Returns:
        0 on success (clean dry run or successful application and verification)
        1 on failure (type conflicts, missing collection, or post-apply verification failure)
    """
    logger.info("=" * 70)
    logger.info("NileGuide mini-rag - Qdrant Payload Index Remediation")
    logger.info("=" * 70)
    logger.info(f"Target Collection : {collection_name}")
    logger.info(f"Execution Mode    : {'APPLY (MUTATING)' if apply else 'DRY RUN (READ ONLY)'}")

    # 1. Verify collection exists
    try:
        if not client.collection_exists(collection_name=collection_name):
            logger.error(f"Collection '{collection_name}' does NOT exist in Qdrant cluster!")
            return 1
    except Exception as e:
        logger.error(f"Failed to check collection existence: {sanitize_error_message(e)}")
        return 1

    # 2. Inspect existing payload indexes
    existing_schema = inspect_collection_schema(client, collection_name)
    logger.info(f"Found {len(existing_schema)} existing payload index(es) on '{collection_name}':")
    for field, dt in sorted(existing_schema.items()):
        logger.info(f"  - '{field}': schema={dt}")

    # 3. Sample payload points for schema confirmation
    observed_types, sample_count = inspect_sample_payloads(client, collection_name, limit=50)
    logger.info(f"Sampled {sample_count} points to analyze stored payload data types.")

    # 4. Evaluate each required index against current state
    missing_indexes: List[str] = []
    type_conflicts: List[str] = []
    already_valid_indexes: List[str] = []

    print("\n" + "-" * 70)
    print(f"{'Target Field':<25} | {'Required':<8} | {'Existing':<10} | {'Sample Type':<12} | {'Status'}")
    print("-" * 70)

    for field, req in REQUIRED_PAYLOAD_INDEXES.items():
        req_schema = req["schema_str"]
        existing = existing_schema.get(field)
        sample_t = ",".join(observed_types.get(field, ["(not in sample)"]))

        if existing is None:
            status = "MISSING"
            missing_indexes.append(field)
        elif existing.lower() == req_schema.lower():
            status = "OK (EXISTS)"
            already_valid_indexes.append(field)
        else:
            status = f"CONFLICT (type mismatch: {existing} != {req_schema})"
            type_conflicts.append(field)

        print(f"{field:<25} | {req_schema:<8} | {str(existing):<10} | {sample_t:<12} | {status}")

    print("-" * 70 + "\n")

    # 5. Handle type conflicts
    if type_conflicts:
        logger.critical(
            f"ABORTING: {len(type_conflicts)} field(s) have incompatible index schema types in Qdrant: {type_conflicts}. "
            "Manual database review required. Utility will NOT drop or alter existing indexes automatically."
        )
        return 1

    # 6. Dry run summary
    if not apply:
        logger.info("Summary of Dry Run:")
        logger.info(f"  - Total confirmed required fields : {len(REQUIRED_PAYLOAD_INDEXES)}")
        logger.info(f"  - Already indexed correctly       : {len(already_valid_indexes)}")
        logger.info(f"  - Missing indexes to create       : {len(missing_indexes)}")

        if missing_indexes:
            logger.info("\nMissing indexes that would be created under --apply:")
            for f in missing_indexes:
                logger.info(f"  + {f} ({REQUIRED_PAYLOAD_INDEXES[f]['schema_str']}): {REQUIRED_PAYLOAD_INDEXES[f]['description']}")
            print("\n[!] TO APPLY THESE INDEXES TO PRODUCTION, RUN:")
            print(f"    python scripts/ensure_qdrant_payload_indexes.py --apply\n")
        else:
            logger.info("All required payload indexes are already present in Qdrant. No action needed.")

        return 0

    # 7. Apply mode
    if not missing_indexes:
        logger.info("All required payload indexes are already present. Nothing to apply.")
        return 0

    logger.info(f"Creating {len(missing_indexes)} missing payload index(es)...")
    for f in missing_indexes:
        schema = REQUIRED_PAYLOAD_INDEXES[f]["schema"]
        schema_str = REQUIRED_PAYLOAD_INDEXES[f]["schema_str"]
        logger.info(f"Creating payload index for '{f}' (schema={schema_str})...")
        try:
            res = client.create_payload_index(
                collection_name=collection_name,
                field_name=f,
                field_schema=schema,
                wait=True
            )
            logger.info(f"Successfully created index '{f}': {res}")
        except Exception as e:
            logger.error(f"Failed to create index for '{f}': {sanitize_error_message(e)}")
            return 1

    # 8. Re-read and verify every required index post-apply
    logger.info("Re-reading collection schema to verify newly created indexes...")
    updated_schema = inspect_collection_schema(client, collection_name)
    verification_failed = []

    for f, req in REQUIRED_PAYLOAD_INDEXES.items():
        actual = updated_schema.get(f)
        expected = req["schema_str"]
        if not actual or actual.lower() != expected.lower():
            verification_failed.append((f, expected, actual))

    if verification_failed:
        logger.error(f"POST-APPLY VERIFICATION FAILED for {len(verification_failed)} index(es):")
        for f, exp, act in verification_failed:
            logger.error(f"  - {f}: expected {exp}, found {act}")
        return 1

    logger.info("=" * 70)
    logger.info("SUCCESS: All required Qdrant payload indexes are verified active!")
    logger.info("=" * 70)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NileGuide Qdrant Payload Index Remediation Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Apply index creation to Qdrant. Defaults to false (dry-run)."
    )
    parser.add_argument(
        "--check-credentials",
        action="store_true",
        default=False,
        help="Preflight check: verifies credential resolution and connection without dry-run or index creation."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Optional Qdrant cluster URL (recommended: set QDRANT_DB_URL or QDRANT_URL environment variable)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional Qdrant cluster API key override (recommended: set QDRANT_DB_API_KEY or QDRANT_API_KEY environment variable)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Qdrant collection name (defaults to 'tourism_knowledge_base')"
    )

    args = parser.parse_args()

    url, api_key, collection, url_source, key_source, coll_source = get_qdrant_credentials(
        cli_url=args.url,
        cli_api_key=args.api_key,
        cli_collection=args.collection
    )

    if not url:
        logger.error("CRITICAL: Qdrant cluster URL is not configured. Set QDRANT_DB_URL or QDRANT_URL environment variable.")
        return 1

    masked_url = mask_url(url)
    logger.info(f"Connecting to Qdrant Cluster: {masked_url} (source: {url_source})")

    if not api_key:
        logger.error(
            "CRITICAL: Qdrant API key is not configured. "
            "Set QDRANT_DB_API_KEY or QDRANT_API_KEY environment variable before running this utility."
        )
        return 1

    logger.info(f"API Key Status             : CONFIGURED (source: {key_source})")
    logger.info(f"Target Collection          : {collection} (source: {coll_source})")

    try:
        client = QdrantClient(url=url, api_key=api_key, timeout=10.0)
    except Exception as e:
        logger.error(f"Failed to initialize QdrantClient: {sanitize_error_message(e)}")
        return 1

    if args.check_credentials:
        logger.info("Executing preflight credential verification...")
        try:
            if client.collection_exists(collection_name=collection):
                info = client.get_collection(collection_name=collection)
                points_cnt = getattr(info, "points_count", "unknown")
                logger.info(f"PREFLIGHT SUCCESS: Connected to Qdrant cluster. Collection '{collection}' found with {points_cnt} points.")
                return 0
            else:
                logger.error(f"PREFLIGHT FAILURE: Connected to Qdrant cluster, but collection '{collection}' does NOT exist.")
                return 1
        except Exception as e:
            logger.error(f"PREFLIGHT FAILURE: Could not connect to Qdrant cluster: {sanitize_error_message(e)}")
            return 1

    return run_remediation(client=client, collection_name=collection, apply=args.apply)


if __name__ == "__main__":
    sys.exit(main())
