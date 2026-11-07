"""The ``models`` registry table and the runtime code fingerprint (ids mint in ``artifacts``)."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from league_predictor.db.rows import ModelRow

if TYPE_CHECKING:
    import sqlite3

# The subtree code_fingerprint covers: everything that affects what the package does when
# imported. Bytecode caches are excluded; git's own ignore rules exclude nothing else here.
_RUNTIME_TREE = "src/league_predictor"


def _repo_root() -> Path:
    """The repo checkout this package runs from (three levels above this file)."""
    return Path(__file__).resolve().parents[3]


def _git_lines(root: Path, *args: str) -> list[str]:
    """One NUL-separated git plumbing call at ``root`` (``-z`` must precede any ``--``).

    Raises:
        RuntimeError: The git call failed; a fingerprint must never degrade to an empty tree.
    """
    proc = subprocess.run(  # noqa: S603 - no shell; private helper, both call sites pass repo literals
        ["git", "-C", str(root), *args],  # noqa: S607 - git resolves from PATH by design
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,  # the returncode is inspected below
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"code_fingerprint: `git {' '.join(args)}` failed at {root}: "
            f"{proc.stderr.strip()} — the runtime source-tree hash needs the git checkout"
        )
    return [line for line in proc.stdout.split("\0") if line]


def _is_cache_path(path: str) -> bool:
    """Bytecode caches never change what the package does when imported."""
    return "__pycache__" in path.split("/") or path.endswith((".pyc", ".pyo"))


def code_fingerprint(*, include_dirty: bool = True) -> str:
    """The one runtime source-tree hash over ``src/league_predictor``, stamped into artifacts.

    Args:
        include_dirty: Hash dirty/untracked runtime files from disk too; False hashes the
            tracked git index alone (the clean-tree identity).

    Returns:
        A content-keyed digest — dirty files go through ``git hash-object``, so a stat-only
        touch never moves it — covering edits, additions and deletions, bytecode excluded.
        Cost: 2-3 git calls, well under a second.

    Raises:
        RuntimeError: git is unavailable; never a silent empty hash.
    """
    root = _repo_root()
    entries: dict[str, str] = {}
    for line in _git_lines(root, "ls-files", "-s", "-z", "--", _RUNTIME_TREE):
        meta, _, path = line.partition("\t")
        entries[path] = meta.split()[1]  # "<mode> <blob sha> <stage>\t<path>"
    if include_dirty:
        dirty = _git_lines(
            root, "ls-files", "-m", "-d", "-o", "--exclude-standard", "-z", "--", _RUNTIME_TREE
        )
        on_disk = [p for p in dict.fromkeys(dirty) if not _is_cache_path(p)]
        for path in on_disk:
            entries[path] = "deleted"  # overwritten below for files that still exist
        present = [p for p in on_disk if (root / p).is_file()]
        if present:
            # hash-object applies the same content filters the index does, so a dirty file
            # whose bytes match its index blob hashes to the SAME sha (content identity).
            hashed = subprocess.run(  # noqa: S603 - argv is a fixed list, no shell
                ["git", "-C", str(root), "hash-object", "--stdin-paths"],  # noqa: S607 - PATH git
                input="\n".join(present) + "\n",
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,  # the returncode is inspected below
            )
            if hashed.returncode != 0:
                raise RuntimeError(
                    f"code_fingerprint: `git hash-object` failed at {root}: {hashed.stderr.strip()}"
                )
            entries.update(zip(present, hashed.stdout.split(), strict=True))
    digest = hashlib.sha256()
    for path in sorted(entries):
        digest.update(f"{path}\x00{entries[path]}\n".encode())
    return digest.hexdigest()


def prune_orphaned_models(
    conn: sqlite3.Connection, kind: str, refs: tuple[tuple[str, str], ...]
) -> int:
    """Delete every ``<kind>:`` registry row that no ``(table, id_column)`` reference holds.

    Args:
        kind: Trusted literal from the owning lifecycle module (identifiers cannot be params).
        refs: Trusted ``(table, id_column)`` literals; a derived table's parent-id column
            counts as a reference. Returns rows removed; one atomic uncommitted DELETE.
    """
    conditions = "".join(
        f"AND NOT EXISTS (SELECT 1 FROM {table} WHERE {table}.{column} = models.model_id) "
        for table, column in refs
    )
    cur = conn.execute(
        f"DELETE FROM models WHERE model_id LIKE ? {conditions}",
        (f"{kind}:%",),
    )
    return cur.rowcount


def get_model(conn: sqlite3.Connection, model_id: str) -> ModelRow | None:
    """Read one ``models`` row, or None if the id is not registered."""
    row = conn.execute(
        "SELECT model_id, kind, artifact_sha FROM models WHERE model_id = ?",
        (model_id,),
    ).fetchone()
    if row is None:
        return None
    return ModelRow(
        model_id=row["model_id"],
        kind=row["kind"],
        artifact_sha=row["artifact_sha"],
    )


def ensure_registered(conn: sqlite3.Connection, row: ModelRow) -> None:
    """Idempotent INSERT into ``models``; does not commit.

    Raises:
        ValueError: The id already exists under a different ``artifact_sha`` — one model_id
            must never denote two artifacts. The single ``ON CONFLICT DO NOTHING`` insert
            makes racing writers fall through to this comparison against the row that won.
    """
    conn.execute(
        "INSERT INTO models (model_id, kind, artifact_sha) VALUES (?, ?, ?) "
        "ON CONFLICT(model_id) DO NOTHING",
        (row.model_id, row.kind, row.artifact_sha),
    )
    registered = get_model(conn, row.model_id)
    if registered is not None and registered.artifact_sha != row.artifact_sha:
        raise ValueError(
            f"model_id {row.model_id!r} already registered with "
            f"artifact_sha {registered.artifact_sha!r}; refusing to re-register "
            f"under {row.artifact_sha!r}"
        )
