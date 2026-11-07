"""Consumer-declared identities for historical facts prefixes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterable
    from datetime import date


@dataclass(frozen=True, slots=True)
class PrefixTable:
    """One result-affecting facts table and its stable row-order keys."""

    name: str
    key_columns: tuple[str, ...]
    optional: bool = False


@dataclass(frozen=True, slots=True)
class PrefixDigestSpec:
    """The complete database input contract for one derived result."""

    version: str
    tables: tuple[PrefixTable, ...]
    series_columns: tuple[str, ...] = (
        "series_id",
        "league_id",
        "team_a",
        "team_b",
        "scheduled_ts",
        "best_of",
    )


SERVED_CORPUS_INPUTS = PrefixDigestSpec(
    version="served-corpus-v1",
    tables=(
        PrefixTable("games", ("game_id",)),
        PrefixTable("game_players", ("game_id", "player_id"), optional=True),
        PrefixTable("game_team_stats", ("game_id", "team_id"), optional=True),
        PrefixTable("draft_scores", ("game_id", "model_id", "context"), optional=True),
    ),
)

RATING_REPLAY_INPUTS = PrefixDigestSpec(
    version="rating-replay-v1",
    tables=(
        PrefixTable("games", ("game_id",)),
        PrefixTable("game_players", ("game_id", "player_id"), optional=True),
    ),
)


def main_db_path(conn: sqlite3.Connection) -> str:
    """Return the main database path, or an empty string for an in-memory database."""
    for _seq, name, path in conn.execute("PRAGMA database_list"):
        if name == "main":
            return str(path or "")
    return ""


def cache_prefix_digests(
    conn: sqlite3.Connection,
    cutoffs: Iterable[date],
    spec: PrefixDigestSpec,
    *,
    policy: Literal["transaction_visible", "file_snapshot"] = "transaction_visible",
) -> dict[date, str]:
    """Return cache identities under an explicit connection-visibility policy.

    Args:
        policy: ``transaction_visible`` reads through Python while a transaction is open, so
            pending writes count; ``file_snapshot`` always reads the file through DuckDB,
            ignoring unrelated pending writes. A file-backed native failure raises; only
            in-memory databases fall back to Python.

    Returns:
        One digest per cutoff over the declared facts rows strictly before it, consumed in
        (event day, stable key) order — a prefix chain: appends after a cutoff preserve its
        digest; correcting ANY digested value before it (even count-preserving) flips it.
    """
    wanted = sorted(set(cutoffs))
    if not wanted:
        return {}
    path = main_db_path(conn)
    use_native = bool(path) and (policy == "file_snapshot" or not conn.in_transaction)
    if use_native:
        try:
            return native_prefix_digests(conn, path, wanted, spec)
        except Exception as exc:
            raise RuntimeError(
                f"DuckDB could not compute the file-backed facts digest: {exc}"
            ) from exc
    return python_prefix_digests(conn, wanted, spec)


def python_prefix_digests(
    conn: sqlite3.Connection, wanted: list[date], spec: PrefixDigestSpec
) -> dict[date, str]:
    """Compute portable ordered SHA-256 cache identities."""
    bounds = [cutoff.isoformat() for cutoff in wanted]
    parts: dict[date, list[str]] = {cutoff: [] for cutoff in wanted}
    for table in spec.tables:
        own_columns = _table_columns(conn, table.name, optional=table.optional)
        running = hashlib.sha256(
            f"{spec.version}\x00{table.name}\x00{','.join(own_columns or ())}".encode()
        )
        if own_columns is None:
            running.update(b"<absent>")
            final = running.hexdigest()
            for cutoff in wanted:
                parts[cutoff].append(final)
            continue
        index = 0
        for row in conn.execute(_stream_sql(table, own_columns, spec)):
            while index < len(bounds) and row[0] >= bounds[index]:
                parts[wanted[index]].append(running.copy().hexdigest())
                index += 1
            if index == len(bounds):
                break
            running.update(repr(tuple(row)).encode("utf-8"))
            running.update(b"\n")
        final = running.hexdigest()
        for cutoff in wanted[index:]:
            parts[cutoff].append(final)
    return {
        cutoff: hashlib.sha256("\x00".join([spec.version, *table_digests]).encode()).hexdigest()
        for cutoff, table_digests in parts.items()
    }


def native_prefix_digests(
    conn: sqlite3.Connection,
    db_path: str,
    wanted: list[date],
    spec: PrefixDigestSpec,
) -> dict[date, str]:
    """Compute vectorized DuckDB cache identities for the declared inputs."""
    import duckdb

    escaped = str(Path(db_path).resolve()).replace("'", "''")
    db = duckdb.connect()
    parts: dict[date, list[str]] = {cutoff: [] for cutoff in wanted}
    try:
        db.execute(f"ATTACH '{escaped}' AS facts_src (TYPE sqlite, READ_ONLY)")
        for table in spec.tables:
            own_columns = _table_columns(conn, table.name, optional=table.optional)
            schema = f"{spec.version}\x00{table.name}\x00{','.join(own_columns or ())}"
            if own_columns is None:
                absent = hashlib.sha256(f"{schema}\x00<absent>".encode()).hexdigest()
                for cutoff in wanted:
                    parts[cutoff].append(absent)
                continue
            row_sql = _native_rows_sql(table.name, own_columns, spec)
            aggregates: list[str] = []
            params: list[str] = []
            for cutoff in wanted:
                aggregates.extend(
                    (
                        "count(*) FILTER (WHERE event_d < ?)",
                        "coalesce(bit_xor(row_hash) FILTER (WHERE event_d < ?), 0)",
                        "coalesce(sum(row_hash::HUGEINT) FILTER (WHERE event_d < ?), 0)",
                    )
                )
                params.extend([cutoff.isoformat()] * 3)
            values = db.execute(
                f"SELECT {', '.join(aggregates)} FROM ({row_sql}) rows", params
            ).fetchone()
            assert values is not None
            for index, cutoff in enumerate(wanted):
                count, xor_hash, sum_hash = values[index * 3 : index * 3 + 3]
                payload = f"{schema}\x00{count}\x00{xor_hash}\x00{sum_hash}"
                parts[cutoff].append(hashlib.sha256(payload.encode()).hexdigest())
        return {
            cutoff: hashlib.sha256("\x00".join([spec.version, *table_digests]).encode()).hexdigest()
            for cutoff, table_digests in parts.items()
        }
    finally:
        try:
            db.execute("DETACH facts_src")
        finally:
            db.close()


def _table_columns(
    conn: sqlite3.Connection, table: str, *, optional: bool
) -> tuple[str, ...] | None:
    rows = conn.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()
    if not rows:
        if optional:
            return None
        raise RuntimeError(f"facts prefix input table {table!r} is missing")
    return tuple(sorted(row[1] for row in rows))


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _stream_sql(table: PrefixTable, own_columns: tuple[str, ...], spec: PrefixDigestSpec) -> str:
    if table.name == "games":
        joins = "FROM games x JOIN series s ON s.series_id = x.series_id"
        winner = "x.winner"
        extra = [f"s.{_quote_identifier(column)}" for column in spec.series_columns]
    else:
        joins = (
            f"FROM {_quote_identifier(table.name)} x "
            "JOIN games g ON g.game_id = x.game_id "
            "JOIN series s ON s.series_id = g.series_id"
        )
        winner = "g.winner"
        extra = []
    columns = ", ".join([f"x.{_quote_identifier(column)}" for column in own_columns] + extra)
    keys = ", ".join(f"x.{_quote_identifier(key)}" for key in table.key_columns)
    return (
        f"SELECT substr(s.scheduled_ts, 1, 10) AS event_d, {columns} {joins} "
        f"WHERE {winner} IS NOT NULL AND s.scheduled_ts IS NOT NULL "
        f"ORDER BY event_d, {keys}"
    )


def _native_rows_sql(table: str, own_columns: tuple[str, ...], spec: PrefixDigestSpec) -> str:
    own = [f"x.{_quote_identifier(column)}" for column in own_columns]
    if table == "games":
        joins = "FROM facts_src.games x JOIN facts_src.series s ON s.series_id = x.series_id"
        winner = "x.winner"
        values = own + [f"s.{_quote_identifier(column)}" for column in spec.series_columns]
    else:
        quoted = _quote_identifier(table)
        joins = (
            f"FROM facts_src.{quoted} x JOIN facts_src.games g ON g.game_id = x.game_id "
            "JOIN facts_src.series s ON s.series_id = g.series_id"
        )
        winner = "g.winner"
        values = own
    return (
        "SELECT substr(s.scheduled_ts, 1, 10) AS event_d, "
        f"hash({', '.join(values)}) AS row_hash {joins} "
        f"WHERE {winner} IS NOT NULL AND s.scheduled_ts IS NOT NULL"
    )
