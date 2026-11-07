"""Download, parse, and ingest Oracle's Elixir CSVs."""  # sightline-ok: 27 - one ingest pipeline

# file-length-ok: one source's ingest pipeline — download, the frozen _OE_USED_COLUMNS
# contract, parse, upsert; every stage is bound to the same CSV column names.

import hashlib
import logging
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar, cast

import httpx
import pandas as pd

from league_predictor import config
from league_predictor.db.rows import GamePlayerRow, GameTeamStatRow
from league_predictor.db.writes import (
    upsert_draft_actions,
    upsert_game,
    upsert_game_players,
    upsert_game_team_stats,
    upsert_series,
)
from league_predictor.ingestion.facts_transform import (
    RawMap,
    game_rows,
    group_series,
    reconstruct_draft_actions,
    series_row,
)
from league_predictor.ingestion.player_reconciler import (
    apply_canonical_ids,
    known_player_map,
)
from league_predictor.vocab.roles import N_ROLES

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


def _resolve_gdrive_redirect(
    content: str,
    *,
    gdrive_id: str,
    year: int,
    local_path: Path,
    has_cached: bool,
) -> str | None:
    """Parse a Drive confirmation page for the download URL; None means use the cached copy."""
    if "quota" in content.lower():
        if has_cached:
            logger.warning(
                "Google Drive quota exceeded for OE %d — using cached copy (%s)",
                year,
                local_path,
            )
            return None
        raise ValueError(
            f"OE download for {year} failed: Google Drive download quota exceeded. "
            "Too many users have downloaded this file recently. "
            "Try again later, or make a personal copy of the file in Google Drive."
        )

    for pattern in [
        r'href="(/uc\?export=download[^"]+)"',
        r'id="download-form" action="([^"]+)"',
        r'href="(https://drive\.usercontent\.google\.com/download[^"]+)"',
    ]:
        match = re.search(pattern, content)
        if match:
            confirm_url = match.group(1).replace("&amp;", "&")
            if confirm_url.startswith("/"):
                confirm_url = "https://drive.google.com" + confirm_url
            return confirm_url

    return f"https://drive.usercontent.google.com/download?id={gdrive_id}&confirm=t"


def _stream_to_file(resp: httpx.Response, tmp_path: Path, year: int) -> None:
    """Stream an httpx response to a temp file, validating the first chunk."""
    with tmp_path.open("wb") as f:
        first_chunk = True
        for chunk in resp.iter_bytes(chunk_size=8192):
            if first_chunk:
                header = chunk[:512].decode("utf-8", errors="replace")
                if header.lstrip().startswith("<!") or "<html" in header.lower():
                    raise ValueError(
                        f"OE download for {year} returned HTML instead of CSV — "
                        "Google Drive redirect may have failed"
                    )
                first_chunk = False
            f.write(chunk)


def download_csv(year: int, target_dir: Path | None = None) -> Path:
    """Download Oracle's Elixir CSV from Google Drive for a given year."""
    target_dir = target_dir or config.RAW_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    gdrive_id = config.OE_GDRIVE_IDS.get(year)
    if not gdrive_id:
        raise ValueError(f"No Google Drive ID configured for year {year}")

    url = f"{config.OE_GDRIVE_BASE}{gdrive_id}&export=download&confirm=t"
    local_path = target_dir / f"oe_{year}.csv"
    logger.info("Downloading OE %d from Google Drive...", year)

    # If a valid cached copy exists, we can fall back to it on failure
    has_cached = local_path.exists() and local_path.stat().st_size > 0

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
        ),
    }
    client = httpx.Client(headers=headers, timeout=120.0, follow_redirects=True)
    tmp_path = local_path.with_suffix(".csv.tmp")
    try:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")

            if "text/html" in ct:
                # Small HTML confirmation page — safe to read fully.
                content = resp.read().decode("utf-8", errors="replace")
                download_url = _resolve_gdrive_redirect(
                    content,
                    gdrive_id=gdrive_id,
                    year=year,
                    local_path=local_path,
                    has_cached=has_cached,
                )
                if download_url is None:
                    return local_path
            else:
                # Already streaming the CSV — write directly to disk.
                _stream_to_file(resp, tmp_path, year)
                tmp_path.replace(local_path)
                download_url = None

        # If we resolved a redirect URL, stream that in a second request.
        if download_url is not None:
            with client.stream("GET", download_url) as resp:
                resp.raise_for_status()
                _stream_to_file(resp, tmp_path, year)
            tmp_path.replace(local_path)

    except (httpx.HTTPError, OSError, ValueError):
        # The expected transport/parse failures fall back to the cached copy;
        # anything else is a bug and propagates.
        tmp_path.unlink(missing_ok=True)
        if has_cached:
            logger.warning(
                "Download failed for OE %d — using cached copy (%s)",
                year,
                local_path,
            )
            return local_path
        raise
    finally:
        client.close()

    size_mb = local_path.stat().st_size / 1024 / 1024
    logger.info("Saved to %s (%.1f MB)", local_path, size_mb)
    return local_path


# Columns referenced anywhere in this module's ingest path; loading only these roughly
# halves parse time and peak memory. A callable usecols tolerates year-to-year drift.
_OE_USED_COLUMNS: frozenset[str] = frozenset(
    {
        # Row identifiers and grouping
        "gameid",
        "position",
        "side",
        "result",
        "teamid",
        "teamname",
        "playerid",
        "playername",
        "champion",
        # Match-level
        "date",
        "league",
        "playoffs",
        "patch",
        "game",
        "gamelength",
        # Player KDA / share
        "kills",
        "deaths",
        "assists",
        "teamkills",
        "teamdeaths",
        "damageshare",
        "killparticipation",
        # Vision
        "vspm",
        "controlwardsbought",
        "wpm",
        "wcpm",
        "wardsplaced",
        "wardskilled",
        # Lane / objective at 10 and 15
        "golddiffat10",
        "csdiffat10",
        "xpdiffat10",
        "golddiffat15",
        "csdiffat15",
        "xpdiffat15",
        "goldat15",
        "csat15",
        "xpat15",
        "killsat15",
        "assistsat15",
        "deathsat15",
        # Per-minute / share
        "dpm",
        "cspm",
        "earnedgoldshare",
        "damagetakenperminute",
        "damagemitigatedperminute",
        # First-event flags (player and team)
        "firstblood",
        "firsttower",
        "firstdragon",
        "firstbaron",
        "firstherald",
        "firstbloodkill",
        "firstbloodassist",
        # Team stat lines
        "dragons",
        "barons",
        "towers",
        "heralds",
        "void_grubs",
        "totalgold",
        "total cs",
        # Draft
        "ban1",
        "ban2",
        "ban3",
        "ban4",
        "ban5",
        "pick1",
        "pick2",
        "pick3",
        "pick4",
        "pick5",
        "firstPick",
    }
)


def load_csv(path: Path) -> pd.DataFrame:
    """Load an OE CSV restricted to the columns we actually use."""
    return pd.read_csv(
        path,
        usecols=lambda col: col in _OE_USED_COLUMNS,
        low_memory=False,
    )


def csv_content_hash(path: Path) -> str:
    """SHA-256 of the CSV bytes (not size/mtime) — the identity the ingest-skip marker uses."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _col_nullable(df: pd.DataFrame, col: str, cast_to: Callable[[Any], _T]) -> list[_T | None]:
    """``df[col]`` through ``cast_to``; NaN and a missing column both become None."""
    if col not in df.columns:
        return [None] * len(df)
    series = df[col]
    mask = series.isna().to_numpy()
    values = series.to_numpy()
    return [None if m else cast_to(v) for m, v in zip(mask, values, strict=True)]


def upsert_oe_team_names(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Store unique teamname→teamid pairs; never commits — the caller owns the transaction."""
    if "teamname" not in df.columns or "teamid" not in df.columns:
        return 0
    teams = df[["teamname", "teamid"]].drop_duplicates().dropna(subset=["teamname", "teamid"])  # pyright: ignore[reportCallIssue]  # pandas-stubs dropna(subset=) overload
    if teams.empty:
        return 0
    params = [
        (str(tid), str(name)) for tid, name in zip(teams["teamid"], teams["teamname"], strict=True)
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO oe_team_names (team_id, team_name) VALUES (?, ?)",
        params,
    )
    return len(params)


def _upsert_player_aliases(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Store unique playername->playerid alias pairs; never commits — the caller owns the txn."""
    if "playername" not in df.columns or "playerid" not in df.columns:
        return 0
    players = (
        df[df["position"] != "team"][["playerid", "playername"]]  # pyright: ignore[reportCallIssue]  # pandas-stubs mask-index overloads
        .drop_duplicates()  # pyright: ignore[reportAttributeAccessIssue]  # pandas-stubs infers ndarray above
        .dropna(subset=["playerid", "playername"])
    )
    if players.empty:
        return 0
    params = [
        (str(pid), "oe", str(name))
        for pid, name in zip(players["playerid"], players["playername"], strict=True)
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO player_aliases (player_id, source, source_name) VALUES (?, ?, ?)",
        params,
    )
    return len(params)


def _maybe_str(val) -> str | None:
    """``str(val)`` with None/NaN → None."""
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    return str(val)


def _team_draft_fields(team_row) -> tuple[list[str | None] | None, list[str | None] | None]:
    """(bans, picks) for one team row: length-5 lists, ``None`` per missing entry or group."""
    bans = [_maybe_str(team_row.get(f"ban{i}")) for i in range(1, 6)]
    picks = [_maybe_str(team_row.get(f"pick{i}")) for i in range(1, 6)]
    return (
        bans if any(b is not None for b in bans) else None,
        picks if any(p is not None for p in picks) else None,
    )


def _exclusive_claimant(blue, red, claims: Callable[[Any], bool | None]) -> str | None:
    """The team_id of the side that alone satisfies ``claims``; None when both or neither do."""
    blue_claims, red_claims = claims(blue), claims(red)
    if blue_claims and not red_claims:
        return str(blue["teamid"])
    if red_claims and not blue_claims:
        return str(red["teamid"])
    return None


def _winner_team_id(blue, red) -> str | None:
    """The winning side's team_id; None unless exactly one side carries ``result == 1``."""

    def _won(row) -> bool:
        v = row.get("result")
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return False
        try:
            return int(v) == 1
        except (TypeError, ValueError):
            return False

    return _exclusive_claimant(blue, red, _won)


def _first_pick_team(blue, red) -> str | None:
    """The pick-priority team_id; None unless exactly one side claims ``firstPick``."""

    def _is_first(row) -> bool | None:
        v = row.get("firstPick")
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        try:
            return bool(int(v))
        except (TypeError, ValueError):
            return None

    return _exclusive_claimant(blue, red, _is_first)


def _int_or_none(val) -> int | None:
    """``int(val)`` with None/NaN → None."""
    if pd.isna(val):
        return None
    return int(val)


def _fact_columns(row_type: type, frame: pd.DataFrame) -> dict[str, list]:
    """Read nullable OE columns once each in the row type's declaration order."""
    aliases = {
        "player_id": "playerid",
        "team_id": "teamid",
        "role": "position",
        "champion_id": "champion",
        "totalcs": "total cs",
    }
    return {
        field.name: _col_nullable(
            frame,
            aliases.get(field.name, field.name),
            int if field.type == "int | None" else float if field.type == "float | None" else str,
        )
        for field in fields(row_type)
        if field.name != "game_id"
    }


def _game_player_rows(game_id: str, player_df: pd.DataFrame) -> list[GamePlayerRow]:
    """OE box scores, including kill participation and death share when derivable."""
    columns = _fact_columns(GamePlayerRow, player_df)
    rows = []
    for values in zip(*columns.values(), strict=True):
        row = dict(zip(columns, values, strict=True))
        if (
            row["killparticipation"] is None
            and row["teamkills"]
            and row["kills"] is not None
            and row["assists"] is not None
        ):
            row["killparticipation"] = (row["kills"] + row["assists"]) / row["teamkills"]
        row["deathshare"] = (
            row["deaths"] / row["teamdeaths"]
            if row["deaths"] is not None and row["teamdeaths"]
            else None
        )
        rows.append(GamePlayerRow(game_id=game_id, **row))
    return rows


def _game_team_stat_rows(game_id: str, team_df: pd.DataFrame) -> list[GameTeamStatRow]:
    """Team event counts and OE's space-named total CS column."""
    columns = _fact_columns(GameTeamStatRow, team_df)
    return [
        GameTeamStatRow(game_id=game_id, **dict(zip(columns, values, strict=True)))
        for values in zip(*columns.values(), strict=True)
    ]


@dataclass(frozen=True)
class _ParsedGame:
    """One OE game parsed off the frame: map fact, player rows, team stats, draft inputs."""

    raw_map: RawMap
    players: list[GamePlayerRow]
    team_stats: list[GameTeamStatRow]
    draft: dict


def _parse_game(gid: str, game_df: pd.DataFrame) -> _ParsedGame | None:
    """Parse one ``gameid`` group into a ``_ParsedGame``; ``None`` on a degenerate group.

    Returns:
        ``draft['first_pick_is_blue']`` is three-state — ``None`` when OE's ``firstPick``
        is missing or ambiguous, which skips ordered draft reconstruction.
    """
    team_rows = cast("pd.DataFrame", game_df[game_df["position"] == "team"])
    player_rows = cast("pd.DataFrame", game_df[game_df["position"] != "team"])
    if team_rows.empty or player_rows.empty:
        return None
    blue_team = team_rows[team_rows["side"] == "Blue"]
    red_team = team_rows[team_rows["side"] == "Red"]
    if blue_team.empty or red_team.empty:
        return None
    blue, red = blue_team.iloc[0], red_team.iloc[0]
    if pd.isna(blue["teamid"]) or pd.isna(red["teamid"]):
        return None  # a team OE never linked; the game rates no one

    gn = _int_or_none(blue.get("game"))
    league = blue.get("league") if not pd.isna(blue.get("league")) else None
    # OE's playoffs flag becomes the game's event_type fact ('playoffs'/'regular').
    playoffs = blue.get("playoffs")
    event_type = (
        "playoffs" if (playoffs is not None and not pd.isna(playoffs) and playoffs) else "regular"
    )

    players = _game_player_rows(gid, player_rows)
    blue_bans, blue_picks = _team_draft_fields(blue)
    red_bans, red_picks = _team_draft_fields(red)
    first_pick_team_id = _first_pick_team(blue, red)
    return _ParsedGame(
        raw_map=RawMap(
            game_id=gid,
            game_no=1 if gn is None else gn,
            date=str(blue["date"]),
            team1_id=str(blue["teamid"]),
            team2_id=str(red["teamid"]),
            winner=_winner_team_id(blue, red),
            blue_team=str(blue["teamid"]),
            league_id=league,
            patch=_maybe_str(blue.get("patch")),
            duration_s=_int_or_none(blue.get("gamelength")),
            event_type=event_type,
        ),
        # A blank OE playerid is a player OE never linked; the draft still uses its champion.
        players=[p for p in players if p.player_id is not None],
        team_stats=_game_team_stat_rows(gid, team_rows),
        draft={
            "blue_picks": blue_picks,
            "red_picks": red_picks,
            "blue_bans": blue_bans or [],
            "red_bans": red_bans or [],
            "first_pick_is_blue": (
                None if first_pick_team_id is None else first_pick_team_id == str(blue["teamid"])
            ),
            "champ_to_role": {p.champion_id: p.role for p in players if p.champion_id and p.role},
        },
    )


_REQUIRED_FIELDS = {
    row_type: tuple(f.name for f in fields(row_type) if "None" not in str(f.type))
    for row_type in (GamePlayerRow, GameTeamStatRow)
}


def _missing_required(parsed: list[_ParsedGame]) -> list[str]:
    """One line per ``table.column`` that parsed rows leave NULL, naming sample games."""
    tables = {GamePlayerRow: "game_players", GameTeamStatRow: "game_team_stats"}
    gaps: dict[str, list[str]] = {}
    for game in parsed:
        for row in (*game.players, *game.team_stats):
            for name in _REQUIRED_FIELDS[type(row)]:
                if getattr(row, name) is None:
                    gaps.setdefault(f"{tables[type(row)]}.{name}", []).append(row.game_id)
    return [
        f"{col}: {len(gids)} rows in {len(set(gids))} games, e.g. {sorted(set(gids))[:3]}"
        for col, gids in sorted(gaps.items())
    ]


def ingest_oe_results(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Idempotently ingest an OE results frame into the facts tables; owns the transaction.

    Returns:
        Games written. Every write commits together at the end; a failure rolls the
        whole batch back.
    """
    try:
        df = apply_canonical_ids(df, known_player_map(conn))
        upsert_oe_team_names(conn, df)
        _upsert_player_aliases(conn, df)
        is_team = df["position"] == "team"
        unlinked_games = df.loc[is_team & df["teamid"].isna(), "gameid"].nunique()
        unlinked_players = int((~is_team & df["playerid"].isna()).sum())
        if unlinked_games or unlinked_players:
            logger.info(
                "Skipping %d games with no OE teamid and %d player rows with no OE playerid",
                unlinked_games,
                unlinked_players,
            )

        parsed = [
            game
            for game_id, game_df in df.groupby("gameid")
            if (game := _parse_game(str(game_id), game_df)) is not None
        ]
        if gaps := _missing_required(parsed):
            raise ValueError("OE rows leave required columns NULL: " + "; ".join(gaps))
        grouped = group_series([g.raw_map for g in parsed])

        # FK dimensions first (INSERT OR IGNORE; names backfilled by their own vocabularies).
        league_ids = {g.raw_map.league_id for g in parsed if g.raw_map.league_id}
        team_ids = {t for g in parsed for t in (g.raw_map.team1_id, g.raw_map.team2_id)}
        team_ids |= {g.raw_map.winner for g in parsed if g.raw_map.winner}
        team_ids |= {p.team_id for g in parsed for p in g.players}
        player_ids = {p.player_id for g in parsed for p in g.players}
        conn.executemany(
            "INSERT OR IGNORE INTO leagues (league_id, name) VALUES (?, ?)",
            [(lg, lg) for lg in league_ids],
        )
        conn.executemany(
            "INSERT OR IGNORE INTO teams (team_id) VALUES (?)", [(t,) for t in team_ids]
        )
        conn.executemany(
            "INSERT OR IGNORE INTO players (player_id) VALUES (?)", [(p,) for p in player_ids]
        )

        n_games = 0
        for g in grouped.groups:
            upsert_series(conn, series_row(g))
            for gr in game_rows(g):
                upsert_game(conn, gr)
                n_games += 1

        for game in parsed:
            gid = game.raw_map.game_id
            pair = grouped.game_to_pair.get(gid)
            if pair is None:
                continue  # its map was degenerate / never written as a game
            upsert_game_players(conn, game.players)
            keep = [s for s in game.team_stats if s.team_id in pair]  # drop off-pair dirt
            if keep:
                upsert_game_team_stats(conn, keep)
            d = game.draft
            # Skip ordered reconstruction when the first-pick side is unknown: every
            # draft_actions seq is a true draft slot, so there is no honest fallback.
            if (
                d["first_pick_is_blue"] is not None
                and d["blue_picks"]
                and d["red_picks"]
                and len(d["blue_picks"]) == N_ROLES
                and len(d["red_picks"]) == N_ROLES
            ):
                actions = reconstruct_draft_actions(
                    gid,
                    blue_picks=d["blue_picks"],
                    red_picks=d["red_picks"],
                    blue_bans=d["blue_bans"],
                    red_bans=d["red_bans"],
                    first_pick_is_blue=d["first_pick_is_blue"],
                    champ_to_role=d["champ_to_role"],
                )
                if actions:
                    upsert_draft_actions(conn, actions)

        conn.commit()
    except Exception:
        conn.rollback()  # transaction owner: a partial batch never leaks
        raise
    else:
        return n_games
