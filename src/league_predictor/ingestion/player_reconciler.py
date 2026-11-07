"""The single home for OE player identity: fold reminted player_ids onto canonical ones."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Mapping

    import pandas as pd

logger = logging.getLogger(__name__)

# Hand-adjudicated remints with no within-match duplicate overlap, so detection cannot
# see them. Each entry is alias_id -> canonical_id (canonical = the senior history).
CONFIRMED_FOLDS: dict[str, str] = {
    # AKIRA (LCK Academy sup): Dplus Kia Academy 24 / T1 Rookies 25 -> DN SOOPers Academy 26.
    "oe:player:a878546de507670f5e83d7763c4c70b": "oe:player:1d3d6f4976411d319538f08ea907726",
    # Bf (NEXO sup): same-day handoff at the same event + shared team.
    "oe:player:66e5b72ca89ff0059c58a9119d7c660": "oe:player:7b8d2f92148f52cad46e331936922e8",
    # child (LRS sup): casing remint (child -> chilD), clean succession
    # (2025-07 -> 2026-07) + shared team (Seven Dark), same scene/role.
    "oe:player:3e399114ebc16721af820c236337309": "oe:player:a82fea383f121c803741f8de1c20973",
    # chips (LDL bot): shared team; the junior id is 3 games nested in the senior span.
    "oe:player:e85bdbd40e46cd880bdff0f74bbe3fe": "oe:player:5b4d15f5fc900a9f1a7d51c9cfc3424",
    # Easylove (VCS bot): Leaguepedia redirects EasyLove -> Easylove (one player).
    "oe:player:290980a65afac4fc497b70359bddb47": "oe:player:8bc5c92cfeab0c2f650afbfcbff035c",
    # En (PCS sup): Dewish/TNU Eagle/HELL PIGS -> SillySilly Gaming, same scene.
    "oe:player:6d416f7786dbe588eaa6ab48666588d": "oe:player:aa85f1840e282d8a8304dbc2c0d7c9b",
    # Fenrir (KT bot): casing remint (FenRir -> Fenrir), shared team (KT Rolster
    # Challengers), clean succession (senior through 2026-08-09, junior from 2026-08-13).
    "oe:player:d384cf5ad7db803f29799f36ebbc3ea": "oe:player:655ff5d5e29058249fdf0ab0e1389f8",
    # Ignar (the LCS/LEC sup): ids INTERLEAVE by vintage; Leaguepedia redirects
    # IgNar -> Ignar, and the career path is one person.
    "oe:player:b6f837c8bd1b510dd3f77ee8c903c30": "oe:player:e22cffb92554f5dde312609feccc6f8",
    # Jactroll (Polish sup): AGO/Vitality/Origen/Zero Tenacity -> Barczaca 2026.
    "oe:player:088a8dda4109f6904aa917153b5db18": "oe:player:3e6a33c44716bae499ef0111a7e60d4",
    # SOLBON (LCK Academy mid): shared team across the split.
    "oe:player:5acf2785e5891a5be39e0be16d88282": "oe:player:f557edcff47190d6f3e4ab271c0c40c",
    # Zoen (LTA-S top, Enzo Ganino): one Leaguepedia page; Pampas/inf -> paiN Academy.
    "oe:player:7edb662a8ada86b7a8a09410b3492d5": "oe:player:0597ac632095e480098c6cb5c1ae596",
}

# Normalized handles whose id pairs were adjudicated as DIFFERENT people — never
# fold, and don't re-flag them in the collision warner.
CONFIRMED_DISTINCT: frozenset[str] = frozenset(
    {
        "taeyoon",  # Kim Tae-yoon (LCK bot) vs Lee Tae-yoon (LCK Academy top) — wiki disambig
        "bean",  # EU bot (Aegis) vs LCK Academy top (Gen.G Scholars)
        "chao",  # EU bot (NEXO/IC) vs CN amateur mid (WLRQ 小爬虫)
        "fisher",  # KR mid (DFM, LCP) vs PT mid (PRMP/LPLOL)
        "juny",  # BR jng (IGNIS/CD) vs KR jng (KeSPA Cup)
        "k1ng",  # OCE/NA bot (retired 2024) vs IT/GR bot (LIT/HLL)
        "kratos",  # VCS/LCP top (MVK) vs RAMPAGE top (different scene)
        "liang",  # PCS jng (Reignfall) vs CN amateur jng (WLRQ 小爬虫) — no shared team
        "mg",  # GR sup (GLLPA 2024) vs LAS/LDL/AL mid — roles and scenes differ
        "mikasa",  # CN jng (EDG Youth, Asia Master) vs CD top — roles differ, no shared team
        "miracle",  # NACL/LRN sup vs HW jng
        "noway",  # VN bot (VCS) vs DE mid (PRM/NLC)
        "random",  # EU mid (LVP/EM) vs BR mid (CBLOLA/LTA S)
        "sorrow",  # CN jng (LPL/LDL) vs ES bot (LVP/IC)
        "storm",  # NEXO sup vs PRMP bot
        "titan",  # BR bot (CBLOL/LTA S) vs NA mid (NACL)
        "tundra",  # EBL sup vs TCL bot
        "whisper",  # LCK Academy bot/mid vs CN jng (LDL/PCS)
    }
)

# The same (game, team, role, champion) slot under two player_ids is one person
# republished. Champion equality keeps the 2020-21 champion-conflict rows out of the fold.
_DUP_REMINT_SQL = """
SELECT g1.player_id AS id_a, g2.player_id AS id_b
FROM game_players g1
JOIN game_players g2
  ON g2.game_id = g1.game_id AND g2.team_id = g1.team_id
 AND g2.role = g1.role AND g2.champion_id = g1.champion_id
 AND g2.player_id < g1.player_id
GROUP BY g1.player_id, g2.player_id
"""


def detect_dup_remints(conn: sqlite3.Connection) -> dict[str, str]:
    """Detect reminted ids from within-game duplicate evidence.

    Returns:
        alias_id -> canonical_id, canonical being the earliest first game, then more
        games, then the smaller id; durability comes from the ``player_id_map`` pin.
    """
    pairs = conn.execute(_DUP_REMINT_SQL).fetchall()
    if not pairs:
        return {}
    ids = sorted({pid for pair in pairs for pid in pair})
    placeholders = ",".join("?" * len(ids))
    seniority = {
        row[0]: (row[1], -row[2], row[0])
        for row in conn.execute(
            f"""SELECT g.player_id, MIN(substr(s.scheduled_ts, 1, 10)), COUNT(*)
                FROM game_players g
                JOIN games gm ON gm.game_id = g.game_id
                JOIN series s ON s.series_id = gm.series_id
                WHERE g.player_id IN ({placeholders})
                GROUP BY g.player_id""",
            ids,
        )
    }
    mapping: dict[str, str] = {}
    for id_a, id_b in pairs:
        canonical, alias = sorted((id_a, id_b), key=lambda pid: seniority[pid])
        mapping[alias] = canonical
    return mapping


def _resolve_chains(mapping: Mapping[str, str]) -> dict[str, str]:
    """Path-compress alias -> ... -> root; a cycle resolves to a no-op fold and is logged."""
    resolved: dict[str, str] = {}
    for alias in mapping:
        seen = {alias}
        target = mapping[alias]
        while target in mapping and target not in seen:
            seen.add(target)
            target = mapping[target]
        if target in mapping and target in seen:
            logger.warning(
                "player_id_map cycle at %s — folding nothing for this chain; "
                "fix CONFIRMED_FOLDS / player_id_map",
                alias,
            )
            target = alias
        resolved[alias] = target
    return resolved


def _stored_map(conn: sqlite3.Connection) -> dict[str, str]:
    """The persisted ``player_id_map`` as an alias -> canonical dict."""
    return dict(conn.execute("SELECT alias_id, canonical_id FROM player_id_map"))


def known_player_map(conn: sqlite3.Connection) -> dict[str, str]:
    """Already-adjudicated folds only: stored ``player_id_map`` plus the curated table."""
    mapping = _stored_map(conn)
    mapping.update(CONFIRMED_FOLDS)
    return _resolve_chains(mapping)


def canonical_player_map(conn: sqlite3.Connection) -> dict[str, str]:
    """The fold's full view: stored map + detected remints + curated, chains compressed."""
    mapping = _stored_map(conn)
    mapping.update(detect_dup_remints(conn))
    mapping.update(CONFIRMED_FOLDS)
    return _resolve_chains(mapping)


def apply_canonical_ids(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Fold the ``playerid`` column of an OE frame onto canonical ids; unchanged when no match."""
    if not mapping or "playerid" not in df.columns:
        return df
    if not df["playerid"].isin(mapping).any():  # pyright: ignore[reportGeneralTypeIssues]  # Series.any() -> numpy bool; __bool__ union is a stub artifact
        return df
    df = df.copy()
    df["playerid"] = df["playerid"].replace(mapping)
    return df


def fold_player_ids(
    conn: sqlite3.Connection, mapping: dict[str, str] | None = None
) -> dict[str, int]:
    """Fold alias player_ids onto their canonical ids across the DB; idempotent, commits.

    Returns:
        Counts of aliases folded, duplicate rows deleted, rows repointed, and same-game
        champion conflicts left in place rather than forced.
    """
    if mapping is None:
        mapping = canonical_player_map(conn)
    report = {
        "aliases_folded": 0,
        "dup_rows_deleted": 0,
        "rows_repointed": 0,
        "conflicts_skipped": 0,
    }
    for alias, canonical in mapping.items():
        if alias == canonical:
            continue
        # Persist the pair FIRST: folding deletes the duplicate evidence it was detected
        # from. An already-correct row is left alone so its evidence label stays sticky.
        stored = conn.execute(
            "SELECT canonical_id FROM player_id_map WHERE alias_id = ?", (alias,)
        ).fetchone()
        if stored is None or stored[0] != canonical:
            conn.execute(
                """INSERT INTO player_id_map (alias_id, canonical_id, evidence)
                   VALUES (?, ?, ?)
                   ON CONFLICT(alias_id) DO UPDATE SET
                       canonical_id = excluded.canonical_id,
                       evidence = excluded.evidence""",
                (alias, canonical, "curated" if alias in CONFIRMED_FOLDS else "dup-remint"),
            )
        cur = conn.execute(
            """DELETE FROM game_players WHERE player_id = ? AND game_id IN (
                   SELECT ga.game_id FROM game_players ga
                   JOIN game_players gc ON gc.game_id = ga.game_id
                    AND gc.player_id = ?
                   WHERE ga.player_id = ? AND ga.team_id = gc.team_id
                     AND ga.role IS gc.role
                     AND ga.champion_id IS gc.champion_id)""",
            (alias, canonical, alias),
        )
        dup_deleted = cur.rowcount
        # OE sometimes remints the TEAM id too, republishing the slot under a team not in
        # the game's series pair; the series pair is the authority on who played.
        cur = conn.execute(
            """DELETE FROM game_players WHERE (game_id, player_id) IN (
                   SELECT bad.game_id, bad.player_id FROM game_players bad
                   JOIN game_players good ON good.game_id = bad.game_id
                    AND good.role IS bad.role
                    AND good.champion_id IS bad.champion_id
                   JOIN games gm ON gm.game_id = bad.game_id
                   JOIN series s ON s.series_id = gm.series_id
                   WHERE bad.team_id NOT IN (s.team_a, s.team_b)
                     AND good.team_id IN (s.team_a, s.team_b)
                     AND ((bad.player_id = :a AND good.player_id = :c)
                       OR (bad.player_id = :c AND good.player_id = :a)))""",
            {"a": alias, "c": canonical},
        )
        dup_deleted += cur.rowcount
        # The NOT IN guard keeps PRIMARY KEY (game_id, player_id) safe and
        # leaves champion-conflict co-occurrences (canonical still present) alone.
        cur = conn.execute(
            """UPDATE game_players SET player_id = ?
               WHERE player_id = ? AND game_id NOT IN (
                   SELECT game_id FROM game_players WHERE player_id = ?)""",
            (canonical, alias, canonical),
        )
        repointed = cur.rowcount
        conflicts = conn.execute(
            "SELECT COUNT(*) FROM game_players WHERE player_id = ?", (alias,)
        ).fetchone()[0]
        if conflicts:
            logger.warning(
                "fold_player_ids: %d champion-conflict rows left under %s "
                "(canonical %s) — investigate before forcing",
                conflicts,
                alias,
                canonical,
            )
        # Handle variants resolve to the canonical id from now on.
        conn.execute(
            "UPDATE player_aliases SET player_id = ? WHERE player_id = ?",
            (canonical, alias),
        )
        # Purge derived per-player state under the dead id; analytics refresh and the
        # roster builder re-derive it under the canonical id.
        for table in ("player_daily_snapshot", "rosters"):
            conn.execute(f"DELETE FROM {table} WHERE player_id = ?", (alias,))
        if dup_deleted or repointed:
            report["aliases_folded"] += 1
            report["dup_rows_deleted"] += dup_deleted
            report["rows_repointed"] += repointed
        report["conflicts_skipped"] += conflicts
    conn.commit()
    return report


def warn_unadjudicated_collisions(conn: sqlite3.Connection) -> list[str]:
    """Warn on OE handles resolving to 2+ ids that are neither folded nor confirmed distinct."""
    # Spacing is stripped along with casing — OE remints on both.
    rows = conn.execute(
        """SELECT replace(lower(trim(a.source_name)), ' ', ''),
                  COUNT(DISTINCT a.player_id)
           FROM player_aliases a
           WHERE a.source = 'oe'
             AND EXISTS (SELECT 1 FROM game_players g WHERE g.player_id = a.player_id)
           GROUP BY replace(lower(trim(a.source_name)), ' ', '')
           HAVING COUNT(DISTINCT a.player_id) > 1"""
    ).fetchall()
    flagged = sorted(norm for norm, _ in rows if norm not in CONFIRMED_DISTINCT)
    for norm in flagged:
        logger.warning(
            "Unadjudicated OE handle collision: %r maps to multiple player_ids "
            "with games — adjudicate in ingestion/player_reconciler.py "
            "(CONFIRMED_FOLDS or CONFIRMED_DISTINCT)",
            norm,
        )
    return flagged
