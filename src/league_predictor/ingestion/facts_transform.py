"""The one shared per-day-group facts transform: raw maps -> series / games / draft."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from league_predictor.db.rows import DraftActionRow, GameRow, SeriesRow
from league_predictor.db.series_identity import mint_series_id
from league_predictor.vocab.draft_order import ordered_picks

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

from league_predictor.vocab.series import WINS_NEEDED


# sightline-ok: 14 - the clump's home is vocab.draft_order.ordered_picks, which these forward to
def _pick_sequence(
    blue_picks: list[str],
    red_picks: list[str],
    *,
    first_pick_is_blue: bool,
    champ_to_role: Mapping[str, str],
) -> list[tuple[str, str, str | None]]:
    """The interleaved 10-pick order as ``(champion, side, role)`` tuples.

    Returns:
        Picks in draft order; roles are all-or-nothing -- if ANY champion lacks a role
        mapping every role is None (order and side are always kept).
    """
    picks = [
        (champ, side, champ_to_role.get(champ))
        for champ, side, _ in ordered_picks(
            blue_picks, red_picks, first_pick_is_blue=first_pick_is_blue
        )
    ]
    if any(role is None for _, _, role in picks):
        return [(champ, side, None) for champ, side, _ in picks]
    return picks


def infer_series_format(map_gns: Sequence[int], map_labels: Sequence[int]) -> str:
    """TRUE best-of from the realised ``game_number`` run -- never a league field (~38% wrong).

    Returns:
        "Bo1"/"Bo3"/"Bo5" from ``map_gns`` (game_number per map) and ``map_labels`` (1 iff
        canonical team_a won); missing maps only under-label, and a true Bo2 reads "Bo3".
    """
    wins_a = sum(map_labels)
    wins_b = len(map_labels) - wins_a
    if max(map_gns) >= 4 or max(wins_a, wins_b) >= 3:  # noqa: PLR2004 - a 4th map or 3rd win
        return "Bo5"
    if max(map_gns) >= 2 or len(map_labels) >= 2:  # noqa: PLR2004 - a 2nd map
        return "Bo3"
    return "Bo1"


def norm_ts(value: str) -> str:
    """Normalize a source timestamp (naive OE input reads as UTC) to ``db.writes`` ISO-8601."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


@dataclass(frozen=True)
class RawMap:
    """One source map, source-neutral and pre-grouping/pre-orientation.

    Attributes:
        team1_id/team2_id: sides as the source named them; grouping re-orients canonically.
        blue_team: the blue-side team id (a per-game fact).
        event_type: the playoffs fact -- ``'regular'`` or ``'playoffs'``.
    """

    game_id: str
    game_no: int
    date: str
    team1_id: str | None
    team2_id: str | None
    winner: str | None
    blue_team: str | None = None
    league_id: str | None = None
    patch: str | None = None
    duration_s: int | None = None
    event_type: str | None = None


@dataclass
class SeriesGroup:
    """A grouped series ready to write: canonical id/orientation, derived outcome, its maps."""

    series_id: str
    league_id: str | None
    team_a: str
    team_b: str
    scheduled_ts: str | None
    best_of: int | None
    winner: str | None
    status: str
    maps: list[RawMap]


@dataclass
class GroupResult:
    """The transform's output: grouped series plus the id maps downstream facts writers key on.

    Attributes:
        degenerate: count of dropped maps (missing side / self-pair).
    """

    groups: list[SeriesGroup] = field(default_factory=list)
    game_to_pair: dict[str, tuple[str, str]] = field(default_factory=dict)
    degenerate: int = 0


def _derive_outcome(
    team_a: str, team_b: str, maps: Iterable[RawMap]
) -> tuple[int, str | None, str]:
    """Series ``(best_of, winner, status)`` from the realised maps.

    Returns:
        ``best_of`` from ``infer_series_format``; the winner is whichever side reaches the
        format's map-win threshold, else ``incomplete`` with a NULL winner.
    """
    map_gns = [m.game_no for m in maps]
    map_labels = [1 if m.winner == team_a else 0 for m in maps]
    fmt = infer_series_format(map_gns, map_labels)
    best_of = int(fmt[2:])
    threshold = WINS_NEEDED[fmt]
    a_wins = sum(1 for m in maps if m.winner == team_a)
    b_wins = sum(1 for m in maps if m.winner == team_b)
    if a_wins >= threshold and b_wins < threshold:
        return best_of, team_a, "completed"
    if b_wins >= threshold and a_wins < threshold:
        return best_of, team_b, "completed"
    return best_of, None, "incomplete"


def group_series(raw_maps: Iterable[RawMap]) -> GroupResult:
    """Group raw maps into canonical series with derived outcomes.

    Returns:
        The grouped series and the id maps downstream writers key on. Maps with a missing
        side or a self-pair are dropped and counted in ``GroupResult.degenerate``.
    """
    daygroups: dict[tuple[str, str, str], list[RawMap]] = {}
    result = GroupResult()
    for m in raw_maps:
        if m.team1_id is None or m.team2_id is None or m.team1_id == m.team2_id:
            result.degenerate += 1
            continue
        team_a = min(m.team1_id, m.team2_id)
        team_b = max(m.team1_id, m.team2_id)
        day = str(m.date)[:10]
        daygroups.setdefault((team_a, team_b, day), []).append(m)

    for (team_a, team_b, day), maps in daygroups.items():
        # split the day-group into series at game_number resets (a game_number that does
        # not strictly exceed the run's last is a new series -- keeps game_no unique)
        maps.sort(key=lambda m: (m.date, m.game_no))
        runs: list[list[RawMap]] = []
        for m in maps:
            if not runs or m.game_no <= runs[-1][-1].game_no:
                runs.append([m])
            else:
                runs[-1].append(m)
        for seq, run in enumerate(runs):
            best_of, winner, status = _derive_outcome(team_a, team_b, run)
            sid = mint_series_id(day, team_a, team_b, seq)
            result.groups.append(
                SeriesGroup(
                    series_id=sid,
                    league_id=run[0].league_id,
                    team_a=team_a,
                    team_b=team_b,
                    scheduled_ts=norm_ts(run[0].date),
                    best_of=best_of,
                    winner=winner,
                    status=status,
                    maps=run,
                )
            )
            for m in run:
                result.game_to_pair[m.game_id] = (team_a, team_b)
    return result


def series_row(group: SeriesGroup) -> SeriesRow:
    """The ``series`` fact for a grouped series (canonical, outcome derived)."""
    return SeriesRow(
        series_id=group.series_id,
        team_a=group.team_a,
        team_b=group.team_b,
        league_id=group.league_id,
        scheduled_ts=group.scheduled_ts,
        best_of=group.best_of,
        winner=group.winner,
        status=group.status,
    )


def game_rows(group: SeriesGroup) -> list[GameRow]:
    """The per-map ``games`` facts for a grouped series.

    Returns:
        Rows whose red side is the pair member the blue-side fact does not name; blue/red
        stay NULL when the source gave no blue side.
    """
    rows: list[GameRow] = []
    for m in group.maps:
        if m.blue_team == m.team1_id:
            red = m.team2_id
        elif m.blue_team == m.team2_id:
            red = m.team1_id
        else:
            red = None  # unknown blue side -- leave both untagged rather than guess
        rows.append(
            GameRow(
                game_id=m.game_id,
                series_id=group.series_id,
                game_no=m.game_no,
                blue_team=m.blue_team,
                red_team=red,
                winner=m.winner,
                patch=m.patch,
                duration_s=m.duration_s,
                event_type=m.event_type,
            )
        )
    return rows


def reconstruct_draft_actions(
    game_id: str,
    *,
    blue_picks: list[str],
    red_picks: list[str],
    blue_bans: list[str | None],
    red_bans: list[str | None],
    first_pick_is_blue: bool,
    champ_to_role: dict[str, str],
) -> list[DraftActionRow]:
    """Full pick/ban action list for one game in true draft order.

    Args:
        first_pick_is_blue: The source first-pick side, True for blue.

    Returns:
        Rows with slot-anchored ``seq`` 1-20 (an empty ban slot leaves a hole rather than
        shifting later actions); bans follow the snake structure in ``vocab/draft_order.py``
        (Ban P1: FP-SP x3, Ban P2: SP-FP x2). A pick whose champion has no known role keeps
        its order/side with ``role=None``.
    """
    if first_pick_is_blue:
        fp_bans, sp_bans, fp_side, sp_side = blue_bans, red_bans, "blue", "red"
    else:
        fp_bans, sp_bans, fp_side, sp_side = red_bans, blue_bans, "red", "blue"

    picks = _pick_sequence(
        blue_picks, red_picks, first_pick_is_blue=first_pick_is_blue, champ_to_role=champ_to_role
    )

    ban_p1 = [
        (fp_side, fp_bans, 0),
        (sp_side, sp_bans, 0),
        (fp_side, fp_bans, 1),
        (sp_side, sp_bans, 1),
        (fp_side, fp_bans, 2),
        (sp_side, sp_bans, 2),
    ]
    ban_p2 = [
        (sp_side, sp_bans, 3),
        (fp_side, fp_bans, 3),
        (sp_side, sp_bans, 4),
        (fp_side, fp_bans, 4),
    ]

    ordered: list[tuple] = []
    ordered += [("ban", s, bl, i) for s, bl, i in ban_p1]
    ordered += [("pick", p) for p in picks[:6]]
    ordered += [("ban", s, bl, i) for s, bl, i in ban_p2]
    ordered += [("pick", p) for p in picks[6:]]

    actions: list[DraftActionRow] = []
    for seq, item in enumerate(ordered, start=1):  # slot-anchored: holes stay holes
        if item[0] == "ban":
            _, side, bl, idx = item
            champ = bl[idx] if idx < len(bl) else None
            if not champ:
                continue  # empty ban slot: the seq is consumed, the row is not written
            actions.append(DraftActionRow(game_id, seq, side, "ban", champ, None))
        else:
            champion, side, role = item[1]
            actions.append(DraftActionRow(game_id, seq, side, "pick", champion, role))
    return actions
