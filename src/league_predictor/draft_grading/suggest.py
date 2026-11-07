"""Next-pick suggestions + pick-ordered scoring."""  # sightline-ok: 27 - one search engine

# file-length-ok: one search engine — DraftSuggestEngine's build, sweep, refine, rescore and
# finish passes mutate the same _BoardSearch objects in order; a split scatters that state.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from league_predictor import clock
from league_predictor.draft_grading.grader.bulk_stats import BulkStatsCache
from league_predictor.draft_grading.grader.features import DEFAULT_MIN_ROLE_GAMES
from league_predictor.draft_grading.grader.predictor import SwapRequest
from league_predictor.draft_grading.recipe import GATE_K
from league_predictor.draft_grading.roles import PlayCountRoles
from league_predictor.draft_grading.state import UNPICKED
from league_predictor.vocab.draft_order import remaining_turns
from league_predictor.vocab.roles import N_ROLES, ROLE_ORDER

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path

    from league_predictor.draft_grading.grader.predictor import (
        DraftScore as DraftScore,  # noqa: PLC0414 - PEP 484 explicit re-export (the suggest surface's score type)
    )

# Stage-1 survivors that get the adversarial (best-enemy-reply) pass: greedy move
# ordering keeps the minimax stage one batched forward wide instead of pool-squared.
_SHORTLIST = 16

# Stage-2 survivors that get a full-draft rollout. The lockstep rollout advances this
# many boards together, so the knob widens each ply's batched search, not wall time.
_ROLLOUT_BREADTH = 8

# Role legality needs real usage, not just an absolute floor: at least this share of the
# champion's own games (Rakan's 26 jungle games out of 16k is not a jungler).
_MIN_ROLE_SHARE = 0.02

# Lifetime counts alone keep a dead regime alive forever, so suggestion legality also
# needs this many games in the role inside the trailing window. Live suggest only.
_MIN_ROLE_RECENT_GAMES = 3
_ROLE_RECENCY_DAYS = 365

# Evidence floor for candidates: the decayed (champion, role) n_eff below which the blend
# gate collapses and the search would exploit cold picks. = GATE_K/4, pinned by test.
_MIN_CANDIDATE_N_EFF = GATE_K / 4


def _playable_role_pools(
    role_counts: Mapping[tuple[str, str], int],
    recent_role_counts: Mapping[tuple[str, str], int],
) -> dict[str, set[str]]:
    """Role -> champions that really play it now: pool floor, share threshold, recency floor."""

    totals: dict[str, int] = {}
    for (champ, _role), n in role_counts.items():
        totals[champ] = totals.get(champ, 0) + n
    pools: dict[str, set[str]] = {role: set() for role in ROLE_ORDER}
    for (champ, role), n in role_counts.items():
        if (
            role in pools
            and n >= DEFAULT_MIN_ROLE_GAMES
            and n >= _MIN_ROLE_SHARE * totals[champ]
            and recent_role_counts.get((champ, role), 0) >= _MIN_ROLE_RECENT_GAMES
        ):
            pools[role].add(champ)
    return pools


def board_after_pick(
    blue_picks: list[str],
    red_picks: list[str],
    side: str,
    champion: str,
) -> tuple[list[str], list[str]]:
    """The (blue, red) pick lists once ``side`` picks ``champion`` — one home for placement."""
    if side == "blue":
        return [*blue_picks, champion], list(red_picks)
    return list(blue_picks), [*red_picks, champion]


def require_legal_draft(
    blue_picks: list[str],
    red_picks: list[str],
    blue_bans: list[str],
    red_bans: list[str],
) -> None:
    """Raise ValueError on an illegal draft: over 5 picks a side, or a champion used twice."""
    if len(blue_picks) > N_ROLES or len(red_picks) > N_ROLES:
        raise ValueError(f"a side has more than {N_ROLES} picks")
    used = blue_picks + red_picks + blue_bans + red_bans
    dupes = {c for c in used if used.count(c) > 1}
    if dupes:
        raise ValueError(f"champion used twice: {', '.join(sorted(dupes))}")


def _p_side(p_blue: float, side: str) -> float:
    """P(``side`` wins) from P(blue wins) — the ordering perspective everywhere."""
    return p_blue if side == "blue" else 1.0 - p_blue


def _our_board(line: Mapping) -> dict[str, str]:
    """The picking side's full role map once a line's candidate lands."""
    return {**line["base"], line["role"]: line["champion"]}


@dataclass
class _BoardSearch:
    """One board's in-flight state through ``_search_many``'s staged passes."""

    side: str
    their_picks: list[str]
    used: set[str]
    current_ours: dict[str, str]
    current_theirs: dict[str, str]
    groups: list
    build_draft: Callable
    lines: list = field(default_factory=list)


class DraftSuggestEngine:
    """A loaded draft grader plus role-legality state; single-threaded, caller owns locking."""

    def __init__(
        self,
        predictor,
        *,
        role_counts: dict[tuple[str, str], int],
        role_pools: dict[str, set[str]],
        stats_cache=None,
        score_cache=None,
    ) -> None:
        self._predictor = predictor
        self._roles = PlayCountRoles(role_counts)
        self._role_pools = role_pools
        self._stats_cache = stats_cache  # BulkStatsCache — suggest's in-memory stat source
        self._score_cache = score_cache  # the score path's cache (see _score_path_stats_cache)
        # (board, bans, side) -> base-policy pick. Entries are tiny strings and the
        # engine reloads on a retrain, so the memo is unbounded by design.
        self._policy_memo: dict[tuple, str | None] = {}

    @classmethod
    def load(cls, conn: sqlite3.Connection, artifact: Path) -> DraftSuggestEngine:
        """Load the grader at ``artifact`` plus its champion role data off ``conn`` (~10s).

        Args:
            conn: must outlive the engine — the stats cache and evidence gate read through it.
        """
        from league_predictor.draft_grading.live import load_grader

        predictor = load_grader(conn, artifact=artifact)
        role_counts = {
            (r["champion_id"], r["role"]): r["n"]
            for r in conn.execute(
                "SELECT champion_id, role, COUNT(*) AS n FROM game_players"
                " WHERE champion_id IS NOT NULL GROUP BY champion_id, role"
            )
        }
        recency_cutoff = (clock.utc_now() - timedelta(days=_ROLE_RECENCY_DAYS)).strftime("%Y-%m-%d")
        recent_role_counts = {
            (r["champion_id"], r["role"]): r["n"]
            for r in conn.execute(
                "SELECT gp.champion_id, gp.role, COUNT(*) AS n FROM game_players gp"
                " JOIN games g ON g.game_id = gp.game_id"
                " JOIN series s ON s.series_id = g.series_id"
                " WHERE gp.champion_id IS NOT NULL AND s.scheduled_ts >= ?"
                " GROUP BY gp.champion_id, gp.role",
                (recency_cutoff,),
            )
        }
        stats_cache = BulkStatsCache(conn, baseline=predictor.pro_excess_baseline)
        return cls(
            predictor,
            role_counts=role_counts,
            role_pools=_playable_role_pools(role_counts, recent_role_counts),
            stats_cache=stats_cache,
            score_cache=stats_cache,
        )

    # ── role assignment: the play-count law, one home (roles.py) ─────────

    # ── scoring the standing board ───────────────────────────────────────

    def score(
        self,
        blue_picks: list[str],
        red_picks: list[str],
        blue_bans: list[str],
        red_bans: list[str],
    ) -> DraftScore:
        """Grade the draft as it stands.

        Returns:
            The grader's ``DraftScore``: ``prob_blue_wins``, ``grade_blue/red``, ``edge_pp_blue``.

        Raises:
            ValueError: on an illegal draft — over 5 picks a side, or a champion used twice.
        """
        from league_predictor.draft_grading.live import build_hypothetical_draft

        require_legal_draft(blue_picks, red_picks, blue_bans, red_bans)
        return self._predictor.predict(
            build_hypothetical_draft(
                blue=self._roles.assign(blue_picks),
                red=self._roles.assign(red_picks),
                blue_bans=blue_bans,
                red_bans=red_bans,
                when=clock.utc_now(),
            ),
            stats_cache=self._score_cache,
        )

    def assigned_roles(
        self, blue_picks: list[str], red_picks: list[str]
    ) -> dict[str, dict[str, str]]:
        """Both sides' champion -> role maps under the play-count assignment, for display."""
        return {
            "blue": {champ: role for role, champ in self._roles.assign(blue_picks).items()},
            "red": {champ: role for role, champ in self._roles.assign(red_picks).items()},
        }

    def _candidate_groups(self, side_picks: list[str], used: set[str]) -> list:
        """Group a side's legal next-pick candidates by the draft they imply.

        Returns:
            ``[(teammate role map, landing role, [champions])]`` — one entry per distinct
            (map, role), each answerable by a single batched swap query.
        """
        if len(side_picks) >= N_ROLES:
            return []
        pool_all: set[str] = set().union(*self._role_pools.values()) if self._role_pools else set()
        known = set(self._predictor.index.known_champions)
        current_roles = {c: r for r, c in self._roles.assign(side_picks).items()}
        candidates = sorted((pool_all & known) - used)
        mappings = self._roles.assign_with(side_picks, candidates)
        groups: dict[tuple, tuple[dict[str, str], str, list[str]]] = {}
        cold: dict[tuple, tuple[dict[str, str], str, list[str]]] = {}
        for champ in candidates:
            mapping = mappings[champ]
            landing = next(role for role, c in mapping.items() if c == champ)
            if champ not in self._role_pools.get(landing, ()):
                continue
            if any(
                c != champ
                and current_roles.get(c) != role
                and c not in self._role_pools.get(role, ())
                for role, c in mapping.items()
            ):
                continue
            base = {role: c for role, c in mapping.items() if c != champ}
            key = (tuple(sorted(base.items())), landing)
            target = groups if self._line_is_evidenced(mapping, current_roles) else cold
            target.setdefault(key, (base, landing, []))[2].append(champ)
        return list(groups.values()) if groups else list(cold.values())

    def _line_is_evidenced(
        self, mapping: Mapping[str, str], current_roles: Mapping[str, str]
    ) -> bool:
        """Whether every cell the line CHANGES clears ``_MIN_CANDIDATE_N_EFF``."""
        if self._stats_cache is None:
            return True
        now = clock.utc_now()
        return all(
            self._stats_cache.n_eff(champ, role, now) >= _MIN_CANDIDATE_N_EFF
            for role, champ in mapping.items()
            if current_roles.get(champ) != role  # only landed/moved cells; user picks stand
        )

    # ── the suggest surface ──────────────────────────────────────────────

    def suggest(
        self,
        blue_picks: list[str],
        red_picks: list[str],
        blue_bans: list[str],
        red_bans: list[str],
        *,
        side: str,
        top_n: int = 5,
        rollout: bool = True,
    ) -> tuple[DraftScore, list[dict[str, Any]]]:
        """Rank the strongest next picks for ``side`` by where the FINISHED draft lands.

        Args:
            rollout: False returns the fast one-ply answer (~1s) with no ``p_blue_end``.

        Returns:
            ``(base, lines)`` — the standing ``DraftScore`` and up to ``top_n`` line dicts
            ``{champion, role, p_blue_end, p_blue, p_blue_now, reply, moved}``, best first.

        Raises:
            ValueError: on an illegal draft or unknown ``side``.
        """
        if side not in ("blue", "red"):
            raise ValueError(f"side must be 'blue' or 'red', got {side!r}")
        require_legal_draft(blue_picks, red_picks, blue_bans, red_bans)

        if not rollout:
            return self._search(blue_picks, red_picks, blue_bans, red_bans, side=side, top_n=top_n)
        base, lines = self._search(
            blue_picks, red_picks, blue_bans, red_bans, side=side, top_n=_ROLLOUT_BREADTH
        )
        ends = self.rollout_values(
            blue_picks,
            red_picks,
            blue_bans,
            red_bans,
            side=side,
            champions=[ln["champion"] for ln in lines],
        )
        for line, end in zip(lines, ends, strict=True):
            line["p_blue_end"] = end
        # The end-value ranking rule; the dashboard's evaluateLines (draft.js)
        # mirrors it client-side over the fast path — keep the two in step.
        lines.sort(key=lambda ln: (-_p_side(ln["p_blue_end"], side), ln["champion"]))
        return base, lines[:top_n]

    # ── stage 3: the lockstep full-draft rollout ─────────────────────────

    def rollout_values(
        self,
        blue_picks: list[str],
        red_picks: list[str],
        blue_bans: list[str],
        red_bans: list[str],
        *,
        side: str,
        champions: Iterable[str],
    ) -> list[float]:
        """End-of-draft P(blue) for each candidate, were ``side`` to pick it now.

        Returns:
            One P(blue) per candidate, aligned to ``champions`` — what ``suggest`` attaches
            as ``p_blue_end``. A side with no legal candidate skips its turn.

        Raises:
            ValueError: on an illegal draft or candidate, or an unknown ``side``.
        """
        if side not in ("blue", "red"):
            raise ValueError(f"side must be 'blue' or 'red', got {side!r}")
        boards: list[tuple[list[str], list[str]]] = []
        for champion in champions:
            blue_after, red_after = board_after_pick(blue_picks, red_picks, side, champion)
            require_legal_draft(blue_after, red_after, blue_bans, red_bans)
            boards.append((blue_after, red_after))
        if not boards:
            return []
        # Lockstep precondition: one pick each on the same standing board.
        assert all((len(b), len(r)) == (len(boards[0][0]), len(boards[0][1])) for b, r in boards)
        bans_key = (tuple(blue_bans), tuple(red_bans))
        for team_index in remaining_turns(len(boards[0][0]), len(boards[0][1])):
            ply_side = "blue" if team_index == 0 else "red"
            keys = [(tuple(b), tuple(r), *bans_key, ply_side) for b, r in boards]
            pending = [i for i, key in enumerate(keys) if key not in self._policy_memo]
            if pending:
                picks = self._policy_picks(
                    [(boards[i][0], boards[i][1], blue_bans, red_bans, ply_side) for i in pending]
                )
                for i, pick in zip(pending, picks, strict=True):
                    self._policy_memo[keys[i]] = pick
            for (b, r), key in zip(boards, keys, strict=True):
                pick = self._policy_memo[key]
                if pick is not None:
                    (b if ply_side == "blue" else r).append(pick)
        drafts = [self._build_scoring_draft(b, r, blue_bans, red_bans) for b, r in boards]
        scores = self._predictor.predict_batch(drafts, stats_cache=self._score_cache)
        return [float(s.prob_blue_wins) for s in scores]

    def rollout_value(
        self,
        blue_picks: list[str],
        red_picks: list[str],
        blue_bans: list[str],
        red_bans: list[str],
        *,
        side: str,
        champion: str,
    ) -> float:
        """One candidate's end-of-draft P(blue) — the single read of :meth:`rollout_values`."""
        return self.rollout_values(
            blue_picks, red_picks, blue_bans, red_bans, side=side, champions=[champion]
        )[0]

    def _build_scoring_draft(
        self,
        blue_picks: list[str],
        red_picks: list[str],
        blue_bans: list[str],
        red_bans: list[str],
    ):
        """A finished board as the champion-only draft the exact score path reads."""
        from league_predictor.draft_grading.live import build_hypothetical_draft

        return build_hypothetical_draft(
            blue=self._roles.assign(blue_picks),
            red=self._roles.assign(red_picks),
            blue_bans=blue_bans,
            red_bans=red_bans,
            when=clock.utc_now(),
        )

    def _policy_picks(self, jobs: list[tuple]) -> list[str | None]:
        """Each job's one-ply top line under the base policy, or None when no legal candidate."""
        # top_n=5 is the policy's own fixed breadth: the memo caches this exact function,
        # so it must never vary with a request's display top_n.
        return [
            lines[0]["champion"] if lines else None
            for _base, lines in self._search_many(list(jobs), top_n=5)
        ]

    # ── stages 1+2: the one-ply search ───────────────────────────────────

    def _search(
        self,
        blue_picks: list[str],
        red_picks: list[str],
        blue_bans: list[str],
        red_bans: list[str],
        *,
        side: str,
        top_n: int,
    ):
        """The single-board read of :meth:`_search_many`, with the standing draft's base score."""
        return self._search_many(
            [(blue_picks, red_picks, blue_bans, red_bans, side)], top_n, with_base=True
        )[0]

    def _search_many(self, jobs: Iterable[tuple], top_n: int, *, with_base: bool = False):
        """The one-ply engine search over MANY boards, each stage batched into one forward.

        Returns:
            ``[(base, lines)]`` aligned to ``jobs``, as :meth:`_finish_boards` builds it.
        """
        boards = self._build_boards(jobs)
        self._sweep_boards(boards)
        self._refine_boards(boards)
        self._rescore_shortlists(boards)
        return self._finish_boards(boards, top_n, with_base=with_base)

    def _build_boards(self, jobs: Iterable[tuple]) -> list[_BoardSearch]:
        """One :class:`_BoardSearch` per ``(blue, red, blue_bans, red_bans, side)`` job."""
        from league_predictor.draft_grading.live import build_hypothetical_draft

        def make_build_draft(side: str, blue_bans: list[str], red_bans: list[str]):
            def build_draft(our_map: dict[str, str], their_map: dict[str, str]):
                blue, red = (our_map, their_map) if side == "blue" else (their_map, our_map)
                return build_hypothetical_draft(
                    blue=blue, red=red, blue_bans=blue_bans, red_bans=red_bans, when=clock.utc_now()
                )

            return build_draft

        boards = []
        for blue_picks, red_picks, blue_bans, red_bans, side in jobs:
            our_picks, their_picks = (
                (blue_picks, red_picks) if side == "blue" else (red_picks, blue_picks)
            )
            used = set(blue_picks + red_picks + blue_bans + red_bans)
            boards.append(
                _BoardSearch(
                    side=side,
                    their_picks=their_picks,
                    used=used,
                    current_ours=self._roles.assign(our_picks),
                    current_theirs=self._roles.assign(their_picks),
                    groups=self._candidate_groups(our_picks, used),
                    build_draft=make_build_draft(side, blue_bans, red_bans),
                )
            )
        return boards

    def _sweep_boards(self, boards: list[_BoardSearch]) -> None:
        """Stage 1 — the greedy sweep, every board's group queries in one forward."""
        entries, spans = [], []
        for board in boards:
            board_entries = self._sweep_requests(board, UNPICKED)
            spans.append((len(entries), len(entries) + len(board_entries)))
            entries.extend(board_entries)
        results = (
            self._predictor.predict_swaps_marginalized_many(entries, stats_cache=self._stats_cache)
            if entries
            else []
        )
        for board, (lo, hi) in zip(boards, spans, strict=True):
            lines = self._lines_from_sweep(board.groups, results[lo:hi])
            lines.sort(key=lambda ln: (-_p_side(ln["p_blue"], board.side), ln["champion"]))
            board.lines = lines[:_SHORTLIST]

    def _refine_boards(self, boards: Iterable[_BoardSearch]) -> None:
        """Stage 2 — minimax refinement, all boards' reply queries in one forward."""
        entries, meta = [], []
        for board in boards:
            if len(board.their_picks) >= N_ROLES or not board.lines:
                continue
            enemy_groups = self._candidate_groups(board.their_picks, board.used)
            board_entries, board_meta = self._reply_requests(board, enemy_groups, UNPICKED)
            entries.extend(board_entries)
            meta.extend(board_meta)
        if entries:
            results = self._predictor.predict_swaps_marginalized_many(
                entries, stats_cache=self._stats_cache
            )
            self._apply_best_replies(meta, results)

    def _rescore_shortlists(self, boards: list[_BoardSearch]) -> None:
        """Re-score every shortlisted line exactly, in one forward, before the ``top_n`` cut."""
        # The sweep's swap path only approximates counterfactual features, and its
        # error can exceed near-tied gaps.
        drafts, plan = [], []
        for board in boards:
            replied = [ln for ln in board.lines if ln["reply"] is not None]
            drafts.extend(
                board.build_draft(_our_board(ln), board.current_theirs) for ln in board.lines
            )
            drafts.extend(
                board.build_draft(_our_board(ln), ln.pop("theirs_after_reply")) for ln in replied
            )
            plan.append(replied)
        exact = (
            self._predictor.predict_batch(drafts, stats_cache=self._score_cache) if drafts else []
        )
        pos = 0
        for board, replied in zip(boards, plan, strict=True):
            for ln in board.lines:
                ln["p_blue_now"] = float(exact[pos].prob_blue_wins)
                pos += 1
                if ln["reply"] is None:  # no enemy reply left: minimax IS the board value
                    ln["p_blue"] = ln["p_blue_now"]
            for ln in replied:
                ln["p_blue"] = float(exact[pos].prob_blue_wins)
                pos += 1

    def _finish_boards(self, boards: Iterable[_BoardSearch], top_n: int, *, with_base: bool):
        """Cut each board to ``top_n`` and attach its role moves.

        Returns:
            ``(base, lines)`` per board — ``base`` is the standing draft's score under
            ``with_base`` else None; each line carries ``p_blue`` and ``p_blue_now``.
        """
        out = []
        for board in boards:
            lines = board.lines
            lines.sort(key=lambda ln: (-_p_side(ln["p_blue"], board.side), ln["champion"]))
            lines = lines[:top_n]
            lines.sort(key=lambda ln: (-_p_side(ln["p_blue_now"], board.side), ln["champion"]))
            role_of = {champ: role for role, champ in board.current_ours.items()}
            for line in lines:
                line["moved"] = [
                    {"champion": champ, "from": role_of[champ], "to": role_to}
                    for role_to, champ in line.pop("base").items()
                    if role_of[champ] != role_to
                ]
            base = (
                self._predictor.predict(
                    board.build_draft(board.current_ours, board.current_theirs),
                    stats_cache=self._stats_cache,
                )
                if with_base
                else None
            )
            out.append((base, lines))
        return out

    def _sweep_requests(self, board: _BoardSearch, unpicked: str) -> list:
        """Stage-1 queries: one batched swap request per candidate group."""

        return [
            (
                board.build_draft(base_map, board.current_theirs),
                [
                    SwapRequest(
                        side=board.side,
                        role=landing,
                        alternatives=candidates,
                        enemy_role=landing,
                        enemy_responses=[(board.current_theirs.get(landing, unpicked), 1.0)],
                    )
                ],
            )
            for base_map, landing, candidates in board.groups
        ]

    def _lines_from_sweep(self, groups: list, results: list) -> list:
        """Stage-1 consume: unranked line dicts from the sweep's answers."""
        lines = []
        for (base_map, landing, candidates), probs in zip(
            groups, (r[0] for r in results), strict=True
        ):
            for champ, p in zip(candidates, probs, strict=True):
                lines.append(
                    {
                        "champion": champ,
                        "role": landing,
                        "p_blue": float(p),  # stage 2 overwrites with the minimax value
                        "p_blue_now": float(p),  # the pick's immediate board value
                        "reply": None,
                        "base": base_map,
                    }
                )
        return lines

    def _reply_requests(self, board: _BoardSearch, enemy_groups: list, unpicked: str):
        """Stage-2 queries: every enemy group's candidates as the reply to every line.

        Returns:
            ``(entries, meta)``, meta one ``(line, side, enemy base map, landing role,
            reply candidates)`` per entry.
        """

        enemy_side = "red" if board.side == "blue" else "blue"
        entries, meta = [], []
        for line in board.lines:
            our_map = _our_board(line)
            for ebase, erole, ecands in enemy_groups:
                # The full pool is scored even when it holds this line's own champion
                # (masked on apply): a stable alternatives tuple keeps the stat memo hot.
                entries.append(
                    (
                        board.build_draft(our_map, ebase),
                        [
                            SwapRequest(
                                side=enemy_side,
                                role=erole,
                                alternatives=ecands,
                                enemy_role=erole,
                                enemy_responses=[(our_map.get(erole, unpicked), 1.0)],
                            )
                        ],
                    )
                )
                meta.append((line, board.side, ebase, erole, ecands))
        return entries, meta

    def _apply_best_replies(self, meta: list, results: list) -> None:
        """Stage-2 consume: value each line at the enemy's best reply; mutates lines in place."""
        best_seen: dict[int, float] = {}
        for (line, side, ebase, erole, replies), probs in zip(
            meta, (r[0] for r in results), strict=True
        ):
            for champ, p in zip(replies, probs, strict=True):
                if champ == line["champion"]:
                    continue  # not a legal reply — kept in the batch for memo stability
                p_blue = float(p)
                ps = _p_side(p_blue, side)
                held = best_seen.get(id(line))
                if held is None or ps < held:
                    best_seen[id(line)] = ps
                    line["p_blue"] = p_blue
                    line["reply"] = {"champion": champ, "role": erole}
                    line["theirs_after_reply"] = {**ebase, erole: champ}
