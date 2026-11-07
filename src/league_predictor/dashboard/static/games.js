/* Matches (Games) — the landing tab. List view is a paginated match
   browser (date, matchup, pick icons, as-of ratings, OOS prediction,
   draft grades); clicking a row opens the per-game detail on the same
   page (/games?id=<game_id>): one draft board (bans + role-ordered picks
   with per-pick grade meters), per-player KDA/CS lines, objective stats,
   prediction. Every model-derived cell renders a dash when null. */

const {esc, fmt, links, fetchJSON, valClass, champIcon, filters, edgePP} = HERALD;

const rootEl = document.getElementById('games-root');
const PAGE = 50;

let listState = {offset: 0, total: 0, q: ''};

/* ── List view ── */
function goGame(gameId) {
    history.pushState(null, '', `/games?id=${encodeURIComponent(gameId)}`);
    loadGame(gameId);
}

async function loadList() {
    rootEl.innerHTML = `
        <div class="page-hero">
            <div>
                <div class="label"><span class="accent">◆</span> Matches</div>
                <div class="entity-name" style="font-size:30px;">Match Browser</div>
            </div>
            <input type="search" id="game-q" placeholder="Filter by team…" value="${esc(listState.q)}" style="width:240px;">
        </div>
        <div class="panel">
            <div class="panel-h">
                <span class="label">Last Games</span>
                <span class="label" id="game-total" style="color:var(--text-muted)"></span>
            </div>
            <div id="game-table"><div class="loading">Loading matches</div></div>
        </div>
        <div class="pager" id="game-pager"></div>`;

    const qEl = document.getElementById('game-q');
    qEl.addEventListener('change', () => { listState.q = qEl.value.trim(); listState.offset = 0; loadList(); });
    qEl.addEventListener('keydown', (e) => { if (e.key === 'Enter') { listState.q = qEl.value.trim(); listState.offset = 0; loadList(); } });

    const qs = filters.params()
        + `&limit=${PAGE}&offset=${listState.offset}`
        + (listState.q ? `&q=${encodeURIComponent(listState.q)}` : '');
    const data = await fetchJSON(`/api/games?${qs}`);
    const tableEl = document.getElementById('game-table');
    if (!tableEl) return;              // navigated away mid-fetch
    if (!data) { tableEl.innerHTML = '<div class="error-msg">Match feed unavailable.</div>'; return; }

    listState.total = data.total || 0;
    document.getElementById('game-total').textContent = `${listState.total} games`;
    HERALD.matchTable(tableEl, data.games || [], goGame);
    renderPager();
}

function renderPager() {
    const pager = document.getElementById('game-pager');
    if (!pager) return;
    const {offset, total} = listState;
    const from = total ? offset + 1 : 0;
    const to = Math.min(offset + PAGE, total);
    const prevDis = offset <= 0 ? 'disabled' : '';
    const nextDis = to >= total ? 'disabled' : '';
    pager.innerHTML = `
        <button class="btn quiet" id="pg-prev" ${prevDis}>Prev</button>
        <span class="pg-info">${from}–${to} of ${total}</span>
        <button class="btn quiet" id="pg-next" ${nextDis}>Next</button>`;
    const prev = pager.querySelector('#pg-prev');
    const next = pager.querySelector('#pg-next');
    if (!prevDis) prev.addEventListener('click', () => { listState.offset = Math.max(0, offset - PAGE); loadList(); });
    if (!nextDis) next.addEventListener('click', () => { listState.offset = offset + PAGE; loadList(); });
}

/* ── Detail view ── */
// Browser mirror of vocab/roles.py::ROLE_ORDER; keep this language-boundary copy aligned.
const ROLE_ORDER = Object.fromEntries(HERALD.ROLES.map((role, index) => [role, index]));

/* Side-by-side lineup comparison: blue left, red right, roles aligned
   row-by-row (TOP vs TOP, …), stat columns mirroring inward to the centered
   role label. GD15/CSD15 keep the green/red sign coloring; the winning team's
   names read green. Missing seats render dashes. */
function lineupComparison(blue, red) {
    const byRole = (side) => Object.fromEntries((side.players || []).map((p) => [p.role, p]));
    const B = byRole(blue), R = byRole(red);

    const kda = (p) => p ? `${p.kills ?? '–'}/${p.deaths ?? '–'}/${p.assists ?? '–'}` : '–';
    const diff = (v) => v == null ? '–' : `<span class="${v >= 0 ? 'val-A' : 'val-D'}">${fmt.signed(v)}</span>`;
    // Stat cells for one player, ordered outward→inward (name is placed separately).
    const stats = (p) => [
        kda(p),
        p ? fmt.num(p.cspm, 1) : '–',
        p ? fmt.int(p.dpm) : '–',
        diff(p ? p.gd15 : null),
        diff(p ? p.csd15 : null),
        p ? fmt.pct(p.kp, 0) : '–',
    ];
    const nameCell = (p, side, won) => {
        const inner = p
            ? `${links.player(p.player_id, p.player_name || p.player_id || '--')}${champIcon(p.champion, 20)}`
            : '<span style="color:var(--text-muted);">—</span>';
        return `<div class="lu-name ${side}${won ? ' won' : ''}">${inner}</div>`;
    };

    const roleRow = (role) => {
        const b = B[role], r = R[role];
        const bs = stats(b).map((v) => `<div class="b num">${v}</div>`).join('');
        // Red cells mirror: innermost (KP) first, outermost (K/D/A) last.
        const rs = stats(r).slice().reverse().map((v) => `<div class="r num">${v}</div>`).join('');
        return `<div class="lu-row">
            ${nameCell(b, 'b', blue.won === true)}${bs}
            <div class="lu-role">${esc(role.toUpperCase())}</div>
            ${rs}${nameCell(r, 'r', red.won === true)}`
            + `</div>`;
    };

    const head = `<div class="lu-row lu-head">
        <div class="b">Player</div><div class="b">K/D/A</div><div class="b">CSPM</div><div class="b">DPM</div><div class="b">GD15</div><div class="b">CSD15</div><div class="b">KP</div>
        <div class="lu-role"></div>
        <div class="r">KP</div><div class="r">CSD15</div><div class="r">GD15</div><div class="r">DPM</div><div class="r">CSPM</div><div class="r">K/D/A</div><div class="r">Player</div>
    </div>`;

    return `
        <div class="panel">
            <div class="panel-h"><span class="label">Lineups</span></div>
            <div class="panel-b">
                <div class="lu-board">
                    <div class="lu-teams">
                        <span class="lu-team blue${blue.won === true ? ' won' : ''}">${links.team(blue.team_id, blue.name || 'Blue')}${blue.won === true ? ' ✓' : ''}</span>
                        <span class="lu-vs">vs</span>
                        <span class="lu-team red${red.won === true ? ' won' : ''}">${red.won === true ? '✓ ' : ''}${links.team(red.team_id, red.name || 'Red')}</span>
                    </div>
                    ${head}
                    ${Object.keys(ROLE_ORDER).map(roleRow).join('')}
                </div>
            </div>
        </div>`;
}


/* One combined objectives scoreboard — blue left, red right, label center.
   First objectives are binary between the teams, so each renders as ONE chip
   naming the side that took it ("not in feed" when the league's source data
   lacks the field — e.g. LPL first blood/baron). */
function objectivesPanel(blue, red) {
    const goldK = (v) => v == null ? null : (v / 1000).toFixed(1) + 'k';
    const rows = [
        ['Kills', blue.kills, red.kills],
        ['Gold', blue.gold, red.gold, goldK],
        ['Towers', blue.towers, red.towers],
        ['Dragons', blue.dragons, red.dragons],
        ['Barons', blue.barons, red.barons],
        ['Heralds', blue.heralds, red.heralds],
        ['Grubs', blue.void_grubs, red.void_grubs],
    ];
    const objRow = ([label, bv, rv, f]) => {
        const show = f || ((v) => v == null ? null : fmt.int(v));
        const lead = (a, b) => a != null && b != null && a > b ? 'lead' : '';
        const led = (a, b) => lead(a, b) ? ' title="Led this stat"' : '';
        return `<div class="obj-row">
            <span class="obj-val blue ${lead(bv, rv)}"${led(bv, rv)}>${show(bv) ?? '–'}</span>
            <span class="obj-mid">${label}</span>
            <span class="obj-val ${lead(rv, bv)}"${led(rv, bv)}>${show(rv) ?? '–'}</span>
        </div>`;
    };
    const firstChip = (label, bHas, rHas) => {
        const state = bHas === true ? 'blue' : rHas === true ? 'red'
            : (bHas == null && rHas == null) ? 'none-feed' : 'none-taken';
        const who = state === 'blue' ? (blue.name || 'Blue') : state === 'red' ? (red.name || 'Red')
            : state === 'none-feed' ? 'not in feed' : 'none';
        return `<span class="first-chip ${state === 'blue' ? 'blue' : state === 'red' ? 'red' : 'none'}">
            First ${label}<span class="who">${esc(who)}</span></span>`;
    };
    const teamCell = (side, cls) => `<span class="obj-val ${cls}" style="font-weight:700;${side.won ? 'color:var(--win);' : 'color:var(--text-bright);'}"${side.won ? ' title="Winner"' : ''}>${esc(side.name || '--')}${side.won ? ' ✓' : ''}</span>`;
    return `
        <div class="panel">
            <div class="panel-h"><span class="label">Objectives</span></div>
            <div class="panel-b">
                <div class="obj-board">
                    <div class="obj-row" style="border-bottom:1px solid var(--line);">
                        ${teamCell(blue, 'blue')}
                        <span class="obj-mid">blue · red</span>
                        ${teamCell(red, '')}
                    </div>
                    ${rows.map(objRow).join('')}
                    <div class="firsts-row">
                        ${firstChip('blood', blue.firstblood, red.firstblood)}
                        ${firstChip('tower', blue.firsttower, red.firsttower)}
                        ${firstChip('dragon', blue.firstdragon, red.firstdragon)}
                        ${firstChip('baron', blue.firstbaron, red.firstbaron)}
                    </div>
                </div>
            </div>
        </div>`;
}

/* Per-pick draft grades, tiered into verdict words. Percentile = share of the
   champions still available at that slot the pick beat by model value. */
const VERDICTS = [
    [0.85, 'Elite', 'val-S'], [0.65, 'Strong', 'val-A'], [0.45, 'Fair', 'val-B'],
    [0.25, 'Shaky', 'val-C'], [-Infinity, 'Poor', 'val-D'],
];
const verdictOf = (p) => VERDICTS.find(([floor]) => p >= floor);

/* One combined draft board: bans + role-ordered picks per side; when the game
   has per-pick scores (data.draft), each pick row carries its grade meter and
   the header carries the team grades + the draft-side blue win probability. */
function draftPanel(draft, blue, red) {
    const scored = draft != null;
    // pick_scores rows carry side+role only; champion/player come from the
    // side blocks' role-keyed picks.
    const scores = {};
    for (const s of (draft && draft.picks) || []) scores[`${s.side}:${s.role}`] = s;

    const pickRow = (sideName) => (pk) => {
        const s = scores[`${sideName}:${pk.role}`];
        const v = s && s.percentile != null ? verdictOf(s.percentile) : null;
        const gapPP = s && s.best_score != null && s.actual_score != null
            ? (s.best_score - s.actual_score) * 100 : null;
        const gapTxt = gapPP == null ? ''
            : gapPP < 0.05 ? 'best available'
            : `best alt +${gapPP.toFixed(1)}pp`;
        const meter = scored ? `
                <div class="dg-meter ${v ? v[2] : ''}"><div class="dg-fill" style="width:${s && s.percentile != null ? (s.percentile * 100).toFixed(0) : 0}%"></div></div>
                <span class="dg-pct ${v ? v[2] : ''}">${s ? fmt.pct(s.percentile, 0) : '–'}</span>
                <div class="dg-verdict">
                    <span class="word ${v ? v[2] : ''}">${v ? v[1] : '–'}</span>
                    <span class="gap">${gapTxt}</span>
                </div>` : '';
        return `
            <div class="dg-row${scored ? '' : ' bare'}">
                <span class="pick-role">${esc((pk.role || '').toUpperCase())}</span>
                ${champIcon(pk.champion, 26)}
                <span class="champ-name" style="min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${pk.champion ? links.champion(pk.champion) : '--'}</span>
                <span class="dg-player">${esc(pk.player_name || '')}</span>
                ${meter}
            </div>`;
    };

    const col = (sideName, team, grade, edge) => {
        const picks = (team.picks || []).slice()
            .sort((a, b) => (ROLE_ORDER[a.role] ?? 9) - (ROLE_ORDER[b.role] ?? 9));
        const bans = `<div class="dband"><span class="dband-label">Bans</span>${(team.bans || []).map((b) => b ? `<span class="ban-ic">${champIcon(b, 28)}</span>` : '').join('')}</div>`;
        const gradeHTML = scored
            ? ` <span class="mb-grade ${valClass(grade)}">${esc(grade || '–')}</span>${edgePP(edge)}` : '';
        return `
        <div class="br-col ${sideName}">
            <div class="br-head ${sideName}">
                <span class="side-tag">${sideName === 'blue' ? 'Blue side' : 'Red side'}</span>
                <span class="team-nm">${links.team(team.team_id, team.name || '--')}${gradeHTML}</span>
            </div>
            ${bans}
            ${picks.map(pickRow(sideName)).join('') || '<span style="color:var(--text-muted);">No picks recorded.</span>'}
        </div>`;
    };

    const redEdge = scored && draft.edge_pp_blue != null ? -draft.edge_pp_blue : null;
    const draftHelp = 'Each pick is scored against every champion still available at that slot: '
        + 'the bar is the share of alternatives it beat (100% = best available by model value), '
        + 'and "best alt" is the win probability the top alternative would have added.';
    return `
        <div class="panel">
            <div class="panel-h"><span class="label">Draft</span><span class="info-hint" title="${esc(draftHelp)}">ⓘ</span>
                ${scored ? `<span class="label" style="color:var(--text-muted)">Draft win probability (Blue) ${fmt.pct(draft.p_blue)}</span>` : ''}</div>
            <div class="panel-b">
                <div class="br-board">
                    ${col('blue', blue, scored ? draft.grade_blue : null, scored ? draft.edge_pp_blue : null)}
                    ${col('red', red, scored ? draft.grade_red : null, redEdge)}
                </div>
            </div>
        </div>`;
}

async function loadGame(gameId) {
    rootEl.innerHTML = '<div class="loading">Opening match</div>';
    const data = await fetchJSON(`/api/games/${encodeURIComponent(gameId)}`);
    if (!data || !data.game) {
        rootEl.innerHTML = `<button class="back-link" id="back">Back to matches</button><div class="error-msg">This match has no sheet.</div>`;
        document.getElementById('back').addEventListener('click', goList);
        return;
    }
    const g = data.game, blue = data.blue || {}, red = data.red || {};
    const seriesTabs = (data.series_games || []).map((sg) => {
        const w = sg.winner_side === 'blue' ? 'Blue' : sg.winner_side === 'red' ? 'Red' : null;
        const dot = w ? `<span class="gc-win ${sg.winner_side}"></span>` : '';
        const title = w ? `Game ${sg.game_no} — ${w} side won` : `Game ${sg.game_no}`;
        return `<button class="chip ${sg.game_id === g.game_id ? 'on' : ''}" data-g="${esc(sg.game_id)}" title="${esc(title)}">Game ${sg.game_no}${dot}</button>`;
    }).join('');
    const pred = data.prediction;
    // Post-draft rests on the in-sample backfill prior when there is no walk-forward
    // served prediction for this game — disclose it (never hide the bar).
    const postInSample = data.post_p_blue != null && (!pred || pred.p_blue == null);
    // Live-model prediction: the game postdates the walk-forward corpus, so the
    // current PRD's row is leak-free evidence — tagged, not blended in silently.
    const livePred = !!(pred && pred.source === 'live');

    rootEl.innerHTML = `
        <button class="back-link" id="back">Back to matches</button>
        <div class="page-hero">
            <div>
                <div class="label"><span class="accent">◆</span> ${links.tournament(g.league_id, g.league_id)} · ${esc(g.date || '')}</div>
                <div class="entity-name" style="font-size:26px;">${links.team(blue.team_id, blue.name || 'Blue')} <span style="color:var(--text-faint);">vs</span> ${links.team(red.team_id, red.name || 'Red')}</div>
                <div class="entity-sub">Best of ${g.best_of ?? '--'} · patch ${esc(g.patch || '--')} · ${g.duration_s != null ? Math.floor(g.duration_s / 60) + 'm ' + (g.duration_s % 60) + 's' : '--'}</div>
            </div>
            <div class="chip-row" id="series-tabs">${seriesTabs}</div>
        </div>
        <div class="panel">
            <div class="panel-h"><span class="label" title="Pre-game prediction from the served model, made before the match">Prediction</span>
                <span class="label" style="color:var(--text-muted)">${pred && pred.asof_date ? (livePred ? 'live model · as of ' : 'as of ') + esc(pred.asof_date) : 'no walk-forward row for this game'}</span></div>
            <div class="panel-b">
                <div style="display:flex;justify-content:space-between;gap:12px;margin-bottom:7px;">
                    <span class="mb-name">${esc(blue.name || 'Blue')} <span class="mb-pow">${fmt.int(blue.rating)}</span></span>
                    <span class="mb-name" style="text-align:right;"><span class="mb-pow">${fmt.int(red.rating)}</span> ${esc(red.name || 'Red')}</span>
                </div>
                ${(() => {
                    const bar = pred ? HERALD.splitBar(pred.p_blue, pred.p_blue != null ? fmt.pct(pred.p_blue) : '–', pred.p_blue != null ? fmt.pct(1 - pred.p_blue) : '–') : HERALD.splitBar(null);
                    return livePred ? `<div class="live-pred" title="Prediction by the current live model — the game postdates the walk-forward corpus">${bar}</div>` : bar;
                })()}
                ${data.post_p_blue != null ? `
                <div style="display:flex;justify-content:space-between;margin:9px 0 3px;">
                    <span class="label" style="color:var(--text-muted)" title="Pre-game prediction adjusted by draft quality">Post-Draft${postInSample ? ' · in-sample prior' : ''}</span></div>
                <div class="pd-bar${postInSample ? ' insample' : ''}"${postInSample ? ' title="post-draft rests on an in-sample pre-draft prior (no walk-forward prediction for this game)"' : ''}>${HERALD.splitBar(data.post_p_blue, fmt.pct(data.post_p_blue), fmt.pct(1 - data.post_p_blue))}</div>` : ''}
            </div>
        </div>
        ${draftPanel(data.draft, blue, red)}
        ${objectivesPanel(blue, red)}
        ${lineupComparison(blue, red)}`;

    document.getElementById('back').addEventListener('click', goList);
    rootEl.querySelectorAll('#series-tabs [data-g]').forEach((btn) => {
        btn.addEventListener('click', () => goGame(btn.dataset.g));
    });
    HERALD.setFooterAsOf(g.date);
}

function goList() {
    history.pushState(null, '', `/games`);
    listState.offset = 0;
    loadList();
}

async function boot() {
    HERALD.initFilterBar(() => {
        // Filter change always returns to the list scope.
        listState.offset = 0;
        const id = new URLSearchParams(window.location.search).get('id');
        if (id) { history.replaceState(null, '', '/games'); }
        loadList();
    });
    const id = new URLSearchParams(window.location.search).get('id');
    if (id) await loadGame(id);
    else await loadList();
}

window.addEventListener('popstate', () => {
    const id = new URLSearchParams(window.location.search).get('id');
    if (id) loadGame(id); else loadList();
});

boot();
