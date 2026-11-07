/* Tournaments — side rail of leagues with games in range;
   detail sheet has four sub-tabs: Overview (stat panels + blue/red split
   bars + top KDA), Ranking (team table with WR bars), Picks & Bans
   (champion presence), Match list (series with scores). The league list
   reuses /api/leagues; the detail reads /api/tournaments/{id}. URL carries
   ?id= for sharing. */

const {esc, fmt, links, fetchJSON, champIcon, filters} = HERALD;

const railEl = document.getElementById('lg-rail');
const detailEl = document.getElementById('tourn-detail');
const searchEl = document.getElementById('lg-search');

let leagues = [];
let selectedId = null;
let activeTab = 'overview';

/* ── Side rail — majors grouped above regional leagues ── */
function renderRail() {
    const q = searchEl.value.trim().toLowerCase();
    const shown = leagues.filter((l) => (l.name || l.league_id || '').toLowerCase().includes(q));
    // No per-row "major" badge: the section header ("Major Leagues") already
    // conveys tier, and every row under it would repeat it.
    const rowHTML = (l) => `
        <button class="side-row ${l.league_id === selectedId ? 'active' : ''}" data-lg="${esc(l.league_id)}">
            <div class="side-row-name"><span>${esc(l.name || l.league_id)}</span></div>
            <div class="side-row-meta">${l.n_games ?? 0} games · ${l.n_series ?? 0} series</div>
        </button>`;
    const section = (label, arr) => arr.length
        ? `<div class="label side-group">${label}</div>${arr.map(rowHTML).join('')}`
        : '';
    railEl.innerHTML = (
        section('Major Leagues', shown.filter((l) => l.is_major))
        + section('Regional Leagues', shown.filter((l) => !l.is_major))
    ) || '<div class="palette-empty">No leagues match.</div>';
    railEl.querySelectorAll('[data-lg]').forEach((btn) => btn.addEventListener('click', () => loadLeague(btn.dataset.lg)));
}

/* ── Sub-tab bodies ── */
function overviewTab(t) {
    const o = t.overview || {};
    const br = o.blue_red || {};
    const gameCard = (g, label) => g
        ? `<div class="mini"><div class="mini-label">${label}</div><div class="mini-value" style="font-size:13px;">${links.game(g.game_id, g.label || '--')}</div><div class="mini-sub">${g.duration_s != null ? Math.floor(g.duration_s / 60) + 'm' : (g.kills != null ? g.kills + ' kills' : '')}</div></div>`
        : `<div class="mini"><div class="mini-label">${label}</div><div class="mini-value">–</div></div>`;
    // A null share with games in range means this stat wasn't recorded for the
    // league (e.g. LPL first blood / first baron) — say so instead of dashes.
    const splitRow = (label, share) => `
        <div class="br-row">
            <span class="metric-sub" style="margin:0;">${label}</span>
            ${share == null && (o.n_games || 0) > 0
                ? '<span class="note-faint">No data</span>'
                : HERALD.splitBar(share)}
        </div>`;
    const kdaGrid = '28px minmax(160px,320px) 100px minmax(150px,1fr)';
    const topKda = (o.top_kda || []).map((p, i) => `
        <div class="trow" style="grid-template-columns:${kdaGrid};">
            <div class="rank">${i + 1}</div>
            <div>${links.player(p.player_id, p.name || p.player_id)}</div>
            <div class="num" style="color:var(--gold-bright);font-weight:700;">${fmt.num(p.kda, 2)}</div>
            <div class="num" style="color:var(--text-muted);">${p.kills ?? '–'}/${p.deaths ?? '–'}/${p.assists ?? '–'}</div>
        </div>`).join('');
    return `
        <section class="stat-grid">
            <div class="panel"><div class="panel-h"><span class="label">Games</span></div><div class="panel-b"><div class="metric tone-gold">${fmt.int(o.n_games)}</div><div class="metric-sub">${fmt.int(o.n_series)} series</div></div></div>
            <div class="panel"><div class="panel-h"><span class="label">Avg duration</span></div><div class="panel-b"><div class="metric tone-arcane">${o.avg_duration_s != null ? Math.floor(o.avg_duration_s / 60) + ':' + String(Math.round(o.avg_duration_s % 60)).padStart(2, '0') : '--'}</div><div class="metric-sub">mm:ss</div></div></div>
            <div class="panel"><div class="panel-h"><span class="label">Avg kills</span></div><div class="panel-b"><div class="metric tone-win">${fmt.num(o.avg_kills, 1)}</div><div class="metric-sub">per game</div></div></div>
            <div class="panel w12"><div class="panel-h"><span class="label">Notable games</span></div><div class="panel-b"><div class="mini-grid" style="grid-template-columns:repeat(3,minmax(0,1fr));">
                ${gameCard(o.shortest_game, 'Shortest')}${gameCard(o.longest_game, 'Longest')}${gameCard(o.most_kills_game, 'Bloodiest')}
            </div></div></div>
            <div class="panel w6"><div class="panel-h"><span class="label">Blue vs red side</span><span class="label" style="color:var(--text-faint)">blue share</span></div><div class="panel-b">
                ${splitRow('Win rate', br.win_rate)}${splitRow('First blood', br.first_blood)}${splitRow('First tower', br.first_tower)}${splitRow('First dragon', br.first_dragon)}${splitRow('First baron', br.first_baron)}
            </div></div>
            <div class="panel w6"><div class="panel-h"><span class="label">Top KDA</span></div>
                <div class="table"><div class="trow thead" style="grid-template-columns:${kdaGrid};"><div>#</div><div>Player</div><div>KDA</div><div>K/D/A</div></div>
                ${topKda || '<div class="panel-b" style="color:var(--text-muted);">No player data.</div>'}</div>
            </div>
        </section>`;
}

function rankingTab(t) {
    const el = document.createElement('div');
    el.className = 'panel w12';
    el.innerHTML = '<div class="panel-h"><span class="label">Team ranking</span></div><div id="rank-body"></div>';
    HERALD.sortableTable(el.querySelector('#rank-body'), [
        {key: 'name', label: 'Team', sortable: true, defaultDir: 'asc', render: (r) => links.team(r.team_id, r.name || '--')},
        {key: 'winrate', label: 'Win rate', align: 'left', defaultDir: 'desc', render: (r) => HERALD.wrBar(r.winrate)},
        {key: 'wins', label: 'W', align: 'center', defaultDir: 'desc', cls: 'num', render: (r) => fmt.int(r.wins)},
        {key: 'losses', label: 'L', align: 'center', defaultDir: 'desc', cls: 'num', render: (r) => fmt.int(r.losses)},
        {key: 'avg_duration_s', label: 'Avg dur', align: 'center', defaultDir: 'asc', cls: 'num', render: (r) => r.avg_duration_s != null ? Math.floor(r.avg_duration_s / 60) + ':' + String(Math.round(r.avg_duration_s % 60)).padStart(2, '0') : '–'},
        {key: 'avg_gdm', label: 'GD/min', align: 'center', defaultDir: 'desc', cls: 'num', render: (r) => fmt.signed(r.avg_gdm)},
    ], t.ranking || [], {grid: 'minmax(180px,1.6fr) minmax(180px,380px) 52px 52px 84px 84px', sortKey: 'winrate', sortDir: 'desc'});
    return el;
}

function picksBansTab(t) {
    const el = document.createElement('div');
    el.className = 'panel w12';
    el.innerHTML = '<div class="panel-h"><span class="label">Picks &amp; bans</span></div><div id="pb-body"></div>';
    HERALD.sortableTable(el.querySelector('#pb-body'), [
        {key: 'champion', label: 'Champion', sortable: true, defaultDir: 'asc', render: (r) => `<div class="champ-cell">${champIcon(r.champion, 22)}<span class="champ-name">${links.champion(r.champion)}</span></div>`},
        {key: 'presence', label: 'Presence', align: 'left', defaultDir: 'desc', render: (r) => HERALD.wrBar(r.presence)},
        {key: 'picks', label: 'Picks', align: 'center', defaultDir: 'desc', cls: 'num', render: (r) => fmt.int(r.picks)},
        {key: 'bans', label: 'Bans', align: 'center', defaultDir: 'desc', cls: 'num', render: (r) => fmt.int(r.bans)},
        {key: 'winrate', label: 'Win rate', align: 'center', defaultDir: 'desc', cls: 'num', render: (r) => fmt.pct(r.winrate, 0)},
    ], t.picks_bans || [], {grid: 'minmax(180px,1.8fr) minmax(180px,380px) 64px 64px 84px', sortKey: 'presence', sortDir: 'desc'});
    return el;
}

function matchesTab(t) {
    const rows = (t.matches || []).map((m) => {
        const a = m.team_a || {}, b = m.team_b || {};
        const aWon = m.winner_team_id && m.winner_team_id === a.team_id;
        const bWon = m.winner_team_id && m.winner_team_id === b.team_id;
        const clickable = m.game1_id ? ` data-game="${esc(m.game1_id)}" title="Open the match sheet"` : '';
        return `
            <div class="trow"${clickable} style="grid-template-columns:92px 1.4fr 90px 1.4fr 80px;${m.game1_id ? 'cursor:pointer;' : ''}">
                <div class="num" style="color:var(--text-secondary);">${esc(m.date || '--')}</div>
                <div style="text-align:right;${aWon ? 'color:var(--win);' : ''}">${links.team(a.team_id, a.name || '--')}</div>
                <div class="score-cell">${m.score_a ?? '–'} : ${m.score_b ?? '–'}</div>
                <div style="${bWon ? 'color:var(--win);' : ''}">${links.team(b.team_id, b.name || '--')}</div>
                <div class="num" style="color:var(--text-muted);text-align:center;">Bo${m.best_of ?? '?'}</div>
            </div>`;
    }).join('');
    const el = document.createElement('div');
    el.className = 'panel w12';
    el.innerHTML = `<div class="panel-h"><span class="label">Match list</span><span class="label" style="color:var(--text-faint)">click a series to open its game sheet</span></div>
        <div class="table"><div class="trow thead" style="grid-template-columns:92px 1.4fr 90px 1.4fr 80px;"><div>Date</div><div style="text-align:right;">Team A</div><div style="text-align:center;">Score</div><div>Team B</div><div style="text-align:center;">Format</div></div>
        ${rows || '<div class="panel-b" style="color:var(--text-muted);">No series in range.</div>'}</div>`;
    el.querySelectorAll('.trow[data-game]').forEach((row) => {
        row.addEventListener('click', () => { window.location.href = `/games?id=${encodeURIComponent(row.dataset.game)}`; });
    });
    return el;
}

/* ── Detail assembly ── */
const TABS = [
    {key: 'overview', label: 'Overview'},
    {key: 'ranking', label: 'Ranking'},
    {key: 'picks_bans', label: 'Picks & Bans'},
    {key: 'matches', label: 'Match list'},
];

function renderDetail(t) {
    const body = document.getElementById('tourn-body');
    if (!body) return;
    if (activeTab === 'overview') body.innerHTML = overviewTab(t);
    else {
        body.innerHTML = '';
        body.appendChild(activeTab === 'ranking' ? rankingTab(t)
            : activeTab === 'matches' ? matchesTab(t) : picksBansTab(t));
    }
}

async function loadLeague(leagueId) {
    selectedId = leagueId;
    activeTab = 'overview';
    history.replaceState(null, '', `/tournaments?id=${encodeURIComponent(leagueId)}`);
    renderRail();
    detailEl.innerHTML = '<div class="loading">Opening league</div>';

    const t = await fetchJSON(`/api/tournaments/${encodeURIComponent(leagueId)}?${filters.dateParams()}`);
    if (!t || !t.league) {
        detailEl.innerHTML = '<div class="error-msg">This league has no games in range.</div>';
        return;
    }
    const lg = t.league;
    const o = t.overview || {};
    detailEl.innerHTML = `
        <section class="entity-hero reveal">
            <div>
                <div class="entity-name">${esc(lg.name || lg.league_id)}</div>
                <div class="entity-sub">${fmt.int(o.n_games)} games · ${fmt.int(o.n_series)} series in range</div>
            </div>
        </section>
        <div class="chip-row" id="tourn-tabs">${TABS.map((tb) => `<button class="chip ${tb.key === activeTab ? 'on' : ''}" data-tab="${tb.key}">${tb.label}</button>`).join('')}</div>
        <div id="tourn-body"></div>`;
    detailEl.querySelectorAll('#tourn-tabs [data-tab]').forEach((btn) => {
        btn.addEventListener('click', () => {
            activeTab = btn.dataset.tab;
            detailEl.querySelectorAll('#tourn-tabs .chip').forEach((c) => c.classList.toggle('on', c.dataset.tab === activeTab));
            renderDetail(t);
        });
    });
    renderDetail(t);
}

async function boot() {
    HERALD.initFilterBar(boot);
    const data = await fetchJSON(`/api/leagues?${filters.params()}`);
    if (!data) { railEl.innerHTML = '<div class="error-msg">League index unavailable.</div>'; return; }
    leagues = data.leagues || [];
    renderRail();
    const urlId = new URLSearchParams(window.location.search).get('id');
    const startId = urlId || (leagues.length ? leagues[0].league_id : null);
    if (startId) await loadLeague(startId);
    else detailEl.innerHTML = '<div class="no-matches">No leagues in this range</div>';
}

searchEl.addEventListener('input', renderRail);
boot();
