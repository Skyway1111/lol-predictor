/* Champions — list view is a sortable champion mega-table
   (picks, bans, presence, WR bar, KDA, DPM, CSPM, lane diffs) with role
   filter chips; clicking a champion opens its detail (/champions?id=Name):
   stat card (bans/picks by side, presence, WR, per-15 diffs), best-players
   table beside a draft-grade presence board, recent match list. Every
   model cell is dash-safe. */

const {esc, fmt, links, fetchJSON, champIcon, filters, valClass} = HERALD;

const rootEl = document.getElementById('champ-root');
const ROLES = ['', ...HERALD.ROLES];
let role = '';
let champTable = null;   /* sortableTable handle, so the search box filters in place */
let allChamps = [];      /* full fetched set the client-side search filters over */
let searchQ = '';        /* live champion-name query, preserved across role switches */

/* ── List view ── */
function goChampion(name) {
    history.pushState(null, '', `/champions?id=${encodeURIComponent(name)}`);
    loadChampion(name);
}

function roleChips() {
    return ROLES.map((r) => `<button class="chip ${r === role ? 'on' : ''}" data-role="${r}">${r ? r.toUpperCase() : 'All'}</button>`).join('');
}

/* Filter the already-fetched table by champion name, in place (keeps sort). */
function applyChampFilter() {
    if (!champTable) return;
    const q = searchQ.trim().toLowerCase();
    const rows = q ? allChamps.filter((c) => (c.champion || '').toLowerCase().includes(q)) : allChamps;
    champTable.update(rows);
}

async function loadTable() {
    rootEl.innerHTML = `
        <div class="page-hero">
            <div>
                <div class="label"><span class="accent">◆</span> Champions</div>
                <div class="entity-name" style="font-size:30px;">Champion browser</div>
            </div>
            <div class="chip-row">
                <input type="search" id="champ-search" placeholder="Search champions" title="Filter the table by champion name" style="width:200px;" value="${esc(searchQ)}">
                <div class="chip-row" id="role-chips">${roleChips()}</div>
            </div>
        </div>
        <div class="panel">
            <div class="panel-h" title="GD15 / CSD15 / XPD15 = gold / CS / XP difference at 15 minutes"><span class="label">Champion Table</span><span class="label" id="champ-n" style="color:var(--text-faint)"></span></div>
            <div id="champ-table"><div class="loading">Loading champions</div></div>
        </div>`;
    rootEl.querySelectorAll('#role-chips [data-role]').forEach((btn) => {
        btn.addEventListener('click', () => { role = btn.dataset.role; loadTable(); });
    });
    const searchEl = document.getElementById('champ-search');
    searchEl.addEventListener('input', () => { searchQ = searchEl.value; applyChampFilter(); });

    const data = await fetchJSON(`/api/champions?${filters.params()}${role ? `&role=${role}` : ''}`);
    const tableEl = document.getElementById('champ-table');
    if (!tableEl) return;
    if (!data) { tableEl.innerHTML = '<div class="error-msg">Champion feed unavailable.</div>'; return; }
    document.getElementById('champ-n').textContent = `${data.n_games ?? 0} games in scope`;

    allChamps = data.champions || [];
    champTable = HERALD.sortableTable(tableEl, [
        {key: 'champion', label: 'Champion', sortable: true, defaultDir: 'asc', render: (r) => `<div class="champ-cell">${champIcon(r.champion, 24)}<span class="champ-name">${esc(r.champion)}</span></div>`},
        {key: 'main_role', label: 'Role', align: 'center', defaultDir: 'asc', render: (r) => `<span class="pick-role">${esc((r.main_role || '--').toUpperCase())}</span>`},
        {key: 'presence', label: 'Presence', align: 'left', defaultDir: 'desc', render: (r) => HERALD.wrBar(r.presence)},
        {key: 'picks', label: 'Picks', align: 'center', defaultDir: 'desc', cls: 'num', render: (r) => fmt.int(r.picks)},
        {key: 'bans', label: 'Bans', align: 'center', defaultDir: 'desc', cls: 'num', render: (r) => fmt.int(r.bans)},
        {key: 'winrate', label: 'Win rate', align: 'left', defaultDir: 'desc', render: (r) => HERALD.wrBar(r.winrate)},
        {key: 'kda', label: 'KDA', align: 'center', defaultDir: 'desc', cls: 'num', render: (r) => fmt.num(r.kda, 2)},
        {key: 'dpm', label: 'DPM', align: 'center', defaultDir: 'desc', cls: 'num', render: (r) => fmt.int(r.dpm)},
        {key: 'cspm', label: 'CSPM', align: 'center', defaultDir: 'desc', cls: 'num', render: (r) => fmt.num(r.cspm, 1)},
        {key: 'gd15', label: 'GD15', align: 'center', defaultDir: 'desc', cls: 'num', render: (r) => fmt.signed(r.gd15)},
        {key: 'csd15', label: 'CSD15', align: 'center', defaultDir: 'desc', cls: 'num', render: (r) => fmt.signed(r.csd15)},
        {key: 'xpd15', label: 'XPD15', align: 'center', defaultDir: 'desc', cls: 'num', render: (r) => fmt.signed(r.xpd15)},
    ], allChamps, {
        grid: '1.5fr 54px 1.3fr 50px 50px 1.3fr 54px 58px 54px 62px 62px 62px',
        sortKey: 'presence', sortDir: 'desc',
        onRowClick: (r) => goChampion(r.champion),
    });
    applyChampFilter();
}

/* ── Detail view ── */
/* Games per draft grade of the side that picked the champion — the same
   S–D badge the match board wears, with a share bar and count per tier. */
function gradePresencePanel(gp) {
    const grades = ['S', 'A', 'B', 'C', 'D'];
    const max = Math.max(...grades.map((g) => (gp && gp[g]) || 0), 1);
    const rows = grades.map((g) => {
        const n = (gp && gp[g]) || 0;
        return `
        <div class="gp-row">
            <span class="mb-grade ${valClass(g)}">${g}</span>
            <div class="dg-meter ${valClass(g)}"><div class="dg-fill" style="width:${((n / max) * 100).toFixed(0)}%"></div></div>
            <span class="dg-pct ${valClass(g)}">${fmt.int(n)}</span>
        </div>`;
    }).join('');
    const ungraded = (gp && gp.ungraded) || 0;
    return `
        <div class="panel w6"><div class="panel-h"><span class="label">Draft grade presence</span><span class="label" style="color:var(--text-faint)">games at each team draft grade</span></div>
            <div class="panel-b">${rows}
            ${ungraded ? `<div class="gp-ungraded">${fmt.int(ungraded)} game${ungraded === 1 ? '' : 's'} without a draft score</div>` : ''}</div>
        </div>`;
}

async function loadChampion(name) {
    rootEl.innerHTML = '<div class="loading">Opening champion</div>';
    const data = await fetchJSON(`/api/champions/${encodeURIComponent(name)}?${filters.params()}`);
    if (!data || !data.stats) {
        rootEl.innerHTML = `<button class="back-link" id="back">Back to champions</button><div class="error-msg">No data for this champion in range.</div>`;
        document.getElementById('back').addEventListener('click', goList);
        return;
    }
    const s = data.stats;
    const mini = (label, val) => `<div class="mini"><div class="mini-label">${label}</div><div class="mini-value">${val}</div></div>`;
    const bestPlayers = (data.best_players || []).map((p, i) => `
        <div class="trow" style="grid-template-columns:28px 1.5fr 0.7fr 1.1fr 0.7fr;">
            <div class="rank">${i + 1}</div>
            <div>${links.player(p.player_id, p.name || p.player_id)}</div>
            <div class="num">${p.games ?? '–'}g</div>
            <div>${HERALD.wrBar(p.winrate)}</div>
            <div class="num" style="color:var(--gold-bright);font-weight:700;">${fmt.num(p.kda, 2)}</div>
        </div>`).join('');
    const recent = (data.recent_games || []).map((g) => `
        <div class="trow"${g.game_id ? ` data-game="${esc(g.game_id)}" title="Open the match sheet"` : ''} style="grid-template-columns:90px 100px 1.2fr 0.9fr 1.2fr 0.6fr;${g.game_id ? 'cursor:pointer;' : ''}">
            <div class="num" style="color:var(--text-secondary);">${esc(g.date || '--')}</div>
            <div style="min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${links.tournament(g.league_id, g.league_id)}</div>
            <div>${links.player(g.player_id, g.player_name || g.player_id)}</div>
            <div class="num">${g.kills ?? '–'}/${g.deaths ?? '–'}/${g.assists ?? '–'}</div>
            <div class="champ-cell">${g.opponent_champion ? champIcon(g.opponent_champion, 18) : ''}<span class="champ-name">${esc(g.opponent_champion || '--')}</span></div>
            <div><span class="wl-pip ${g.result === 'win' ? 'w' : g.result === 'loss' ? 'l' : ''}">${g.result === 'win' ? 'W' : g.result === 'loss' ? 'L' : '–'}</span></div>
        </div>`).join('');

    rootEl.innerHTML = `
        <button class="back-link" id="back">Back to champions</button>
        <section class="entity-hero reveal">
            <div style="display:flex;gap:16px;align-items:center;">
                ${champIcon(s.champion, 56)}
                <div>
                    <div class="entity-name">${esc(s.champion)}</div>
                    <div class="entity-sub">${fmt.int(s.picks)} picks · ${fmt.int(s.bans)} bans · presence ${fmt.pct(s.presence, 0)}</div>
                </div>
            </div>
            <div style="min-width:220px;">${HERALD.wrBar(s.winrate, s.winrate != null ? `${(s.winrate * 100).toFixed(1)}%` : '--')}</div>
        </section>
        <section class="stat-grid">
            <div class="panel w12"><div class="panel-h"><span class="label">Profile</span></div><div class="panel-b"><div class="mini-grid">
                ${mini('Picks (B/R)', `${fmt.int(s.picks_blue)}/${fmt.int(s.picks_red)}`)}
                ${mini('Bans (B/R)', `${fmt.int(s.bans_blue)}/${fmt.int(s.bans_red)}`)}
                ${mini('Wins/Losses', `${fmt.int(s.wins)}/${fmt.int(s.losses)}`)}
                ${mini('KDA', fmt.num(s.kda, 2))}
                ${mini('Avg K/D/A', `${fmt.num(s.avg_kills, 1)}/${fmt.num(s.avg_deaths, 1)}/${fmt.num(s.avg_assists, 1)}`)}
                ${mini('DPM', fmt.int(s.dpm))}
                ${mini('CSPM', fmt.num(s.cspm, 1))}
                ${mini('GD@15', fmt.signed(s.gd15))}
                ${mini('CSD@15', fmt.signed(s.csd15))}
                ${mini('XPD@15', fmt.signed(s.xpd15))}
            </div></div></div>
            <div class="panel w6"><div class="panel-h"><span class="label">Best players</span><span class="label" style="color:var(--text-faint)">by win rate</span></div>
                <div class="table"><div class="trow thead" style="grid-template-columns:28px 1.5fr 0.7fr 1.1fr 0.7fr;"><div>#</div><div>Player</div><div>Games</div><div>Win rate</div><div>KDA</div></div>
                ${bestPlayers || '<div class="panel-b" style="color:var(--text-muted);">No qualifying players.</div>'}</div>
            </div>
            ${gradePresencePanel(data.grade_presence)}
            <div class="panel w12"><div class="panel-h"><span class="label">Recent games</span><span class="label" style="color:var(--text-faint)">click a row to open the match sheet</span></div>
                <div class="table"><div class="trow thead" style="grid-template-columns:90px 100px 1.2fr 0.9fr 1.2fr 0.6fr;"><div>Date</div><div>League</div><div>Player</div><div>K/D/A</div><div>Opp</div><div>Res</div></div>
                ${recent || '<div class="panel-b" style="color:var(--text-muted);">No recent games.</div>'}</div>
            </div>
        </section>`;
    document.getElementById('back').addEventListener('click', goList);
    rootEl.querySelectorAll('.trow[data-game]').forEach((row) => {
        row.addEventListener('click', () => { window.location.href = `/games?id=${encodeURIComponent(row.dataset.game)}`; });
    });
}

function goList() {
    history.pushState(null, '', `/champions`);
    loadTable();
}

async function boot() {
    HERALD.initFilterBar(() => {
        const id = new URLSearchParams(window.location.search).get('id');
        if (id) loadChampion(id); else loadTable();
    });
    const id = new URLSearchParams(window.location.search).get('id');
    if (id) await loadChampion(id);
    else await loadTable();
}

window.addEventListener('popstate', () => {
    const id = new URLSearchParams(window.location.search).get('id');
    if (id) loadChampion(id); else loadTable();
});

boot();
