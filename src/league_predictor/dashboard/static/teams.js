/* Teams — power-ranked side rail of every active team, detail sheet per team:
   power trace over time, stat panels, and linked roster. URL carries ?id=
   for sharing. */

const {esc, fmt, links, fetchJSON} = HERALD;

const teamListEl = document.getElementById('team-list');
const teamDetailEl = document.getElementById('team-detail');
const teamSearchEl = document.getElementById('team-search');
const regionFilterEl = document.getElementById('team-region-filter');

let teams = [];
let selectedTeamId = null;

/* ── Side rail ── */
function renderTeamList() {
    const q = teamSearchEl.value.trim().toLowerCase();
    const region = regionFilterEl.value;
    const filtered = teams.filter((t) =>
        (t.display_name || '').toLowerCase().includes(q)
        && (!region || t.region === region));
    teamListEl.innerHTML = filtered.map((t) => `
        <button class="side-row ${t.team_id === selectedTeamId ? 'active' : ''}" data-team-id="${esc(t.team_id)}">
            <div class="side-row-name">
                <span><span class="mono" style="color:var(--text-muted);margin-right:8px;">${t.global_rank ?? '--'}</span>${esc(t.display_name)}</span>
                <span class="mono" style="color:var(--gold);">${fmt.int(t.power_rating)}</span>
            </div>
            <div class="side-row-meta">
                <span>${esc(t.region || '--')}</span>
            </div>
        </button>`).join('') || '<div class="palette-empty">No teams match.</div>';
    teamListEl.querySelectorAll('[data-team-id]').forEach((btn) => {
        btn.addEventListener('click', () => loadTeam(btn.dataset.teamId));
    });
}

function populateRegionFilter() {
    const prev = regionFilterEl.value;
    const regions = [...new Set(teams.map((t) => t.region).filter(Boolean))].sort();
    regionFilterEl.innerHTML = '<option value="">All regions</option>'
        + regions.map((r) => `<option value="${esc(r)}">${esc(r)}</option>`).join('');
    if (regions.includes(prev)) regionFilterEl.value = prev;
}

/* ── Power trace sparkline (rating-history endpoint) ── */
function sparklineSVG(history, w = 240, h = 48) {
    const pts = (history || []).filter((p) => p.rating != null);
    if (pts.length < 2) return '';
    const ratings = pts.map((p) => p.rating);
    const lo = Math.min(...ratings), hi = Math.max(...ratings);
    const span = hi - lo || 1;
    const x = (i) => (i / (pts.length - 1)) * (w - 4) + 2;
    const y = (r) => h - 6 - ((r - lo) / span) * (h - 12);
    const line = pts.map((p, i) => `${x(i).toFixed(1)},${y(p.rating).toFixed(1)}`).join(' ');
    const area = `2,${h - 2} ${line} ${w - 2},${h - 2}`;
    const last = pts[pts.length - 1];
    return `
        <svg class="spark-svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
            <title>Team power over time</title>
            <defs><linearGradient id="sparkfill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="rgba(200,170,110,0.28)"/>
                <stop offset="100%" stop-color="rgba(200,170,110,0)"/>
            </linearGradient></defs>
            <polygon points="${area}" fill="url(#sparkfill)"/>
            <polyline points="${line}" fill="none" stroke="var(--gold)" stroke-width="1.5"/>
            <circle cx="${x(pts.length - 1).toFixed(1)}" cy="${y(last.rating).toFixed(1)}" r="2.5" fill="var(--gold-bright)"/>
        </svg>`;
}

/* ── Detail sheet ── */
function metricPanel(label, value, sub, tone = '', span = '') {
    return `
        <div class="panel reveal ${span}">
            <div class="panel-h"><span class="label">${label}</span></div>
            <div class="panel-b"><div class="metric ${tone}">${value}</div><div class="metric-sub">${sub || ''}</div></div>
        </div>`;
}

async function loadTeam(teamId) {
    selectedTeamId = teamId;
    history.replaceState(null, '', `/teams?id=${encodeURIComponent(teamId)}`);
    renderTeamList();
    teamDetailEl.innerHTML = '<div class="loading">Opening team sheet</div>';

    const [overview, roster, ratingHistory] = await Promise.all([
        fetchJSON(`/api/teams/${encodeURIComponent(teamId)}/overview`),
        fetchJSON(`/api/teams/${encodeURIComponent(teamId)}/roster?${HERALD.filters.dateParams()}`),
        fetchJSON(`/api/teams/${encodeURIComponent(teamId)}/rating-history`),
    ]);
    if (!overview || !roster) {
        teamDetailEl.innerHTML = '<div class="error-msg">This team has no data in the selected range.</div>';
        return;
    }

    const p = overview.power, f = overview.form, mac = overview.macro, c = overview.composite;
    // Draft utilization: the post-draft stack's draft-logit weight for this team (beta + u).
    const du = overview.draft_utilization;
    const utilPanel = du
        ? metricPanel('Draft Utilization', fmt.num(du.value), `weight on the draft grade · ${du.n_oos} walk-forward games`, du.value > 1 ? 'tone-win' : '', 'w6')
        : '';
    const hist = ((ratingHistory && ratingHistory.history) || []).filter((h) => h.rating != null);
    const spark = sparklineSVG(hist);
    // Start → end power flanks the sparkline so the squiggle carries real values.
    const sparkCap = hist.length >= 2
        ? `${fmt.int(hist[0].rating)} → ${fmt.int(hist[hist.length - 1].rating)}`
        : '';

    // Role | Player | Grade | Impact | Games — fills the panel width.
    const rosterGrid = '56px minmax(130px,2fr) 74px 74px 74px';
    const rosterRows = (roster.players || []).map((pl) => `
        <div class="trow" style="grid-template-columns:${rosterGrid};">
            <div style="font-family:var(--font-mono);font-size:10px;color:var(--arcane);">${esc((pl.role || '--').toUpperCase())}</div>
            <div>${links.player(pl.player_id, pl.display_name || pl.player_id)}</div>
            <div class="num ${HERALD.valClass(pl.grade)}">${esc(pl.grade || '—')}</div>
            <div class="num">${pl.impact != null ? `${pl.impact > 0 ? '+' : ''}${pl.impact.toFixed(1)}` : '—'}</div>
            <div class="num" style="color:var(--text-secondary);">${pl.games_played ?? '--'}</div>
        </div>`).join('') || '<div class="panel-b" style="color:var(--text-muted);">No roster on the latest snapshot.</div>';

    teamDetailEl.innerHTML = `
        <section class="entity-hero reveal">
            <div>
                <div class="entity-name">${esc(overview.display_name)}</div>
                <div class="entity-sub">${esc(overview.region || 'Unknown region')}</div>
                <div class="entity-chips" style="margin-top:12px;">
                    <div class="tag" title="Overall global power ranking">Global Rank<strong>#${p.global_rank ?? '--'} · ${fmt.int(p.rating)}</strong></div>
                    <span class="label" style="color:var(--text-faint);">As of ${esc(overview.as_of_date || '--')}</span>
                </div>
            </div>
            ${spark ? `<div class="spark-wrap" title="Team power over time">${spark}<span class="spark-cap">${sparkCap}</span></div>` : ''}
        </section>
        <section class="stat-grid">
            ${metricPanel('Team Power', fmt.int(p.rating), 'latent strength + league anchor', 'tone-gold')}
            ${metricPanel('Recent Form', fmt.pct(f.win_rate_10), `win rate last 10 · last 5 ${fmt.pct(f.win_rate_5)} · streak ${f.win_streak || 0}W / ${f.loss_streak || 0}L`, 'tone-win')}
            ${metricPanel('First-Objective Control', fmt.pct(mac.first_dragon_rate, 0), `first dragon · first tower ${fmt.pct(mac.first_tower_rate, 0)} · first baron ${fmt.pct(mac.first_baron_rate, 0)}`, 'tone-arcane')}
            ${metricPanel('Lane Diff @15', c.avg_gold_diff_15 != null ? `${c.avg_gold_diff_15.toFixed(0)}g` : '--', `CS ${c.avg_cs_diff_15?.toFixed(1) ?? '--'} · XP ${c.avg_xp_diff_15?.toFixed(0) ?? '--'}`, 'tone-azure', 'w6')}
            ${metricPanel('Objectives Per Game', fmt.num(mac.dragon_rate, 1), `dragons · barons ${fmt.num(mac.baron_rate, 1)} · towers ${fmt.num(mac.tower_rate, 1)}`, '', 'w6')}
            ${utilPanel}
            <div class="panel w12 reveal">
                <div class="panel-h"><span class="label">Roster</span><span class="label" style="color:var(--text-faint)">as of ${esc(roster.as_of_date || '--')}</span></div>
                <div class="table">
                    <div class="trow thead" style="grid-template-columns:${rosterGrid};"><div>Role</div><div>Player</div><div title="Impact grade: percentile band inside the role's premier field">Grade</div><div title="Impact Rating — map-win% added over the role's average premier starter">Impact</div><div>Games</div></div>
                    ${rosterRows}
                </div>
            </div>
            <div class="panel w12 reveal">
                <div class="panel-h"><span class="label">Recent Matches</span></div>
                <div id="team-matches"><div class="loading">Loading matches</div></div>
            </div>
        </section>`;

    HERALD.setFooterAsOf(overview.as_of_date);
    loadTeamMatches(teamId);
}

/* Team match list — same row shape as /games, scoped to the team. */
async function loadTeamMatches(teamId) {
    const el = document.getElementById('team-matches');
    if (!el) return;
    const data = await fetchJSON(`/api/teams/${encodeURIComponent(teamId)}/matches?${HERALD.filters.params()}&limit=20&offset=0`);
    if (!el.isConnected) return;
    HERALD.matchTable(el, (data && data.games) || [], (gameId) => {
        window.location.href = `/games?id=${encodeURIComponent(gameId)}`;
    }, {perspective: teamId});
}

async function boot() {
    HERALD.initFilterBar(boot);
    const data = await fetchJSON(`/api/teams?${HERALD.filters.params()}`);
    if (!data) {
        teamListEl.innerHTML = '<div class="error-msg">Team list unavailable.</div>';
        return;
    }
    teams = data.teams || [];
    HERALD.setFooterAsOf(data.as_of_date);
    populateRegionFilter();
    renderTeamList();
    const urlId = new URLSearchParams(window.location.search).get('id');
    if (urlId && !teams.some((t) => t.team_id === urlId)) {
        // Deep link to a team outside the current filter (e.g. minor region): load it anyway
        await loadTeam(urlId);
        return;
    }
    const startId = urlId || (teams.length ? teams[0].team_id : null);
    if (startId) await loadTeam(startId);
}

teamSearchEl.addEventListener('input', renderTeamList);
regionFilterEl.addEventListener('change', renderTeamList);
boot();
