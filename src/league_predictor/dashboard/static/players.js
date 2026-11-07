/* Players — side rail with search/region/role/sort, detail sheet:
   Impact header, role pillars, the 5-axis draft radar, champion pool.
   Team affiliation links back to the team page; URL carries ?id=. */

const {esc, links, fetchJSON, valClass, champIcon} = HERALD;

const playerListEl = document.getElementById('player-list');
const playerDetailEl = document.getElementById('player-detail');
const playerSearchEl = document.getElementById('player-search');

let players = [];
let selectedPlayerId = null;

/* Pillar cosmetics: label / accent tone / one-line meaning per pillar name.
   Top/jng/mid/bot report laning/combat/vision; support has its own four. */
const PILLARS = {
    laning:     {label: 'Laning',     tone: 'tone-azure',  blurb: 'gold / CS / XP lane profile'},
    combat:     {label: 'Combat',     tone: 'tone-gold',   blurb: 'damage, KDA, fight participation'},
    vision:     {label: 'Vision',     tone: 'tone-win',    blurb: 'vision score profile'},
    lane:       {label: 'Lane',       tone: 'tone-azure',  blurb: 'gold / XP lane profile'},
    playmaking: {label: 'Playmaking', tone: 'tone-gold',   blurb: 'KDA, damage, kill part., first blood'},
    utility:    {label: 'Utility',    tone: 'tone-plum',   blurb: 'gold generation efficiency'},
};

function renderPillars(pillars) {
    if (!Array.isArray(pillars) || !pillars.length) return '';
    const cards = pillars.map((p) => {
        const meta = PILLARS[p.name] || {label: p.name, tone: '', blurb: 'pillar score'};
        const badge = p.data_quality && p.data_quality !== 'full'
            ? `<span class="pillar-badge ${esc(p.data_quality)}">${esc(p.data_quality)}</span>` : '';
        return `
            <div class="panel reveal">
                <div class="panel-h"><span class="label">${esc(meta.label)}${badge}</span></div>
                <div class="panel-b">
                    <div class="metric ${meta.tone}">${p.rating != null ? p.rating.toFixed(2) : 'N/A'}</div>
                    <div class="metric-sub">${esc(meta.blurb)}</div>
                </div>
            </div>`;
    }).join('');
    return `<div class="pillars-grid ${pillars.length === 4 ? 'cols-4' : ''}">${cards}</div>`;
}

/* Spoke order and labels are served, so the axis list has one home:
   analytics/player_draft.py RADAR_AXES. */
function servedAxes(draftProfile) {
    return Object.entries(draftProfile.axes || {})
        .map(([key, ax]) => ({key, label: ax.label || key}));
}

/* 5-spoke draft radar. NULL axes collapse to center, labels dim with "n/a";
   sample size lives on each label's hover tooltip. */
function renderRadar(draftProfile) {
    // Width leaves ~120px per side so the long axis labels don't clip.
    const W = 460, H = 290, CX = W / 2, CY = H / 2, R = 96;
    const axes = draftProfile.axes || {};
    const spokeAxes = servedAxes(draftProfile);
    const angle = (i) => (i * 2 * Math.PI) / spokeAxes.length - Math.PI / 2;
    const polar = (i, r) => [CX + Math.cos(angle(i)) * r, CY + Math.sin(angle(i)) * r];

    const rings = [0.25, 0.5, 0.75, 1.0].map((frac) => {
        const pts = spokeAxes.map((_, i) => polar(i, R * frac).map((n) => n.toFixed(1)).join(',')).join(' ');
        return `<polygon points="${pts}" class="radar-ring" />`;
    }).join('');

    const spokes = spokeAxes.map((_, i) => {
        const [x, y] = polar(i, R);
        return `<line x1="${CX}" y1="${CY}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" class="radar-spoke" />`;
    }).join('');

    const valuePoints = spokeAxes.map((ax, i) => {
        const v = axes[ax.key]?.value;
        const frac = (v == null) ? 0 : Math.max(0, Math.min(100, v)) / 100;
        return polar(i, R * frac);
    });
    const shape = valuePoints.map((p) => p.map((n) => n.toFixed(1)).join(',')).join(' ');
    const dots = valuePoints.map(([x, y], i) =>
        axes[spokeAxes[i].key]?.value != null
            ? `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.5" class="radar-dot" />` : ''
    ).join('');

    const labels = spokeAxes.map((ax, i) => {
        const [x, y] = polar(i, R + 24);
        const value = axes[ax.key]?.value;
        const sample = axes[ax.key]?.sample ?? 0;
        const dim = (value == null) ? 'opacity:0.42' : '';
        // Sample size moves to the hover tooltip; null axes keep a subtle "n/a" marker.
        const tip = (value == null) ? `${ax.label}: no data in range` : `${ax.label}: based on ${sample} games`;
        const na = (value == null) ? '<tspan class="radar-sub" dx="4">n/a</tspan>' : '';
        const anchor = (x < CX - 10) ? 'end' : (x > CX + 10 ? 'start' : 'middle');
        return `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" text-anchor="${anchor}" style="${dim}" class="radar-label"><title>${esc(tip)}</title>${ax.label}${na}</text>`;
    }).join('');

    return `<svg viewBox="0 0 ${W} ${H}" class="radar-svg">
        ${rings}${spokes}
        <polygon points="${shape}" class="radar-shape" />
        ${dots}${labels}
    </svg>`;
}

function draftProfileHTML(draftProfile) {
    if (!draftProfile) return '';
    const overall = draftProfile.overall_quality;
    const axes = draftProfile.axes || {};
    const weights = draftProfile.weights || {};
    const axisList = servedAxes(draftProfile);
    // The blend is stated from the served weights, never restated as literals here.
    const blend = axisList
        .filter((ax) => weights[ax.key])
        .map((ax) => `${ax.label.toLowerCase()} ${weights[ax.key].toFixed(2)}`)
        .join(' / ');
    const cells = axisList.map((ax) => {
        const v = axes[ax.key]?.value;
        const sample = axes[ax.key]?.sample ?? 0;
        // Sample size moves off the card face into its hover tooltip.
        const tip = v == null ? `${ax.label}: no data in range` : `${ax.label}: based on ${sample} games`;
        return `<div class="mini" title="${esc(tip)}" style="${v == null ? 'opacity:0.42' : ''}">
            <div class="mini-label">${ax.label}</div>
            <div class="mini-value">${v == null ? 'n/a' : v.toFixed(0)}</div>
        </div>`;
    }).join('');
    return `
        <div class="panel reveal">
            <div class="panel-h">
                <span class="label">Draft Profile</span>
                <span class="label" style="color:var(--text-faint)">${esc(String(draftProfile.year ?? ''))}</span>
            </div>
            <div class="panel-b radar-layout">
                <div>${renderRadar(draftProfile)}</div>
                <div>
                    <div class="tag" title="${esc(`Overall Quality blends the draft axes (${blend}); weights renormalize over non-NULL axes`)}">Overall Quality<strong>${overall == null ? '--' : overall.toFixed(0)}</strong></div>
                    <div class="mini-grid" style="margin-top:10px;">${cells}</div>
                </div>
            </div>
        </div>`;
}

function championPoolHTML(details) {
    const pool = (details || {}).champion_pool || {};
    const champions = pool.champions || [];
    const chips = champions.map((c) =>
        `<span class="champ-chip">${champIcon(c.champion_id, 20)}${links.champion(c.champion_id)} <span class="n">×${c.games}</span></span>`).join('');
    return `
        <div class="panel reveal">
            <div class="panel-h">
                <span class="label">Champion Pool</span>
                <span class="label" style="color:var(--text-faint)" title="Distinct champions played this season">Diversity ${pool.champion_diversity ?? '--'}</span>
            </div>
            <div class="panel-b">
                ${chips ? `<div class="champ-chips">${chips}</div>` : '<span style="color:var(--text-muted);font-family:var(--font-mono);font-size:11px;">No champion data in range.</span>'}
            </div>
        </div>`;
}

/* ── Side rail ── */
const ROLE_LABEL = {top: 'TOP', jng: 'JNG', mid: 'MID', bot: 'BOT', sup: 'SUP'};

/* Impact reads as a signed win-percentage lift, so the sign is part of the number;
   a sub-threshold player has no Impact at all and shows an em dash, never a 0. */
function fmtImpact(v, digits = 1) {
    if (v == null) return '—';
    return `${v > 0 ? '+' : ''}${v.toFixed(digits)}`;
}

/* The one header number: Impact pp, its grade sigil, a percentile chip, and an
   info tooltip carrying the stats/results split and what the number is measured against. */
function impactHeaderHTML(o) {
    // The bar comes from the server's promoted recipe — never restate its numbers here.
    const bar = o.impact_threshold;
    if (o.impact == null) {
        const needs = bar
            ? `needs ${bar.min_games} games in ${bar.window_months}mo`
            : 'below the qualification bar';
        return `
            <div class="grade-sigil none" title="No Impact rating — ${esc(needs)}">·</div>
            <div class="impact-head">
                <div class="impact-value">—</div>
                <div class="impact-sub">Impact · ${esc(needs)}</div>
            </div>`;
    }
    const tip = [
        "Impact — map-win% this player adds over the role's average premier starter,",
        'on a team of five premier averages.',
        `Stats ${fmtImpact(o.stats_pp, 1)} pp (box-score prior) + Results ${fmtImpact(o.results_pp, 1)} pp`,
        '(what winning and losing with real teammates moved it).',
    ].join(' ');
    return `
        <div class="grade-sigil ${valClass(o.grade) ? o.grade : 'none'}" title="Grade ${esc(o.grade || '--')}">${esc(o.grade || '·')}</div>
        <div class="impact-head">
            <div class="impact-value ${valClass(o.grade)}">${fmtImpact(o.impact)}<span class="impact-unit">pp</span>
                <span class="impact-info" title="${esc(tip)}">ⓘ</span>
            </div>
            <div class="impact-sub">
                Impact
                ${o.percentile != null ? `<span class="impact-chip">${o.percentile.toFixed(0)}th pct</span>` : ''}
                <span class="impact-split">stats ${fmtImpact(o.stats_pp)} · results ${fmtImpact(o.results_pp)}</span>
            </div>
        </div>`;
}

function visiblePlayers() {
    const q = playerSearchEl.value.trim().toLowerCase();
    const region = document.getElementById('region-filter').value;
    const role = document.getElementById('role-filter').value;
    const sortBy = document.getElementById('sort-by').value;
    const out = players.filter((p) =>
        (p.display_name || '').toLowerCase().includes(q)
        && (!region || p.region === region)
        && (!role || p.role === role));
    const cmp = {
        name: (a, b) => (a.display_name || '').localeCompare(b.display_name || ''),
        games: (a, b) => (b.games_played || 0) - (a.games_played || 0),
        impact: (a, b) => (b.impact ?? -1e9) - (a.impact ?? -1e9),
    }[sortBy] || (() => 0);
    return out.sort(cmp);
}

function renderPlayerList() {
    const filtered = visiblePlayers();
    playerListEl.innerHTML = filtered.map((p) => `
        <button class="side-row ${p.player_id === selectedPlayerId ? 'active' : ''}" data-player-id="${esc(p.player_id)}">
            <div class="side-row-name">
                <span>${esc(p.display_name)}</span>
                <span class="mono ${valClass(p.grade)}" style="font-weight:700;">${fmtImpact(p.impact)}</span>
            </div>
            <div class="side-row-meta">
                <span>${ROLE_LABEL[p.role] || '--'}</span>
                <span>${esc(p.team_display_name || '--')}</span>
                ${p.region ? `<span>${esc(p.region)}</span>` : ''}
                <span>${p.games_played || 0} g</span>
            </div>
        </button>`).join('') || '<div class="palette-empty">No players match these filters.</div>';
    playerListEl.querySelectorAll('[data-player-id]').forEach((btn) => {
        btn.addEventListener('click', () => loadPlayer(btn.dataset.playerId));
    });
}

function populateRegions() {
    const sel = document.getElementById('region-filter');
    const prev = sel.value;
    const regions = [...new Set(players.map((p) => p.region).filter(Boolean))].sort();
    sel.innerHTML = '<option value="">All regions</option>'
        + regions.map((r) => `<option value="${esc(r)}">${esc(r)}</option>`).join('');
    if (regions.includes(prev)) sel.value = prev;
}

/* ── Detail sheet ── */
async function loadPlayer(playerId) {
    selectedPlayerId = playerId;
    history.replaceState(null, '', `/players?id=${encodeURIComponent(playerId)}`);
    renderPlayerList();
    playerDetailEl.innerHTML = '<div class="loading">Opening player profile</div>';

    const overview = await fetchJSON(`/api/players/${encodeURIComponent(playerId)}/overview?${HERALD.filters.dateParams()}`);
    if (!overview) {
        playerDetailEl.innerHTML = '<div class="error-msg">This player has no profile in the selected range.</div>';
        return;
    }

    const role = overview.role || 'unknown';
    const pillars = Array.isArray(overview.pillars) ? overview.pillars : [];

    playerDetailEl.innerHTML = `
        <section class="entity-hero reveal">
            <div style="display:flex;gap:18px;align-items:flex-start;">
                ${impactHeaderHTML(overview)}
                <div>
                    <div class="entity-name">${esc(overview.display_name)}</div>
                    <div class="entity-sub">
                        ${esc(role.charAt(0).toUpperCase() + role.slice(1))} ·
                        ${links.team(overview.team_id, overview.team_display_name || overview.team_id || 'No team')}
                    </div>
                    <div class="entity-chips" style="margin-top:12px;">
                        <div class="tag">${(overview.games_played || 0).toLocaleString()} games</div>
                        ${overview.impact_asof ? `<div class="tag">as of ${esc(overview.impact_asof)}</div>` : ''}
                    </div>
                </div>
            </div>
        </section>
        ${renderPillars(pillars)}
        ${draftProfileHTML(overview.draft_profile)}
        ${championPoolHTML(overview.details)}`;
}

async function boot() {
    HERALD.initFilterBar(boot);
    const data = await fetchJSON(`/api/players?${HERALD.filters.params()}`);
    if (!data) {
        playerListEl.innerHTML = '<div class="error-msg">Player data unavailable.</div>';
        return;
    }
    players = data.players || [];
    HERALD.setFooterAsOf(data.as_of_date);
    populateRegions();
    renderPlayerList();
    const urlId = new URLSearchParams(window.location.search).get('id');
    if (urlId && !players.some((p) => p.player_id === urlId)) {
        // Deep link outside the current filter — load the sheet anyway
        await loadPlayer(urlId);
        return;
    }
    const startId = urlId || (players.length ? players[0].player_id : null);
    if (startId) await loadPlayer(startId);
}

playerSearchEl.addEventListener('input', renderPlayerList);
['region-filter', 'role-filter', 'sort-by'].forEach((id) => {
    document.getElementById(id).addEventListener('change', renderPlayerList);
});
boot();
