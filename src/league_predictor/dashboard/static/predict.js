/* Oracle (Predict) — two searchable team pickers + an optional as-of date →
   POST /api/predict/matchup → P(team A) at Bo1/Bo3/Bo5. The teams are
   evaluated as snapshotted on the as-of date (default: today). The server
   warms the served model's feature indexes on first use (a few minutes);
   while it answers status:'warming' the page retries automatically. */

const {esc, fmt, fetchJSON, postJSON} = HERALD;

const rootEl = document.getElementById('predict-root');
const RETRY_MS = 5000;
let teams = [];
let teamIndex = HERALD.teamNameIndex([]);
let retryTimer = null;

function teamDatalist() {
    return `<datalist id="team-options">${teams.map((t) => `<option value="${esc(t.display_name)}"></option>`).join('')}</datalist>`;
}

function probRow(label, pA, aName, bName) {
    return `
        <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:14px;align-items:center;text-align:center;">
            <div class="metric tone-gold" title="${esc(aName)}">${fmt.pct(pA)}</div>
            <div class="label" style="min-width:40px;">${esc(label)}</div>
            <div class="metric tone-arcane" title="${esc(bName)}">${fmt.pct(1 - pA)}</div>
        </div>`;
}

async function rate() {
    if (retryTimer) { clearTimeout(retryTimer); retryTimer = null; }
    const aEl = document.getElementById('team-a');
    const bEl = document.getElementById('team-b');
    const dateEl = document.getElementById('as-of');
    const resEl = document.getElementById('predict-result');
    const aId = teamIndex.resolve(aEl.value), bId = teamIndex.resolve(bEl.value);
    if (!aId || !bId) {
        resEl.innerHTML = '<div class="error-msg">Pick two teams from the list.</div>';
        return;
    }
    resEl.innerHTML = '<div class="loading">Consulting the oracle</div>';
    const body = {team_a_id: aId, team_b_id: bId};
    if (dateEl.value) body.as_of = dateEl.value;
    const bo = document.getElementById('series-bo').value;
    if (bo) {   // a series in progress: the oracle prices the NEXT map at this score
        body.best_of = Number(bo);
        body.score_a = Number(document.getElementById('score-a').value || 0);
        body.score_b = Number(document.getElementById('score-b').value || 0);
    }
    const out = await postJSON('/api/predict/matchup', body);
    if (!out) { resEl.innerHTML = '<div class="error-msg">Oracle unavailable.</div>'; return; }

    if (out.available === false) {
        if (out.status === 'warming') {
            resEl.innerHTML = `<div class="loading">${esc(out.note)}</div>`;
            retryTimer = setTimeout(rate, RETRY_MS);
        } else {
            resEl.innerHTML = `<div class="error-msg">${esc(out.note || 'Oracle unavailable.')}</div>`;
        }
        return;
    }

    const inSeries = Boolean(bo) && Array.isArray(out.score);
    const stateNote = inSeries
        ? `Series at ${out.score[0]}–${out.score[1]} (Bo${bo}); the map line is the next map. `
        : '';
    resEl.innerHTML = `
        <div class="panel"><div class="panel-h"><span class="label">Matchup forecast</span></div><div class="panel-b">
            <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:14px;align-items:center;text-align:center;">
                <div class="entity-sub">${esc(aEl.value)}</div>
                <div class="metric-sub" style="margin:0;min-width:40px;">vs</div>
                <div class="entity-sub">${esc(bEl.value)}</div>
            </div>
            ${probRow(inSeries ? 'Next map' : 'Bo1', out.p_a, aEl.value, bEl.value)}
            ${probRow('Bo3', out.p_a_bo3, aEl.value, bEl.value)}
            ${probRow('Bo5', out.p_a_bo5, aEl.value, bEl.value)}
            <div class="metric-sub" style="margin-top:14px;text-align:center;">${esc(stateNote)}Teams as of ${esc(out.as_of)}</div>
        </div></div>`;
}

function render() {
    rootEl.innerHTML = `
        <div class="page-hero">
            <div>
                <div class="label"><span class="accent">◆</span> Oracle</div>
                <div class="entity-name" style="font-size:30px;">Forecast a matchup</div>
            </div>
        </div>
        <div class="panel"><div class="panel-b">
            <div class="draft-input">
                <div class="combo"><label>Team A</label><input type="text" id="team-a" list="team-options" placeholder="Search team…" autocomplete="off"></div>
                <div class="combo"><label>Team B</label><input type="text" id="team-b" list="team-options" placeholder="Search team…" autocomplete="off"></div>
            </div>
            <div class="combo" style="margin-top:12px;max-width:220px;"><label>As of (optional — defaults to today)</label><input type="date" id="as-of"></div>
            <div class="draft-input" style="margin-top:12px;">
                <div class="combo"><label>Series in progress (optional)</label>
                    <select id="series-bo"><option value="">Pre-series</option><option value="3">Bo3</option><option value="5">Bo5</option></select></div>
                <div class="combo"><label>Maps won: Team A</label><input type="number" id="score-a" min="0" max="2" value="0"></div>
                <div class="combo"><label>Maps won: Team B</label><input type="number" id="score-b" min="0" max="2" value="0"></div>
            </div>
            ${teamDatalist()}
            <div style="margin-top:16px;"><button class="btn" id="rate-btn">Forecast matchup</button></div>
        </div></div>
        <div id="predict-result"></div>`;
    document.getElementById('rate-btn').addEventListener('click', rate);
}

async function boot() {
    // Full-history team list (fixed range, deliberately not the filter bar):
    // the as-of date can forecast historical teams, so the pickers must offer them.
    const data = await fetchJSON('/api/teams?major_only=false&start_date=2020-01-01&end_date=2100-01-01');
    teams = (data && data.teams) || [];
    teamIndex = HERALD.teamNameIndex(teams);
    render();
}

boot();
