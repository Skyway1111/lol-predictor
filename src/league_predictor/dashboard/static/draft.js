/* War Room (Draft) — a live draft board: pick slots B1-B5 / R1-R5, optional
   ban rows, a champion pool, and an engine panel of top lines. Clicking a
   champion fills the active slot; picks auto-advance through the standard
   tournament pick order (B1 → R1 R2 → B2 B3 → R3 → R4 → B4 B5 → R5); ban
   slots are filled by clicking them first (bans are optional). Used champions
   grey out in the pool.

   TWO labeled values, never mixed: the scoreboard's headline %, edge (pp)
   and letter grades all describe the CURRENT board — one calibrator, one
   board, always mutually consistent. The engine's PROJECTION — P(win) at the
   END of the draft — is its own labeled row beneath the bar. A board change
   POSTs the partial draft to /api/predict/draft (grades + board value + edge,
   painting instantly; each filled slot shows the role the scorer assigned it)
   and to /api/predict/draft/suggest (the candidate names land in ~1s, values
   pending). One batched /api/predict/draft/suggest/rollout call then plays
   every line out to the end of the draft in lockstep; the lines show and sort
   by that realized value — so a pick that gets countered down the line ranks
   by the countered draft — and the projection row settles to the BEST line's
   value for the side on the clock (minimax-consistent: assume whoever picks
   next plays the strongest move the rollout found, never that they miss it).
   Clicking a line carries its end value over as the new board's projection
   until the new rollout re-settles it, so the row never blanks mid-click. A
   board change abandons the pending batch. The immediate board value, the
   enemy's best reply, and any teammate role moves show on hover. */

const {esc, fmt, fetchJSON, postJSON, valClass, edgePP, splitBar, champIcon} = HERALD;

const rootEl = document.getElementById('draft-root');
const {ROLES} = HERALD;
// Standard tournament pick order, as [side, slot index] pairs.
const PICK_SEQ = [
    ['blue', 0], ['red', 0], ['red', 1], ['blue', 1], ['blue', 2],
    ['red', 2], ['red', 3], ['blue', 3], ['blue', 4], ['red', 4],
];

let champs = [];                                   // [{champion, main_role}]
const picks = {blue: Array(5).fill(null), red: Array(5).fill(null)};
const bans = {blue: Array(5).fill(null), red: Array(5).fill(null)};
let active = {kind: 'pick', side: 'blue', idx: 0}; // the slot the next click fills
let roleFilter = null;
let scoreSeq = 0;                                  // drops out-of-order score responses
let suggestSeq = 0;                                // drops out-of-order suggest responses
let suggestInFlight = false;                       // latest-wins throttle: one suggest at a time
let suggestQueued = false;                         //   a change during flight re-suggests after
let suggestTarget = null;                          // the pick slot the engine lines fill
let engineLines = null;                            // the current (board, target)'s lines
let assignedRoles = {blue: {}, red: {}};           // champion -> role, from the scorer

function nextEmptyPick() {
    for (const [side, idx] of PICK_SEQ) {
        if (!picks[side][idx]) return {kind: 'pick', side, idx};
    }
    return null;
}

function usedChampions() {
    return new Set([...picks.blue, ...picks.red, ...bans.blue, ...bans.red].filter(Boolean));
}

// ── Scoreboard ────────────────────────────────────────────────────────
let lastScore = null;  // the standing board's score payload, kept for the settle repaint
let projectedP = null; // the standing board's projection; null until its rollout lands
                       // (cleared on every board/target change — the score and rollout
                       // responses race on the server lock, so whichever lands second
                       // must repaint with BOTH pieces, never stomp the other)

function paintScore(out) {
    // Headline %, edge and grades all describe the CURRENT board — they share
    // one calibrator and must never disagree. The projection (end-of-draft
    // value, see the header comment) is its own labeled row once known.
    const el = document.getElementById('db-score');
    const proj = projectedP != null
        ? `<div class="db-score-proj">Projected end of draft: <strong>${fmt.pct(projectedP)}</strong> blue</div>`
        : '';
    el.innerHTML = `
        <div class="db-score-line">
            <span class="db-grade blue ${valClass(out.grade_blue)}">${esc(out.grade_blue)}</span>
            <span class="db-score-p"><strong>${fmt.pct(out.p_blue)}</strong><span class="metric-sub" style="margin:0;">Blue to win${edgePP(out.edge_pp_blue)}</span></span>
            <span class="db-grade red ${valClass(out.grade_red)}">${esc(out.grade_red)}</span>
        </div>
        ${splitBar(out.p_blue)}
        ${proj}`;
}

async function score() {
    const seq = ++scoreSeq;
    lastScore = null;
    const out = await postJSON('/api/predict/draft', {
        blue: {picks: picks.blue.filter(Boolean)},
        red: {picks: picks.red.filter(Boolean)},
        bans: {blue: bans.blue.filter(Boolean), red: bans.red.filter(Boolean)},
    });
    if (seq !== scoreSeq) return; // a newer pick already re-scored
    if (!out || out.available === false) {
        document.getElementById('db-score').innerHTML =
            `<div class="error-msg">${esc((out && out.note) || 'Draft scorer unavailable.')}</div>`;
        return;
    }
    assignedRoles = out.roles || {blue: {}, red: {}};
    paintSlots(); // stamp the scorer's role assignment onto the filled slots
    lastScore = out;
    paintScore(out); // paints the settle too, if the rollout won the race
}

// ── Engine panel (top lines for the pick on the clock) ────────────────
function engineTarget() {
    if (active && active.kind === 'pick') return {side: active.side, idx: active.idx};
    return nextEmptyPick(); // a ban slot is selected (or nothing): suggest for the next pick
}

// Lines fetched fast and then evaluated in one batched rollout. The panel
// shows the one-ply top 5; the rollout re-ranks within them (a pick outside
// the one-ply top 5 is deliberately never considered — panel size is also
// the lockstep rollout's width, so this knob is the panel's latency).
const EVAL_LINES = 5;

function draftPayload() {
    return {
        blue: {picks: picks.blue.filter(Boolean)},
        red: {picks: picks.red.filter(Boolean)},
        bans: {blue: bans.blue.filter(Boolean), red: bans.red.filter(Boolean)},
    };
}

async function suggest() {
    // Every (board, target) change invalidates all in-flight chains FIRST —
    // even when the request is deferred (throttle) or skipped (draft
    // complete), a stale fast suggest or rollout must never pass its seq
    // check and paint.
    const seq = ++suggestSeq;
    engineLines = null; // until this (board, target)'s lines land
    const linesEl = document.getElementById('engine-lines');
    const slotEl = document.getElementById('engine-slot');
    const orientEl = document.getElementById('engine-orient');
    suggestTarget = engineTarget();
    if (!suggestTarget) {
        slotEl.textContent = '--';
        slotEl.className = 'db-engine-slot';
        orientEl.textContent = '';
        linesEl.innerHTML = '<div class="db-engine-note">Draft complete.</div>';
        // A finished board's projection IS the headline — drop the redundant row.
        projectedP = null;
        if (lastScore) paintScore(lastScore);
        return;
    }
    const t = suggestTarget;
    slotEl.textContent = `${t.side === 'blue' ? 'B' : 'R'}${t.idx + 1}`;
    slotEl.className = `db-engine-slot ${t.side}`;
    orientEl.textContent = `${t.side} to win`;
    // Latest-wins throttle: never stack suggest requests — remember that the
    // board moved and re-suggest (from the LATEST state) when this returns.
    if (suggestInFlight) { suggestQueued = true; return; }
    linesEl.classList.add('stale');
    suggestInFlight = true;
    // Snapshot the board NOW: the rollout below must query exactly the board
    // these lines were suggested for, never a re-read of mutated state.
    const payload = {...draftPayload(), side: t.side};
    let out;
    try {
        out = await postJSON('/api/predict/draft/suggest', {...payload, top_n: EVAL_LINES});
    } finally {
        suggestInFlight = false;
        if (suggestQueued) { suggestQueued = false; suggest(); }
    }
    if (seq !== suggestSeq) return; // a newer board state already re-suggested
    linesEl.classList.remove('stale');
    if (!out || out.available === false) {
        linesEl.innerHTML = `<div class="db-engine-note">${esc((out && out.note) || 'Suggestions unavailable.')}</div>`;
        return;
    }
    if (!out.suggestions.length) {
        linesEl.innerHTML = '<div class="db-engine-note">No suggestions for this pick.</div>';
        return;
    }
    engineLines = out.suggestions; // evaluateLines fills end values into this same array
    renderLines(linesEl, out.suggestions, t);
    evaluateLines(seq, payload, out.suggestions, t, linesEl);
}

async function evaluateLines(seq, payload, lines, t, linesEl) {
    // One batched lockstep rollout values every line's finished draft in
    // ~the time one candidate used to take; abandon if the board moved on,
    // then rank by where the finished draft actually lands.
    const r = await postJSON('/api/predict/draft/suggest/rollout',
        {...payload, champions: lines.map((s) => s.champion)});
    if (seq !== suggestSeq) return;
    if (r && r.available) {
        const ends = new Map(r.values.map((v) => [v.champion, v.p_blue_end]));
        for (const s of lines) {
            if (ends.has(s.champion)) s.p_blue_end = ends.get(s.champion);
        }
    }
    const pSide = (p) => (p == null ? -1 : (t.side === 'blue' ? p : 1 - p));
    // Mirrors the engine's end-value ranking rule (suggest.py suggest()) —
    // keep the two in step.
    lines.sort((a, b) => (pSide(b.p_blue_end) - pSide(a.p_blue_end))
        || a.champion.localeCompare(b.champion)); // the engine's tie-break
    renderLines(linesEl, lines, t);
    // Settle the projection minimax-consistently: the side on the clock plays
    // the BEST line the rollout found, never a weaker continuation it can see
    // past (an assumption the panel itself would contradict). Sorted above, so
    // the first evaluated line is that best line.
    const best = lines.find((s) => s.p_blue_end != null);
    if (best) {
        projectedP = best.p_blue_end;
        if (lastScore) paintScore(lastScore);
    }
}

function renderLines(linesEl, lines, t) {
    linesEl.innerHTML = lines.map((s, i) => {
        // The ONE displayed number is the line's end-of-draft value — the same
        // metric the list is sorted by and the scoreboard settles to, so ranks
        // always read in order ("…" until the rollout batch lands). The
        // immediate board value, the enemy's best reply, and any role moves
        // live in the hover.
        const evaluated = s.p_blue_end != null;
        const pEnd = t.side === 'blue' ? s.p_blue_end : 1 - s.p_blue_end;
        const pReply = t.side === 'blue' ? s.p_blue : 1 - s.p_blue;
        const pNow = t.side === 'blue' ? s.p_blue_now : 1 - s.p_blue_now;
        const sub = [
            esc(s.role.toUpperCase()),
            ...(s.moved || []).map((m) => `${esc(m.champion)} → ${esc(m.to.toUpperCase())}`),
        ].filter(Boolean).join(' · ');
        const hover = `Play ${s.champion} (${s.role}) at ${t.side === 'blue' ? 'B' : 'R'}${t.idx + 1}`
            + ` — board becomes ${fmt.pct(pNow)} for ${t.side}`
            + (s.reply ? `, ${fmt.pct(pReply)} after they answer ${s.reply.champion} (${s.reply.role})` : '')
            + (evaluated ? `; drafted out on-model, the draft ends ${fmt.pct(pEnd)}` : '; drafting it out…')
            + (s.moved || []).map((m) => `; ${m.champion} moves ${m.from} → ${m.to}`).join('');
        return `
        <div class="db-line" data-champ="${esc(s.champion)}" title="${esc(hover)}">
            <span class="db-line-rank">${i + 1}</span>
            ${champIcon(s.champion, 34)}
            <div class="db-line-main">
                <span class="db-line-name">${esc(s.champion)}</span>
                <span class="db-line-role">${sub}</span>
            </div>
            <div class="db-line-val">
                <strong class="${evaluated ? t.side : 'pending'}">${evaluated ? fmt.pct(pEnd) : '…'}</strong>
            </div>
        </div>`;
    }).join('');
    linesEl.querySelectorAll('.db-line').forEach((el) => el.addEventListener('click', () => {
        if (!suggestTarget) return;
        setSlot('pick', suggestTarget.side, suggestTarget.idx, el.dataset.champ);
    }));
}

// ── Board state changes ───────────────────────────────────────────────
function setSlot(kind, side, idx, champ) {
    // Optimistic projection carry: playing one of the engine's evaluated lines
    // puts an end value for the new board already in hand — that line's — so
    // the projection row needn't blank for the ~2s the new rollout takes to
    // re-settle it.
    const played = (kind === 'pick' && champ && engineLines && suggestTarget
        && suggestTarget.side === side && suggestTarget.idx === idx)
        ? engineLines.find((s) => s.champion === champ) : null;
    (kind === 'pick' ? picks : bans)[side][idx] = champ;
    active = champ ? (nextEmptyPick() || null) : {kind, side, idx};
    projectedP = played && played.p_blue_end != null ? played.p_blue_end : null;
    paintSlots();
    paintGrid();
    score();
    suggest();
}

function onChampClick(name) {
    if (!active || usedChampions().has(name)) return;
    setSlot(active.kind, active.side, active.idx, name);
}

function onSlotClick(kind, side, idx) {
    const cur = (kind === 'pick' ? picks : bans)[side][idx];
    if (cur) { setSlot(kind, side, idx, null); return; } // click a filled slot to clear it
    active = {kind, side, idx};
    projectedP = null; // the projection follows the side on the clock
    paintSlots();
    if (lastScore) paintScore(lastScore); // drop the old target's projection row
    suggest(); // the slot on the clock changed — re-aim the engine lines
}

function reset() {
    for (const side of ['blue', 'red']) { picks[side].fill(null); bans[side].fill(null); }
    active = {kind: 'pick', side: 'blue', idx: 0};
    projectedP = null;
    paintSlots();
    paintGrid();
    score();
    suggest();
}

// ── Painting ──────────────────────────────────────────────────────────
function isActive(kind, side, idx) {
    return active && active.kind === kind && active.side === side && active.idx === idx;
}

function paintSlots() {
    for (const side of ['blue', 'red']) {
        picks[side].forEach((champ, i) => {
            const el = document.getElementById(`pick-${side}-${i}`);
            const role = champ && assignedRoles[side] && assignedRoles[side][champ];
            el.className = `db-slot ${side} ${isActive('pick', side, i) ? 'active' : ''} ${champ ? 'filled' : ''}`;
            el.innerHTML = champ
                ? champIcon(champ, 64) + (role ? `<span class="db-slot-role">${esc(role.toUpperCase())}</span>` : '')
                : '<span class="db-slot-helm">⛨</span>';
            el.title = champ
                ? `${champ}${role ? ` (${role})` : ''} — click to clear`
                : `${side} pick ${i + 1}`;
        });
        bans[side].forEach((champ, i) => {
            const el = document.getElementById(`ban-${side}-${i}`);
            el.className = `db-ban ${isActive('ban', side, i) ? 'active' : ''} ${champ ? 'filled' : ''}`;
            el.innerHTML = champ ? `${champIcon(champ, 40)}<span class="db-ban-x">⊘</span>` : '';
            el.title = champ ? `${champ} banned — click to clear` : `${side} ban ${i + 1}`;
        });
    }
}

function paintGrid() {
    const q = document.getElementById('champ-search').value.trim().toLowerCase();
    const used = usedChampions();
    document.querySelectorAll('#champ-grid .db-champ').forEach((el) => {
        const name = el.dataset.champ;
        const role = el.dataset.role;
        const hide = (q && !name.toLowerCase().includes(q)) || (roleFilter && role !== roleFilter);
        el.classList.toggle('hidden', hide);
        el.classList.toggle('used', used.has(name));
    });
}

function render() {
    const banBoxes = (side) => [0, 1, 2, 3, 4].map((i) =>
        `<div class="db-ban" id="ban-${side}-${i}"></div>`).join('');
    const pickSlots = (side) => [0, 1, 2, 3, 4].map((i) => `
        <div class="db-pick-row ${side}">
            <span class="db-slot-tag ${side}">${side === 'blue' ? 'B' : 'R'}${i + 1}</span>
            <div class="db-slot ${side}" id="pick-${side}-${i}"></div>
        </div>`).join('');
    const roleBtns = ROLES.map((r) =>
        `<button class="db-role-btn" data-role="${r}">${r.toUpperCase()}</button>`).join('');
    const grid = champs.map((c) => `
        <div class="db-champ" data-champ="${esc(c.champion)}" data-role="${esc(c.main_role || '')}">
            ${champIcon(c.champion, 56)}
            <span class="db-champ-name">${esc(c.champion)}</span>
        </div>`).join('');

    rootEl.innerHTML = `
        <div class="page-hero">
            <div>
                <div class="label"><span class="accent">◆</span> War Room</div>
                <div class="entity-name" style="font-size:30px;">Draft board</div>
            </div>
            <button class="btn quiet" id="reset-btn">Reset draft</button>
        </div>
        <div class="panel"><div class="panel-b">
            <div class="db-sidebar-row">
                <div class="db-side-bar blue">Blue side</div>
                <div class="db-side-bar red">Red side</div>
            </div>
            <div class="db-top">
                <div class="db-bans" id="bans-blue">${banBoxes('blue')}</div>
                <div class="db-score" id="db-score"><div class="loading">Scoring</div></div>
                <div class="db-bans red" id="bans-red">${banBoxes('red')}</div>
            </div>
            <div class="db-board">
                <div class="db-picks">${pickSlots('blue')}</div>
                <div class="db-pool">
                    <div class="db-pool-bar">
                        <input type="text" id="champ-search" placeholder="Search" autocomplete="off">
                        <div class="db-role-btns">${roleBtns}</div>
                    </div>
                    <div class="db-grid" id="champ-grid">${grid}</div>
                </div>
                <div class="db-picks">${pickSlots('red')}</div>
                <div class="db-engine">
                    <div class="db-engine-head">
                        <span class="label" style="margin:0;"><span class="accent">◆</span> Oracle lines</span>
                        <span class="label" id="engine-orient" style="margin:0 auto 0 0;color:var(--text-faint);"></span>
                        <span class="db-engine-slot" id="engine-slot">--</span>
                    </div>
                    <div class="db-engine-lines" id="engine-lines"><div class="loading">Consulting</div></div>
                </div>
            </div>
        </div></div>`;

    document.getElementById('reset-btn').addEventListener('click', reset);
    document.getElementById('champ-search').addEventListener('input', paintGrid);
    document.querySelectorAll('.db-role-btn').forEach((b) => b.addEventListener('click', () => {
        roleFilter = roleFilter === b.dataset.role ? null : b.dataset.role;
        document.querySelectorAll('.db-role-btn').forEach((x) =>
            x.classList.toggle('on', x.dataset.role === roleFilter));
        paintGrid();
    }));
    document.querySelectorAll('#champ-grid .db-champ').forEach((el) =>
        el.addEventListener('click', () => onChampClick(el.dataset.champ)));
    for (const side of ['blue', 'red']) {
        for (let i = 0; i < 5; i++) {
            document.getElementById(`pick-${side}-${i}`)
                .addEventListener('click', () => onSlotClick('pick', side, i));
            document.getElementById(`ban-${side}-${i}`)
                .addEventListener('click', () => onSlotClick('ban', side, i));
        }
    }
    paintSlots();
    paintGrid();
    score();
    suggest();
}

async function boot() {
    // Full-history champion pool (fixed range, deliberately not the filter bar:
    // a draft can pick any champion, not just this season's).
    const data = await fetchJSON('/api/champions?major_only=false&start_date=2020-01-01&end_date=2100-01-01');
    champs = ((data && data.champions) || [])
        .map((c) => ({champion: c.champion, main_role: c.main_role}))
        .sort((a, b) => a.champion.localeCompare(b.champion));
    render();
}

boot();
