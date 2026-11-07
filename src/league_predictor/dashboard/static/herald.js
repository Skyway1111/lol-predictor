/* HERALD desk kit — the shared shell every tab loads first.
   Owns: header/nav/footer rendering, cross-entity link builders, the
   Ctrl-K command palette, the shared filter bar, and small formatting
   helpers. Page scripts consume this via the global HERALD object. */

const HERALD = (() => {
    // ── HTML escaping — every dynamic string goes through esc() ──
    const esc = (s) => String(s ?? '').replace(/[&<>"']/g,
        (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));

    // ── Formatting ──
    const fmt = {
        pct: (v, dp = 1) => v == null ? '--' : `${(v * 100).toFixed(dp)}%`,
        num: (v, dp = 2) => v == null ? '--' : v.toFixed(dp),
        int: (v) => v == null ? '--' : Math.round(v).toString(),
        signed: (v, dp = 0) => v == null ? '--' : `${v > 0 ? '+' : ''}${v.toFixed(dp)}`,
        dateISO: (d) => {
            const m = String(d.getMonth() + 1).padStart(2, '0');
            const day = String(d.getDate()).padStart(2, '0');
            return `${d.getFullYear()}-${m}-${day}`;
        },
    };

    // ── Cross-entity links — one home for "click a name, get the entity" ──
    const links = {
        team: (id, name, cls = '') => id
            ? `<a href="/teams?id=${encodeURIComponent(id)}" class="link link-team ${cls}" onclick="event.stopPropagation()">${esc(name)}</a>`
            : esc(name),
        player: (id, name, cls = '') => id
            ? `<a href="/players?id=${encodeURIComponent(id)}" class="link link-player ${cls}" onclick="event.stopPropagation()">${esc(name)}</a>`
            : esc(name),
        champion: (name, cls = '') => name
            ? `<a href="/champions?id=${encodeURIComponent(name)}" class="link link-champ ${cls}" onclick="event.stopPropagation()">${esc(name)}</a>`
            : esc(name),
        tournament: (id, name, cls = '') => id
            ? `<a href="/tournaments?id=${encodeURIComponent(id)}" class="link link-tourn ${cls}" onclick="event.stopPropagation()">${esc(name ?? id)}</a>`
            : esc(name ?? id),
        game: (id, label, cls = '') => id
            ? `<a href="/games?id=${encodeURIComponent(id)}" class="link link-game ${cls}" onclick="event.stopPropagation()">${esc(label ?? id)}</a>`
            : esc(label ?? id),
    };

    /* Grade letter → the .val-S/.val-A/… tint class (whitelisted). */
    const valClass = (g) => ['S', 'A', 'B', 'C', 'D'].includes(g) ? `val-${g}` : '';

    /* Calibrated expected edge beside a grade letter: "(+4.2pp)" / "(-4.2pp)".
       Blue's value is stored; red's is its exact negation. Empty string (no
       layout shift) on null — a row scored by a pre-edge grader. */
    const edgePP = (v) => v == null ? ''
        : ` <span class="metric-sub">(${v >= 0 ? '+' : '-'}${Math.abs(v).toFixed(1)}pp)</span>`;

    // Network/parse failures return null (not throw) so a page's existing
    // null-handling error state fires instead of the view sticking on "Loading…".
    async function fetchJSON(url) {
        try {
            const resp = await fetch(url);
            if (!resp.ok) return null;
            return await resp.json();
        } catch (e) { return null; }
    }

    async function postJSON(url, body) {
        try {
            const resp = await fetch(url, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(body),
            });
            if (!resp.ok) return null;
            return await resp.json();
        } catch (e) { return null; }
    }

    // ── Shared date range ──
    // The rendered bar is the live source of truth; localStorage only seeds
    // the initial render ("from" persists across sessions, "to" resets to
    // today). Pages without a bar fall back to the defaults.
    const dateRange = {
        defaults() {
            const y = new Date().getFullYear();
            return {
                from: localStorage.getItem('lol_date_from') || `${y}-01-01`,
                to: fmt.dateISO(new Date()),
            };
        },
    };

    // ── Shared filter state (date range + leagues multiselect + major) ──
    // The single source of truth for list/aggregate queries. The rendered
    // filter bar seeds the date inputs and major toggle; leagues selection
    // is a CSV persisted in localStorage ("" = all). Every page composes
    // its list queries from HERALD.filters.params().
    const LG_KEY = 'lol_leagues_csv';
    const filters = {
        from() {
            const el = document.getElementById('date-from');
            return el ? el.value : dateRange.defaults().from;
        },
        to() {
            const el = document.getElementById('date-to');
            return el ? el.value : dateRange.defaults().to;
        },
        majorOnly() {
            const t = document.getElementById('major-toggle');
            return t && !t.checked ? 'false' : 'true';
        },
        leaguesCSV() { return localStorage.getItem(LG_KEY) || ''; },
        setLeaguesCSV(csv) { localStorage.setItem(LG_KEY, csv || ''); },
        leagueSet() {
            const csv = this.leaguesCSV();
            return new Set(csv ? csv.split(',').filter(Boolean) : []);
        },
        /* start_date + end_date only (for endpoints that take no league scope). */
        dateParams() {
            return `start_date=${encodeURIComponent(this.from())}&end_date=${encodeURIComponent(this.to())}`;
        },
        /* Full scope: start_date, end_date, leagues (omitted when empty), major_only. */
        params() {
            const p = new URLSearchParams();
            p.set('start_date', this.from());
            p.set('end_date', this.to());
            const lg = this.leaguesCSV();
            if (lg) p.set('leagues', lg);
            p.set('major_only', this.majorOnly());
            return p.toString();
        },
    };

    /* Render the shared filter bar (From/To/Apply + Leagues multiselect +
       major-leagues toggle) into #date-bar. Renders ONCE — pages call this
       from boot(), and boot() re-runs on Apply/toggle/leagues, so a
       re-render here would reset the very control state (unchecked toggle,
       edited To date, open dropdown) the refetch is reading. Exposes the
       selection via HERALD.filters.params(). */
    function initFilterBar(onApply) {
        const bar = document.getElementById('date-bar');
        if (!bar || bar.dataset.wired) return;
        bar.dataset.wired = '1';
        const {from, to} = dateRange.defaults();
        bar.innerHTML = `
            <span class="label">From</span>
            <input type="date" id="date-from" value="${from}">
            <span class="label">To</span>
            <input type="date" id="date-to" value="${to}">
            <button id="date-apply" class="btn">Apply</button>
            <div class="lg-select">
                <button class="btn quiet" id="lg-btn" type="button">Leagues · <span id="lg-count">All</span></button>
                <div class="lg-drop" id="lg-drop">
                    <button class="lg-all" id="lg-all" type="button">All leagues (clear)</button>
                    <div class="lg-list" id="lg-list"><div class="loading">Loading</div></div>
                    <div class="lg-foot"><button class="btn" id="lg-apply" type="button">Apply leagues</button></div>
                </div>
            </div>
            <div class="toggle-wrap">
                <input type="checkbox" id="major-toggle" class="switch" checked>
                <label for="major-toggle">Major leagues only</label>
            </div>`;

        const drop = bar.querySelector('#lg-drop');
        const listEl = bar.querySelector('#lg-list');
        const countEl = bar.querySelector('#lg-count');
        let loaded = false;

        const paintCount = () => {
            const n = filters.leagueSet().size;
            countEl.textContent = n ? String(n) : 'All';
        };
        paintCount();

        async function loadLeagues() {
            listEl.innerHTML = '<div class="loading">Loading</div>';
            // Always the full league list (an explicit selection is the way to
            // reach non-majors, whatever the toggle says); majors sort first.
            const data = await fetchJSON(`/api/leagues?${filters.dateParams()}&major_only=false`);
            const sel = filters.leagueSet();
            const leagues = ((data && data.leagues) || []).slice().sort((a, b) =>
                (b.is_major ? 1 : 0) - (a.is_major ? 1 : 0) || (b.n_games ?? 0) - (a.n_games ?? 0));
            listEl.innerHTML = leagues.length
                ? leagues.map((l) => `
                    <label class="lg-item">
                        <input type="checkbox" value="${esc(l.league_id)}" ${sel.has(String(l.league_id)) ? 'checked' : ''}>
                        <span class="lg-item-name">${esc(l.name || l.league_id)}</span>
                        ${l.is_major ? '<span class="lg-mj">Major</span>' : ''}
                        <span class="lg-item-n">${l.n_games ?? 0}</span>
                    </label>`).join('')
                : '<div class="palette-empty">No leagues in range</div>';
            loaded = true;
        }

        const closeDrop = () => drop.classList.remove('open');
        bar.querySelector('#lg-btn').addEventListener('click', (e) => {
            e.stopPropagation();
            const opening = !drop.classList.contains('open');
            drop.classList.toggle('open');
            if (opening) loadLeagues();       // refetch each open → reflects current range
        });
        drop.addEventListener('click', (e) => e.stopPropagation());
        document.addEventListener('click', () => closeDrop());
        bar.querySelector('#lg-all').addEventListener('click', () => {
            listEl.querySelectorAll('input[type=checkbox]').forEach((c) => { c.checked = false; });
        });
        bar.querySelector('#lg-apply').addEventListener('click', () => {
            const csv = loaded
                ? [...listEl.querySelectorAll('input[type=checkbox]:checked')].map((c) => c.value).join(',')
                : filters.leaguesCSV();
            filters.setLeaguesCSV(csv);
            paintCount();
            closeDrop();
            onApply();
        });

        bar.querySelector('#date-apply').addEventListener('click', () => {
            localStorage.setItem('lol_date_from', bar.querySelector('#date-from').value);
            onApply();
        });
        bar.querySelector('#major-toggle').addEventListener('change', onApply);
    }

    // ── Header / nav / footer shell ──
    const ROLES = ['top', 'jng', 'mid', 'bot', 'sup'];
    const NAV = [
        {path: '/games', word: 'Games'},
        {path: '/upcoming', word: 'Upcoming'},
        {path: '/tournaments', word: 'Tournaments'},
        {path: '/teams', word: 'Teams'},
        {path: '/players', word: 'Players'},
        {path: '/champions', word: 'Champions'},
        {path: '/predict', word: 'Predict'},
        {path: '/draft', word: 'Draft'},
    ];

    function renderHeader() {
        const header = document.getElementById('desk-header');
        if (!header) return;
        const here = window.location.pathname;
        header.innerHTML = `
            <a class="masthead" href="/games">
                <div class="masthead-sigil">H</div>
                <div>
                    <div class="masthead-word">HERALD</div>
                </div>
            </a>
            <nav class="desk-nav">
                ${NAV.map((n) => `
                    <a class="nav-tab ${n.path === here ? 'active' : ''}" href="${n.path}">
                        <span class="nav-tab-word">${n.word}</span>
                    </a>`).join('')}
            </nav>
            <div class="desk-status">
                <button class="desk-btn" id="palette-btn" title="Search teams, players, champions">Search <kbd>Ctrl K</kbd></button>
            </div>`;
    }

    function renderFooter() {
        const footer = document.getElementById('desk-footer');
        if (!footer) return;
        footer.innerHTML = `
            <span>HERALD</span>
            <span id="footer-asof"></span>`;
    }

    function setFooterAsOf(dateStr) {
        const el = document.getElementById('footer-asof');
        if (el && dateStr) el.innerHTML = `Snapshot <span class="lit">${esc(dateStr)}</span>`;
    }

    // ── Command palette: fuzzy search across every team and player ──
    let _paletteIndex = null; // [{kind, id, name, meta}], built on first open

    async function buildPaletteIndex() {
        if (_paletteIndex) return _paletteIndex;
        const [teams, players] = await Promise.all([
            fetchJSON(`/api/teams?major_only=false&${filters.dateParams()}`),
            fetchJSON(`/api/players?major_only=true&${filters.dateParams()}`),
        ]);
        _paletteIndex = [
            ...((teams && teams.teams) || []).map((t) => ({
                kind: 'team', id: t.team_id, name: t.display_name,
                meta: [t.region, t.global_rank ? `#${t.global_rank}` : null].filter(Boolean).join(' · '),
            })),
            ...((players && players.players) || []).map((p) => ({
                kind: 'player', id: p.player_id, name: p.display_name,
                meta: [(p.role || '').toUpperCase(), p.team_display_name].filter(Boolean).join(' · '),
            })),
        ];
        return _paletteIndex;
    }

    function renderPalette() {
        const root = document.createElement('div');
        root.className = 'palette-overlay';
        root.id = 'palette-overlay';
        root.innerHTML = `
            <div class="palette">
                <input class="palette-input" id="palette-input" type="text"
                       placeholder="Search teams, players, champions…" autocomplete="off" spellcheck="false">
                <div class="palette-results" id="palette-results">
                    <div class="palette-empty">Type to search</div>
                </div>
            </div>`;
        document.body.appendChild(root);

        const input = root.querySelector('#palette-input');
        const results = root.querySelector('#palette-results');
        let rows = [];
        let sel = 0;

        const close = () => { root.classList.remove('open'); input.value = ''; };
        const open = async () => {
            root.classList.add('open');
            input.focus();
            await buildPaletteIndex();
            // Re-filter anything typed while the index was still loading
            input.dispatchEvent(new Event('input'));
        };

        const go = (entry) => {
            close();
            window.location.href = entry.kind === 'team'
                ? `/teams?id=${encodeURIComponent(entry.id)}`
                : `/players?id=${encodeURIComponent(entry.id)}`;
        };

        const paint = () => {
            if (!rows.length) {
                results.innerHTML = `<div class="palette-empty">${input.value ? 'No matches' : 'Type to search'}</div>`;
                return;
            }
            results.innerHTML = rows.map((r, i) => `
                <div class="palette-row ${i === sel ? 'sel' : ''}" data-i="${i}">
                    <span class="palette-kind ${r.kind}">${r.kind === 'team' ? 'Team' : 'Player'}</span>
                    <span class="palette-name">${esc(r.name)}</span>
                    <span class="palette-meta">${esc(r.meta)}</span>
                </div>`).join('');
            results.querySelectorAll('.palette-row').forEach((el) => {
                el.addEventListener('click', () => go(rows[parseInt(el.dataset.i, 10)]));
            });
        };

        input.addEventListener('input', () => {
            const q = input.value.trim().toLowerCase();
            sel = 0;
            if (!q || !_paletteIndex) { rows = []; paint(); return; }
            const starts = [];
            const contains = [];
            for (const e of _paletteIndex) {
                const n = e.name.toLowerCase();
                if (n.startsWith(q)) starts.push(e);
                else if (n.includes(q)) contains.push(e);
                if (starts.length >= 12) break;
            }
            rows = [...starts, ...contains].slice(0, 12);
            paint();
        });
        input.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowDown') { sel = Math.min(sel + 1, rows.length - 1); paint(); e.preventDefault(); }
            else if (e.key === 'ArrowUp') { sel = Math.max(sel - 1, 0); paint(); e.preventDefault(); }
            else if (e.key === 'Enter' && rows[sel]) go(rows[sel]);
            else if (e.key === 'Escape') close();
        });
        root.addEventListener('click', (e) => { if (e.target === root) close(); });
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); open(); }
        });
        const btn = document.getElementById('palette-btn');
        if (btn) btn.addEventListener('click', open);
    }

    // ── Champion icons (Data Dragon) ──────────────────────────────────
    // Champion ids in the DB are display names ("Kai'Sa"); Data Dragon
    // keys differ, so map the irregular ones and strip punctuation for the
    // rest. Version resolves once per session (cached in localStorage);
    // renders fall back to a pinned recent patch until it lands, and any
    // broken image collapses to a 2-letter text chip via onerror.
    const DD_PINNED = '15.13.1';
    let _ddVersion = localStorage.getItem('dd_version') || null;

    async function resolveDDVersion() {
        if (_ddVersion) return _ddVersion;
        try {
            const resp = await fetch('https://ddragon.leagueoflegends.com/api/versions.json');
            if (resp.ok) {
                const arr = await resp.json();
                if (Array.isArray(arr) && arr[0]) {
                    _ddVersion = arr[0];
                    localStorage.setItem('dd_version', arr[0]);
                    return arr[0];
                }
            }
        } catch (e) { /* offline / blocked — fall back to pinned */ }
        _ddVersion = DD_PINNED;
        return _ddVersion;
    }

    const CHAMP_KEY_MAP = {
        "Kai'Sa": 'Kaisa', "Kha'Zix": 'Khazix', "Vel'Koz": 'Velkoz', "Cho'Gath": 'Chogath',
        "Bel'Veth": 'Belveth', "K'Sante": 'KSante', "Kog'Maw": 'KogMaw', "Rek'Sai": 'RekSai',
        'LeBlanc': 'Leblanc', 'Wukong': 'MonkeyKing', 'Nunu & Willump': 'Nunu',
        'Renata Glasc': 'Renata', 'Dr. Mundo': 'DrMundo', 'Jarvan IV': 'JarvanIV',
        'Lee Sin': 'LeeSin', 'Master Yi': 'MasterYi', 'Miss Fortune': 'MissFortune',
        'Twisted Fate': 'TwistedFate', 'Xin Zhao': 'XinZhao', 'Aurelion Sol': 'AurelionSol',
        'Tahm Kench': 'TahmKench',
    };
    const champKey = (name) => {
        if (!name) return '';
        if (CHAMP_KEY_MAP[name]) return CHAMP_KEY_MAP[name];
        return String(name).replace(/['.\s&]/g, '');
    };
    const champTwoLetter = (name) => String(name || '?').replace(/[^A-Za-z]/g, '').slice(0, 2).toUpperCase() || '?';

    /* Champion icon: <img> from Data Dragon with a text-chip onerror fallback. */
    function champIcon(name, size = 22) {
        const ver = _ddVersion || DD_PINNED;
        const src = `https://ddragon.leagueoflegends.com/cdn/${ver}/img/champion/${champKey(name)}.png`;
        return `<span class="champ-ic" style="width:${size}px;height:${size}px" title="${esc(name || '')}">`
            + `<img src="${esc(src)}" width="${size}" height="${size}" loading="lazy" alt="${esc(name || '')}"`
            + ` onerror="this.style.display='none';this.nextElementSibling.style.display='flex';">`
            + `<span class="champ-ic-fb" style="display:none;">${esc(champTwoLetter(name))}</span></span>`;
    }

    // ── Win/loss + blue/red split bars (hand-drawn, gol.gg-style) ─────
    /* Green-win / red-loss proportion bar. winrate in 0..1 or null. */
    function wrBar(winrate, label) {
        if (winrate == null) {
            return '<div class="wr-bar"><div class="wr-track"></div><span class="wr-num">--</span></div>';
        }
        const pct = Math.max(0, Math.min(1, winrate)) * 100;
        const cap = label != null ? label : `${pct.toFixed(0)}%`;
        return `<div class="wr-bar"><div class="wr-track">`
            + `<div class="wr-win" style="width:${pct.toFixed(1)}%"></div>`
            + `<div class="wr-loss" style="width:${(100 - pct).toFixed(1)}%"></div>`
            + `</div><span class="wr-num">${esc(cap)}</span></div>`;
    }

    /* Blue-side / red-side split bar. share = blue-side fraction 0..1 or null. */
    function splitBar(share, blueLabel, redLabel) {
        if (share == null) {
            return '<div class="split-bar"><span class="sb-tag">--</span><div class="sb-track"></div><span class="sb-tag">--</span></div>';
        }
        const pct = Math.max(0, Math.min(1, share)) * 100;
        const bl = blueLabel != null ? blueLabel : `${pct.toFixed(0)}%`;
        const rl = redLabel != null ? redLabel : `${(100 - pct).toFixed(0)}%`;
        return `<div class="split-bar"><span class="sb-tag blue">${esc(bl)}</span>`
            + `<div class="sb-track"><div class="sb-blue" style="width:${pct.toFixed(1)}%"></div>`
            + `<div class="sb-red" style="width:${(100 - pct).toFixed(1)}%"></div></div>`
            + `<span class="sb-tag red">${esc(rl)}</span></div>`;
    }

    /* Team-name → id resolver. Build once from a /api/teams list; resolve()
       maps a trimmed, case-insensitive display name to its team_id (null when
       blank or unknown). Shared by the Predict and Draft team pickers. */
    function teamNameIndex(teams) {
        const byName = new Map((teams || []).map((t) => [(t.display_name || '').toLowerCase(), t.team_id]));
        return {
            resolve(name) {
                const key = (name || '').trim().toLowerCase();
                return key ? (byName.get(key) || null) : null;
            },
        };
    }

    // ── Sortable table helper (client-side sort on header click) ──────
    /* Renders columns/rows into `container` as a .table grid and wires
       click-to-sort on headers. columns: {key,label,get,render,cls,align,
       sortable,defaultDir}. opts: {grid,sortKey,sortDir,onRowClick}. Nulls
       always sort last. Returns {update(rows)}. */
    function sortableTable(container, columns, rows, opts = {}) {
        const grid = opts.grid || columns.map(() => '1fr').join(' ');
        let sortKey = opts.sortKey || (columns.find((c) => c.sortable !== false) || columns[0]).key;
        let sortDir = opts.sortDir || 'desc';
        const getCol = (k) => columns.find((c) => c.key === k);
        const valOf = (c, r) => (c.get ? c.get(r) : r[c.key]);

        function sortRows() {
            const col = getCol(sortKey);
            if (!col) return rows.slice();
            const dir = sortDir === 'asc' ? 1 : -1;
            return rows.slice().sort((a, b) => {
                const va = valOf(col, a), vb = valOf(col, b);
                if (va == null && vb == null) return 0;
                if (va == null) return 1;
                if (vb == null) return -1;
                if (typeof va === 'string' || typeof vb === 'string') {
                    return String(va).localeCompare(String(vb)) * dir;
                }
                return (va - vb) * dir;
            });
        }

        function paint() {
            const data = sortRows();
            const head = columns.map((c) => {
                const active = c.key === sortKey;
                const arrow = active ? (sortDir === 'asc' ? '▲' : '▼') : '';
                const sortable = c.sortable !== false;
                const align = c.align ? `text-align:${c.align};` : '';
                return `<div class="${sortable ? 'sortable' : ''} ${active ? 'sorted' : ''}" data-k="${esc(c.key)}" style="${align}">`
                    + `${esc(c.label)}<span class="sort-caret">${arrow}</span></div>`;
            }).join('');
            const body = data.map((r, i) => {
                const cells = columns.map((c) => {
                    const html = c.render ? c.render(r) : esc(valOf(c, r) ?? '--');
                    const align = c.align ? `text-align:${c.align};` : '';
                    return `<div class="${c.cls || ''}" style="${align}">${html}</div>`;
                }).join('');
                const rowAttrs = opts.onRowClick
                    ? ` data-idx="${i}" style="grid-template-columns:${grid};cursor:pointer;"`
                    : ` style="grid-template-columns:${grid};"`;
                return `<div class="trow"${rowAttrs}>${cells}</div>`;
            }).join('');
            container.innerHTML = `<div class="table"><div class="trow thead" style="grid-template-columns:${grid};">${head}</div>${body}</div>`;
            container.querySelectorAll('.thead .sortable').forEach((el) => {
                el.addEventListener('click', () => {
                    const k = el.dataset.k;
                    if (k === sortKey) sortDir = sortDir === 'asc' ? 'desc' : 'asc';
                    else { sortKey = k; sortDir = getCol(k).defaultDir || 'desc'; }
                    paint();
                });
            });
            if (opts.onRowClick) {
                container.querySelectorAll('.trow[data-idx]').forEach((el) => {
                    el.addEventListener('click', () => opts.onRowClick(data[parseInt(el.dataset.idx, 10)]));
                });
            }
        }

        paint();
        return { update(next) { rows = next; paint(); } };
    }

    // ── Shared match table (games list + team match list) ────────────
    /* Series-grouped scoreboard over /api/games-shaped rows. Rows arrive
       newest-first ordered (scheduled_ts DESC, game_no DESC), so consecutive
       rows sharing a series_id form one series block: a header line
       (date · league · Bo · series score over the listed games) above its
       games sorted G1→Gn. A game row reads
       strength · name / picks · grade   ⟨winner arrow · served P(blue)⟩
       mirrored for red; every model cell is dash-safe. Row click →
       onClickGame(game_id). opts.perspective (a team_id — the team sheet)
       adds that team's W/L per series and the power delta realized after it
       (next-newer pre-series strength minus this one's; includes drift
       between series, and the newest visible series has no successor → –). */

    function seriesBlocks(games) {
        const blocks = [];
        for (const g of games) {
            const last = blocks[blocks.length - 1];
            if (last && g.series_id != null && last.series_id === g.series_id) last.games.push(g);
            else blocks.push({series_id: g.series_id, games: [g]});
        }
        for (const b of blocks) b.games.sort((x, y) => (x.game_no ?? 0) - (y.game_no ?? 0));
        return blocks;
    }

    /* Map wins over a block's listed games; sides swap between games, so the
       count keys on team_id. Orientation is stable: G1's blue side first. */
    function seriesScore(games) {
        const wins = new Map(), names = new Map();
        for (const g of games) {
            for (const side of [g.blue, g.red]) {
                if (!side || !side.team_id) continue;
                if (!names.has(side.team_id)) names.set(side.team_id, side.name || side.team_id);
                wins.set(side.team_id, (wins.get(side.team_id) || 0) + (side.won === true ? 1 : 0));
            }
        }
        const ids = [...names.keys()];
        if (ids.length !== 2) return null;
        const first = games[0].blue && games[0].blue.team_id === ids[1] ? [ids[1], ids[0]] : ids;
        return {
            a: {id: first[0], name: names.get(first[0]), w: wins.get(first[0]) || 0},
            b: {id: first[1], name: names.get(first[1]), w: wins.get(first[1]) || 0},
        };
    }

    /* The team's pre-series strength, read off a games row. Null when unrated. */
    const povStrength = (g, teamId) => g.blue && g.blue.team_id === teamId ? g.rating_blue
        : g.red && g.red.team_id === teamId ? g.rating_red : null;

    function seriesHeaderHTML(block, opts, deltaNext) {
        const g0 = block.games[0];
        const score = seriesScore(block.games);
        const partial = (g0.game_no ?? 1) > 1;   // earlier games fell on the previous page
        const scoreHTML = score ? `
            <span class="sh-score" ${partial ? 'title="Earlier games of this series are on the previous page"' : ''}>${partial ? '⋯ ' : ''}
                <span class="${score.a.w > score.b.w ? 'w' : ''}">${esc(score.a.name)} ${score.a.w}</span>
                –
                <span class="${score.b.w > score.a.w ? 'w' : ''}">${score.b.w} ${esc(score.b.name)}</span>
            </span>` : '';
        let right = '';
        if (opts.perspective && score) {
            const mine = score.a.id === opts.perspective ? score.a
                : score.b.id === opts.perspective ? score.b : null;
            if (mine) {
                const theirs = mine === score.a ? score.b : score.a;
                // A definitive W/L needs a clinch over the VISIBLE games — a
                // page-cut series must not present a partial tally as a result.
                const clinched = Math.max(mine.w, theirs.w) > (g0.best_of ?? block.games.length) / 2;
                const res = !clinched ? '⋯' : mine.w > theirs.w ? 'W' : 'L';
                const dCls = deltaNext == null ? '' : deltaNext >= 0 ? 'up' : 'down';
                right = `<span class="sh-right">
                    <span class="sh-res ${res === 'W' ? 'w' : res === 'L' ? 'l' : ''}" ${clinched ? '' : 'title="Series result not decided by the games listed here"'}>${res} ${mine.w}–${theirs.w}</span>
                    <span class="sh-delta ${dCls}" title="Team power going into the next series minus power going into this one (includes drift between series)">Δ ${deltaNext == null ? '–' : fmt.signed(deltaNext)}</span>
                </span>`;
            }
        }
        return `<div class="series-hdr">
            <span>${esc(g0.date || '--')}</span>
            ${links.tournament(g0.league_id, g0.league_id)}
            <span>Bo${g0.best_of ?? '?'}</span>
            ${scoreHTML}${right}
        </div>`;
    }

    function matchRowHTML(g) {
        const blue = g.blue || {}, red = g.red || {};
        const icons = (arr) => `<div class="pick-row">`
            + (arr || []).map((c) => champIcon(c, 26)).join('') + '</div>';
        const grade = (letter, edge) => `<span class="mb-grade ${valClass(letter)}"
            title="Draft grade${edge != null ? ` · calibrated edge ${edge >= 0 ? '+' : '−'}${Math.abs(edge).toFixed(1)}pp` : ''}">${esc(letter || '–')}</span>`;
        const sideCls = (s) => s.won === true ? 'won' : s.won === false ? 'lost' : '';
        const dur = g.duration_s != null ? `${Math.round(g.duration_s / 60)}m` : '';
        // Post-draft bar. When the served parent came from the in-sample backfill row
        // (post present, walk-forward pred absent), hatch + mute it so it never reads as evidence.
        const postInSample = g.post_p_blue != null && g.pred_p_blue == null;
        const postBar = g.post_p_blue != null
            ? `<div class="mb-post${postInSample ? ' insample' : ''}"${postInSample ? ' title="post-draft rests on an in-sample pre-draft prior (no walk-forward prediction for this game)"' : ''}>${splitBar(g.post_p_blue)}</div>`
            : '';
        // Live-model prediction: the game postdates the walk-forward corpus, so the
        // current PRD's row is leak-free evidence — tagged, not blended in silently.
        const predBar = g.pred_source === 'live'
            ? `<div class="live-pred" title="Pre-draft P(blue) from the current live model — the game postdates the walk-forward corpus">${splitBar(g.pred_p_blue)}</div>`
            : splitBar(g.pred_p_blue);
        return `
            <div class="mrow" data-game="${esc(g.game_id)}">
                <div class="mb-gno"><span class="g">G${g.game_no ?? '?'}</span></div>
                <div class="mb-team blue ${sideCls(blue)}">
                    <span class="mb-pow" title="Team power rating going into the game">${fmt.int(g.rating_blue)}</span>
                    ${grade(g.grade_blue, g.edge_pp_blue)}
                    ${icons(blue.picks)}
                    <span class="mb-name" style="text-align:right;">${links.team(blue.team_id, blue.name || '--')}</span>
                </div>
                <div class="mb-center" title="Model win probability at pick time. Top bar: pre-draft P(blue) — dotted underline marks a live-model prediction (game is newer than the walk-forward corpus). Lower bar (when present): post-draft P(blue) = pre-draft × draft grade. Arrow marks the winner.">
                    <span class="mb-warrow" title="Game winner">${blue.won === true ? '◂' : ''}</span>
                    <div class="mb-probs">
                        ${predBar}
                        ${postBar}
                    </div>
                    <span class="mb-warrow" title="Game winner">${red.won === true ? '▸' : ''}</span>
                </div>
                <div class="mb-team red ${sideCls(red)}">
                    <span class="mb-name">${links.team(red.team_id, red.name || '--')}</span>
                    ${icons(red.picks)}
                    ${grade(g.grade_red, g.edge_pp_blue == null ? null : -g.edge_pp_blue)}
                    <span class="mb-pow" title="Team power rating going into the game">${fmt.int(g.rating_red)}</span>
                </div>
                <div class="mb-gno"><span class="d" title="Game duration">${esc(dur)}</span></div>
            </div>`;
    }

    function matchTable(container, games, onClickGame, opts = {}) {
        if (!games || !games.length) {
            container.innerHTML = '<div class="no-matches">No matches in this range</div>';
            return;
        }
        const blocks = seriesBlocks(games);
        const html = blocks.map((b, i) => {
            let delta = null;
            if (opts.perspective && i > 0) {
                const now = povStrength(b.games[0], opts.perspective);
                const next = povStrength(blocks[i - 1].games[0], opts.perspective);
                if (now != null && next != null) delta = next - now;
            }
            return `<div class="sblock ${i % 2 ? 'alt' : ''}">${seriesHeaderHTML(b, opts, delta)}${b.games.map(matchRowHTML).join('')}</div>`;
        }).join('');
        container.innerHTML = `<div class="matchboard">${html}</div>`;
        if (onClickGame) {
            container.querySelectorAll('.mrow[data-game]').forEach((el) => {
                el.addEventListener('click', () => onClickGame(el.dataset.game));
            });
        }
    }

    // ── Boot the shared shell ──
    function bootShell() {
        renderHeader();
        renderFooter();
        renderPalette();
        resolveDDVersion();       // warm the Data Dragon version before icons render
    }

    return {
        esc, fmt, links, valClass, edgePP, fetchJSON, postJSON,
        ROLES, filters, initFilterBar,
        champIcon, wrBar, splitBar, teamNameIndex, sortableTable, matchTable,
        setFooterAsOf, bootShell,
    };
})();

HERALD.bootShell();
