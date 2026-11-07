/* Upcoming slate — scheduled matches with the served model's pre-match call.
   Day groups over /api/upcoming rows; Today / This-week presets drive the
   shared filter bar's date inputs (leagues + major toggle work as everywhere).
   Times render in the viewer's local timezone, so the request pads the date
   window by a day each side and the rows re-filter to the local-day window. */

(() => {
    const {esc, fmt, links, splitBar, fetchJSON, filters, initFilterBar} = HERALD;

    const VIEW_KEY = 'lol_upcoming_view';   // 'today' | 'week'
    const board = document.getElementById('slate-board');

    const localDay = (iso) => fmt.dateISO(new Date(iso));
    const plusDays = (dayISO, n) => {
        const d = new Date(`${dayISO}T12:00:00`);   // noon dodges DST edges
        d.setDate(d.getDate() + n);
        return fmt.dateISO(d);
    };

    function setInputs(from, to) {
        const f = document.getElementById('date-from');
        const t = document.getElementById('date-to');
        if (f) f.value = from;
        if (t) t.value = to;
    }

    function applyView(view) {
        localStorage.setItem(VIEW_KEY, view);
        const today = fmt.dateISO(new Date());
        setInputs(today, view === 'week' ? plusDays(today, 6) : today);
        document.getElementById('view-today').classList.toggle('on', view === 'today');
        document.getElementById('view-week').classList.toggle('on', view === 'week');
        loadList();
    }

    /* One fixture line. Left team is team_a (P(team_a) = p_a). */
    function rowHTML(m) {
        const time = new Date(m.scheduled_ts)
            .toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
        const favA = m.p_a != null && m.p_a >= 0.5;
        const teamCell = (t, isA) => {
            const isFav = m.p_a != null && (isA ? favA : !favA);
            return `<span class="${isFav ? 'fav' : ''}">${links.team(t.team_id, t.name || 'TBD')}</span>`;
        };
        let meta = `Bo${m.best_of ?? '?'}`;
        let dim = '';
        // A running row carries the feed's score and a prediction for its NEXT map; a
        // not_started row begun in the last 8h with no ingested result is "live" by clock.
        const startedMs = Date.now() - new Date(m.scheduled_ts).getTime();
        if (m.ps_status === 'canceled' || m.ps_status === 'postponed') {
            meta = m.ps_status;
            dim = 'dim';
        } else if (m.result) {
            const {winner_team_id, score_a, score_b} = m.result;
            const called = m.p_a != null && winner_team_id
                ? (favA === (winner_team_id === m.team_a.team_id)
                    ? ' <span class="hit" title="The pre-match favorite won">✓</span>'
                    : ' <span class="miss" title="The pre-match favorite lost">✗</span>')
                : '';
            meta = `${score_a}–${score_b}${called}`;
        } else if (m.ps_status === 'running' && m.score) {
            const [a, b] = m.score;
            meta = `<span class="live" title="Maps won so far; the bar is the next map">LIVE ${a}–${b}</span>`;
        } else if (startedMs > 0 && startedMs < 8 * 3600 * 1000) {
            meta = '<span class="live">LIVE</span>';
        }
        const pcts = m.p_a == null ? null
            : [fmt.pct(m.p_a, 0), fmt.pct(1 - m.p_a, 0)];
        return `<div class="urow ${dim}">
            <span class="u-time">${esc(time)}</span>
            <span class="u-league" title="${esc(m.league_name)}">${esc(m.league_id || m.league_name)}</span>
            <span class="u-team a">${teamCell(m.team_a, true)}</span>
            ${pcts ? splitBar(m.p_a, pcts[0], pcts[1]) : splitBar(null)}
            <span class="u-team">${teamCell(m.team_b, false)}</span>
            <span class="u-meta">${meta}</span>
        </div>`;
    }

    function render(matches, from, to) {
        const inWindow = matches.filter((m) => {
            const day = localDay(m.scheduled_ts);
            return day >= from && day <= to;
        });
        if (!inWindow.length) {
            board.innerHTML = '<div class="no-matches">No matches scheduled in this window</div>';
            return;
        }
        const byDay = new Map();
        for (const m of inWindow) {
            const day = localDay(m.scheduled_ts);
            if (!byDay.has(day)) byDay.set(day, []);
            byDay.get(day).push(m);
        }
        board.innerHTML = [...byDay.entries()].map(([day, rows]) => {
            const label = new Date(`${day}T12:00:00`).toLocaleDateString([], {
                weekday: 'long', month: 'short', day: 'numeric',
            });
            return `<div class="slate-day series-hdr"><span>${esc(label)}</span>
                    <span>${rows.length} match${rows.length === 1 ? '' : 'es'}</span></div>`
                + rows.map(rowHTML).join('');
        }).join('');
    }

    async function loadList() {
        board.innerHTML = '<div class="loading">Loading the slate</div>';
        const from = filters.from();
        const to = filters.to();
        // Pad a day each side: the server windows on UTC dates, rows regroup locally.
        const p = new URLSearchParams();
        p.set('start_date', plusDays(from, -1));
        p.set('end_date', plusDays(to, 1));
        const lg = filters.leaguesCSV();
        if (lg) p.set('leagues', lg);
        p.set('major_only', filters.majorOnly());
        const data = await fetchJSON(`/api/upcoming?${p.toString()}`);
        if (!data) {
            board.innerHTML = '<div class="no-matches">The slate is unavailable</div>';
            return;
        }
        render(data.matches || [], from, to);
    }

    initFilterBar(loadList);
    document.getElementById('view-today').addEventListener('click', () => applyView('today'));
    document.getElementById('view-week').addEventListener('click', () => applyView('week'));
    applyView(localStorage.getItem(VIEW_KEY) === 'week' ? 'week' : 'today');
})();
