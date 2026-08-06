const API = 'http://127.0.0.1:8000';

/* ============================================================
   CHART.JS GLOBAL DEFAULTS
   ============================================================ */
function applyChartDefaults() {
    Chart.defaults.color = 'hsl(210,15%,65%)';
    Chart.defaults.borderColor = 'hsla(210,30%,60%,0.1)';
    Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
    Chart.defaults.font.size = 12;
}

const MODEL_COLORS = {
    xgboost:             { border: 'hsl(199,98%,55%)',   bg: 'hsla(199,98%,55%,0.2)' },
    logistic_regression: { border: 'hsl(260,80%,68%)',   bg: 'hsla(260,80%,68%,0.2)' },
    rnn_lstm:            { border: 'hsl(152,72%,50%)',   bg: 'hsla(152,72%,50%,0.2)' },
    qsofa_baseline:      { border: 'hsl(38,95%,55%)',    bg: 'hsla(38,95%,55%,0.2)'  },
};

const MODEL_LABELS = {
    xgboost:             'XGBoost',
    logistic_regression: 'Logistic Reg.',
    rnn_lstm:            'RNN-LSTM',
    qsofa_baseline:      'qSOFA Baseline',
};

const ORDER = ['xgboost', 'logistic_regression', 'rnn_lstm', 'qsofa_baseline'];

function chartTooltipPlugin() {
    return {
        backgroundColor: 'hsl(222,24%,11%)',
        borderColor:      'hsla(210,40%,60%,0.2)',
        borderWidth: 1,
        titleColor:  'hsl(210,30%,94%)',
        bodyColor:   'hsl(210,15%,65%)',
        padding: 10,
        cornerRadius: 8,
    };
}

/* ============================================================
   RENDER: AUROC / AUPRC BAR CHARTS
   ============================================================ */
function renderBarChart(canvasId, models, metricKey, label, yMax = 1) {
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx) return;

    const labels  = ORDER.filter(k => models[k] && Object.keys(models[k]).length > 0)
                         .map(k => MODEL_LABELS[k]);
    const values  = ORDER.filter(k => models[k] && Object.keys(models[k]).length > 0)
                         .map(k => {
                             const m = models[k];
                             return m[metricKey] ?? m.roc_auc ?? m.pr_auc ?? null;
                         });
    const colors  = ORDER.filter(k => models[k] && Object.keys(models[k]).length > 0)
                         .map(k => MODEL_COLORS[k]);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label,
                data: values,
                backgroundColor: colors.map(c => c.bg),
                borderColor:     colors.map(c => c.border),
                borderWidth: 2,
                borderRadius: 8,
                borderSkipped: false,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    ...chartTooltipPlugin(),
                    callbacks: {
                        label: ctx => ` ${label}: ${ctx.raw?.toFixed(4) ?? '—'}`,
                    },
                },
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { font: { size: 11 } },
                },
                y: {
                    min: 0,
                    max: yMax,
                    grid: { color: 'hsla(210,30%,60%,0.08)' },
                    ticks: {
                        stepSize: 0.1,
                        callback: v => v.toFixed(2),
                    },
                },
            },
            animation: { duration: 900, easing: 'easeOutQuart' },
        },
    });
}

/* ============================================================
   RENDER: MULTI-METRIC RADAR
   ============================================================ */
function renderRadarChart(canvasId, models) {
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx) return;

    const radarMetrics = ['auroc', 'sensitivity', 'specificity', 'f1_score', 'auprc'];
    const radarLabels  = ['AUROC', 'Sensitivity', 'Specificity', 'F1 Score', 'AUPRC'];

    const datasets = ORDER
        .filter(k => models[k] && Object.keys(models[k]).length > 0)
        .map(k => {
            const m = models[k];
            const c = MODEL_COLORS[k];
            return {
                label: MODEL_LABELS[k],
                data: radarMetrics.map(metric => {
                    const v = m[metric] ?? m.f1 ?? m.roc_auc ?? 0;
                    return typeof v === 'number' ? +v.toFixed(4) : 0;
                }),
                backgroundColor: c.bg,
                borderColor:     c.border,
                borderWidth: 2,
                pointBackgroundColor: c.border,
                pointRadius: 4,
                pointHoverRadius: 6,
            };
        });

    new Chart(ctx, {
        type: 'radar',
        data: { labels: radarLabels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { boxWidth: 12, padding: 12, font: { size: 11 } },
                },
                tooltip: { ...chartTooltipPlugin() },
            },
            scales: {
                r: {
                    min: 0,
                    max: 1,
                    grid:       { color: 'hsla(210,30%,60%,0.12)' },
                    angleLines: { color: 'hsla(210,30%,60%,0.12)' },
                    pointLabels: { font: { size: 11 }, color: 'hsl(210,15%,65%)' },
                    ticks: {
                        stepSize: 0.25,
                        backdropColor: 'transparent',
                        callback: v => v.toFixed(2),
                    },
                },
            },
            animation: { duration: 900, easing: 'easeOutQuart' },
        },
    });
}

/* ============================================================
   RENDER: SENSITIVITY vs SPECIFICITY GROUPED BAR
   ============================================================ */
function renderSensSpecChart(canvasId, models) {
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx) return;

    const keys   = ORDER.filter(k => models[k] && Object.keys(models[k]).length > 0);
    const labels = keys.map(k => MODEL_LABELS[k]);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'Sensitivity',
                    data:  keys.map(k => +(models[k].sensitivity ?? 0).toFixed(4)),
                    backgroundColor: 'hsla(199,98%,55%,0.25)',
                    borderColor:     'hsl(199,98%,55%)',
                    borderWidth: 2,
                    borderRadius: 6,
                    borderSkipped: false,
                },
                {
                    label: 'Specificity',
                    data:  keys.map(k => +(models[k].specificity ?? 0).toFixed(4)),
                    backgroundColor: 'hsla(152,72%,50%,0.25)',
                    borderColor:     'hsl(152,72%,50%)',
                    borderWidth: 2,
                    borderRadius: 6,
                    borderSkipped: false,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    align: 'end',
                    labels: { boxWidth: 12, padding: 12, font: { size: 11 } },
                },
                tooltip: {
                    ...chartTooltipPlugin(),
                    callbacks: {
                        label: ctx => ` ${ctx.dataset.label}: ${(ctx.raw * 100).toFixed(1)}%`,
                    },
                },
            },
            scales: {
                x: { grid: { display: false }, ticks: { font: { size: 11 } } },
                y: {
                    min: 0, max: 1,
                    grid: { color: 'hsla(210,30%,60%,0.08)' },
                    ticks: { callback: v => (v * 100).toFixed(0) + '%' },
                },
            },
            animation: { duration: 900, easing: 'easeOutQuart' },
        },
    });
}

/* ============================================================
   RENDER: SHAP FEATURE IMPORTANCE HORIZONTAL BAR
   ============================================================ */
function renderFeatureChart(canvasId, shapFeatures) {
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx || !shapFeatures?.length) return;

    const top = shapFeatures.slice(0, 15);
    const labels = top.map(f => f.feature);
    const values = top.map(f => +f.mean_abs_shap.toFixed(5));

    // Gradient from accent to purple
    const grad = ctx.createLinearGradient(400, 0, 0, 0);
    grad.addColorStop(0,   'hsl(199,98%,55%)');
    grad.addColorStop(0.5, 'hsl(230,80%,65%)');
    grad.addColorStop(1,   'hsl(260,80%,68%)');

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Mean |SHAP|',
                data: values,
                backgroundColor: grad,
                borderColor:     'transparent',
                borderRadius: 5,
                borderSkipped: false,
            }],
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    ...chartTooltipPlugin(),
                    callbacks: {
                        label: ctx => ` Mean |SHAP|: ${ctx.raw.toFixed(5)}`,
                    },
                },
            },
            scales: {
                x: {
                    grid: { color: 'hsla(210,30%,60%,0.08)' },
                    ticks: { callback: v => v.toFixed(3) },
                },
                y: {
                    grid: { display: false },
                    ticks: { font: { size: 11 } },
                },
            },
            animation: { duration: 900, easing: 'easeOutQuart' },
        },
    });
}

/* ============================================================
   HELPERS
   ============================================================ */
function fmt(v, decimals = 3) {
    if (v === null || v === undefined || isNaN(v)) return '—';
    return Number(v).toFixed(decimals);
}

function fmtPct(v) {
    if (v === null || v === undefined || isNaN(v)) return '—';
    return (Number(v) * 100).toFixed(1) + '%';
}

function fmtBig(v) {
    if (v === undefined || v === null) return '—';
    return Number(v).toLocaleString();
}

function scoreClass(v) {
    if (v >= 0.75) return 'good';
    if (v >= 0.55) return 'mid';
    return 'low';
}

function pillClass(v) {
    if (v >= 0.75) return 'pill-good';
    if (v >= 0.55) return 'pill-mid';
    return 'pill-low';
}

function colorForModel(key) {
    const map = {
        xgboost:             '#00c9ff',
        logistic_regression: '#a78bfa',
        rnn_lstm:            '#4ade80',
        qsofa_baseline:      '#fb923c',
    };
    return map[key] || '#94a3b8';
}

function labelForModel(key) {
    const map = {
        xgboost:             'XGBoost',
        logistic_regression: 'Logistic Regression',
        rnn_lstm:            'RNN-LSTM',
        qsofa_baseline:      'qSOFA Baseline',
    };
    return map[key] || key;
}

function iconForModel(key) {
    const map = {
        xgboost:             '🌳',
        logistic_regression: '📈',
        rnn_lstm:            '🔁',
        qsofa_baseline:      '🏥',
    };
    return map[key] || '🤖';
}

function subtitleForModel(key) {
    const map = {
        xgboost:             'Gradient Boosted Trees',
        logistic_regression: 'Linear Classifier',
        rnn_lstm:            'Sequential Deep Learning',
        qsofa_baseline:      'Clinical Rule-Based Score',
    };
    return map[key] || '';
}

/* ============================================================
   LIGHTBOX
   ============================================================ */
function openLightbox(src) {
    const lb = document.createElement('div');
    lb.className = 'lightbox';
    lb.innerHTML = `
        <button class="lightbox-close" title="Close">✕</button>
        <img src="${src}" alt="Chart">
    `;
    lb.addEventListener('click', (e) => {
        if (e.target === lb || e.target.classList.contains('lightbox-close')) {
            lb.remove();
        }
    });
    document.body.appendChild(lb);
}

/* ============================================================
   SIDEBAR ACTIVE LINK
   ============================================================ */
function initSidebarScroll() {
    const sections = ['overview', 'models', 'interactive', 'explainability', 'charts', 'runner'];
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                sections.forEach(id => {
                    const link = document.getElementById(`nav-${id}`);
                    if (link) link.classList.toggle('active', id === entry.target.id);
                });
            }
        });
    }, { threshold: 0.3 });
    sections.forEach(id => {
        const el = document.getElementById(id);
        if (el) observer.observe(el);
    });
}

/* ============================================================
   RENDER: DATASET STATS
   ============================================================ */
function renderStats(stats) {
    const grid = document.getElementById('stats-grid');
    const items = [
        { label: 'Total Patients',   value: fmtBig(stats.total_patients),  color: 'blue',   icon: '👤' },
        { label: 'ICU Stays',        value: fmtBig(stats.total_stays),      color: 'purple', icon: '🏥' },
        { label: 'Sepsis Stays',     value: fmtBig(stats.stays_with_sepsis),color: 'red',    icon: '🦠' },
        { label: 'Total Rows',       value: fmtBig(stats.total_rows),       color: 'green',  icon: '📊' },
        { label: 'Positive Rows',    value: fmtBig(stats.positive_rows),    color: 'orange', icon: '⚠️' },
        {
            label: 'Positive Rate',
            value: stats.positive_rate !== undefined ? (stats.positive_rate * 100).toFixed(3) + '%' : '—',
            color: 'orange',
            icon: '📉'
        },
    ];
    grid.innerHTML = items.map(item => `
        <div class="stat-card ${item.color}">
            <div class="stat-icon">${item.icon}</div>
            <div class="stat-value">${item.value}</div>
            <div class="stat-label">${item.label}</div>
        </div>
    `).join('');
}

/* ============================================================
   RENDER: MODEL METRIC CARDS
   ============================================================ */
function renderModelCards(models) {
    const grid = document.getElementById('models-grid');
    const order = ['xgboost', 'logistic_regression', 'rnn_lstm', 'qsofa_baseline'];

    grid.innerHTML = order.map(key => {
        const m = models[key];
        if (!m || Object.keys(m).length === 0) return '';

        const auroc  = m.auroc   ?? m.roc_auc ?? null;
        const auprc  = m.auprc   ?? m.pr_auc  ?? null;
        const sens   = m.sensitivity ?? null;
        const spec   = m.specificity ?? null;
        const prec   = m.precision   ?? null;
        const f1     = m.f1_score ?? m.f1 ?? null;

        const color  = colorForModel(key);

        return `
        <div class="model-card" style="--card-accent: ${color}22;">
            <div class="model-card-header">
                <div class="model-icon" style="background:${color}22; color:${color}; font-size:1.1rem;">
                    ${iconForModel(key)}
                </div>
                <div>
                    <div class="model-card-title">${labelForModel(key)}</div>
                    <div class="model-card-subtitle">${subtitleForModel(key)}</div>
                </div>
            </div>

            <div class="auroc-badge" style="color:${color}">
                ${auroc !== null ? fmt(auroc) : '—'}
                <span style="font-size:0.85rem; color:var(--text-muted); font-weight:500; margin-left:0.2rem;">AUROC</span>
            </div>

            <div class="metric-row">
                <span class="metric-name">AUPRC</span>
                <span class="metric-value ${auprc !== null ? scoreClass(auprc) : 'neutral'}">${fmt(auprc)}</span>
            </div>
            <div class="metric-row">
                <span class="metric-name">Sensitivity</span>
                <span class="metric-value ${sens !== null ? scoreClass(sens) : 'neutral'}">${fmtPct(sens)}</span>
            </div>
            <div class="metric-row">
                <span class="metric-name">Specificity</span>
                <span class="metric-value ${spec !== null ? scoreClass(spec) : 'neutral'}">${fmtPct(spec)}</span>
            </div>
            <div class="metric-row">
                <span class="metric-name">Precision</span>
                <span class="metric-value neutral">${fmtPct(prec)}</span>
            </div>
            <div class="metric-row">
                <span class="metric-name">F1 Score</span>
                <span class="metric-value ${f1 !== null ? scoreClass(f1) : 'neutral'}">${fmt(f1)}</span>
            </div>
            ${m.threshold !== undefined ? `
            <div class="metric-row">
                <span class="metric-name">Threshold</span>
                <span class="metric-value neutral">${fmt(m.threshold)}</span>
            </div>` : ''}
        </div>`;
    }).join('');
}

/* ============================================================
   RENDER: COMPARISON TABLE
   ============================================================ */
function renderComparisonTable(models) {
    const order = ['xgboost', 'logistic_regression', 'rnn_lstm', 'qsofa_baseline'];
    const headers = ['Model', 'AUROC', 'AUPRC', 'Sensitivity', 'Specificity', 'F1 Score'];

    const table = document.getElementById('comparison-table');
    const thead = `<thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead>`;

    const rows = order.map(key => {
        const m = models[key];
        if (!m || Object.keys(m).length === 0) return '';
        const auroc = m.auroc ?? m.roc_auc ?? null;
        const auprc = m.auprc ?? m.pr_auc ?? null;
        const sens  = m.sensitivity ?? null;
        const spec  = m.specificity ?? null;
        const f1    = m.f1_score ?? m.f1 ?? null;
        const color = colorForModel(key);

        return `<tr>
            <td>
                <div class="model-name-cell">
                    <span class="model-dot" style="background:${color}; box-shadow:0 0 6px ${color};"></span>
                    ${labelForModel(key)}
                </div>
            </td>
            <td><span class="pill-value ${auroc !== null ? pillClass(auroc) : 'pill-neutral'}">${fmt(auroc)}</span></td>
            <td><span class="pill-value ${auprc !== null ? pillClass(auprc) : 'pill-neutral'}">${fmt(auprc)}</span></td>
            <td>${fmtPct(sens)}</td>
            <td>${fmtPct(spec)}</td>
            <td>${fmt(f1)}</td>
        </tr>`;
    }).join('');

    table.innerHTML = `${thead}<tbody>${rows}</tbody>`;
}

/* ============================================================
   RENDER: FEATURE IMPORTANCE BARS
   ============================================================ */
function renderBars(containerId, items, nameKey, valueKey, color) {
    const el = document.getElementById(containerId);
    if (!items || items.length === 0) {
        el.innerHTML = '<p style="color:var(--text-muted); font-size:0.82rem;">No data available.</p>';
        return;
    }
    const max = Math.max(...items.map(i => i[valueKey]));
    el.innerHTML = items.slice(0, 15).map(item => {
        const pct = ((item[valueKey] / max) * 100).toFixed(1);
        return `
        <div class="feat-bar-row">
            <span class="feat-name" title="${item[nameKey]}">${item[nameKey]}</span>
            <div class="feat-bar-bg">
                <div class="feat-bar-fill" style="width:${pct}%; background:${color};"></div>
            </div>
            <span class="feat-score">${Number(item[valueKey]).toFixed(4)}</span>
        </div>`;
    }).join('');
}

/* ============================================================
   RENDER: IMAGE GALLERY
   ============================================================ */
function renderImages(containerId, imageMap, keys, labels) {
    const el = document.getElementById(containerId);
    const items = keys
        .filter(k => imageMap[k])
        .map(k => ({ src: API + imageMap[k], label: labels[k] || k }));

    if (items.length === 0) {
        el.innerHTML = '<p style="color:var(--text-muted); font-size:0.82rem;">No images available from backend.</p>';
        return;
    }

    el.innerHTML = items.map(({ src, label }) => `
        <div class="img-frame" onclick="openLightbox('${src}')">
            <img src="${src}" alt="${label}" loading="lazy">
            <div class="img-frame-label">${label}</div>
        </div>
    `).join('');
}


/* ============================================================
   RUNNER — state
   ============================================================ */
let activeReader   = null;   // ReadableStreamDefaultReader
let elapsedTimer   = null;   // setInterval handle
let runStartTime   = null;
let lineCount      = 0;
let activeScriptKey = null;

/* ============================================================
   RENDER: SCRIPT CARDS
   ============================================================ */
async function renderScriptCards() {
    const grid = document.getElementById('runner-grid');
    try {
        const res     = await fetch(`${API}/api/scripts`);
        const scripts = await res.json();

        grid.innerHTML = scripts.map(s => `
            <div class="script-card" style="--script-color: ${s.color};">
                <div class="script-card-header">
                    <span class="script-icon">${s.icon}</span>
                    <span class="script-label">${s.label}</span>
                    <span class="script-est">${s.estimated}</span>
                </div>
                <p class="script-desc">${s.description}</p>
                <button
                    class="script-run-btn"
                    id="run-btn-${s.key}"
                    onclick="startRun('${s.key}', '${s.label}', '${s.color}')"
                    ${s.available ? '' : 'disabled title="Script file not found"'}
                >
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                    ${s.available ? 'Run' : 'Unavailable'}
                </button>
            </div>
        `).join('');
    } catch {
        grid.innerHTML = '<p style="color:var(--text-muted);font-size:0.83rem;">Could not load script list from backend.</p>';
    }
}

/* ============================================================
   RUNNER — Terminal helpers
   ============================================================ */
function classifyLine(text) {
    if (/─{3,}/.test(text))                          return 'tline-sep';
    if (/\[EXIT:\d+\]/.test(text))                   return 'tline-sep';
    if (/✅|completed successfully|saved|done/i.test(text)) return 'tline-success';
    if (/❌|error|traceback|exception|failed/i.test(text))  return 'tline-error';
    if (/warning|warn/i.test(text))                  return 'tline-warn';
    if (/▶|⏱|📂|step|loading|starting/i.test(text)) return 'tline-info';
    if (/^─+$/.test(text.trim()))                    return 'tline-dim';
    return 'tline-normal';
}

function appendLine(text) {
    const body = document.getElementById('terminal-body');
    // Remove cursor if present
    const cursor = body.querySelector('.term-cursor');
    if (cursor) cursor.remove();

    const span = document.createElement('span');
    span.className = `tline ${classifyLine(text)}`;
    span.textContent = text;
    body.appendChild(span);

    // Re-add blinking cursor
    const cur = document.createElement('span');
    cur.className = 'term-cursor';
    body.appendChild(cur);

    lineCount++;
    document.getElementById('status-lines').textContent = `${lineCount} lines`;

    // Auto-scroll to bottom
    body.scrollTop = body.scrollHeight;
}

function setRunningState(isRunning, scriptKey) {
    const stopBtn = document.getElementById('btn-stop');
    stopBtn.disabled = !isRunning;

    // Disable/enable all run buttons
    document.querySelectorAll('.script-run-btn').forEach(btn => {
        btn.disabled = isRunning;
        btn.classList.toggle('running', isRunning && btn.id === `run-btn-${scriptKey}`);
    });

    if (!isRunning) {
        document.getElementById('status-script').textContent = 'Idle';
        document.getElementById('status-elapsed').textContent = '';
        clearInterval(elapsedTimer);
    }
}

function startElapsedTimer() {
    runStartTime = Date.now();
    elapsedTimer = setInterval(() => {
        const s = Math.floor((Date.now() - runStartTime) / 1000);
        const m = Math.floor(s / 60);
        const ss = String(s % 60).padStart(2, '0');
        document.getElementById('status-elapsed').textContent = `${m}:${ss} elapsed`;
    }, 1000);
}

/* ============================================================
   RUNNER — Start / Stop
   ============================================================ */
async function startRun(scriptKey, label, color) {
    // Stop any existing run
    if (activeReader) {
        try { await activeReader.cancel(); } catch (_) {}
        activeReader = null;
    }

    activeScriptKey = scriptKey;
    lineCount = 0;

    // Reset terminal
    const body = document.getElementById('terminal-body');
    body.innerHTML = '';

    document.getElementById('terminal-title').textContent = `python src/${scriptKey}.py`;
    document.getElementById('status-script').textContent  = label;
    document.getElementById('status-lines').textContent   = '0 lines';

    setRunningState(true, scriptKey);
    startElapsedTimer();

    try {
        const res = await fetch(`${API}/api/run/${scriptKey}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const reader = res.body.getReader();
        activeReader = reader;
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Parse SSE — split on double-newline
            const parts = buffer.split('\n\n');
            buffer = parts.pop(); // last incomplete chunk stays in buffer

            for (const part of parts) {
                const line = part.startsWith('data: ')
                    ? part.slice(6)
                    : part;

                if (line.startsWith('[EXIT:')) {
                    const code = parseInt(line.match(/\[EXIT:(\d+)\]/)?.[1] ?? '1');
                    finishRun(code);
                    return;
                }
                if (line.trim()) appendLine(line);
            }
        }

        finishRun(0);

    } catch (err) {
        appendLine(`[ERROR] ${err.message}`);
        finishRun(1);
    }
}

async function finishRun(exitCode) {
    activeReader = null;
    clearInterval(elapsedTimer);

    // Remove blinking cursor
    const cursor = document.getElementById('terminal-body').querySelector('.term-cursor');
    if (cursor) cursor.remove();

    setRunningState(false, null);

    const elapsed = runStartTime
        ? Math.floor((Date.now() - runStartTime) / 1000)
        : 0;
    const m = Math.floor(elapsed / 60);
    const ss = String(elapsed % 60).padStart(2, '0');
    document.getElementById('status-elapsed').textContent =
        exitCode === 0 ? `✅ Done in ${m}:${ss}` : `❌ Failed after ${m}:${ss}`;

    // Auto-refresh results if successful
    if (exitCode === 0) {
        appendLine('');
        appendLine('🔄  Refreshing dashboard results…');
        await refreshResults();
        appendLine('✅  Dashboard updated with latest outputs.');
    }
}

async function refreshResults() {
    try {
        const res  = await fetch(`${API}/api/results/all`);
        if (!res.ok) return;
        const data = await res.json();

        // Refresh metric cards + charts
        renderModelCards(data.models || {});
        renderComparisonTable(data.models || {});

        // Destroy and redraw Chart.js charts
        Chart.helpers.each(Chart.instances, c => c.destroy());
        applyChartDefaults();
        renderBarChart('chart-auroc',    data.models || {}, 'auroc', 'AUROC', 1.0);
        renderBarChart('chart-auprc',    data.models || {}, 'auprc', 'AUPRC', 0.5);
        renderRadarChart('chart-radar',  data.models || {});
        renderSensSpecChart('chart-sens-spec', data.models || {});
        renderFeatureChart('chart-features', (data.shap || {}).top_20_features || []);

        // Refresh images
        const imgs = data.images || {};
        const imgLabels = {
            shap_summary_bar:           'SHAP Summary (Bar)',
            shap_summary_beeswarm:      'SHAP Summary (Beeswarm)',
            shap_waterfall_positive:    'SHAP Waterfall — Positive Case',
            shap_dependence_lactate:    'SHAP Dependence — Lactate',
            shap_dependence_sofa:       'SHAP Dependence — SOFA',
            xgboost_feature_importance: 'XGBoost Feature Importance',
            feature_importances:        'Feature Importances',
            roc_pr_curves:              'ROC & PR Curves',
            roc_pr_comparison:          'ROC / PR Comparison',
            confusion_matrices:         'Confusion Matrices',
            confusion_matrices_comparison: 'Confusion Matrix Comparison',
            xgboost_performance:        'XGBoost Performance Report',
            lr_performance:             'Logistic Regression Performance',
            rnn_performance:            'RNN Performance',
            lstm_performance:           'LSTM Performance',
            lstm_fold_aurocs:           'LSTM Fold AUROCs',
            baseline_performance:       'Baseline Performance',
            tabular_only_performance:   'Tabular Only (Ablation)',
            multimodal_performance:     'Multimodal (Ablation)',
            modality_only_performance:  'Modality Only (Ablation)',
        };
        // Add cache-bust to force browser to reload images
        const bust = `?t=${Date.now()}`;
        const bustedImgs = {};
        for (const [k, v] of Object.entries(imgs)) bustedImgs[k] = v + bust;

        renderImages('shap-images', bustedImgs,
            ['shap_summary_bar','shap_summary_beeswarm','shap_waterfall_positive','shap_dependence_lactate','shap_dependence_sofa'],
            imgLabels);
        renderImages('roc-images', bustedImgs,
            ['roc_pr_curves','roc_pr_comparison'], imgLabels);
        renderImages('cm-images', bustedImgs,
            ['confusion_matrices','confusion_matrices_comparison'], imgLabels);
        renderImages('perf-images', bustedImgs,
            ['xgboost_performance','lr_performance','rnn_performance','lstm_performance','lstm_fold_aurocs','baseline_performance','xgboost_feature_importance','feature_importances'],
            imgLabels);
        renderImages('ablation-images', bustedImgs,
            ['tabular_only_performance','multimodal_performance','modality_only_performance'],
            imgLabels);

        // Refresh SHAP bars
        const shap = data.shap || {};
        renderBars('shap-bars', shap.top_20_features || [], 'feature', 'mean_abs_shap', '#00c9ff');
        renderBars('perm-bars', data.feature_importance || [], 'feature', 'importance_mean', '#a78bfa');

    } catch (err) {
        console.warn('Auto-refresh failed:', err);
    }
}

// Stop button
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('btn-stop')?.addEventListener('click', async () => {
        if (activeReader) {
            try { await activeReader.cancel(); } catch (_) {}
            activeReader = null;
        }
        appendLine('');
        appendLine('⏹  Run stopped by user.');
        finishRun(1);
    });

    document.getElementById('btn-clear')?.addEventListener('click', () => {
        const body = document.getElementById('terminal-body');
        body.innerHTML = `
            <div class="terminal-placeholder">
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>
                <p>No output yet. Pick a script and hit <strong>Run</strong>.</p>
            </div>`;
        lineCount = 0;
        document.getElementById('status-lines').textContent = '0 lines';
    });
});

/* ============================================================
   MAIN
   ============================================================ */
async function init() {
    const loadingEl = document.getElementById('loading-state');
    const errorEl   = document.getElementById('error-state');
    const dashboard = document.getElementById('dashboard');
    const statusPill = document.getElementById('status-pill');
    const statusDot  = statusPill.querySelector('.status-dot');
    const statusText = document.getElementById('status-text');

    try {
        const res  = await fetch(`${API}/api/results/all`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        // Update status
        loadingEl.classList.add('hidden');
        statusDot.classList.remove('loading');
        statusDot.classList.add('ok');
        statusText.textContent = 'Backend Connected';

        // 1. Dataset Stats
        renderStats(data.dataset_stats || {});

        // 2. Model Cards + comparison table
        renderModelCards(data.models || {});
        renderComparisonTable(data.models || {});

        // 3. Interactive Charts (Chart.js)
        applyChartDefaults();
        renderBarChart('chart-auroc',    data.models || {}, 'auroc',   'AUROC',  1.0);
        renderBarChart('chart-auprc',    data.models || {}, 'auprc',   'AUPRC',  0.5);
        renderRadarChart('chart-radar',  data.models || {});
        renderSensSpecChart('chart-sens-spec', data.models || {});
        renderFeatureChart('chart-features', (data.shap || {}).top_20_features || []);

        // 3. SHAP / Feature importance
        const shap = data.shap || {};
        renderBars('shap-bars', shap.top_20_features || [], 'feature', 'mean_abs_shap', '#00c9ff');

        if (shap.computation_date || shap.sample_size) {
            document.getElementById('shap-meta').textContent =
                `Computed on ${shap.sample_size?.toLocaleString()} samples · ${shap.positive_samples} positives · ${shap.total_features} features · Date: ${shap.computation_date || 'N/A'}`;
        }

        renderBars('perm-bars', data.feature_importance || [], 'feature', 'importance_mean', '#a78bfa');

        // 4. Images
        const imgs = data.images || {};
        const imgLabels = {
            shap_summary_bar:         'SHAP Summary (Bar)',
            shap_summary_beeswarm:    'SHAP Summary (Beeswarm)',
            shap_waterfall_positive:  'SHAP Waterfall — Positive Case',
            shap_dependence_lactate:  'SHAP Dependence — Lactate',
            shap_dependence_sofa:     'SHAP Dependence — SOFA',
            xgboost_feature_importance: 'XGBoost Feature Importance',
            feature_importances:       'Feature Importances',
            roc_pr_curves:             'ROC & PR Curves',
            roc_pr_comparison:         'ROC / PR Comparison',
            confusion_matrices:        'Confusion Matrices',
            confusion_matrices_comparison: 'Confusion Matrix Comparison',
            xgboost_performance:       'XGBoost Performance Report',
            lr_performance:            'Logistic Regression Performance',
            rnn_performance:           'RNN Performance',
            lstm_performance:          'LSTM Performance',
            lstm_fold_aurocs:          'LSTM Fold AUROCs',
            baseline_performance:      'Baseline Performance',
            tabular_only_performance:  'Tabular Only (Ablation)',
            multimodal_performance:    'Multimodal (Ablation)',
            modality_only_performance: 'Modality Only (Ablation)',
        };

        renderImages('shap-images', imgs,
            ['shap_summary_bar','shap_summary_beeswarm','shap_waterfall_positive','shap_dependence_lactate','shap_dependence_sofa'],
            imgLabels);

        renderImages('roc-images', imgs,
            ['roc_pr_curves','roc_pr_comparison'],
            imgLabels);

        renderImages('cm-images', imgs,
            ['confusion_matrices','confusion_matrices_comparison'],
            imgLabels);

        renderImages('perf-images', imgs,
            ['xgboost_performance','lr_performance','rnn_performance','lstm_performance','lstm_fold_aurocs','baseline_performance','xgboost_feature_importance','feature_importances'],
            imgLabels);

        renderImages('ablation-images', imgs,
            ['tabular_only_performance','multimodal_performance','modality_only_performance'],
            imgLabels);

        // 5. Script runner cards
        await renderScriptCards();

        // Show dashboard & init sidebar scroll
        dashboard.classList.remove('hidden');
        initSidebarScroll();

    } catch (err) {
        console.error('API error:', err);
        loadingEl.classList.add('hidden');
        errorEl.classList.remove('hidden');
        statusDot.classList.remove('loading');
        statusDot.classList.add('error');
        statusText.textContent = 'Backend Offline';
    }
}

document.addEventListener('DOMContentLoaded', init);
