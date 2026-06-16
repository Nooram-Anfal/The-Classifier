/* script.js — The Classifier, full multi-tab JS */

// ── State ─────────────────────────────────────────────────────────────────────
let activeModel  = "naive_bayes";
let evalData     = {};
let evalModel    = "naive_bayes";
let batchResults = [];
let fnROCChart   = null;  // Chart.js instance for ROC curve

const MODEL_INFO = {
  naive_bayes:          { label: "Naive Bayes",         desc: "Multinomial Naive Bayes. Fast, probabilistic, ideal for sparse TF-IDF features." },
  logistic_regression:  { label: "Logistic Regression", desc: "Linear classifier with L2 regularisation. Strong baseline, well-calibrated probabilities." },
  linear_svc:           { label: "Linear SVC",          desc: "Support Vector Classifier with linear kernel. Maximises classification margin." },
  sgd:                  { label: "SGD Classifier",       desc: "Stochastic Gradient Descent with log-loss. Fast and scalable to large corpora." },
};

const CAT_ICONS = {
  "World":    '<i class="fa-solid fa-earth-americas"></i>',
  "Sports":   '<i class="fa-solid fa-trophy"></i>',
  "Business": '<i class="fa-solid fa-building-columns"></i>',
  "Sci/Tech": '<i class="fa-solid fa-flask"></i>',
};

// ── Auth helpers ──────────────────────────────────────────────────────────────
function getToken() {
  return localStorage.getItem("token") || null;
}

function authHeaders() {
  const token = getToken();
  return token
    ? { "Authorization": "Bearer " + token, "Content-Type": "application/json" }
    : { "Content-Type": "application/json" };
}

function isLoggedIn() {
  // Check server-injected value first, then fall back to localStorage
  if (typeof isAuthenticated !== "undefined" && isAuthenticated === true) return true;
  return !!getToken();
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  setupTabs();
  setupModelPills();
  setupClassify();
  setupBatch();
  setupFakeNews();
  setDate();
  updateAuthUI();
  evalData = await fetchEval();
  buildEvalTab(evalData);
});

// ── Date ──────────────────────────────────────────────────────────────────────
function setDate() {
  const el = document.getElementById("liveDate");
  if (el) el.textContent = new Date().toLocaleDateString("en-GB", {
    weekday: "long", year: "numeric", month: "long", day: "numeric"
  });
}

// ── Auth UI ───────────────────────────────────────────────────────────────────
function updateAuthUI() {
  const fakeTab = document.getElementById("fakeNewsTab");
  const nav     = document.querySelector(".masthead-nav");

  if (isLoggedIn()) {
    // Show Fake News tab
    if (fakeTab) fakeTab.style.display = "";

    // Add logout button to nav if not already there
    if (nav && !document.getElementById("authNavBtn")) {
      const logoutBtn = document.createElement("a");
      logoutBtn.id        = "authNavBtn";
      logoutBtn.className = "nav-auth-link";
      logoutBtn.href      = "#";
      logoutBtn.innerHTML = '<i class="fa-solid fa-right-from-bracket"></i> Logout';
      logoutBtn.addEventListener("click", e => {
        e.preventDefault();
        localStorage.removeItem("token");
        localStorage.removeItem("token_type");
        window.location.reload();
      });
      nav.appendChild(logoutBtn);
    }
  } else {
    // Hide Fake News tab
    if (fakeTab) fakeTab.style.display = "none";

    // Add login link to nav
    if (nav && !document.getElementById("authNavBtn")) {
      const loginBtn = document.createElement("a");
      loginBtn.id        = "authNavBtn";
      loginBtn.className = "nav-auth-link";
      loginBtn.href      = "/auth/login";
      loginBtn.innerHTML = '<i class="fa-solid fa-right-to-bracket"></i> Login';
      nav.appendChild(loginBtn);
    }
  }
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
function setupTabs() {
  document.querySelectorAll(".nav-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;

      // Guard: if fake-news tab clicked and not logged in → redirect
      if (tab === "fake-news" && !isLoggedIn()) {
        window.location.href = "/auth/login";
        return;
      }

      document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(t => t.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById("tab-" + tab).classList.add("active");

      // Load metrics + history when fake-news tab is opened
      if (tab === "fake-news") {
        loadFakeNewsMetrics();
        loadFakeNewsHistory();
      }
    });
  });
}

// ── Model pills ───────────────────────────────────────────────────────────────
function setupModelPills() {
  document.querySelectorAll(".mpill").forEach(pill => {
    pill.addEventListener("click", () => {
      document.querySelectorAll(".mpill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      activeModel = pill.dataset.key;
      document.getElementById("modelDesc").textContent = MODEL_INFO[activeModel].desc;
    });
  });
}

// ── Classify ──────────────────────────────────────────────────────────────────
function setupClassify() {
  const textarea  = document.getElementById("newsInput");
  const charCount = document.getElementById("charCount");
  const btn       = document.getElementById("classifyBtn");
  const errorMsg  = document.getElementById("errorMsg");

  textarea.addEventListener("input", () => { charCount.textContent = textarea.value.length; });
  textarea.addEventListener("keydown", e => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") btn.click();
  });

  document.querySelectorAll(".eg-btn").forEach(b => {
    b.addEventListener("click", () => {
      textarea.value = b.dataset.text;
      charCount.textContent = textarea.value.length;
      textarea.focus();
    });
  });

  btn.addEventListener("click", async () => {
    const text = textarea.value.trim();
    errorMsg.style.display = "none";
    if (!text) return showError(errorMsg, "Please enter some news text.");
    if (text.length < 10) return showError(errorMsg, "Please enter at least 10 characters.");

    setLoading(btn, true, "Analysing...", "Classify Article");
    try {
      const r = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model: activeModel }),
      });
      const data = await r.json();
      if (!r.ok) return showError(errorMsg, data.detail || data.error || "Server error.");
      renderResult(data);
    } catch (e) {
      showError(errorMsg, "Network error — is the server running?");
    } finally {
      setLoading(btn, false, "Analysing...", "Classify Article");
    }
  });
}

function renderResult(data) {
  document.getElementById("idleWrap").style.display   = "none";
  document.getElementById("resultWrap").style.display = "block";

  const banner = document.getElementById("verdictBanner");
  banner.dataset.stamp = MODEL_INFO[data.model_key]?.short || "NB";
  document.getElementById("verdictKicker").textContent  = "CLASSIFIED AS";
  document.getElementById("verdictHeadline").innerHTML  = (CAT_ICONS[data.category] || "") + " " + data.category;
  document.getElementById("verdictByline").textContent  =
    `Model: ${data.model_label} · Confidence: ${data.confidence}%`;

  document.getElementById("confPct").textContent = data.confidence + "%";
  requestAnimationFrame(() => requestAnimationFrame(() => {
    document.getElementById("confFill").style.width = data.confidence + "%";
  }));

  const probRows = document.getElementById("probRows");
  probRows.innerHTML = "";
  const sorted = Object.entries(data.all_probs).sort((a,b) => b[1]-a[1]);
  sorted.forEach(([cls, pct]) => {
    const isWin = cls === data.category;
    const row   = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span class="prob-name">${CAT_ICONS[cls] || ""} ${cls}</span>
      <div class="prob-track"><div class="prob-fill${isWin?" winner":""}" style="width:0" data-w="${pct}"></div></div>
      <span class="prob-pct${isWin?" winner":""}">${pct}%</span>
    `;
    probRows.appendChild(row);
  });
  requestAnimationFrame(() => requestAnimationFrame(() => {
    probRows.querySelectorAll(".prob-fill").forEach(b => { b.style.width = b.dataset.w + "%"; });
  }));

  const cloud = document.getElementById("keywordCloud");
  cloud.innerHTML = "";
  const maxScore = data.keywords[0]?.score || 1;
  data.keywords.forEach(kw => {
    const pct  = kw.score / maxScore;
    const tier = pct > 0.66 ? "tier-1" : pct > 0.33 ? "tier-2" : "tier-3";
    const tag  = document.createElement("span");
    tag.className   = `kw-tag ${tier}`;
    tag.textContent = kw.word;
    tag.title       = `TF-IDF: ${kw.score}`;
    cloud.appendChild(tag);
  });
}

// ── Batch ─────────────────────────────────────────────────────────────────────
function setupBatch() {
  const btn       = document.getElementById("batchBtn");
  const exportBtn = document.getElementById("exportBtn");
  const errorEl   = document.getElementById("batchError");

  btn.addEventListener("click", async () => {
    const raw = document.getElementById("batchInput").value.trim();
    errorEl.style.display = "none";
    if (!raw) return showError(errorEl, "Please enter at least one line.");

    setLoading(btn, true, "Analysing...", "Classify All");
    try {
      const r = await fetch("/api/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: raw, model: activeModel }),
      });
      const data = await r.json();
      if (!r.ok) return showError(errorEl, data.detail || data.error || "Server error.");
      batchResults = data;
      renderBatch(data);
      exportBtn.style.display = "flex";
    } catch(e) {
      showError(errorEl, "Network error.");
    } finally {
      setLoading(btn, false, "Analysing...", "Classify All");
    }
  });

  exportBtn.addEventListener("click", () => exportCSV(batchResults));
}

function renderBatch(results) {
  const container = document.getElementById("batchResults");
  container.innerHTML = "";

  const header = document.createElement("div");
  header.className = "batch-header";
  header.innerHTML = `<span>ARTICLE</span><span>CATEGORY</span><span>CONF.</span>`;
  container.appendChild(header);

  results.forEach(r => {
    const row = document.createElement("div");
    row.className = "batch-row";
    row.innerHTML = `
      <span class="b-text" title="${esc(r.input)}">${esc(r.input)}</span>
      <span class="b-cat">${CAT_ICONS[r.category]||""} ${r.category}</span>
      <span class="b-conf">${r.confidence}%</span>
    `;
    container.appendChild(row);
  });
}

function exportCSV(results) {
  const header = "Input,Category,Confidence,Model\n";
  const rows   = results.map(r =>
    `"${r.input.replace(/"/g,'""')}","${r.category}","${r.confidence}%","${r.model_label}"`
  ).join("\n");
  const blob   = new Blob([header + rows], { type: "text/csv" });
  const url    = URL.createObjectURL(blob);
  const a      = Object.assign(document.createElement("a"), { href: url, download: "classifications.csv" });
  a.click();
  URL.revokeObjectURL(url);
}

// ── Evaluation ────────────────────────────────────────────────────────────────
async function fetchEval() {
  try {
    const r = await fetch("/api/evaluation");
    return await r.json();
  } catch(e) { return {}; }
}

function buildEvalTab(data) {
  if (!data || !Object.keys(data).length) return;

  const keys = Object.keys(data);
  const accs = keys.map(k => data[k].accuracy);
  const best = Math.max(...accs);

  // ── Cards ──────────────────────────────────────────────────────────────────
  const cards = document.getElementById("evalCards");
  cards.innerHTML = "";
  keys.forEach(k => {
    const d    = data[k];
    const card = document.createElement("div");
    card.className = "eval-card" + (k === "naive_bayes" ? " selected" : "");
    card.innerHTML = `
      <div class="eval-card-model">${MODEL_INFO[k]?.label || k}</div>
      <div class="eval-card-acc">${d.accuracy}%</div>
      <div class="eval-card-label">ACCURACY</div>
    `;
    card.addEventListener("click", () => {
      document.querySelectorAll(".eval-card").forEach(c => c.classList.remove("selected"));
      card.classList.add("selected");
      evalModel = k;
      renderPerClassMetrics(data, k);
      renderConfMatrix(data, k);
    });
    cards.appendChild(card);
  });

  // ── Accuracy chart ─────────────────────────────────────────────────────────
  const accChart = document.getElementById("accChart");
  accChart.innerHTML = "";
  keys.forEach(k => {
    const d   = data[k];
    const row = document.createElement("div");
    row.className = "acc-row";
    const isBest = d.accuracy === best;
    row.innerHTML = `
      <span class="acc-label">${MODEL_INFO[k]?.label || k}</span>
      <div class="acc-bar-track">
        <div class="acc-bar-fill${isBest?" best":""}" style="width:0" data-w="${d.accuracy}"></div>
      </div>
      <span class="acc-val">${d.accuracy}%</span>
    `;
    accChart.appendChild(row);
  });
  setTimeout(() => {
    accChart.querySelectorAll(".acc-bar-fill").forEach(b => {
      b.style.width = (parseFloat(b.dataset.w) / 100 * 100) + "%";
    });
  }, 100);

  // ── Per-class model selector ───────────────────────────────────────────────
  const sel = document.getElementById("evalModelSelect");
  sel.innerHTML = "";
  keys.forEach(k => {
    const opt = document.createElement("option");
    opt.value = k;
    opt.textContent = MODEL_INFO[k]?.label || k;
    sel.appendChild(opt);
  });
  sel.addEventListener("change", () => {
    evalModel = sel.value;
    renderPerClassMetrics(data, evalModel);
    renderConfMatrix(data, evalModel);
  });

  renderPerClassMetrics(data, "naive_bayes");
  renderConfMatrix(data, "naive_bayes");
  buildF1Chart(data);
}

function renderPerClassMetrics(data, key) {
  const report = data[key]?.report || {};
  const table  = document.getElementById("metricsTable");
  const cats   = ["World", "Sports", "Business", "Sci/Tech"];
  table.innerHTML = `
    <tr>
      <th>Category</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1-Score</th>
      <th>Support</th>
    </tr>
  `;
  cats.forEach(cat => {
    const m = report[cat] || {};
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="cat">${CAT_ICONS[cat]||""} ${cat}</td>
      <td class="num">${((m.precision||0)*100).toFixed(1)}%</td>
      <td class="num">${((m.recall||0)*100).toFixed(1)}%</td>
      <td class="num">${((m["f1-score"]||0)*100).toFixed(1)}%</td>
      <td class="num">${m.support||0}</td>
    `;
    table.appendChild(tr);
  });
  const ma = report["macro avg"] || {};
  const maRow = document.createElement("tr");
  maRow.style.borderTop = "2px solid var(--rule)";
  maRow.innerHTML = `
    <td class="cat" style="font-style:italic">Macro Avg</td>
    <td class="num">${((ma.precision||0)*100).toFixed(1)}%</td>
    <td class="num">${((ma.recall||0)*100).toFixed(1)}%</td>
    <td class="num">${((ma["f1-score"]||0)*100).toFixed(1)}%</td>
    <td class="num"></td>
  `;
  table.appendChild(maRow);
}

function renderConfMatrix(data, key) {
  const d      = data[key];
  if (!d) return;
  const cm     = d.confusion;
  const labels = d.labels;
  const wrap   = document.getElementById("cmWrap");
  wrap.innerHTML = "";

  const maxVal = Math.max(...cm.flat());
  const table  = document.createElement("table");
  table.className = "cm-table";

  const thead = table.createTHead();
  const hrow  = thead.insertRow();
  hrow.insertCell().innerHTML = `<th class="row-header">Actual \\ Pred.</th>`;
  labels.forEach(l => {
    const th = document.createElement("th");
    th.innerHTML = `${CAT_ICONS[l]||""} ${l}`;
    hrow.appendChild(th);
  });

  const tbody = table.createTBody();
  cm.forEach((row, i) => {
    const tr = tbody.insertRow();
    const lh = document.createElement("th");
    lh.className = "row-header";
    lh.innerHTML = `${CAT_ICONS[labels[i]]||""} ${labels[i]}`;
    tr.appendChild(lh);
    row.forEach((val, j) => {
      const td      = tr.insertCell();
      td.textContent = val;
      const isDiag  = i === j;
      td.className  = isDiag ? "diag" : "";
      const alpha   = (val / maxVal);
      if (isDiag) {
        td.style.background = `rgba(13,13,13,${alpha * 0.25 + 0.08})`;
        td.style.fontWeight = "700";
      } else if (val > 0) {
        td.style.background = `rgba(139,26,26,${alpha * 0.18})`;
        td.style.color      = alpha > 0.4 ? "#8b1a1a" : "inherit";
      }
    });
  });

  wrap.appendChild(table);
}

function buildF1Chart(data) {
  const cats = ["World", "Sports", "Business", "Sci/Tech"];
  const wrap = document.getElementById("f1Chart");
  wrap.innerHTML = "";

  cats.forEach(cat => {
    const group = document.createElement("div");
    group.className = "f1-group";
    group.innerHTML = `<div class="f1-group-label">${CAT_ICONS[cat]||""} ${cat}</div><div class="f1-bars" id="f1-${cat.replace("/","")}"></div>`;
    wrap.appendChild(group);

    const barsEl = group.querySelector(".f1-bars");
    const vals   = Object.keys(data).map(k => ({
      key: k,
      f1: (data[k].report?.[cat]?.["f1-score"] || 0) * 100,
    }));
    const maxF1 = Math.max(...vals.map(v => v.f1));

    vals.forEach(v => {
      const row = document.createElement("div");
      row.className = "f1-row";
      const isBest = v.f1 === maxF1;
      row.innerHTML = `
        <span class="f1-label">${MODEL_INFO[v.key]?.label || v.key}</span>
        <div class="f1-track"><div class="f1-fill${isBest?" best":""}" style="width:0" data-w="${v.f1}"></div></div>
        <span class="f1-val">${v.f1.toFixed(1)}%</span>
      `;
      barsEl.appendChild(row);
    });
  });

  setTimeout(() => {
    wrap.querySelectorAll(".f1-fill").forEach(b => {
      b.style.width = parseFloat(b.dataset.w) + "%";
    });
  }, 150);
}

// ── Fake News Detection ───────────────────────────────────────────────────────
function setupFakeNews() {
  const textarea  = document.getElementById("fakeNewsInput");
  const charCount = document.getElementById("fnCharCount");
  const btn       = document.getElementById("fnAnalyseBtn");
  const errorMsg  = document.getElementById("fnErrorMsg");

  if (!textarea || !btn) return; // guard if elements don't exist

  textarea.addEventListener("input", () => { charCount.textContent = textarea.value.length; });

  btn.addEventListener("click", async () => {
    const text = textarea.value.trim();
    errorMsg.style.display = "none";
    if (!text) return showError(errorMsg, "Please enter a news article or headline.");
    if (text.length < 10) return showError(errorMsg, "Please enter at least 10 characters.");

    if (!isLoggedIn()) {
      showError(errorMsg, "You must be logged in to use Fake News Detection. Redirecting...");
      setTimeout(() => { window.location.href = "/auth/login"; }, 1500);
      return;
    }

    setLoading(btn, true, "Analysing...", "Analyse");
    try {
      const r = await fetch("/api/fake-news/predict", {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ text }),
      });

      if (r.status === 401) {
        localStorage.removeItem("token");
        window.location.href = "/auth/login";
        return;
      }

      const data = await r.json();
      if (!r.ok) return showError(errorMsg, data.detail || "Server error.");
      renderFakeNewsResult(data);
      // Refresh history after successful prediction
      loadFakeNewsHistory();
    } catch(e) {
      showError(errorMsg, "Network error — is the server running?");
    } finally {
      setLoading(btn, false, "Analysing...", "Analyse");
    }
  });
}

function renderFakeNewsResult(data) {
  const resultEl = document.getElementById("fnResult");
  resultEl.style.display = "block";

  // Verdict headline — data-verdict drives CSS colour (ink vs stamp-red)
  const headlineEl = document.getElementById("fnVerdictHeadline");
  headlineEl.textContent     = data.label_text.toUpperCase();
  headlineEl.dataset.verdict = data.label_text;

  document.getElementById("fnVerdictByline").textContent =
    `Logistic Regression · Confidence: ${data.confidence}%`;

  // Confidence bar
  document.getElementById("fnConfPct").textContent = data.confidence + "%";
  requestAnimationFrame(() => requestAnimationFrame(() => {
    document.getElementById("fnConfFill").style.width = data.confidence + "%";
  }));

  // Prob rows (Real / Fake breakdown)
  const probRows = document.getElementById("fnProbRows");
  probRows.innerHTML = "";
  Object.entries(data.all_probs).forEach(([cls, pct]) => {
    const isWin = cls === data.label_text;
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span class="prob-name">${cls}</span>
      <div class="prob-track"><div class="prob-fill${isWin?" winner":""}" style="width:0" data-w="${pct}"></div></div>
      <span class="prob-pct${isWin?" winner":""}">${pct}%</span>
    `;
    probRows.appendChild(row);
  });
  requestAnimationFrame(() => requestAnimationFrame(() => {
    probRows.querySelectorAll(".prob-fill").forEach(b => { b.style.width = b.dataset.w + "%"; });
  }));

  // Top 5 keywords
  const cloud = document.getElementById("fnKeywordCloud");
  cloud.innerHTML = "";
  const maxScore = data.keywords[0]?.score || 1;
  (data.keywords || []).slice(0, 5).forEach(kw => {
    const pct  = kw.score / maxScore;
    const tier = pct > 0.66 ? "tier-1" : pct > 0.33 ? "tier-2" : "tier-3";
    const tag  = document.createElement("span");
    tag.className   = `kw-tag ${tier}`;
    tag.textContent = kw.word;
    tag.title       = `TF-IDF: ${kw.score}`;
    cloud.appendChild(tag);
  });
}

// ── Fake News Metrics ─────────────────────────────────────────────────────────
async function loadFakeNewsMetrics() {
  if (!isLoggedIn()) return;

  try {
    const r = await fetch("/api/fake-news/metrics", { headers: authHeaders() });
    if (r.status === 401) { localStorage.removeItem("token"); return; }
    if (!r.ok) return;

    const data = await r.json();

    // Metric cards
    const setCard = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = val !== undefined ? val.toFixed(1) : "—";
    };
    setCard("fnAccuracy",  data.accuracy);
    setCard("fnPrecision", data.precision);
    setCard("fnRecall",    data.recall);
    setCard("fnF1Score",   data.f1_score);

    // 2×2 confusion matrix
    if (data.confusion_matrix && data.labels) {
      renderFakeConfMatrix(data.confusion_matrix, data.labels);
    }

    // ROC curve
    if (data.roc_curve) {
      renderROCCurve(data.roc_curve, data.roc_auc);
    }
  } catch(e) {
    console.warn("Failed to load fake news metrics:", e);
  }
}

function renderFakeConfMatrix(cm, labels) {
  if (!cm || !labels) return;
  const wrap = document.getElementById("fnCMWrap");
  if (!wrap) return;
  wrap.innerHTML = "";

  const maxVal = Math.max(...cm.flat());
  const table  = document.createElement("table");
  table.className = "cm-table";

  const thead = table.createTHead();
  const hrow  = thead.insertRow();
  hrow.insertCell().innerHTML = `<th class="row-header">Actual \\ Pred.</th>`;
  labels.forEach(l => {
    const th = document.createElement("th");
    th.textContent = l;
    hrow.appendChild(th);
  });

  const tbody = table.createTBody();
  cm.forEach((row, i) => {
    const tr = tbody.insertRow();
    const lh = document.createElement("th");
    lh.className   = "row-header";
    lh.textContent = labels[i];
    tr.appendChild(lh);
    row.forEach((val, j) => {
      const td     = tr.insertCell();
      td.textContent = val.toLocaleString();
      const isDiag = i === j;
      td.className = isDiag ? "diag" : "";
      const alpha  = val / maxVal;
      if (isDiag) {
        td.style.background = `rgba(13,13,13,${alpha * 0.25 + 0.08})`;
        td.style.fontWeight = "700";
      } else if (val > 0) {
        td.style.background = `rgba(139,26,26,${alpha * 0.18})`;
        td.style.color      = alpha > 0.4 ? "#8b1a1a" : "inherit";
      }
    });
  });

  wrap.appendChild(table);
}

function renderROCCurve(rocData, aucScore) {
  const canvas = document.getElementById("fnROCChart");
  if (!canvas || !window.Chart) return;

  // Destroy previous instance to avoid overlay
  if (fnROCChart) { fnROCChart.destroy(); fnROCChart = null; }

  // Downsample to ≤200 points for performance
  const total  = rocData.fpr.length;
  const step   = Math.max(1, Math.floor(total / 200));
  const points = [];
  for (let i = 0; i < total; i += step) {
    points.push({ x: rocData.fpr[i], y: rocData.tpr[i] });
  }

  fnROCChart = new Chart(canvas, {
    type: "line",
    data: {
      datasets: [
        {
          label: `ROC Curve (AUC = ${aucScore})`,
          data: points,
          borderColor: "#0d0d0d",
          borderWidth: 2,
          pointRadius: 0,
          fill: false,
          tension: 0.1,
        },
        {
          label: "Random Classifier (AUC = 0.50)",
          data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
          borderColor: "#c8c0b0",
          borderWidth: 1,
          borderDash: [4, 4],
          pointRadius: 0,
          fill: false,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      scales: {
        x: {
          type: "linear",
          title: {
            display: true,
            text: "False Positive Rate",
            font: { family: "'IBM Plex Mono', monospace", size: 11 }
          },
          min: 0, max: 1,
          grid: { color: "#e4ddd2" },
          ticks: { font: { family: "'IBM Plex Mono', monospace", size: 10 } }
        },
        y: {
          type: "linear",
          title: {
            display: true,
            text: "True Positive Rate",
            font: { family: "'IBM Plex Mono', monospace", size: 11 }
          },
          min: 0, max: 1,
          grid: { color: "#e4ddd2" },
          ticks: { font: { family: "'IBM Plex Mono', monospace", size: 10 } }
        }
      },
      plugins: {
        legend: {
          labels: {
            font: { family: "'IBM Plex Mono', monospace", size: 11 },
            color: "#2e2e2e"
          }
        },
        tooltip: { enabled: false }
      }
    }
  });
}

// ── Fake News History ─────────────────────────────────────────────────────────
async function loadFakeNewsHistory() {
  if (!isLoggedIn()) return;

  try {
    const r = await fetch("/api/fake-news/history", { headers: authHeaders() });
    if (r.status === 401) { localStorage.removeItem("token"); return; }
    if (!r.ok) return;

    const logs = await r.json();
    if (!logs || !logs.length) return;

    const historyEl = document.getElementById("fnHistory");
    const listEl    = document.getElementById("fnHistoryList");
    if (!historyEl || !listEl) return;

    historyEl.style.display = "block";
    listEl.innerHTML = "";

    // Show last 5
    logs.slice(0, 5).forEach(log => {
      const isFake    = log.result_label === "Fake";
      const badgeCls  = isFake ? "fake" : "real";
      const truncated = log.input_text.length > 60
        ? log.input_text.substring(0, 60) + "…"
        : log.input_text;

      let timeStr = "";
      try {
        timeStr = new Date(log.timestamp).toLocaleString("en-GB", {
          day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit"
        });
      } catch(e) {}

      const item = document.createElement("div");
      item.className = "fn-history-item";
      item.innerHTML = `
        <span class="fn-h-text" title="${esc(log.input_text)}">${esc(truncated)}</span>
        <span class="fn-badge ${badgeCls}">${esc(log.result_label)}</span>
        <span class="fn-h-conf">${log.confidence}%</span>
        <span class="fn-h-time">${esc(timeStr)}</span>
      `;
      listEl.appendChild(item);
    });
  } catch(e) {
    console.warn("Failed to load fake news history:", e);
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function setLoading(btn, on, loadingText, idleText) {
  btn.disabled = on;
  btn.classList.toggle("loading", on);
  const textEl = btn.querySelector(".btn-text");
  if (textEl) textEl.textContent = on ? loadingText : idleText;
}

function showError(el, msg) {
  el.textContent = msg;
  el.style.display = "block";
}

function esc(str) {
  return String(str)
    .replace(/&/g,"&amp;")
    .replace(/</g,"&lt;")
    .replace(/>/g,"&gt;")
    .replace(/"/g,"&quot;");
}