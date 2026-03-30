/* script.js — The Classifier, full multi-tab JS */

// ── State ────────────────────────────────────────────────────────────────────
let activeModel = "naive_bayes";
let evalData    = {};
let evalModel   = "naive_bayes";
let batchResults = [];

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

// ── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  setupTabs();
  setupModelPills();
  setupClassify();
  setupBatch();
  setDate();
  evalData = await fetchEval();
  buildEvalTab(evalData);
});

// ── Date ─────────────────────────────────────────────────────────────────────
function setDate() {
  const el = document.getElementById("liveDate");
  if (el) el.textContent = new Date().toLocaleDateString("en-GB", { weekday: "long", year: "numeric", month: "long", day: "numeric" });
}

// ── Tabs ─────────────────────────────────────────────────────────────────────
function setupTabs() {
  document.querySelectorAll(".nav-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(t => t.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById("tab-" + tab).classList.add("active");
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

// ── Classify ─────────────────────────────────────────────────────────────────
function setupClassify() {
  const textarea    = document.getElementById("newsInput");
  const charCount   = document.getElementById("charCount");
  const btn         = document.getElementById("classifyBtn");
  const errorMsg    = document.getElementById("errorMsg");

  textarea.addEventListener("input", () => { charCount.textContent = textarea.value.length; });
  textarea.addEventListener("keydown", e => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") btn.click(); });

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

    setLoading(btn, true);
    try {
      const r = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model: activeModel }),
      });
      const data = await r.json();
      if (!r.ok) return showError(errorMsg, data.error || "Server error.");
      renderResult(data);
    } catch (e) {
      showError(errorMsg, "Network error — is the Flask server running?");
    } finally {
      setLoading(btn, false);
    }
  });
}

function renderResult(data) {
  document.getElementById("idleWrap").style.display   = "none";
  document.getElementById("resultWrap").style.display = "block";

  // Verdict banner
  const banner = document.getElementById("verdictBanner");
  banner.dataset.stamp = MODEL_INFO[data.model_key]?.short || "NB";
  document.getElementById("verdictKicker").textContent   = "CLASSIFIED AS";
  document.getElementById("verdictHeadline").innerHTML   = (CAT_ICONS[data.category] || "") + " " + data.category;
  document.getElementById("verdictByline").textContent   =
    `Model: ${data.model_label} · Confidence: ${data.confidence}%`;

  // Confidence bar
  document.getElementById("confPct").textContent = data.confidence + "%";
  requestAnimationFrame(() => requestAnimationFrame(() => {
    document.getElementById("confFill").style.width = data.confidence + "%";
  }));

  // Prob rows
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

  // Keywords
  const cloud = document.getElementById("keywordCloud");
  cloud.innerHTML = "";
  const maxScore = data.keywords[0]?.score || 1;
  data.keywords.forEach((kw, i) => {
    const pct  = kw.score / maxScore;
    const tier = pct > 0.66 ? "tier-1" : pct > 0.33 ? "tier-2" : "tier-3";
    const tag  = document.createElement("span");
    tag.className = `kw-tag ${tier}`;
    tag.textContent = kw.word;
    tag.title = `TF-IDF: ${kw.score}`;
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

    setLoading(btn, true);
    try {
      const r = await fetch("/api/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: raw, model: activeModel }),
      });
      const data = await r.json();
      if (!r.ok) return showError(errorEl, data.error || "Server error.");
      batchResults = data;
      renderBatch(data);
      exportBtn.style.display = "flex";
    } catch(e) {
      showError(errorEl, "Network error.");
    } finally {
      setLoading(btn, false);
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
  // Macro avg
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

  // Header row
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

// ── Helpers ───────────────────────────────────────────────────────────────────
function setLoading(btn, on) {
  btn.disabled = on;
  btn.classList.toggle("loading", on);
  btn.querySelector(".btn-text").textContent = on ? "Analysing..." : "Classify All";
  // Reset text for single classify btn
  if (btn.id === "classifyBtn") {
    btn.querySelector(".btn-text").textContent = on ? "Analysing..." : "Classify Article";
  }
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