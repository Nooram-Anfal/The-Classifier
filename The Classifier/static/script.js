/* script.js — News Classifier frontend logic */

const textarea    = document.getElementById('newsInput');
const charCount   = document.getElementById('charCount');
const classifyBtn = document.getElementById('classifyBtn');
const errorBox    = document.getElementById('errorBox');
const idleState   = document.getElementById('idleState');
const resultState = document.getElementById('resultState');

// Category → CSS class mapping (slug)
const SLUGS = {
  'World':    'world',
  'Sports':   'sports',
  'Business': 'business',
  'Sci/Tech': 'scitech',
};

// ── Character counter ────────────────────────────────────────────────────────
textarea.addEventListener('input', () => {
  charCount.textContent = textarea.value.length;
});

// ── Example pills ────────────────────────────────────────────────────────────
document.querySelectorAll('.pill').forEach(btn => {
  btn.addEventListener('click', () => {
    textarea.value = btn.dataset.text;
    charCount.textContent = textarea.value.length;
    textarea.focus();
  });
});

// ── Classify ─────────────────────────────────────────────────────────────────
classifyBtn.addEventListener('click', classify);

textarea.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') classify();
});

async function classify() {
  const text = textarea.value.trim();

  hideError();

  if (!text) {
    showError('Please enter some news text before classifying.');
    return;
  }

  // Loading state
  classifyBtn.disabled = true;
  classifyBtn.classList.add('loading');
  classifyBtn.querySelector('.btn-text').textContent = 'Analysing…';

  try {
    const res = await fetch('/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ text }),
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data.error || 'Server error. Please try again.');
      return;
    }

    renderResult(data, text);

  } catch (err) {
    showError('Network error — make sure the Flask server is running.');
  } finally {
    classifyBtn.disabled = false;
    classifyBtn.classList.remove('loading');
    classifyBtn.querySelector('.btn-text').textContent = 'Classify Article';
  }
}

// ── Render result ─────────────────────────────────────────────────────────────
function renderResult(data, rawText) {
  const { category, icon, confidence, all_probs } = data;
  const slug = SLUGS[category] || 'world';

  // Switch panels
  idleState.style.display   = 'none';
  resultState.style.display = 'block';

  // Badge
  const badge = document.getElementById('categoryBadge');
  badge.className = `category-badge ${slug}`;
  document.getElementById('badgeIcon').textContent  = icon;
  document.getElementById('badgeLabel').textContent = category;

  // Confidence
  document.getElementById('confValue').textContent = confidence + '%';
  setTimeout(() => {
    document.getElementById('confBarFill').style.width = confidence + '%';
  }, 60);

  // All probabilities
  const grid = document.getElementById('probsGrid');
  grid.innerHTML = '';

  const sortedProbs = Object.entries(all_probs).sort((a, b) => b[1] - a[1]);

  sortedProbs.forEach(([cls, pct]) => {
    const clsSlug    = SLUGS[cls] || 'world';
    const isWinner   = cls === category;

    const row = document.createElement('div');
    row.className = 'prob-row';
    row.innerHTML = `
      <span class="prob-name">${getIcon(cls)} ${cls}</span>
      <div class="prob-track">
        <div class="prob-fill ${clsSlug}" style="width:0%"
             data-target="${pct}"></div>
      </div>
      <span class="prob-pct ${isWinner ? 'highlight' : ''}">${pct}%</span>
    `;
    grid.appendChild(row);
  });

  // Animate bars after DOM paint
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      document.querySelectorAll('.prob-fill').forEach(bar => {
        bar.style.width = bar.dataset.target + '%';
      });
    });
  });

  // Token preview (simple client-side tokenisation preview)
  const tokens = rawText
    .toLowerCase()
    .replace(/[^a-z\s]/g, ' ')
    .split(/\s+/)
    .filter(t => t.length > 2)
    .slice(0, 40)
    .join('  ·  ');

  document.getElementById('tokenPreview').textContent = tokens || '—';
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function getIcon(cls) {
  const icons = { 'World': '🌍', 'Sports': '⚽', 'Business': '💼', 'Sci/Tech': '🔬' };
  return icons[cls] || '📰';
}

function showError(msg) {
  errorBox.textContent = '⚠ ' + msg;
  errorBox.style.display = 'block';
}

function hideError() {
  errorBox.style.display = 'none';
  errorBox.textContent   = '';
}
