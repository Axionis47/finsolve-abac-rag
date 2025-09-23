const form = document.getElementById('chat-form');
const messageEl = document.getElementById('message');
const topkEl = document.getElementById('topk');
const rerankerEl = document.getElementById('reranker');
const statusEl = document.getElementById('status');
const answerSection = document.getElementById('answer-section');
const answerEl = document.getElementById('answer');
const corrEl = document.getElementById('correlation');
const citationsSection = document.getElementById('citations-section');
const citationsEl = document.getElementById('citations');
const metricsSection = document.getElementById('metrics-section');
const metricsEl = document.getElementById('metrics');

// Admin
const adminPanel = document.getElementById('admin-panel');
const btnStatus = document.getElementById('btn-status');
const btnReindexSparse = document.getElementById('btn-reindex-sparse');
const btnReindexDense = document.getElementById('btn-reindex-dense');
const adminStatus = document.getElementById('admin-status');

function show(el) { el?.classList.remove('hidden'); }
function hide(el) { el?.classList.add('hidden'); }

// Show admin panel for c_level
const bodyEl = document.querySelector('body');
const role = bodyEl?.getAttribute('data-user-role') || '';
if (role === 'c_level') {
  show(adminPanel);
}

btnStatus?.addEventListener('click', async () => {
  adminStatus.textContent = 'Loading status...';
  try {
    const res = await fetch('/admin/status');
    const corr = res.headers.get('X-Correlation-ID') || '';
    const data = await res.json();
    adminStatus.textContent = JSON.stringify({
      ...data,
      corr,
    }, null, 2);
  } catch (e) {
    adminStatus.textContent = 'Failed to load status';
  }
});

btnReindexSparse?.addEventListener('click', async () => {
  adminStatus.textContent = 'Reindexing sparse...';
  try {
    const res = await fetch('/admin/reindex', { method: 'POST' });
    const data = await res.json();
    adminStatus.textContent = 'Sparse reindex complete: ' + JSON.stringify(data);
  } catch (e) {
    adminStatus.textContent = 'Failed to reindex sparse';
  }
});

btnReindexDense?.addEventListener('click', async () => {
  adminStatus.textContent = 'Reindexing dense...';
  try {
    const res = await fetch('/admin/reindex_dense', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ persist_dir: '.chroma', base_dir: 'resources/data' }),
    });
    const data = await res.json();
    adminStatus.textContent = 'Dense reindex complete: ' + JSON.stringify(data);
  } catch (e) {
    adminStatus.textContent = 'Failed to reindex dense';
  }
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  hide(answerSection); hide(citationsSection); hide(metricsSection);
  statusEl.textContent = 'Asking...';
  const payload = {
    message: messageEl.value.trim(),
    top_k: Number(topkEl.value || 5),
  };
  const rk = rerankerEl.value;
  if (rk && rk !== 'none') {
    payload.rerank = true;
    payload.reranker = rk;
  }

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const corr = res.headers.get('X-Correlation-ID') || '';
    const data = await res.json();
    statusEl.textContent = '';

    if (!res.ok) {
      answerEl.textContent = `Error: ${res.status} - ${JSON.stringify(data)}`;
      show(answerSection);
      corrEl.textContent = corr ? `Correlation ID: ${corr}` : '';
      return;
    }

    answerEl.textContent = data.answer || '';
    corrEl.textContent = corr ? `Correlation ID: ${corr}` : '';
    show(answerSection);

    // Citations
    citationsEl.innerHTML = '';
    (data.citations || []).forEach((c, i) => {
      const li = document.createElement('li');
      const src = `${c.source_path || ''}${c.section_path ? '#' + c.section_path : ''}`;
      li.textContent = `${src} (rule: ${c.rule || 'n/a'})`;
      citationsEl.appendChild(li);
    });
    show(citationsSection);

    // Metrics
    metricsEl.textContent = JSON.stringify(data.metrics || {}, null, 2);
    show(metricsSection);
  } catch (err) {
    statusEl.textContent = 'Network error';
    console.error(err);
  }
});

