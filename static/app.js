/**
 * FinSolve Chat Application
 * Handles chat UI interactions and admin functions.
 */
(function() {
  'use strict';

  // DOM Elements
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

  // Admin elements
  const adminPanel = document.getElementById('admin-panel');
  const btnStatus = document.getElementById('btn-status');
  const btnReindexSparse = document.getElementById('btn-reindex-sparse');
  const btnReindexDense = document.getElementById('btn-reindex-dense');
  const adminStatus = document.getElementById('admin-status');

  // Utility functions
  function show(el) { el?.classList.remove('hidden'); }
  function hide(el) { el?.classList.add('hidden'); }

  function setStatus(msg, isError = false) {
    if (statusEl) {
      statusEl.textContent = msg;
      statusEl.style.color = isError ? 'var(--error)' : 'var(--accent-secondary)';
    }
  }

  function setAdminStatus(msg, isError = false) {
    if (adminStatus) {
      adminStatus.textContent = msg;
      adminStatus.style.color = isError ? 'var(--error)' : 'var(--text-muted)';
    }
  }

  function formatDuration(ms) {
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  }

  // Show admin panel for c_level users
  const role = document.body?.getAttribute('data-user-role') || '';
  if (role === 'c_level') {
    show(adminPanel);
  }

  // Admin: System Status
  btnStatus?.addEventListener('click', async () => {
    setAdminStatus('Loading system status...');
    try {
      const res = await fetch('/admin/status');
      const data = await res.json();
      const formatted = {
        sparse_index: data.sparse_count || 0,
        dense_index: data.dense_count || 0,
        cache: data.rag_cache || {},
        openai_connected: data.openai?.ok || false,
        timestamp: new Date().toISOString(),
      };
      setAdminStatus(JSON.stringify(formatted, null, 2));
    } catch (e) {
      setAdminStatus('Failed to load status: ' + e.message, true);
    }
  });

  // Admin: Rebuild Text Index
  btnReindexSparse?.addEventListener('click', async () => {
    setAdminStatus('Rebuilding text index...');
    try {
      const res = await fetch('/admin/reindex', { method: 'POST' });
      const data = await res.json();
      setAdminStatus(`Text index rebuilt: ${data.count || 0} documents indexed`);
    } catch (e) {
      setAdminStatus('Failed to rebuild text index: ' + e.message, true);
    }
  });

  // Admin: Rebuild Vector Index
  btnReindexDense?.addEventListener('click', async () => {
    setAdminStatus('Rebuilding vector index (this may take a minute)...');
    try {
      const res = await fetch('/admin/reindex_dense', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ persist_dir: '.chroma', base_dir: 'resources/data' }),
      });
      const data = await res.json();
      setAdminStatus(`Vector index rebuilt: ${data.count || 0} chunks indexed`);
    } catch (e) {
      setAdminStatus('Failed to rebuild vector index: ' + e.message, true);
    }
  });

  // Chat form submission
  form?.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Clear previous results
    hide(answerSection);
    hide(citationsSection);
    hide(metricsSection);
    setStatus('Searching documents...');

    const message = messageEl.value.trim();
    if (!message) {
      setStatus('Please enter a question', true);
      return;
    }

    const payload = {
      message: message,
      top_k: Number(topkEl.value || 5),
    };

    // Map user-friendly reranker values to backend values
    const rerankerMap = {
      'none': null,
      'simple': 'simple',
      'openai': 'openai',  // AI-powered uses embedding-based reranker
    };

    const selectedReranker = rerankerEl?.value || 'none';
    if (selectedReranker && selectedReranker !== 'none') {
      payload.rerank = true;
      payload.reranker = rerankerMap[selectedReranker] || selectedReranker;
    }

    try {
      setStatus('Generating answer...');

      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const correlationId = res.headers.get('X-Correlation-ID') || '';
      const data = await res.json();

      setStatus('');

      if (!res.ok) {
        const errorMsg = data.detail || JSON.stringify(data);
        answerEl.textContent = `Error: ${errorMsg}`;
        show(answerSection);
        corrEl.textContent = correlationId ? `Request ID: ${correlationId}` : '';
        return;
      }

      // Display answer
      answerEl.textContent = data.answer || 'No answer generated.';
      corrEl.textContent = correlationId ? `Request ID: ${correlationId}` : '';
      show(answerSection);

      // Display citations
      citationsEl.innerHTML = '';
      const citations = data.citations || [];
      if (citations.length > 0) {
        citations.forEach((c) => {
          const li = document.createElement('li');
          const sourcePath = c.source_path || 'Unknown';
          const section = c.section_path ? `#${c.section_path}` : '';
          li.textContent = `${sourcePath}${section}`;
          citationsEl.appendChild(li);
        });
        show(citationsSection);
      }

      // Display metrics (simplified)
      if (data.metrics) {
        const metrics = data.metrics;
        const metricsText = [
          `Search: ${formatDuration(metrics.retrieve_ms)}`,
          `Generation: ${formatDuration(metrics.llm_ms)}`,
          `Total: ${formatDuration(metrics.total_ms)}`,
        ].join(' • ');
        metricsEl.textContent = metricsText;
        show(metricsSection);
      }

    } catch (err) {
      setStatus('Connection error. Please try again.', true);
      console.error('Chat error:', err);
    }
  });

  // Allow Cmd/Ctrl+Enter to submit
  messageEl?.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      form?.dispatchEvent(new Event('submit'));
    }
  });

})();

