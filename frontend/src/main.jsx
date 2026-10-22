import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

async function api(path, options = {}) {
  let response;
  try { response = await fetch(`/api${path}`, options); }
  catch { throw new Error('Cannot reach the backend. Check that FastAPI is running on port 8000.'); }
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new Error(typeof data?.detail === 'string' ? data.detail : `Request failed (${response.status}). Check that FastAPI is running.`);
  return data;
}

function App() {
  const [documents, setDocuments] = useState([]);
  const [selected, setSelected] = useState('');
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const input = useRef(null);
  const busy = uploading || asking;
  const active = documents.find(doc => doc.document_id === selected);

  async function refresh(preferred) {
    setLoading(true);
    try {
      const saved = await api('/documents');
      setDocuments(saved);
      setSelected(current => {
        const candidate = preferred || current;
        return saved.some(doc => doc.document_id === candidate) ? candidate : (saved[0]?.document_id || '');
      });
    } finally { setLoading(false); }
  }
  useEffect(() => { refresh().catch(err => setError(err.message)); }, []);

  async function upload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setError(''); setNotice('');
    if (!file.name.toLowerCase().endsWith('.pdf') || file.size === 0 || file.size > 10 * 1024 * 1024) {
      setError('Choose a nonempty PDF no larger than 10 MB.'); event.target.value = ''; return;
    }
    setUploading(true); setResult(null);
    try {
      const form = new FormData(); form.append('file', file);
      const saved = await api('/documents/ingest', { method: 'POST', body: form });
      // Keep the newly saved upload selectable even if refreshing the list fails.
      setDocuments(current => [...current, saved]); setSelected(saved.document_id);
      setNotice(`${saved.filename} is ready. ${saved.chunk_count} passages saved.`);
      await refresh(saved.document_id);
    } catch (err) { setError(err.message); }
    finally { setUploading(false); if (input.current) input.current.value = ''; }
  }

  async function ask(event) {
    event.preventDefault();
    if (!selected || !question.trim() || busy) return;
    setAsking(true); setError(''); setNotice(''); setResult(null);
    const asked = question.trim();
    try {
      const answer = await api(`/documents/${selected}/ask`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question: asked, top_k: 3}) });
      setResult({...answer, question: asked});
    } catch (err) { setError(err.message); }
    finally { setAsking(false); }
  }

  return <div className="app">
    <header><a className="brand" href="/" aria-label="CareSource home"><span className="brand-icon">✳</span> CareSource<span className="brand-sub">DOCUMENT ASSISTANT</span></a><span className="local-badge"><i /> Local workspace</span></header>
    <div className="layout">
      <aside>
        <div className="sidebar-heading"><span>YOUR LIBRARY</span><span>{documents.length}</span></div>
        <input ref={input} id="pdf-upload" type="file" accept=".pdf,application/pdf" onChange={upload} disabled={busy} hidden />
        <button className="upload" onClick={() => input.current?.click()} disabled={busy}><span>＋</span>{uploading ? 'Preparing document…' : 'Upload a PDF'}</button>
        <p className="upload-note">Synthetic documents only · Up to 10 MB<br />Upload sends extracted text to Google.</p>
        <div className="library-toolbar"><h2>Saved documents</h2><button className="text-button" disabled={busy || loading} onClick={() => { setError(''); refresh().catch(err => setError(err.message)); }}>Refresh</button></div>
        {loading && <p role="status" className="muted">Loading library…</p>}
        {!loading && documents.length === 0 && <div className="empty-library">Your library is empty.<br />Upload a fictional policy or workflow to begin.</div>}
        <nav aria-label="Saved documents">{documents.map(doc => <button key={doc.document_id} className={`document ${selected === doc.document_id ? 'selected' : ''}`} aria-pressed={selected === doc.document_id} disabled={busy} onClick={() => { setSelected(doc.document_id); setResult(null); setError(''); setNotice(''); }}>
          <span className="pdf-icon">PDF</span><span><strong>{doc.filename}</strong><small>{doc.chunk_count} passages · {doc.document_id.slice(0, 8)}</small></span>
        </button>)}</nav>
        <div className="sidebar-footer"><span className="tiny-label">BUILT FOR EVIDENCE</span><p>Read the source.<br />Understand the answer.</p><small>Saved documents remain available after a refresh.</small></div>
      </aside>
      <main>
        <div className="eyebrow">YOUR DOCUMENTS, MADE CLEAR</div>
        <h1>Find answers.<br /><span>Follow the evidence.</span></h1>
        <p className="intro">Ask a question about a healthcare document. Get a focused answer with passages you can check.</p>
        <div className="scope"><span className="scope-dot" />{active ? <>Searching <strong>{active.filename}</strong></> : 'Select or upload a document to start'}</div>
        {error && <div className="error" role="alert">{error}</div>}
        {notice && <p className="notice" role="status">{notice}</p>}
        <form onSubmit={ask} className="question-card">
          <label htmlFor="question">What would you like to know?</label>
          <textarea id="question" maxLength={2000} rows={3} placeholder="Who should staff notify about missing inspection entries?" value={question} onChange={event => setQuestion(event.target.value)} disabled={busy} />
          <div className="composer-footer"><small>Answers use only the selected document.</small><button className="primary" disabled={!selected || !question.trim() || busy}>{asking ? 'Finding evidence…' : 'Ask document'} <span aria-hidden="true">↗</span></button></div>
        </form>
        <div className="suggestions"><span>TRY ASKING</span>{['Where are equipment inspections recorded?', 'Who receives reports of missing entries?'].map(text => <button key={text} disabled={busy} onClick={() => setQuestion(text)}>{text} <span aria-hidden="true">↗</span></button>)}</div>
        <div aria-live="polite" aria-busy={asking}>
          {asking && <div className="waiting"><span className="pulse" />Searching passages and preparing your answer. This can take a minute.</div>}
          {result && <section className="answer-card"><div className="answer-heading"><span className="eyebrow">{result.abstained ? 'INSUFFICIENT EVIDENCE' : 'DOCUMENT ANSWER'}</span><span className="result-badge">{result.abstained ? 'No supported answer' : 'Sources attached'}</span></div><h2>{result.question}</h2><p className="answer-text">{result.answer}</p>
            {result.abstained && <p className="muted">Try a question directly covered by the document, or select another document.</p>}
            {result.citations?.length > 0 && <div className="sources"><h3>Check the evidence <span>{result.citations.length}</span></h3>{result.citations.map(citation => <article className="citation" key={citation.citation_number}><div><span className="citation-number">{citation.citation_number}</span><strong>{citation.filename}</strong><span className="page">Page {citation.page_number}</span></div><blockquote>{citation.quote}</blockquote></article>)}<small>A citation supports a statement only when its passage contains the relevant evidence.</small></div>}
          </section>}
          {!result && !asking && <div className="how"><div><span>01</span><h3>Add a document</h3><p>A synthetic policy, summary, or workflow.</p></div><div><span>02</span><h3>Ask in your words</h3><p>Search by meaning, beyond exact keywords.</p></div><div><span>03</span><h3>Check the source</h3><p>Review the quoted passage and page.</p></div></div>}
        </div>
        <footer>For document retrieval and summarization. Not for diagnosis or treatment advice.<br />Questions and retrieved passages are sent to Google. Use synthetic data only.</footer>
      </main>
    </div>
  </div>;
}

createRoot(document.getElementById('root')).render(<App />);
