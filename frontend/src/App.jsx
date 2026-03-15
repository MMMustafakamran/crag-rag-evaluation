import React, { useState, useEffect } from 'react';
import './App.css';

const PIPELINES = [
  { id: 'rag_fusion', name: 'RAG Fusion', desc: 'Multi-query + RRF' },
  { id: 'hyde', name: 'HyDE', desc: 'Hypothetical Doc' },
  { id: 'crag', name: 'CRAG', desc: 'Confidence-Gated + Citations' },
  { id: 'graph_rag', name: 'Graph RAG', desc: 'Similarity Graph BFS' },
];

function App() {
  const [query, setQuery] = useState('');
  const [pipeline, setPipeline] = useState('rag_fusion');
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [samples, setSamples] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('http://localhost:5000/api/samples')
      .then(res => res.json())
      .then(data => setSamples(data))
      .catch(err => console.error('Failed to fetch samples', err));
  }, []);

  const handleRun = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const resp = await fetch('http://localhost:5000/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, pipeline, top_k: topK }),
      });
      const data = await resp.json();
      if (data.error) throw new Error(data.error);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header>
        <h1>RAG in the Wild</h1>
        <p className="subtitle">Exploring Advanced Retrieval-Augmented Generation</p>
      </header>

      <main>
        <div className="card input-card">
          <div className="field-group">
            <label>Question</label>
            <textarea 
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a factual question..."
              rows={3}
            />
          </div>

          <div className="samples">
            {samples.map(s => (
              <button key={s.id} onClick={() => setQuery(s.query)} className="sample-btn">
                {s.query.length > 40 ? s.query.substring(0, 40) + '...' : s.query}
              </button>
            ))}
          </div>

          <div className="controls">
            <div className="field-group">
              <label>Pipeline Strategy</label>
              <div className="pipeline-grid">
                {PIPELINES.map(p => (
                  <button 
                    key={p.id}
                    className={`pipeline-btn ${pipeline === p.id ? 'active' : ''}`}
                    onClick={() => setPipeline(p.id)}
                  >
                    <strong>{p.name}</strong>
                    <span>{p.desc}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="run-section">
               <div className="field-group">
                <label>Top K Chunks</label>
                <input type="number" value={topK} onChange={(e) => setTopK(parseInt(e.target.value))} min="1" max="20" />
              </div>
              <button className="run-btn" onClick={handleRun} disabled={loading}>
                {loading ? 'Processing...' : 'Run Pipeline'}
              </button>
            </div>
          </div>
        </div>

        {error && <div className="error-msg">Error: {error}</div>}

        {result && (
          <div className="results-container">
            <div className="card answer-card">
              <h3>Answer</h3>
              <div className="answer-text">
                {result.answer}
              </div>
              {result.meta && result.meta.confidence !== undefined && (
                <div className="confidence-badge">
                  Confidence: {(result.meta.confidence * 100).toFixed(1)}%
                </div>
              )}
            </div>

            <div className="card chunks-card">
              <h3>Retrieved Chunks</h3>
              <div className="chunks-list">
                {result.retrieved?.map((chunk, i) => (
                  <div key={i} className="chunk-item">
                    <div className="chunk-header">
                      <span className="rank">#{i + 1}</span>
                      <span className="score">Score: {chunk.score.toFixed(4)}</span>
                      {chunk.is_seed && <span className="seed-badge">Seed</span>}
                    </div>
                    <div className="chunk-content">{chunk.text}</div>
                    <div className="chunk-footer">
                      <a href={chunk.page_url} target="_blank" rel="noreferrer">
                        {chunk.page_name || 'View Source'}
                      </a>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {result.meta && result.meta.hypothetical_doc && (
              <div className="card meta-card">
                <h3>Hypothetical Document (HyDE)</h3>
                <p className="meta-text">{result.meta.hypothetical_doc}</p>
              </div>
            )}
            
            {result.meta && result.meta.variants && (
              <div className="card meta-card">
                <h3>Query Variants (RAG Fusion)</h3>
                <ul className="meta-list">
                  {result.meta.variants.map((v, i) => <li key={i}>{v}</li>)}
                </ul>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
