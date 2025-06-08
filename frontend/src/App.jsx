import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import { 
  Search, Globe, FileText, AlertCircle, Loader2, CheckCircle, 
  Terminal, ChevronDown, ChevronUp, Clock, Database, Download,
  Zap, HardDrive, Cpu, Cloud, Key, Settings
} from 'lucide-react';

// Simple markdown parser for basic formatting
const SimpleMarkdown = ({ children }) => {
  if (!children) return null;
  
  // Convert markdown-style bold to HTML
  const processText = (text) => {
    // Handle bold text
    let processed = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Handle line breaks
    processed = processed.split('\n').map((line, i) => (
      <span key={i}>
        {i > 0 && <br />}
        <span dangerouslySetInnerHTML={{ __html: line }} />
      </span>
    ));
    
    return processed;
  };
  
  return <div>{processText(children)}</div>;
};

function App() {
  const [url, setUrl] = useState('https://genai.owasp.org');
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [logs, setLogs] = useState([]);
  const [showLogs, setShowLogs] = useState(false);
  const [forceRefresh, setForceRefresh] = useState(false);
  const [expandedChunks, setExpandedChunks] = useState({});
  const [showSettings, setShowSettings] = useState(false);
  
  // LLM Settings
  const [llmMode, setLlmMode] = useState('local');
  const [apiKey, setApiKey] = useState('');
  const [selectedModel, setSelectedModel] = useState('transformers');
  
  const logsEndRef = useRef(null);

  // Auto-scroll logs
  useEffect(() => {
    if (logsEndRef.current && showLogs) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, showLogs]);

  // Determine cache status from logs
  const isCacheHit = logs.some(log => 
    log.message.includes('[CACHE] Using cached data') || 
    log.message.includes('[CACHE] Found valid cache')
  );

  const isCrawling = logs.some(log => 
    log.message.includes('[CRAWL]') || 
    log.message.includes('[DOWNLOAD]')
  );

  const toggleChunkExpansion = (index) => {
    setExpandedChunks(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!url || !query) {
      setError('Please provide both URL and query');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);
    setLogs([]);

    try {
      const response = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          url: url.trim(),
          query: query.trim(),
          force_refresh: forceRefresh,
          embedding_model: "bge-base",
          max_pages: 5,
          max_depth: 2,
          top_k: 5,
          llm_mode: llmMode === 'api' ? 'gemini' : selectedModel,
          api_key: llmMode === 'api' ? apiKey : null,
          debug: true
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to process request');
      }

      const data = await response.json();
      console.log('Response:', data);
      
      setResults(data);
      
      if (data.metadata && data.metadata.logs) {
        setLogs(data.metadata.logs);
      }
      
    } catch (err) {
      console.error('Error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Custom component to render answer with special formatting
  const AnswerContent = ({ answer }) => {
    if (!answer) return null;
    
    // Check if answer contains risks
    const text = answer.toString();
    
    if (text.includes('risks include')) {
      const parts = text.split('risks include');
      const beforeRisks = parts[0];
      const risksAndAfter = parts[1];
      
      if (risksAndAfter) {
        // Extract the risks part (until the period)
        const risksSentence = risksAndAfter.split('.')[0];
        const afterRisks = risksAndAfter.substring(risksSentence.length);
        
        // Split risks by commas
        const risks = risksSentence.split(',').map(r => r.trim());
        
        return (
          <>
            <p className="answer-paragraph">{beforeRisks}risks include:</p>
            <ul className="answer-list">
              {risks.map((risk, idx) => (
                <li key={idx} className="answer-list-item">
                  <strong>{risk.replace(' and', '').replace('.', '')}</strong>
                </li>
              ))}
            </ul>
            {afterRisks && <p className="answer-paragraph">{afterRisks}</p>}
          </>
        );
      }
    }
    
    // Default rendering with simple markdown support
    return <SimpleMarkdown>{text}</SimpleMarkdown>;
  };

  return (
    <div className="app">
      <header className="header">
        <h1>
          <Search className="header-icon" />
          RAG Knowledge Extraction System
        </h1>
        <p className="subtitle">Extract knowledge from websites using Retrieval-Augmented Generation</p>
      </header>

      <main className="main-content">
        <section className="input-section">
          <form onSubmit={handleSubmit}>
            {/* LLM Settings Section */}
            <div className="settings-section">
              <button
                type="button"
                className="settings-toggle"
                onClick={() => setShowSettings(!showSettings)}
              >
                <Settings size={18} />
                LLM Settings
                {showSettings ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </button>

              {showSettings && (
                <div className="settings-content">
                  <div className="llm-mode-selector">
                    <label className="llm-mode-label">
                      <input
                        type="radio"
                        name="llmMode"
                        value="local"
                        checked={llmMode === 'local'}
                        onChange={(e) => setLlmMode(e.target.value)}
                      />
                      <div className="llm-mode-card">
                        <Cpu size={24} />
                        <span className="llm-mode-title">Local LLM</span>
                        <span className="llm-mode-desc">Run models on your machine</span>
                      </div>
                    </label>

                    <label className="llm-mode-label">
                      <input
                        type="radio"
                        name="llmMode"
                        value="api"
                        checked={llmMode === 'api'}
                        onChange={(e) => setLlmMode(e.target.value)}
                      />
                      <div className="llm-mode-card">
                        <Cloud size={24} />
                        <span className="llm-mode-title">Gemini API</span>
                        <span className="llm-mode-desc">Use Google's Gemini model</span>
                      </div>
                    </label>
                  </div>

                  {llmMode === 'local' && (
                    <div className="local-model-selector">
                      <label htmlFor="localModel">Local Model:</label>
                      <select
                        id="localModel"
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                      >
                        <option value="transformers">Mistral 7B (Transformers)</option>
                        {/* <option value="local">TinyLlama (GGUF)</option> */}
                      </select>
                    </div>
                  )}

                  {llmMode === 'api' && (
                    <div className="api-key-input">
                      <label htmlFor="apiKey">
                        <Key size={18} />
                        Gemini API Key
                      </label>
                      <input
                        type="password"
                        id="apiKey"
                        value={apiKey}
                        onChange={(e) => setApiKey(e.target.value)}
                        placeholder="Enter your Gemini API key"
                        className="api-key-field"
                      />
                      <p className="api-key-help">
                        Get your API key from{' '}
                        <a href="https://makersuite.google.com/app/apikey" target="_blank" rel="noopener noreferrer">
                          Google AI Studio
                        </a>
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="url">
                <Globe size={18} />
                Website URL
              </label>
              <input
                type="url"
                id="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com"
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="query">
                <FileText size={18} />
                Your Question
              </label>
              <textarea
                id="query"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="What would you like to know about this website?"
                rows={3}
                required
              />
            </div>

            <div className="form-options">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={forceRefresh}
                  onChange={(e) => setForceRefresh(e.target.checked)}
                />
                Force fresh crawl (ignore cache)
              </label>
            </div>

            <div className="form-buttons">
              <button type="submit" disabled={loading} className="submit-button">
                {loading ? (
                  <>
                    <Loader2 className="spin" size={18} />
                    Processing...
                  </>
                ) : (
                  <>
                    <Search size={18} />
                    Submit Query
                  </>
                )}
              </button>

              <button 
                type="button" 
                onClick={() => setShowLogs(!showLogs)}
                className="toggle-logs-button"
              >
                <Terminal size={18} />
                {showLogs ? 'Hide' : 'Show'} Logs
              </button>
            </div>
          </form>

          {error && (
            <div className="error-message">
              <AlertCircle size={18} />
              {error}
            </div>
          )}
        </section>

        {/* Enhanced Loading Status */}
        {loading && (
          <section className="crawl-status">
            {isCacheHit ? (
              <>
                <h3>
                  <Zap className="spin" size={18} />
                  Loading from Cache...
                </h3>
                <div className="cache-indicator cached">
                  <HardDrive size={16} className="cache-icon cached" />
                  <span>Using cached content - super fast response!</span>
                </div>
              </>
            ) : isCrawling ? (
              <>
                <h3>
                  <Download className="spin" size={18} />
                  Crawling Website...
                </h3>
                <div className="crawl-progress">
                  <p>Downloading and analyzing content from {url}</p>
                  <p className="crawl-note">This may take a moment for the first time...</p>
                </div>
              </>
            ) : (
              <>
                <h3>
                  <Loader2 className="spin" size={18} />
                  Processing Query...
                </h3>
                <div className="crawl-progress">
                  <p>Initializing search...</p>
                </div>
              </>
            )}
            
            {/* Show which LLM is being used */}
            <div className="llm-status">
              {llmMode === 'api' ? (
                <span><Cloud size={16} /> Using Gemini API</span>
              ) : (
                <span><Cpu size={16} /> Using {selectedModel === 'transformers' ? 'Mistral 7B' : 'TinyLlama'}</span>
              )}
            </div>
          </section>
        )}

        {results && !loading && (
          <section className="results-section">
            {/* Enhanced Metadata with Cache Status */}
            {results.metadata && (
              <div className="metadata-card">
                <h3>Search Statistics</h3>
                
                {/* Cache Status Banner */}
                {results.metadata.cache_used ? (
                  <div className="cache-status-banner cached">
                    <div className="cache-icon-wrapper">
                      <HardDrive size={24} />
                    </div>
                    <div className="cache-status-content">
                      <strong>⚡ Lightning Fast - Cached Results</strong>
                      <p>Content was already indexed. No crawling needed!</p>
                      {results.metadata.cache_age && (
                        <p className="cache-age">
                          <Clock size={14} />
                          Cache age: {results.metadata.cache_age}
                        </p>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="cache-status-banner fresh">
                    <div className="cache-icon-wrapper">
                      <Download size={24} />
                    </div>
                    <div className="cache-status-content">
                      <strong>🔄 Fresh Content Indexed</strong>
                      <p>Website crawled and indexed for your search.</p>
                      <p className="cache-note">Future searches will be instant!</p>
                    </div>
                  </div>
                )}
                
                <div className="metadata-grid">
                  <div className="metadata-item">
                    <span className="metadata-label">Pages Indexed</span>
                    <span className="metadata-value">
                      {results.metadata.total_pages_crawled || 0}
                    </span>
                  </div>
                  <div className="metadata-item">
                    <span className="metadata-label">Chunks Created</span>
                    <span className="metadata-value">
                      {results.metadata.chunks_found || 0}
                    </span>
                  </div>
                  <div className="metadata-item">
                    <span className="metadata-label">Unique Sources</span>
                    <span className="metadata-value">
                      {results.metadata.unique_sources || 0}
                    </span>
                  </div>
                  <div className="metadata-item">
                    <span className="metadata-label">LLM Used</span>
                    <span className="metadata-value">
                      {results.metadata.llm_used || 'Unknown'}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Answer with better formatting */}
            <div className="answer-card">
              <div className="answer-header">
                <h2>
                  <CheckCircle className="section-icon" />
                  Answer
                </h2>
                {results.answer && results.answer.toLowerCase().includes('risks') && (
                  <span className="response-type-badge list">
                    Risk Analysis
                  </span>
                )}
              </div>
              
              <div className="answer-content">
                <AnswerContent answer={results.answer || 'No answer generated'} />
              </div>
              
              {/* Answer quality indicator */}
              {results.metadata && results.metadata.unique_sources >= 3 && (
                <div className="answer-quality-note">
                  <CheckCircle size={16} />
                  Answer synthesized from {results.metadata.unique_sources} sources
                </div>
              )}
            </div>

            {/* Retrieved Chunks */}
            {results.detailed_sources && results.detailed_sources.length > 0 && (
              <div className="sources-section">
                <h2>
                  <FileText className="section-icon" />
                  Retrieved Chunks ({results.detailed_sources.length})
                </h2>
                <p style={{color: '#718096', fontSize: '0.875rem', marginBottom: '1rem'}}>
                  These are the exact text chunks used to generate the answer above.
                </p>
                <div className="sources-grid">
                  {results.detailed_sources.map((source, index) => (
                    <div key={index} className="source-card">
                      <div className="source-header">
                        <span className="source-id">Chunk {source.source_id}</span>
                        <span className="relevance-score">
                          {(source.relevance_score * 100).toFixed(0)}% relevant
                        </span>
                      </div>
                      <a 
                        href={source.url} 
                        target="_blank" 
                        rel="noopener noreferrer" 
                        className="source-url"
                      >
                        {source.url}
                      </a>
                      <div className="source-meta">
                        Chunk {source.chunk_id + 1} of {source.total_chunks || 'unknown'}
                      </div>
                      
                      <div className="chunk-content-container">
                        <div className={`source-content ${expandedChunks[index] ? 'expanded' : 'collapsed'}`}>
                          {expandedChunks[index] ? (
                            <pre style={{whiteSpace: 'pre-wrap', fontFamily: 'inherit'}}>
                              {source.full_content}
                            </pre>
                          ) : (
                            <div>{source.preview}</div>
                          )}
                        </div>
                        
                        {source.full_content && source.full_content.length > 200 && (
                          <button 
                            className="expand-button"
                            onClick={() => toggleChunkExpansion(index)}
                          >
                            {expandedChunks[index] ? (
                              <>
                                <ChevronUp size={16} />
                                Show Less
                              </>
                            ) : (
                              <>
                                <ChevronDown size={16} />
                                Show Full Chunk
                              </>
                            )}
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        )}

        {showLogs && (
          <section className="logs-section">
            <h3>
              <Terminal size={18} />
              Backend Logs
            </h3>
            <div className="logs-container">
              {logs.length === 0 ? (
                <div className="log-entry">No logs yet.</div>
              ) : (
                logs.map((log, index) => (
                  <div key={index} className={`log-entry log-${log.level.toLowerCase()}`}>
                    <span className="log-timestamp">
                      {new Date(log.timestamp).toLocaleTimeString()}
                    </span>
                    <span className="log-level">[{log.level}]</span>
                    <span className="log-message">{log.message}</span>
                  </div>
                ))
              )}
              <div ref={logsEndRef} />
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;