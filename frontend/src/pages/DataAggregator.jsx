import React, { useState, useRef } from 'react';
import { api } from '../services/api';

export default function DataAggregator() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  // Table State
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategoryFilter, setSelectedCategoryFilter] = useState('ALL');
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);

  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0];
    if (selected) {
      if (selected.size > 10 * 1024 * 1024) {
        setError('Selected file exceeds the 10 MB maximum size limit.');
        return;
      }
      setFile(selected);
      setError(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) {
      if (dropped.size > 10 * 1024 * 1024) {
        setError('Dropped file exceeds the 10 MB maximum size limit.');
        return;
      }
      setFile(dropped);
      setError(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleProcessUpload = async (fileToUpload) => {
    const targetFile = fileToUpload || file;
    if (!targetFile) {
      setError('Please select or drop a valid file first.');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const data = await api.uploadAndAggregate(targetFile);
      setResult(data);
      setCurrentPage(1);
      setSearchQuery('');
      setSelectedCategoryFilter('ALL');
    } catch (err) {
      console.error('Aggregation processing error:', err);
      setError(err.message || 'Failed to parse and aggregate the dataset.');
    } finally {
      setLoading(false);
    }
  };

  const handleLoadSample = async (sampleType) => {
    setLoading(true);
    setError(null);
    try {
      const samples = await api.getSampleDatasets();
      const rawText = sampleType === 'financial' ? samples.financial_sample : samples.non_financial_sample;
      const sampleFileName = sampleType === 'financial' ? 'sample_market_ticks.csv' : 'sample_customer_reviews.csv';
      const sampleBlob = new Blob([rawText], { type: 'text/csv' });
      const sampleFile = new File([sampleBlob], sampleFileName, { type: 'text/csv' });
      
      setFile(sampleFile);
      await handleProcessUpload(sampleFile);
    } catch (err) {
      console.error('Failed to load sample dataset:', err);
      setError(err.message || 'Could not load sample dataset.');
    } finally {
      setLoading(false);
    }
  };

  // Export Helpers
  const downloadCSV = () => {
    if (!result?.records || result.records.length === 0) return;
    const records = result.records;
    const headers = Object.keys(records[0]);
    const csvRows = [
      headers.join(','),
      ...records.map(row => 
        headers.map(h => {
          const val = row[h];
          if (val === null || val === undefined) return '';
          const str = String(val).replace(/"/g, '""');
          return `"${str}"`;
        }).join(',')
      )
    ];
    const blob = new Blob([csvRows.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `aggregated_${result.filename || 'dataset'}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const downloadJSON = () => {
    if (!result?.records || result.records.length === 0) return;
    const jsonStr = JSON.stringify(result.records, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `aggregated_${result.filename || 'dataset'}.json`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Filtered Table Data
  const rawRecords = result?.records || [];
  const filteredRecords = rawRecords.filter((rec) => {
    // Category Filter
    if (selectedCategoryFilter !== 'ALL') {
      const matchLabel = rec.Primary_Label === selectedCategoryFilter || rec.Category === selectedCategoryFilter;
      if (!matchLabel) return false;
    }
    // Search Query
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      const rowMatches = Object.values(rec).some(v => 
        v !== null && v !== undefined && String(v).toLowerCase().includes(q)
      );
      if (!rowMatches) return false;
    }
    return true;
  });

  const totalPages = Math.max(1, Math.ceil(filteredRecords.length / pageSize));
  const paginatedRecords = filteredRecords.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  const tableColumns = result?.column_schema?.map(c => c.name) || (rawRecords.length > 0 ? Object.keys(rawRecords[0]) : []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Header */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '0.35rem' }}>
          <span className="badge badge-internal">Decoupled Data Architecture</span>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Automated Ingestion &bull; Dynamic Clustering &bull; Quality Auditing
          </span>
        </div>
        <h2 style={{ fontSize: '2rem' }}>Data Aggregator Studio</h2>
        <p style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', maxWidth: '860px' }}>
          Ingest and structure raw data files (<strong>.csv, .tsv, .json, .txt</strong>). Automatically audits data quality, detects duplicates, classifies content via domain taxonomies or unsupervised NLP clustering, aggregates related information, and exports clean structured results.
        </p>
      </div>

      {/* Error Alert */}
      {error && (
        <div
          style={{
            padding: '1rem',
            borderRadius: 'var(--radius-md)',
            background: 'var(--color-sell-bg)',
            border: '1px solid var(--color-sell-border)',
            color: 'var(--color-sell)',
            fontSize: '0.88rem',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <span>{error}</span>
          <button
            type="button"
            onClick={() => setError(null)}
            style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', fontWeight: 'bold' }}
          >
            &times;
          </button>
        </div>
      )}

      {/* Upload Zone & Quick Sample Triggers */}
      <div
        className="glass-panel"
        style={{
          padding: '2.5rem 2rem',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '1.25rem',
          border: '2px dashed var(--border-active)',
          cursor: 'pointer',
          textAlign: 'center',
          background: 'linear-gradient(135deg, rgba(17, 24, 43, 0.6) 0%, rgba(14, 20, 36, 0.4) 100%)',
        }}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.tsv,.json,.txt"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />

        <div style={{ width: '52px', height: '52px', borderRadius: '50%', background: 'var(--color-brand-glow)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--color-brand)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
        </div>

        <div>
          <h3 style={{ fontSize: '1.25rem', marginBottom: '0.35rem' }}>
            {file ? file.name : 'Drag & Drop your dataset here, or click to browse'}
          </h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Supported formats: <strong>CSV, TSV, JSON, TXT</strong> &bull; Max file size: 10 MB
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginTop: '0.5rem' }} onClick={(e) => e.stopPropagation()}>
          <button
            type="button"
            className="btn btn-primary"
            disabled={loading}
            onClick={() => handleProcessUpload()}
            style={{ padding: '0.65rem 1.5rem', fontSize: '0.9rem' }}
          >
            {loading ? <span className="spinner" /> : 'Inspect & Aggregate File'}
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            disabled={loading}
            onClick={() => handleLoadSample('financial')}
            style={{ padding: '0.65rem 1rem', fontSize: '0.85rem' }}
          >
            Load Financial Sample
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            disabled={loading}
            onClick={() => handleLoadSample('non_financial')}
            style={{ padding: '0.65rem 1rem', fontSize: '0.85rem' }}
          >
            Load Non-Financial Text Sample
          </button>
        </div>
      </div>

      {/* Results View */}
      {result && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          {/* Quality Audit Dashboard */}
          <div className="glass-panel" style={{ padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
              <div>
                <h3 style={{ fontSize: '1.2rem' }}>Data Quality Audit & Provenance</h3>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  File: <strong>{result.filename}</strong> &bull; Format: <span className="badge badge-internal">{result.detected_format}</span>
                </p>
              </div>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button type="button" className="btn btn-secondary" onClick={downloadCSV} style={{ fontSize: '0.8rem', padding: '0.45rem 0.85rem' }}>
                  Download CSV
                </button>
                <button type="button" className="btn btn-secondary" onClick={downloadJSON} style={{ fontSize: '0.8rem', padding: '0.45rem 0.85rem' }}>
                  Download JSON
                </button>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem' }}>
              <div style={{ padding: '1rem', background: 'rgba(0, 0, 0, 0.3)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Raw Records</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 'bold' }}>{result.quality_audit.raw_rows}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{result.quality_audit.raw_columns} columns</div>
              </div>

              <div style={{ padding: '1rem', background: 'rgba(0, 0, 0, 0.3)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Clean Records</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 'bold', color: 'var(--color-buy)' }}>{result.quality_audit.clean_rows}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Empty dropped: {result.quality_audit.empty_rows_dropped}</div>
              </div>

              <div style={{ padding: '1rem', background: 'rgba(0, 0, 0, 0.3)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Duplicates Removed</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 'bold', color: result.quality_audit.duplicates_removed > 0 ? 'var(--color-hold)' : 'var(--text-primary)' }}>
                  {result.quality_audit.duplicates_removed}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Deduplication rate: {result.quality_audit.uniqueness_score_pct}%</div>
              </div>

              <div style={{ padding: '1rem', background: 'rgba(0, 0, 0, 0.3)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Missing Cells</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 'bold' }}>{result.quality_audit.missing_cells_count}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Completeness: {result.quality_audit.completeness_score_pct}%</div>
              </div>

              <div style={{ padding: '1rem', background: 'rgba(0, 0, 0, 0.3)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Overall Quality Score</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 'bold', color: 'var(--color-brand)' }}>{result.quality_audit.overall_quality_score_pct}%</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Audit Confidence</div>
              </div>
            </div>
          </div>

          {/* Classification & Generated Labels Panel */}
          <div className="glass-panel" style={{ padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
              <div>
                <h3 style={{ fontSize: '1.2rem' }}>Classification & Meaningful Labels</h3>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  Modality: <strong>{result.classification.inferred_modality}</strong> &bull; Pipeline Mode:{' '}
                  <span className={`badge ${result.classification.is_financial ? 'badge-internal' : 'badge-external'}`}>
                    {result.classification.labeling_mode === 'domain_taxonomy' ? 'Domain Taxonomy' : 'Unsupervised Automatic Labeling'}
                  </span>
                </p>
              </div>
              {result.classification.unsupervised_metrics?.silhouette_score && (
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Silhouette Cohesion Score: <strong style={{ color: 'var(--color-brand)' }}>{result.classification.unsupervised_metrics.silhouette_score}</strong>
                </div>
              )}
            </div>

            {result.classification.labeling_mode === 'unsupervised_clustering' && (
              <div
                style={{
                  padding: '0.65rem 0.95rem',
                  borderRadius: 'var(--radius-sm)',
                  background: 'rgba(99, 102, 241, 0.08)',
                  border: '1px solid rgba(99, 102, 241, 0.25)',
                  fontSize: '0.78rem',
                  color: 'var(--text-secondary)',
                }}
              >
                <strong>Methodological Notice:</strong> Generated labels are synthesized via unsupervised TF-IDF clustering and statistical distribution heuristics. They represent exploratory semantic groupings discovered from text/numeric patterns, not verified ground-truth classifications.
              </div>
            )}

            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
              {result.classification.generated_labels?.map((lbl, idx) => (
                <div
                  key={idx}
                  style={{
                    padding: '0.75rem 1rem',
                    borderRadius: 'var(--radius-md)',
                    background: 'rgba(255, 255, 255, 0.04)',
                    border: '1px solid var(--border-subtle)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.25rem',
                    minWidth: '220px',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <strong style={{ fontSize: '0.9rem', color: 'var(--text-primary)' }}>{lbl.label}</strong>
                    <span className="badge badge-internal">{lbl.count}</span>
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    Share: {lbl.percentage}% &bull; {lbl.type}
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', fontStyle: 'italic' }}>
                    {lbl.criteria}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Aggregated Groups Summary */}
          {result.aggregated_groups && result.aggregated_groups.length > 0 && (
            <div className="glass-panel" style={{ padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <h3 style={{ fontSize: '1.2rem' }}>Aggregated Group Summaries</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
                {result.aggregated_groups.map((grp, idx) => (
                  <div
                    key={idx}
                    style={{
                      padding: '1rem',
                      borderRadius: 'var(--radius-md)',
                      background: 'rgba(0, 0, 0, 0.3)',
                      border: '1px solid var(--border-subtle)',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '0.5rem',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <strong style={{ fontSize: '0.95rem', color: 'var(--color-brand)' }}>{grp.group_name}</strong>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{grp.record_count} items ({grp.percentage}%)</span>
                    </div>
                    {grp.numeric_averages && Object.keys(grp.numeric_averages).length > 0 && (
                      <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                        <strong>Averages: </strong>
                        {Object.entries(grp.numeric_averages).map(([k, v]) => `${k}: ${v}`).join(' | ')}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Searchable, Filterable Structured Output Table */}
          <div className="glass-panel" style={{ padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
              <h3 style={{ fontSize: '1.2rem' }}>
                Structured Records Output ({filteredRecords.length} / {rawRecords.length})
              </h3>
              
              <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', flexWrap: 'wrap' }}>
                {/* Search Bar */}
                <input
                  type="text"
                  placeholder="Search in table..."
                  value={searchQuery}
                  onChange={(e) => { setSearchQuery(e.target.value); setCurrentPage(1); }}
                  className="form-input"
                  style={{ padding: '0.45rem 0.85rem', fontSize: '0.85rem', width: '220px' }}
                />

                {/* Category Filter */}
                <select
                  value={selectedCategoryFilter}
                  onChange={(e) => { setSelectedCategoryFilter(e.target.value); setCurrentPage(1); }}
                  className="form-select"
                  style={{ padding: '0.45rem 0.85rem', fontSize: '0.85rem' }}
                >
                  <option value="ALL">All Categories</option>
                  {result.classification.generated_labels?.map((l, idx) => (
                    <option key={idx} value={l.label}>{l.label}</option>
                  ))}
                </select>

                {/* Page Size */}
                <select
                  value={pageSize}
                  onChange={(e) => { setPageSize(Number(e.target.value)); setCurrentPage(1); }}
                  className="form-select"
                  style={{ padding: '0.45rem 0.5rem', fontSize: '0.85rem' }}
                >
                  <option value={10}>10 / page</option>
                  <option value={25}>25 / page</option>
                  <option value={50}>50 / page</option>
                </select>
              </div>
            </div>

            {/* Table */}
            <div style={{ overflowX: 'auto', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', textAlign: 'left' }}>
                <thead>
                  <tr style={{ background: 'rgba(255, 255, 255, 0.04)', borderBottom: '1px solid var(--border-subtle)' }}>
                    {tableColumns.map((col, idx) => (
                      <th key={idx} style={{ padding: '0.75rem 1rem', fontWeight: '600', color: 'var(--text-secondary)' }}>
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {paginatedRecords.length === 0 ? (
                    <tr>
                      <td colSpan={tableColumns.length} style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                        No records match the active search filter.
                      </td>
                    </tr>
                  ) : (
                    paginatedRecords.map((row, rIdx) => (
                      <tr
                        key={rIdx}
                        style={{
                          borderBottom: '1px solid var(--border-subtle)',
                          background: rIdx % 2 === 0 ? 'transparent' : 'rgba(255, 255, 255, 0.015)',
                        }}
                      >
                        {tableColumns.map((col, cIdx) => {
                          const val = row[col];
                          // Render badges for special categorization fields
                          if (col === 'Primary_Label' || col === 'Category') {
                            return (
                              <td key={cIdx} style={{ padding: '0.75rem 1rem' }}>
                                <span className="badge badge-internal">{val !== null && val !== undefined ? String(val) : '—'}</span>
                              </td>
                            );
                          }
                          if (col === 'Sentiment') {
                            const badgeCls = val === 'POSITIVE' ? 'badge-buy' : val === 'NEGATIVE' ? 'badge-sell' : 'badge-hold';
                            return (
                              <td key={cIdx} style={{ padding: '0.75rem 1rem' }}>
                                <span className={`badge ${badgeCls}`}>{String(val)}</span>
                              </td>
                            );
                          }
                          if (col === 'Anomaly_Status' && val === 'Outlier') {
                            return (
                              <td key={cIdx} style={{ padding: '0.75rem 1rem' }}>
                                <span className="badge badge-sell">Outlier</span>
                              </td>
                            );
                          }
                          return (
                            <td key={cIdx} style={{ padding: '0.75rem 1rem', color: 'var(--text-primary)' }}>
                              {val !== null && val !== undefined ? String(val) : <span style={{ color: 'var(--text-muted)' }}>null</span>}
                            </td>
                          );
                        })}
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>

            {/* Pagination Controls */}
            {totalPages > 1 && (
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.5rem' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Page {currentPage} of {totalPages} ({filteredRecords.length} items)
                </span>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button
                    type="button"
                    className="btn btn-secondary"
                    disabled={currentPage <= 1}
                    onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                    style={{ padding: '0.35rem 0.75rem', fontSize: '0.8rem' }}
                  >
                    Previous
                  </button>
                  <button
                    type="button"
                    className="btn btn-secondary"
                    disabled={currentPage >= totalPages}
                    onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                    style={{ padding: '0.35rem 0.75rem', fontSize: '0.8rem' }}
                  >
                    Next
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
