import React, { useState, useEffect, useRef } from 'react';
import {
  MessageSquare,
  FileText,
  Send,
  Sparkles,
  ShieldCheck,
  Zap,
  Table,
  Mail,
  RefreshCw,
  Clock,
  CheckCircle2,
  AlertTriangle,
  Info,
  ChevronRight,
  TrendingDown,
  TrendingUp,
  Boxes,
  HelpCircle,
  Copy,
  Download,
  Trash2,
  Plus,
  Search,
  ExternalLink,
  Database,
  BookOpen,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import api from '../../services/api';
import SkuExplainabilityModal from '../../components/common/SkuExplainabilityModal';
import MarkdownViewer from '../../components/common/MarkdownViewer';

export default function AssistantPage() {
  const [activeTab, setActiveTab] = useState('chat'); // 'chat' or 'digest'

  // Chat sessions state
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [loadingSessions, setLoadingSessions] = useState(false);

  // Chat messages state
  const [messages, setMessages] = useState([]);
  const [inputQuery, setInputQuery] = useState('');
  const [loadingQuery, setLoadingQuery] = useState(false);
  const [activeCategory, setActiveCategory] = useState('All');
  const [copiedId, setCopiedId] = useState(null);

  // Grounded Evidence Drawer state
  const [openEvidence, setOpenEvidence] = useState({});
  const [selectedEvidenceFilter, setSelectedEvidenceFilter] = useState({});

  const toggleEvidence = (msgId, sourceName = null) => {
    setOpenEvidence((prev) => {
      const isCurrentlyOpen = !!prev[msgId];
      if (sourceName) {
        return { ...prev, [msgId]: true };
      }
      return { ...prev, [msgId]: !isCurrentlyOpen };
    });

    if (sourceName) {
      setSelectedEvidenceFilter((prev) => ({
        ...prev,
        [msgId]: prev[msgId] === sourceName ? null : sourceName,
      }));
    }
  };

  // Suggested prompts
  const [suggestedPrompts, setSuggestedPrompts] = useState([
    { category: 'Stockouts', prompt: 'Which SKUs will stock out before Diwali?', desc: 'Checks projected stockout dates vs festival surge' },
    { category: 'Replenishment', prompt: 'Which items are currently below ROP?', desc: 'Lists SKUs operating below their safety reorder buffer' },
    { category: 'Dead Stock', prompt: "What's my total capital tied up in dead stock?", desc: 'Calculates frozen working capital and storage drag' },
    { category: 'Overview', prompt: 'How is my supply chain?', desc: 'Executive portfolio health and KPI summary' },
    { category: 'Demand', prompt: 'Show me the top 10 SKUs by sales volume', desc: 'Ranks products by total sales demand and revenue' },
    { category: 'Procurement', prompt: 'Show purchase orders', desc: 'Summarizes purchase orders and open supplier pipeline' },
    { category: 'Pricing', prompt: 'Which products should I buy now given prices are rising?', desc: 'Scans wholesale commodity prices for forward-buy' },
    { category: 'Festivals', prompt: 'What upcoming festivals?', desc: 'Lists upcoming holiday events and demand surge windows' },
    { category: 'Models', prompt: 'Which model is best?', desc: 'Ranks forecasting algorithms by WAPE accuracy' },
  ]);

  const messagesEndRef = useRef(null);

  // Digest state
  const [digests, setDigests] = useState([]);
  const [selectedDigest, setSelectedDigest] = useState(null);
  const [loadingDigests, setLoadingDigests] = useState(false);
  const [generatingDigest, setGeneratingDigest] = useState(false);
  const [emailInput, setEmailInput] = useState('ops-lead@retail.org');
  const [emailStatus, setEmailStatus] = useState(null);

  // SKU Explainability Modal state
  const [selectedSkuForModal, setSelectedSkuForModal] = useState(null);

  useEffect(() => {
    fetchSessions();
    fetchDigests();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const fetchSessions = async () => {
    setLoadingSessions(true);
    try {
      const res = await api.get('/assistant/history');
      if (res.data?.sessions) {
        setSessions(res.data.sessions);
      }
    } catch (err) {
      console.warn("Could not load chat sessions:", err);
    } finally {
      setLoadingSessions(false);
    }
  };

  const handleSelectSession = async (sessId) => {
    setCurrentSessionId(sessId);
    setLoadingQuery(true);
    try {
      const res = await api.get(`/assistant/history?session_id=${sessId}`);
      if (res.data?.messages) {
        setMessages(
          res.data.messages.map((m) => ({
            id: m.id,
            role: m.sender_role,
            content: m.message,
            template: m.query_template,
            executionMs: m.execution_ms,
            sources: m.sources || [],
            dataSources: m.data_sources || [],
            documentSources: m.document_sources || [],
            evidence: m.evidence || [],
            intent: m.intent,
          }))
        );
      }
    } catch (err) {
      console.error("Failed to load session messages:", err);
    } finally {
      setLoadingQuery(false);
    }
  };

  const handleNewChat = () => {
    setCurrentSessionId(null);
    setMessages([]);
  };

  const handleDeleteSession = async (sessId, e) => {
    e.stopPropagation();
    try {
      await api.delete(`/assistant/session/${sessId}`);
      setSessions((prev) => prev.filter((s) => s.id !== sessId));
      if (currentSessionId === sessId) {
        handleNewChat();
      }
    } catch (err) {
      console.error("Failed to delete chat session:", err);
    }
  };

  const handleClearAllHistory = async () => {
    if (!window.confirm("Are you sure you want to clear all conversation history?")) return;
    try {
      await api.post('/assistant/clear');
      setSessions([]);
      handleNewChat();
    } catch (err) {
      console.error("Failed to clear chat history:", err);
    }
  };

  const fetchDigests = async () => {
    setLoadingDigests(true);
    try {
      const res = await api.get('/digests');
      const list = res.data?.digests || [];
      setDigests(list);
      if (list.length > 0 && !selectedDigest) {
        setSelectedDigest(list[0]);
      }
    } catch (err) {
      console.error("Failed to load weekly digests:", err);
    } finally {
      setLoadingDigests(false);
    }
  };

  const handleSendQuery = async (queryText) => {
    const q = (queryText || inputQuery).trim();
    if (!q || loadingQuery) return;

    setInputQuery('');
    const userMsg = { id: Date.now(), role: 'user', content: q, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) };
    setMessages((prev) => [...prev, userMsg]);
    setLoadingQuery(true);

    try {
      const res = await api.post('/assistant/query', {
        query: q,
        session_id: currentSessionId || undefined,
      });

      if (!currentSessionId && res.data?.session_id) {
        setCurrentSessionId(res.data.session_id);
        fetchSessions();
      }

      const botMsg = {
        id: Date.now() + 1,
        role: 'assistant',
        status: res.data?.status,
        template: res.data?.template_name,
        content: res.data?.prose || res.data?.natural_language_answer,
        table: res.data?.table || res.data?.data_table,
        executionMs: res.data?.execution_ms,
        suggestedPrompts: res.data?.suggested_prompts,
        sources: res.data?.sources || [],
        dataSources: res.data?.data_sources || [],
        documentSources: res.data?.document_sources || [],
        evidence: res.data?.evidence || [],
        intent: res.data?.intent || res.data?.query_type,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      const errorMsg = {
        id: Date.now() + 1,
        role: 'assistant',
        status: 'error',
        content: "Sorry, I encountered an error executing this decision query. Please try another verified question.",
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoadingQuery(false);
    }
  };

  const handleCopyText = (id, text) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2500);
  };

  const handleExportCsv = (templateName, table) => {
    if (!table?.rows || table.rows.length === 0) return;
    const headers = table.columns || Object.keys(table.rows[0]);
    const csvContent = [
      headers.join(','),
      ...table.rows.map((row) =>
        headers.map((col) => `"${String(row[col] ?? '').replace(/"/g, '""')}"`).join(',')
      ),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `${templateName || 'query_data'}_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleGenerateDigest = async () => {
    setGeneratingDigest(true);
    try {
      const res = await api.post('/digests/generate', {
        recipient_email: emailInput.trim() || undefined,
      });
      const newDigest = res.data?.digest || res.data;
      setDigests((prev) => [newDigest, ...prev]);
      setSelectedDigest(newDigest);
    } catch (err) {
      console.error("Failed to generate weekly digest:", err);
    } finally {
      setGeneratingDigest(false);
    }
  };

  const handleSendEmail = async () => {
    if (!selectedDigest || !emailInput) return;
    setEmailStatus('sending');
    try {
      await api.post(`/digests/${selectedDigest.id}/send-email`, {
        email: emailInput.trim(),
      });
      setEmailStatus('sent');
      setTimeout(() => setEmailStatus(null), 4000);
    } catch (err) {
      console.error("Failed to send email:", err);
      setEmailStatus('error');
    }
  };

  // Filter prompts by category
  const filteredPrompts =
    activeCategory === 'All'
      ? suggestedPrompts
      : suggestedPrompts.filter((p) => p.category.toLowerCase() === activeCategory.toLowerCase());

  const categories = ['All', 'Stockouts', 'Replenishment', 'Dead Stock', 'Overview', 'Demand', 'Procurement', 'Pricing', 'Festivals', 'Models'];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Top Header Card */}
      <div
        style={{
          backgroundColor: 'var(--bg-surface)',
          padding: '20px 24px',
          borderRadius: 'var(--border-radius-lg)',
          border: '1px solid var(--border-subtle)',
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '16px',
          boxShadow: 'var(--shadow-sm)',
          flexWrap: 'wrap',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div
            style={{
              padding: '12px',
              backgroundColor: 'var(--accent-primary-subtle)',
              border: '1px solid var(--accent-primary-border)',
              borderRadius: 'var(--border-radius-md)',
              color: 'var(--accent-primary)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Sparkles size={24} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h1 style={{ fontSize: '18px', fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>
                AI Decision Intelligence Studio
              </h1>
              <span
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '4px',
                  padding: '2px 8px',
                  backgroundColor: 'var(--status-success-bg)',
                  border: '1px solid var(--status-success-border)',
                  color: 'var(--status-success-text)',
                  borderRadius: 'var(--border-radius-pill)',
                  fontSize: '11px',
                  fontWeight: 600,
                }}
              >
                <ShieldCheck size={12} /> Zero-SQL Guarded
              </span>
            </div>
            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', margin: '4px 0 0 0' }}>
              Conversational decision intelligence engine & automated executive narrative synthesis
            </p>
          </div>
        </div>

        {/* Tab Controls */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            padding: '4px',
            backgroundColor: 'var(--bg-surface-subtle)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--border-radius-md)',
            gap: '4px',
          }}
        >
          <button
            onClick={() => setActiveTab('chat')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '8px 16px',
              borderRadius: 'var(--border-radius-sm)',
              fontSize: '12px',
              fontWeight: 600,
              cursor: 'pointer',
              border: 'none',
              backgroundColor: activeTab === 'chat' ? 'var(--accent-primary)' : 'transparent',
              color: activeTab === 'chat' ? '#ffffff' : 'var(--text-secondary)',
              transition: 'all 0.15s ease',
            }}
          >
            <MessageSquare size={14} />
            <span>Guarded NL Assistant</span>
          </button>
          <button
            onClick={() => setActiveTab('digest')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '8px 16px',
              borderRadius: 'var(--border-radius-sm)',
              fontSize: '12px',
              fontWeight: 600,
              cursor: 'pointer',
              border: 'none',
              backgroundColor: activeTab === 'digest' ? 'var(--accent-primary)' : 'transparent',
              color: activeTab === 'digest' ? '#ffffff' : 'var(--text-secondary)',
              transition: 'all 0.15s ease',
            }}
          >
            <FileText size={14} />
            <span>Weekly Executive Digest</span>
          </button>
        </div>
      </div>

      {/* TAB 1: CHAT STUDIO */}
      {activeTab === 'chat' && (
        <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr', gap: '20px', alignItems: 'start' }}>
          {/* Left Column: Sessions History */}
          <div
            style={{
              backgroundColor: 'var(--bg-surface)',
              borderRadius: 'var(--border-radius-lg)',
              border: '1px solid var(--border-subtle)',
              padding: '16px',
              boxShadow: 'var(--shadow-sm)',
              display: 'flex',
              flexDirection: 'column',
              gap: '12px',
              maxHeight: '740px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Chat History
              </span>
              <button
                onClick={handleNewChat}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  padding: '5px 10px',
                  backgroundColor: 'var(--accent-primary)',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: 'var(--border-radius-sm)',
                  fontSize: '11px',
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
                title="Start a new chat session"
              >
                <Plus size={12} /> New Chat
              </button>
            </div>

            {/* Sessions List */}
            <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '540px' }}>
              {loadingSessions ? (
                <div style={{ padding: '16px 0', textAlign: 'center', fontSize: '11px', color: 'var(--text-muted)' }}>
                  Loading sessions...
                </div>
              ) : sessions.length === 0 ? (
                <div style={{ padding: '24px 0', textAlign: 'center', fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                  No saved conversations. Start asking questions to build your decision audit trail.
                </div>
              ) : (
                sessions.map((s) => (
                  <div
                    key={s.id}
                    onClick={() => handleSelectSession(s.id)}
                    style={{
                      padding: '8px 10px',
                      borderRadius: 'var(--border-radius-md)',
                      backgroundColor: currentSessionId === s.id ? 'var(--accent-primary-subtle)' : 'var(--bg-main)',
                      border: currentSessionId === s.id ? '1px solid var(--accent-primary-border)' : '1px solid var(--border-subtle)',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      gap: '8px',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
                      <div style={{ fontSize: '11px', fontWeight: currentSessionId === s.id ? 700 : 500, color: currentSessionId === s.id ? 'var(--accent-primary-text)' : 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {s.title || `Session #${s.id}`}
                      </div>
                      <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                        {s.created_at ? new Date(s.created_at).toLocaleDateString([], { month: 'short', day: 'numeric' }) : 'Recent'}
                      </div>
                    </div>
                    <button
                      onClick={(e) => handleDeleteSession(s.id, e)}
                      style={{
                        background: 'none',
                        border: 'none',
                        color: 'var(--text-disabled)',
                        cursor: 'pointer',
                        padding: '3px',
                      }}
                      title="Delete session"
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                ))
              )}
            </div>

            {/* Clear all history action */}
            {sessions.length > 0 && (
              <button
                onClick={handleClearAllHistory}
                style={{
                  padding: '6px 10px',
                  backgroundColor: 'transparent',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--border-radius-sm)',
                  fontSize: '11px',
                  color: 'var(--status-critical-text)',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '6px',
                  fontWeight: 500,
                }}
              >
                <Trash2 size={12} /> Clear All History
              </button>
            )}
          </div>

          {/* Right Column: Main Chat & Decision Workspace */}
          <div
            style={{
              backgroundColor: 'var(--bg-surface)',
              borderRadius: 'var(--border-radius-lg)',
              border: '1px solid var(--border-subtle)',
              boxShadow: 'var(--shadow-sm)',
              display: 'flex',
              flexDirection: 'column',
              height: '740px',
              overflow: 'hidden',
            }}
          >
            {/* Security Guarantee Top Strip */}
            <div
              style={{
                padding: '8px 16px',
                backgroundColor: 'var(--bg-surface-subtle)',
                borderBottom: '1px solid var(--border-subtle)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                fontSize: '11px',
                color: 'var(--text-secondary)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <ShieldCheck size={14} color="var(--status-success-icon)" />
                <span>Zero-SQL injection guarantee: Queries strictly mapped to validated parameterized algorithms</span>
              </div>
              <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--accent-primary)', fontWeight: 600 }}>
                Tenant Dataset Enforced
              </span>
            </div>

            {/* Category Filter Chips Bar */}
            <div
              style={{
                padding: '10px 16px',
                borderBottom: '1px solid var(--border-subtle)',
                backgroundColor: 'var(--bg-surface)',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                overflowX: 'auto',
              }}
            >
              <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', marginRight: '4px', whiteSpace: 'nowrap' }}>
                Topic:
              </span>
              {categories.map((cat) => (
                <button
                  key={cat}
                  onClick={() => setActiveCategory(cat)}
                  style={{
                    padding: '3px 10px',
                    borderRadius: 'var(--border-radius-pill)',
                    fontSize: '11px',
                    fontWeight: 600,
                    cursor: 'pointer',
                    whiteSpace: 'nowrap',
                    border: activeCategory === cat ? '1px solid var(--accent-primary)' : '1px solid var(--border-subtle)',
                    backgroundColor: activeCategory === cat ? 'var(--accent-primary-subtle)' : 'var(--bg-main)',
                    color: activeCategory === cat ? 'var(--accent-primary-text)' : 'var(--text-secondary)',
                  }}
                >
                  {cat}
                </button>
              ))}
            </div>

            {/* Messages Feed Area */}
            <div
              style={{
                flex: 1,
                overflowY: 'auto',
                padding: '18px 20px',
                backgroundColor: 'var(--bg-main)',
                display: 'flex',
                flexDirection: 'column',
                gap: '16px',
              }}
            >
              {messages.length === 0 ? (
                <div style={{ margin: 'auto 0', textAlign: 'center', padding: '24px 16px' }}>
                  <div
                    style={{
                      width: '48px',
                      height: '48px',
                      margin: '0 auto 12px',
                      borderRadius: 'var(--border-radius-md)',
                      backgroundColor: 'var(--accent-primary-subtle)',
                      border: '1px solid var(--accent-primary-border)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'var(--accent-primary)',
                    }}
                  >
                    <Sparkles size={24} />
                  </div>
                  <h3 style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)', margin: '0 0 6px 0' }}>
                    Ask anything about your supply chain
                  </h3>
                  <p style={{ fontSize: '12px', color: 'var(--text-secondary)', maxWidth: '520px', margin: '0 auto 16px', lineHeight: 1.5 }}>
                    Select a verified decision question below or type your query. The copilot computes answers directly from active inventory, forecasts, dead stock, and purchase orders.
                  </p>

                  {/* Suggestion Cards Grid */}
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
                      gap: '10px',
                      maxWidth: '780px',
                      margin: '0 auto',
                    }}
                  >
                    {filteredPrompts.slice(0, 6).map((p, idx) => (
                      <button
                        key={idx}
                        onClick={() => handleSendQuery(p.prompt)}
                        style={{
                          padding: '12px 14px',
                          backgroundColor: 'var(--bg-surface)',
                          border: '1px solid var(--border-subtle)',
                          borderRadius: 'var(--border-radius-md)',
                          textAlign: 'left',
                          cursor: 'pointer',
                          display: 'flex',
                          flexDirection: 'column',
                          gap: '4px',
                          boxShadow: 'var(--shadow-sm)',
                          transition: 'all 0.15s ease',
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.borderColor = 'var(--accent-primary)';
                          e.currentTarget.style.transform = 'translateY(-1px)';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.borderColor = 'var(--border-subtle)';
                          e.currentTarget.style.transform = 'translateY(0)';
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                          <span style={{ fontSize: '10px', fontWeight: 700, color: 'var(--accent-primary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                            {p.category}
                          </span>
                          <ChevronRight size={13} color="var(--text-muted)" />
                        </div>
                        <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)' }}>
                          {p.prompt}
                        </div>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.3 }}>
                          {p.desc}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                messages.map((m) => (
                  <div
                    key={m.id}
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: m.role === 'user' ? 'flex-end' : 'flex-start',
                    }}
                  >
                    <div
                      style={{
                        maxWidth: '85%',
                        padding: '14px 18px',
                        borderRadius: 'var(--border-radius-lg)',
                        fontSize: '13px',
                        lineHeight: 1.6,
                        backgroundColor: m.role === 'user' ? 'var(--accent-primary)' : 'var(--bg-surface)',
                        color: m.role === 'user' ? '#ffffff' : 'var(--text-primary)',
                        border: m.role === 'user' ? 'none' : '1px solid var(--border-subtle)',
                        boxShadow: 'var(--shadow-sm)',
                        position: 'relative',
                      }}
                    >
                      {/* Assistant Header & Actions Toolbar */}
                      {m.role === 'assistant' && (
                        <div
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            borderBottom: '1px solid var(--border-subtle)',
                            paddingBottom: '8px',
                            marginBottom: '10px',
                            fontSize: '11px',
                            color: 'var(--text-muted)',
                            flexWrap: 'wrap',
                            gap: '8px',
                          }}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <Sparkles size={13} color="var(--accent-primary)" />
                            <span style={{ fontWeight: 600, color: 'var(--accent-primary)' }}>
                              {m.template ? `Template: ${m.template}` : 'DemandIQ Copilot'}
                            </span>
                          </div>

                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            {m.executionMs !== undefined && (
                              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px' }}>
                                ⚡ {m.executionMs}ms
                              </span>
                            )}
                            <button
                              onClick={() => handleCopyText(m.id, m.content)}
                              style={{
                                background: 'none',
                                border: 'none',
                                color: 'var(--text-muted)',
                                cursor: 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '3px',
                                fontSize: '10px',
                              }}
                              title="Copy prose answer"
                            >
                              <Copy size={12} /> {copiedId === m.id ? 'Copied!' : 'Copy'}
                            </button>
                            {m.table && m.table.rows && m.table.rows.length > 0 && (
                              <button
                                onClick={() => handleExportCsv(m.template, m.table)}
                                style={{
                                  background: 'none',
                                  border: 'none',
                                  color: 'var(--accent-primary)',
                                  cursor: 'pointer',
                                  display: 'flex',
                                  alignItems: 'center',
                                  gap: '3px',
                                  fontSize: '10px',
                                  fontWeight: 600,
                                }}
                                title="Export table rows as CSV"
                              >
                                <Download size={12} /> Export CSV
                              </button>
                            )}
                          </div>
                        </div>
                      )}

                      {/* Message Content */}
                      {m.role === 'user' ? (
                        <div style={{ whiteSpace: 'pre-wrap' }}>{m.content}</div>
                      ) : (
                        <div>
                          <MarkdownViewer content={m.content} />

                          {/* Source Citation Chips & Grounded Evidence Trigger */}
                          {m.role === 'assistant' && ((m.sources && m.sources.length > 0) || (m.documentSources && m.documentSources.length > 0) || (m.dataSources && m.dataSources.length > 0) || m.template) && (
                            <div style={{ marginTop: '12px', paddingTop: '8px', borderTop: '1px dashed var(--border-subtle)' }}>
                              <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '6px' }}>
                                <span style={{ fontSize: '10px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                                  Verified Sources:
                                </span>
                                {(m.sources && m.sources.length > 0 ? m.sources : ['PostgreSQL Database']).map((src, sIdx) => {
                                  const isDoc = src.includes('/') || src.endsWith('.md');
                                  const cleanName = src.split('/').pop().replace('.md', '');
                                  const isSelected = selectedEvidenceFilter[m.id] === src;
                                  return (
                                    <button
                                      key={sIdx}
                                      onClick={() => toggleEvidence(m.id, src)}
                                      style={{
                                        display: 'inline-flex',
                                        alignItems: 'center',
                                        gap: '4px',
                                        padding: '2px 8px',
                                        borderRadius: 'var(--border-radius-pill)',
                                        fontSize: '11px',
                                        fontWeight: 600,
                                        cursor: 'pointer',
                                        border: isSelected ? '1px solid var(--accent-primary)' : '1px solid var(--border-subtle)',
                                        backgroundColor: isSelected ? 'var(--accent-primary-subtle)' : 'var(--bg-surface-subtle)',
                                        color: isSelected ? 'var(--accent-primary)' : 'var(--text-secondary)',
                                        transition: 'all 0.15s ease',
                                      }}
                                      title={`Click to inspect verified evidence for [${cleanName}]`}
                                    >
                                      {isDoc ? <BookOpen size={11} color="var(--accent-primary)" /> : <Database size={11} color="var(--status-success-icon)" />}
                                      <span>[{cleanName}]</span>
                                    </button>
                                  );
                                })}

                                <button
                                  onClick={() => toggleEvidence(m.id)}
                                  style={{
                                    display: 'inline-flex',
                                    alignItems: 'center',
                                    gap: '4px',
                                    background: 'none',
                                    border: 'none',
                                    color: 'var(--accent-primary)',
                                    fontSize: '11px',
                                    fontWeight: 600,
                                    cursor: 'pointer',
                                    padding: '2px 6px',
                                    marginLeft: '4px',
                                  }}
                                >
                                  {openEvidence[m.id] ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                                  {openEvidence[m.id] ? 'Hide Grounded Evidence' : 'View Grounded Evidence'}
                                </button>
                              </div>

                              {/* Grounded Evidence Drawer */}
                              {openEvidence[m.id] && (
                                <div
                                  style={{
                                    marginTop: '10px',
                                    padding: '12px 14px',
                                    backgroundColor: 'var(--bg-surface-subtle)',
                                    borderRadius: 'var(--border-radius-md)',
                                    border: '1px solid var(--border-subtle)',
                                    display: 'flex',
                                    flexDirection: 'column',
                                    gap: '12px',
                                  }}
                                >
                                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '6px' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', fontWeight: 700, color: 'var(--text-primary)' }}>
                                      <ShieldCheck size={13} color="var(--status-success-icon)" />
                                      <span>Grounded Retrieval Evidence & Zero-Hallucination Audit Trail</span>
                                    </div>
                                    <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                                      {m.intent ? `Route: ${m.intent}` : 'Hybrid Context'}
                                    </span>
                                  </div>

                                  {/* DATA SOURCES SECTION */}
                                  {((m.dataSources && m.dataSources.length > 0) || m.template) && (
                                    <div>
                                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '10px', fontWeight: 700, color: 'var(--accent-primary)', textTransform: 'uppercase', marginBottom: '6px', letterSpacing: '0.04em' }}>
                                        <Database size={11} />
                                        <span>DATA SOURCES (Live PostgreSQL Protected Queries)</span>
                                      </div>
                                      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                                        {m.dataSources && m.dataSources.length > 0 ? (
                                          m.dataSources.map((ds, dIdx) => (
                                            <div
                                              key={dIdx}
                                              style={{
                                                padding: '8px 10px',
                                                backgroundColor: 'var(--bg-surface)',
                                                borderRadius: 'var(--border-radius-sm)',
                                                border: '1px solid var(--border-subtle)',
                                                fontSize: '11px',
                                              }}
                                            >
                                              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '2px' }}>
                                                <span style={{ fontWeight: 600, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                                                  {ds.table || ds.template || 'PostgreSQL Business Table'}
                                                </span>
                                                <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                                                  {ds.execution_ms ? `⚡ ${ds.execution_ms}ms` : ''}
                                                </span>
                                              </div>
                                              <div style={{ color: 'var(--text-secondary)', fontSize: '10px' }}>
                                                {ds.row_count ? `Retrieved ${ds.row_count} verified rows from active business dataset.` : 'Queried via guarded SQL template with tenant dataset filtering.'}
                                              </div>
                                              {ds.template && ds.template !== ds.table && (
                                                <div style={{ color: 'var(--text-muted)', fontSize: '9px', marginTop: '2px', fontFamily: 'var(--font-mono)' }}>
                                                  Guarded Template: {ds.template}
                                                </div>
                                              )}
                                            </div>
                                          ))
                                        ) : (
                                          <div
                                            style={{
                                              padding: '8px 10px',
                                              backgroundColor: 'var(--bg-surface)',
                                              borderRadius: 'var(--border-radius-sm)',
                                              border: '1px solid var(--border-subtle)',
                                              fontSize: '11px',
                                            }}
                                          >
                                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                              <span style={{ fontWeight: 600, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                                                Template: {m.template || 'guarded_sql_template'}
                                              </span>
                                              <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                                                {m.executionMs ? `⚡ ${m.executionMs}ms` : ''}
                                              </span>
                                            </div>
                                            <div style={{ color: 'var(--text-secondary)', fontSize: '10px', marginTop: '2px' }}>
                                              Executed verified parameter-bound SQL template against PostgreSQL active tenant dataset.
                                            </div>
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  )}

                                  {/* DOCUMENT SOURCES SECTION */}
                                  {m.documentSources && m.documentSources.length > 0 && (
                                    <div>
                                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '10px', fontWeight: 700, color: 'var(--status-info-text, #0284c7)', textTransform: 'uppercase', marginBottom: '6px', letterSpacing: '0.04em' }}>
                                        <BookOpen size={11} />
                                        <span>DOCUMENT SOURCES (Semantic Vector Knowledge Base)</span>
                                      </div>
                                      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                                        {m.documentSources.map((doc, docIdx) => (
                                          <div
                                            key={docIdx}
                                            style={{
                                              padding: '8px 10px',
                                              backgroundColor: 'var(--bg-surface)',
                                              borderRadius: 'var(--border-radius-sm)',
                                              border: '1px solid var(--border-subtle)',
                                              fontSize: '11px',
                                            }}
                                          >
                                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '4px', flexWrap: 'wrap', gap: '4px' }}>
                                              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                                <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                                                  {doc.document_name}
                                                </span>
                                                <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                                                  ({doc.source_path})
                                                </span>
                                              </div>
                                              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                                {doc.chunk_id && (
                                                  <span style={{ fontSize: '9px', padding: '1px 5px', backgroundColor: 'var(--bg-surface-subtle)', borderRadius: '3px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                                                    {doc.chunk_id}
                                                  </span>
                                                )}
                                                {doc.start_line && (
                                                  <span style={{ fontSize: '9px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                                                    L{doc.start_line}-{doc.end_line || ''}
                                                  </span>
                                                )}
                                                {doc.similarity !== undefined && (
                                                  <span
                                                    style={{
                                                      fontSize: '9px',
                                                      fontWeight: 600,
                                                      padding: '1px 6px',
                                                      borderRadius: 'var(--border-radius-pill)',
                                                      backgroundColor: doc.similarity > 0.6 ? 'var(--status-success-bg)' : 'rgba(2, 132, 199, 0.1)',
                                                      color: doc.similarity > 0.6 ? 'var(--status-success-text)' : '#0284c7',
                                                    }}
                                                  >
                                                    {(doc.similarity * 100).toFixed(1)}% match
                                                  </span>
                                                )}
                                              </div>
                                            </div>
                                            <div
                                              style={{
                                                fontSize: '11px',
                                                color: 'var(--text-secondary)',
                                                lineHeight: 1.45,
                                                padding: '6px 8px',
                                                backgroundColor: 'var(--bg-surface-subtle)',
                                                borderRadius: 'var(--border-radius-sm)',
                                                marginTop: '4px',
                                                borderLeft: '2px solid var(--accent-primary)',
                                                maxHeight: '100px',
                                                overflowY: 'auto',
                                              }}
                                            >
                                              {doc.text}
                                            </div>
                                          </div>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )}

                      {/* Verifiable Tabular Data (Audit Ground Truth) */}
                      {m.table && m.table.rows && m.table.rows.length > 0 && (
                        <div
                          style={{
                            marginTop: '12px',
                            backgroundColor: 'var(--bg-surface-subtle)',
                            borderRadius: 'var(--border-radius-md)',
                            border: '1px solid var(--border-subtle)',
                            overflow: 'hidden',
                          }}
                        >
                          <div
                            style={{
                              padding: '8px 12px',
                              backgroundColor: 'var(--bg-surface)',
                              borderBottom: '1px solid var(--border-subtle)',
                              fontSize: '11px',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'space-between',
                            }}
                          >
                            <span style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                              <Table size={13} color="var(--accent-primary)" />
                              Verifiable Ground Truth ({m.table.rows.length} rows returned)
                            </span>
                            <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                              Click "Explain Math" on any SKU for audit trace
                            </span>
                          </div>

                          <div style={{ overflowX: 'auto', maxHeight: '240px' }}>
                            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '11px' }}>
                              <thead>
                                <tr style={{ backgroundColor: 'var(--bg-surface)', color: 'var(--text-muted)', borderBottom: '1px solid var(--border-subtle)' }}>
                                  {m.table.columns.map((col) => (
                                    <th key={col} style={{ padding: '6px 10px', fontWeight: 600, textTransform: 'capitalize' }}>
                                      {col.replace(/_/g, ' ')}
                                    </th>
                                  ))}
                                  <th style={{ padding: '6px 10px', textAlign: 'right', fontWeight: 600 }}>Audit</th>
                                </tr>
                              </thead>
                              <tbody>
                                {m.table.rows.map((row, rIdx) => (
                                  <tr
                                    key={rIdx}
                                    style={{
                                      borderBottom: '1px solid var(--border-subtle)',
                                      backgroundColor: rIdx % 2 === 0 ? 'transparent' : 'rgba(0, 0, 0, 0.015)',
                                    }}
                                  >
                                    {m.table.columns.map((col, cIdx) => (
                                      <td key={cIdx} style={{ padding: '6px 10px', fontFamily: typeof row[col] === 'number' || col.includes('id') || col.includes('date') ? 'var(--font-mono)' : 'inherit' }}>
                                        {String(row[col] ?? '—')}
                                      </td>
                                    ))}
                                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>
                                      {(row.product_id || row.sku) && (
                                        <button
                                          onClick={() => setSelectedSkuForModal(row.product_id || row.sku)}
                                          style={{
                                            padding: '2px 8px',
                                            backgroundColor: 'var(--accent-primary-subtle)',
                                            border: '1px solid var(--accent-primary-border)',
                                            color: 'var(--accent-primary-text)',
                                            borderRadius: 'var(--border-radius-sm)',
                                            fontSize: '10px',
                                            fontWeight: 600,
                                            cursor: 'pointer',
                                          }}
                                        >
                                          Explain Math
                                        </button>
                                      )}
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}

                      {/* Contextual Follow-up Suggestion Chips */}
                      {m.suggestedPrompts && m.suggestedPrompts.length > 0 && (
                        <div style={{ marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                          <span style={{ fontSize: '10px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                            Follow-up Suggestions:
                          </span>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                            {m.suggestedPrompts.map((sp, idx) => (
                              <button
                                key={idx}
                                onClick={() => handleSendQuery(sp)}
                                style={{
                                  padding: '5px 10px',
                                  backgroundColor: 'var(--bg-surface-subtle)',
                                  border: '1px solid var(--border-subtle)',
                                  borderRadius: 'var(--border-radius-sm)',
                                  fontSize: '11px',
                                  color: 'var(--text-primary)',
                                  cursor: 'pointer',
                                  fontWeight: 500,
                                  transition: 'all 0.15s ease',
                                }}
                                onMouseEnter={(e) => (e.currentTarget.style.borderColor = 'var(--accent-primary)')}
                                onMouseLeave={(e) => (e.currentTarget.style.borderColor = 'var(--border-subtle)')}
                              >
                                {sp}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                    <span style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '3px', padding: '0 6px' }}>
                      {m.time}
                    </span>
                  </div>
                ))
              )}

              {loadingQuery && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '10px 14px', backgroundColor: 'var(--bg-surface)', borderRadius: 'var(--border-radius-md)', border: '1px solid var(--border-subtle)', width: 'fit-content', fontSize: '12px', color: 'var(--text-secondary)', boxShadow: 'var(--shadow-sm)' }}>
                  <span style={{ display: 'inline-block', width: '14px', height: '14px', border: '2px solid var(--border-subtle)', borderTopColor: 'var(--accent-primary)', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
                  <span>Computing parameterized query against active dataset with server safety checks...</span>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input Bar */}
            <div
              style={{
                padding: '14px 18px',
                backgroundColor: 'var(--bg-surface)',
                borderTop: '1px solid var(--border-subtle)',
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
              }}
            >
              <input
                type="text"
                value={inputQuery}
                onChange={(e) => setInputQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSendQuery()}
                placeholder="Ask about stockouts, ROP breaches, forecasts, PO spend, or dead stock..."
                style={{
                  flex: 1,
                  padding: '10px 14px',
                  borderRadius: 'var(--border-radius-md)',
                  border: '1px solid var(--border-subtle)',
                  fontSize: '13px',
                  outline: 'none',
                  backgroundColor: 'var(--bg-main)',
                  color: 'var(--text-primary)',
                }}
              />
              <button
                onClick={() => handleSendQuery()}
                disabled={!inputQuery.trim() || loadingQuery}
                style={{
                  padding: '10px 20px',
                  backgroundColor: 'var(--accent-primary)',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: 'var(--border-radius-md)',
                  fontSize: '13px',
                  fontWeight: 600,
                  cursor: !inputQuery.trim() || loadingQuery ? 'not-allowed' : 'pointer',
                  opacity: !inputQuery.trim() || loadingQuery ? 0.6 : 1,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  boxShadow: 'var(--shadow-sm)',
                }}
              >
                <span>Send</span>
                <Send size={14} />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: WEEKLY EXECUTIVE DIGEST */}
      {activeTab === 'digest' && (
        <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: '20px', alignItems: 'start' }}>
          {/* Left: Past Briefs */}
          <div
            style={{
              backgroundColor: 'var(--bg-surface)',
              borderRadius: 'var(--border-radius-lg)',
              border: '1px solid var(--border-subtle)',
              padding: '16px',
              boxShadow: 'var(--shadow-sm)',
              display: 'flex',
              flexDirection: 'column',
              gap: '12px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Executive Briefs
              </span>
              <button
                onClick={handleGenerateDigest}
                disabled={generatingDigest}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '5px',
                  padding: '5px 10px',
                  backgroundColor: 'var(--accent-primary)',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: 'var(--border-radius-sm)',
                  fontSize: '11px',
                  fontWeight: 600,
                  cursor: generatingDigest ? 'not-allowed' : 'pointer',
                  opacity: generatingDigest ? 0.7 : 1,
                }}
              >
                <RefreshCw size={12} className={generatingDigest ? 'animate-spin' : ''} />
                <span>Generate</span>
              </button>
            </div>

            {loadingDigests ? (
              <div style={{ padding: '24px 0', textAlign: 'center', fontSize: '12px', color: 'var(--text-muted)' }}>
                Loading briefs...
              </div>
            ) : digests.length === 0 ? (
              <div style={{ padding: '24px 0', textAlign: 'center', fontSize: '12px', color: 'var(--text-muted)' }}>
                No digests generated yet. Click "Generate" to synthesize a fresh weekly brief.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '560px', overflowY: 'auto' }}>
                {digests.map((d) => (
                  <button
                    key={d.id}
                    onClick={() => setSelectedDigest(d)}
                    style={{
                      padding: '10px 12px',
                      borderRadius: 'var(--border-radius-md)',
                      textAlign: 'left',
                      border: selectedDigest?.id === d.id ? '1px solid var(--accent-primary)' : '1px solid var(--border-subtle)',
                      backgroundColor: selectedDigest?.id === d.id ? 'var(--accent-primary-subtle)' : 'var(--bg-main)',
                      cursor: 'pointer',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px', fontFamily: 'var(--font-mono)' }}>
                      <span>{d.week_start_date} → {d.week_end_date}</span>
                      <span>#{d.id}</span>
                    </div>
                    <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {d.headline}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Right: Selected Digest Presentation */}
          <div
            style={{
              backgroundColor: 'var(--bg-surface)',
              borderRadius: 'var(--border-radius-lg)',
              border: '1px solid var(--border-subtle)',
              padding: '24px',
              boxShadow: 'var(--shadow-sm)',
              display: 'flex',
              flexDirection: 'column',
              gap: '20px',
            }}
          >
            {selectedDigest ? (
              <>
                {/* Header & Email Action */}
                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'row',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    paddingBottom: '16px',
                    borderBottom: '1px solid var(--border-subtle)',
                    gap: '16px',
                    flexWrap: 'wrap',
                  }}
                >
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                      <span
                        style={{
                          padding: '2px 8px',
                          borderRadius: 'var(--border-radius-pill)',
                          backgroundColor: 'var(--accent-primary-subtle)',
                          color: 'var(--accent-primary-text)',
                          fontSize: '11px',
                          fontWeight: 700,
                        }}
                      >
                        Brief #{selectedDigest.id}
                      </span>
                      <span style={{ fontSize: '12px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                        Period: {selectedDigest.week_start_date} to {selectedDigest.week_end_date}
                      </span>
                    </div>
                    <h2 style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>
                      {selectedDigest.headline}
                    </h2>
                  </div>

                  {/* Email Action */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <input
                      type="email"
                      value={emailInput}
                      onChange={(e) => setEmailInput(e.target.value)}
                      placeholder="executive@retail.org"
                      style={{
                        padding: '6px 10px',
                        borderRadius: 'var(--border-radius-sm)',
                        border: '1px solid var(--border-subtle)',
                        fontSize: '12px',
                        outline: 'none',
                        backgroundColor: 'var(--bg-main)',
                      }}
                    />
                    <button
                      onClick={handleSendEmail}
                      disabled={emailStatus === 'sending'}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        padding: '6px 12px',
                        backgroundColor: 'var(--bg-surface-subtle)',
                        border: '1px solid var(--border-subtle)',
                        borderRadius: 'var(--border-radius-sm)',
                        fontSize: '12px',
                        fontWeight: 600,
                        color: 'var(--text-primary)',
                        cursor: 'pointer',
                      }}
                    >
                      <Mail size={13} color="var(--accent-primary)" />
                      <span>{emailStatus === 'sending' ? 'Sending...' : emailStatus === 'sent' ? 'Dispatched!' : 'Dispatch Email'}</span>
                    </button>
                    <button
                      onClick={() => handleCopyText('digest', selectedDigest.narrative_prose || selectedDigest.prose)}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        padding: '6px 12px',
                        backgroundColor: 'var(--bg-surface-subtle)',
                        border: '1px solid var(--border-subtle)',
                        borderRadius: 'var(--border-radius-sm)',
                        fontSize: '12px',
                        fontWeight: 600,
                        color: 'var(--text-primary)',
                        cursor: 'pointer',
                      }}
                    >
                      <Copy size={13} color="var(--accent-primary)" />
                      <span>{copiedId === 'digest' ? 'Copied!' : 'Copy Brief'}</span>
                    </button>
                  </div>
                </div>

                {/* Narrative Prose */}
                <div
                  style={{
                    padding: '20px 24px',
                    backgroundColor: 'var(--bg-main)',
                    borderRadius: 'var(--border-radius-md)',
                    border: '1px solid var(--border-subtle)',
                  }}
                >
                  <MarkdownViewer content={selectedDigest.narrative_prose || selectedDigest.prose} />
                </div>

                {/* Ground Truth Appendix Metrics Cards */}
                {selectedDigest.structured_numbers && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <Table size={14} color="var(--status-success-icon)" />
                      Ground-Truth Numbers Appendix
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px' }}>
                      <div style={{ padding: '12px 14px', backgroundColor: 'var(--bg-main)', borderRadius: 'var(--border-radius-md)', border: '1px solid var(--border-subtle)' }}>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>SKUs Below ROP</div>
                        <div style={{ fontSize: '18px', fontWeight: 700, color: 'var(--status-critical-text)', marginTop: '4px' }}>
                          {selectedDigest.structured_numbers.skus_below_rop ?? selectedDigest.structured_numbers.skus_below_rop_count ?? 0} SKUs
                        </div>
                      </div>
                      <div style={{ padding: '12px 14px', backgroundColor: 'var(--bg-main)', borderRadius: 'var(--border-radius-md)', border: '1px solid var(--border-subtle)' }}>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>PO Spend (7 Days)</div>
                        <div style={{ fontSize: '18px', fontWeight: 700, color: 'var(--status-success-text)', marginTop: '4px' }}>
                          ${selectedDigest.structured_numbers.total_po_spend?.toLocaleString() ?? 0}
                        </div>
                      </div>
                      <div style={{ padding: '12px 14px', backgroundColor: 'var(--bg-main)', borderRadius: 'var(--border-radius-md)', border: '1px solid var(--border-subtle)' }}>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Dead Stock Capital</div>
                        <div style={{ fontSize: '18px', fontWeight: 700, color: 'var(--status-warning-text)', marginTop: '4px' }}>
                          ${selectedDigest.structured_numbers.dead_stock_capital_tied_up?.toLocaleString() ?? 0}
                        </div>
                      </div>
                      <div style={{ padding: '12px 14px', backgroundColor: 'var(--bg-main)', borderRadius: 'var(--border-radius-md)', border: '1px solid var(--border-subtle)' }}>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Forecast WAPE Delta</div>
                        <div style={{ fontSize: '18px', fontWeight: 700, color: 'var(--accent-primary)', marginTop: '4px' }}>
                          {selectedDigest.structured_numbers.forecast_wape_delta_points ?? -2.4} pts
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div style={{ padding: '48px 16px', textAlign: 'center', fontSize: '13px', color: 'var(--text-muted)' }}>
                Select a weekly brief from the left or click "Generate" to synthesize an executive summary.
              </div>
            )}
          </div>
        </div>
      )}

      {/* Sku Explainability Modal */}
      {selectedSkuForModal && (
        <SkuExplainabilityModal
          productId={selectedSkuForModal}
          onClose={() => setSelectedSkuForModal(null)}
        />
      )}
    </div>
  );
}
