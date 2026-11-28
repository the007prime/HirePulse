import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  clearAuthSession,
  deleteThread,
  fetchCurrentUser,
  fetchThreadMessages,
  fetchThreads,
  getStoredToken,
  getStoredUser,
  streamChat,
  uploadDocuments,
} from "./api";
import AuthScreen from "./components/AuthScreen";
import {
  Bot,
  User,
  Plus,
  Trash2,
  Square,
  ArrowUp,
  MessageSquare,
  AlertCircle,
  X,
  Paperclip,
  FileText,
  Loader2,
  CheckCircle2,
  Copy,
  Check,
  RotateCcw,
  Clock,
  LogOut,
  ExternalLink,
  Globe,
} from "lucide-react";
import "./App.css";

function PreBlock({ children, ...props }) {
  const [copied, setCopied] = useState(false);
  const codeChild = children;
  const className = codeChild?.props?.className || "";
  const match = /language-(\w+)/.exec(className);
  const lang = match ? match[1] : "";
  const rawCode = codeChild?.props?.children
    ? String(codeChild.props.children).replace(/\n$/, "")
    : "";

  async function handleCopyCode() {
    if (!rawCode) return;
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(rawCode);
      } else {
        const textArea = document.createElement("textarea");
        textArea.value = rawCode;
        textArea.style.position = "fixed";
        textArea.style.opacity = "0";
        textArea.style.left = "-9999px";
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        document.execCommand("copy");
        document.body.removeChild(textArea);
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy code:", err);
    }
  }

  return (
    <div className="code-block-wrapper">
      <div className="code-block-header">
        <span className="code-block-lang">{lang || "code"}</span>
        <button
          type="button"
          className="code-copy-btn"
          onClick={handleCopyCode}
          title="Copy code"
        >
          {copied ? (
            <>
              <Check size={12} className="copy-success-icon" />
              <span>Copied</span>
            </>
          ) : (
            <>
              <Copy size={12} />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>
      <pre className="code-block-pre" {...props}>
        {children}
      </pre>
    </div>
  );
}

function getSourceName(href, children) {
  if (!href) return children || "Link";
  let hostname = "";
  try {
    const url = new URL(href, window.location.origin);
    hostname = url.hostname.replace(/^www\./i, "");
  } catch {
    hostname = String(href).replace(/^https?:\/\//i, "").replace(/^www\./i, "").split("/")[0];
  }

  const knownSources = {
    "github.com": "GitHub",
    "wikipedia.org": "Wikipedia",
    "en.wikipedia.org": "Wikipedia",
    "linkedin.com": "LinkedIn",
    "twitter.com": "Twitter",
    "x.com": "X (Twitter)",
    "youtube.com": "YouTube",
    "medium.com": "Medium",
    "stackoverflow.com": "Stack Overflow",
    "reddit.com": "Reddit",
    "bloomberg.com": "Bloomberg",
    "reuters.com": "Reuters",
    "techcrunch.com": "TechCrunch",
    "theverge.com": "The Verge",
    "wsj.com": "Wall Street Journal",
    "nytimes.com": "New York Times",
    "cnbc.com": "CNBC",
    "cnn.com": "CNN",
    "bbc.com": "BBC",
    "bbc.co.uk": "BBC",
    "forbes.com": "Forbes",
    "duckduckgo.com": "DuckDuckGo",
    "google.com": "Google",
    "news.google.com": "Google News",
    "ycombinator.com": "Y Combinator",
    "news.ycombinator.com": "Hacker News",
    "docs.python.org": "Python Docs",
    "python.org": "Python",
    "careers.google.com": "Google Careers",
  };

  const domainLower = hostname.toLowerCase();
  let cleanSource = knownSources[domainLower];
  if (!cleanSource) {
    const parts = domainLower.split(".");
    if (parts.length >= 2) {
      const mainPart = parts[parts.length - 2];
      cleanSource = mainPart.charAt(0).toUpperCase() + mainPart.slice(1);
    } else {
      cleanSource = hostname;
    }
  }

  let text = "";
  if (typeof children === "string") text = children.trim();
  else if (Array.isArray(children)) text = children.map((c) => (typeof c === "string" ? c : "")).join("").trim();
  else if (children?.props?.children) text = String(children.props.children).trim();

  const isUrlText =
    !text ||
    /^https?:\/\//i.test(text) ||
    /^www\./i.test(text) ||
    text.includes("http://") ||
    text.includes("https://") ||
    text === href;

  if (isUrlText) {
    return cleanSource || hostname || "Source";
  }
  return text;
}

function SourceLink({ href, children, ...props }) {
  const [isHovered, setIsHovered] = useState(false);
  const sourceName = getSourceName(href, children);

  return (
    <span
      className="source-link-wrapper"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="source-link-pill"
        {...props}
      >
        <ExternalLink size={12} className="source-link-icon" />
        <span className="source-link-text">{sourceName}</span>
      </a>

      {isHovered && href && (
        <span className="url-popup-card" role="tooltip">
          <span className="url-popup-header">
            <Globe size={11} className="url-popup-icon" />
            <span className="url-popup-domain">{sourceName}</span>
          </span>
          <span className="url-popup-href">{href}</span>
          <span className="url-popup-hint">Click to open link ↗</span>
          <span className="url-popup-arrow" />
        </span>
      )}
    </span>
  );
}

const markdownComponents = {
  pre: PreBlock,
  table: ({ node, ...props }) => (
    <div className="table-responsive">
      <table {...props} />
    </div>
  ),
  a: SourceLink,
};

function cleanMessageContent(content) {
  if (!content) return "";
  let text = String(content);
  // Remove <!-- out_of_context --> marker
  text = text.replace(/<!--\s*out_of_context\s*-->/gi, "").trim();
  // Strip any leading out-of-context phrases from the beginning of the response
  text = text
    .replace(
      /^(your\s+request\s+appears\s+to\s+be\s+outside\s+(my\s+)?(designated\s+)?scope[.\s!]*|this\s+(request|question|topic)\s+is\s+(outside\s+(my\s+)?scope|out\s+of\s+context)[.\s!]*)/i,
      ""
    )
    .trim();
  return text;
}

function isOutOfContextMessage(content) {
  if (!content) return false;
  const lower = String(content).toLowerCase();
  return (
    lower.includes("<!-- out_of_context -->") ||
    lower.includes("outside my designated scope") ||
    lower.includes("start a new chat below") ||
    lower.includes("i specialize strictly in career and job search")
  );
}

export default function App() {
  const [user, setUser] = useState(null);
  const [isCheckingAuth, setIsCheckingAuth] = useState(true);
  const [authMessage, setAuthMessage] = useState("");
  const [threads, setThreads] = useState([]);
  const [activeThreadId, setActiveThreadId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [activeFiles, setActiveFiles] = useState([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");
  const [copiedIdx, setCopiedIdx] = useState(null);
  const [isLoadingThread, setIsLoadingThread] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState("");
  const [error, setError] = useState("");

  const messagesEndRef = useRef(null);
  const abortRef = useRef(null);
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);
  const inputStartTimeRef = useRef(null);
  const lastTokenTimeRef = useRef(null);

  function handleLogout(reason = "") {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    clearAuthSession();
    setUser(null);
    setThreads([]);
    setActiveThreadId(null);
    setMessages([]);
    setActiveFiles([]);
    setStreamingContent("");
    setIsStreaming(false);
    setError("");
    setUploadSuccess("");
    setAuthMessage(reason);
  }

  async function loadThreads(autoSelectLatest = false) {
    try {
      const list = await fetchThreads();
      setThreads(list);
      if (autoSelectLatest && list.length > 0) {
        handleSelectThread(list[0].id);
      }
    } catch (err) {
      if (err?.status === 401) {
        handleLogout("Your session expired. Please sign in again.");
        return;
      }
      setError(err.message);
    }
  }

  async function handleAuthenticated(authedUser) {
    setAuthMessage("");
    setError("");
    setActiveThreadId(null);
    setMessages([]);
    setActiveFiles([]);
    setUser(authedUser);
    await loadThreads(true);
  }

  useEffect(() => {
    let cancelled = false;

    async function initAuth() {
      const token = getStoredToken();
      if (!token) {
        if (!cancelled) setIsCheckingAuth(false);
        return;
      }

      try {
        const me = await fetchCurrentUser();
        if (cancelled) return;
        const stored = getStoredUser();
        setUser({
          id: me?.id || stored?.id,
          email: me?.email || stored?.email || "",
        });
        await loadThreads(true);
      } catch {
        if (!cancelled) {
          clearAuthSession();
          setUser(null);
        }
      } finally {
        if (!cancelled) setIsCheckingAuth(false);
      }
    }

    initAuth();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent]);

  async function handleSelectThread(threadId) {
    if (isStreaming || threadId === activeThreadId) return;
    setError("");
    setUploadSuccess("");
    setActiveThreadId(threadId);
    setIsLoadingThread(true);
    setStreamingContent("");
    try {
      const { messages: history, files } = await fetchThreadMessages(threadId);
      setMessages(history);
      setActiveFiles(files);
    } catch (err) {
      if (err?.status === 401) {
        handleLogout("Your session expired. Please sign in again.");
        return;
      }
      setError(err.message);
    } finally {
      setIsLoadingThread(false);
    }
  }

  function handleNewChat() {
    if (isStreaming) return;
    setError("");
    setUploadSuccess("");
    setActiveThreadId(null);
    setMessages([]);
    setActiveFiles([]);
    setStreamingContent("");
    textareaRef.current?.focus();
  }

  async function handleDeleteThread(e, threadId) {
    e.stopPropagation();
    if (isStreaming) return;
    try {
      await deleteThread(threadId);
      setThreads((prev) => prev.filter((t) => t.id !== threadId));
      if (activeThreadId === threadId) {
        handleNewChat();
      }
    } catch (err) {
      if (err?.status === 401) {
        handleLogout("Your session expired. Please sign in again.");
        return;
      }
      setError(err.message);
    }
  }

  function handleStop() {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    const finalTokenTimestamp = lastTokenTimeRef.current || performance.now();
    const durationSec = inputStartTimeRef.current
      ? ((finalTokenTimestamp - inputStartTimeRef.current) / 1000).toFixed(2)
      : null;

    if (streamingContent) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: streamingContent,
          lastTokenDuration: durationSec,
        },
      ]);
    }
    setStreamingContent("");
    setIsStreaming(false);
  }

  async function handleFileUpload(e) {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;

    setError("");
    setUploadSuccess("");
    setIsUploading(true);

    try {
      const res = await uploadDocuments(files, activeThreadId);
      if (!activeThreadId && res.thread_id) {
        setActiveThreadId(res.thread_id);
      }
      setActiveFiles(res.all_thread_files || []);
      setUploadSuccess(
        `Uploaded & indexed ${res.uploaded_files.length} file(s) into chat memory.`
      );
      await loadThreads();
    } catch (err) {
      if (err?.status === 401) {
        handleLogout("Your session expired. Please sign in again.");
        return;
      }
      setError(err.message || "Failed to upload document");
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function executeChatStream(promptText, threadId = activeThreadId) {
    setError("");
    setUploadSuccess("");
    setIsStreaming(true);
    setStreamingContent("");
    inputStartTimeRef.current = performance.now();
    lastTokenTimeRef.current = null;

    const controller = new AbortController();
    abortRef.current = controller;
    let accumulated = "";
    let currentThreadId = threadId;

    try {
      await streamChat({
        threadId: currentThreadId,
        message: promptText,
        signal: controller.signal,
        onMeta: (evt) => {
          if (evt.thread_id && !currentThreadId) {
            currentThreadId = evt.thread_id;
            setActiveThreadId(evt.thread_id);
          }
        },
        onToken: (token) => {
          accumulated += token;
          lastTokenTimeRef.current = performance.now();
          setStreamingContent((prev) => prev + token);
        },
        onError: (errMsg) => {
          setError(errMsg || "Streaming error encountered");
        },
      });

      const finalTokenTimestamp = lastTokenTimeRef.current || performance.now();
      const durationSec = inputStartTimeRef.current
        ? ((finalTokenTimestamp - inputStartTimeRef.current) / 1000).toFixed(2)
        : null;

      if (accumulated) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: accumulated,
            lastTokenDuration: durationSec,
          },
        ]);
      }
      setStreamingContent("");
      await loadThreads();
    } catch (err) {
      if (err?.status === 401) {
        handleLogout("Your session expired. Please sign in again.");
        return;
      }
      if (err.name !== "AbortError") {
        setError(err.message || "Failed to stream message");
      }
    } finally {
      abortRef.current = null;
      setIsStreaming(false);
      setStreamingContent("");
    }
  }

  async function handleSend(e) {
    e?.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isStreaming) return;

    setInput("");
    const userMsg = { role: "user", content: trimmed };
    setMessages((prev) => [...prev, userMsg]);

    await executeChatStream(trimmed, activeThreadId);
  }

  async function handleRegenerate(targetIdx = null) {
    if (isStreaming) return;

    let assistantIndex = targetIdx;
    if (assistantIndex === null || assistantIndex === undefined) {
      for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i].role === "assistant") {
          assistantIndex = i;
          break;
        }
      }
    }

    if (assistantIndex === null || assistantIndex === undefined || assistantIndex < 0) {
      const lastUser = [...messages].reverse().find((m) => m.role === "user");
      if (lastUser) {
        await executeChatStream(lastUser.content, activeThreadId);
      }
      return;
    }

    let promptText = "";
    for (let i = assistantIndex - 1; i >= 0; i--) {
      if (messages[i].role === "user") {
        promptText = messages[i].content;
        break;
      }
    }

    if (!promptText) return;

    // Remove the assistant message being regenerated and any following messages
    setMessages((prev) => prev.slice(0, assistantIndex));

    await executeChatStream(promptText, activeThreadId);
  }

  async function handleCopy(text, idx) {
    if (!text) return;
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        const textArea = document.createElement("textarea");
        textArea.value = text;
        textArea.style.position = "fixed";
        textArea.style.opacity = "0";
        textArea.style.left = "-9999px";
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        document.execCommand("copy");
        document.body.removeChild(textArea);
      }
      setCopiedIdx(idx);
      setTimeout(() => {
        setCopiedIdx((current) => (current === idx ? null : current));
      }, 2000);
    } catch (err) {
      console.error("Failed to copy text:", err);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  const activeThread = threads.find((t) => t.id === activeThreadId);
  const canRegenerateLast =
    !isStreaming &&
    messages.length > 0 &&
    messages[messages.length - 1].role === "assistant";

  if (isCheckingAuth) {
    return (
      <div className="auth-screen">
        <div className="auth-loading">
          <Loader2 size={20} className="spin" />
          <span>Loading session...</span>
        </div>
      </div>
    );
  }

  if (!user) {
    return (
      <AuthScreen
        onAuthenticated={handleAuthenticated}
        initialMessage={authMessage}
      />
    );
  }

  return (
    <div className="app-shell">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="brand-row">
            <span className="brand-dot" />
            <span className="brand">Agentic RAG Chat</span>
          </div>
          <button
            type="button"
            className="new-chat-btn"
            onClick={handleNewChat}
            disabled={isStreaming}
          >
            <Plus size={16} />
            <span>New Chat</span>
          </button>
        </div>

        <div className="thread-list">
          {threads.length === 0 ? (
            <div className="thread-empty">No conversations yet</div>
          ) : (
            threads.map((thread) => (
              <div
                key={thread.id}
                role="button"
                tabIndex={0}
                className={`thread-item ${
                  thread.id === activeThreadId ? "active" : ""
                }`}
                onClick={() => handleSelectThread(thread.id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    handleSelectThread(thread.id);
                  }
                }}
              >
                <MessageSquare size={15} className="thread-icon" />
                <div className="thread-info-stack">
                  <span className="thread-title">{thread.title}</span>
                  {thread.files && thread.files.length > 0 && (
                    <span className="thread-file-badge">
                      <FileText size={10} />
                      {thread.files.length} file(s)
                    </span>
                  )}
                </div>
                <button
                  type="button"
                  className="delete-thread-btn"
                  title="Delete thread"
                  onClick={(e) => handleDeleteThread(e, thread.id)}
                  disabled={isStreaming}
                >
                  <Trash2 size={13} />
                </button>
              </div>
            ))
          )}
        </div>

        <div className="sidebar-footer">
          <div className="user-pill" title={user.email}>
            <div className="user-avatar">
              <User size={14} />
            </div>
            <span className="user-email">{user.email}</span>
          </div>
          <button
            type="button"
            className="logout-btn"
            title="Sign out"
            onClick={() => handleLogout("")}
            disabled={isStreaming}
          >
            <LogOut size={15} />
          </button>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="chat-main">
        <header className="chat-header">
          <div className="chat-header-info">
            <h1>{activeThread ? activeThread.title : "New Chat"}</h1>
            {activeThreadId && (
              <span className="thread-badge" title={activeThreadId}>
                ID: {activeThreadId.slice(0, 8)}
              </span>
            )}
          </div>
          <div className="chat-header-actions">
            {isStreaming && (
              <div className="streaming-badge">
                <span className="pulse-dot" />
                <span>Streaming</span>
              </div>
            )}
          </div>
        </header>

        {/* Active Files Pill Banner */}
        {activeFiles.length > 0 && (
          <div className="thread-context-banner">
            <span className="context-label">Chat Memory Context:</span>
            <div className="file-pills">
              {activeFiles.map((file, idx) => (
                <span key={idx} className="file-pill">
                  <FileText size={12} />
                  {file}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Error or Success Banners */}
        {error && (
          <div className="error-banner">
            <div className="error-msg">
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
            <button
              type="button"
              className="error-close"
              onClick={() => setError("")}
            >
              <X size={14} />
            </button>
          </div>
        )}

        {uploadSuccess && (
          <div className="success-banner">
            <div className="success-msg">
              <CheckCircle2 size={16} />
              <span>{uploadSuccess}</span>
            </div>
            <button
              type="button"
              className="error-close"
              onClick={() => setUploadSuccess("")}
            >
              <X size={14} />
            </button>
          </div>
        )}

        {/* Messages */}
        <div className="messages-container">
          {isLoadingThread ? (
            <div className="empty-state">Loading conversation...</div>
          ) : messages.length === 0 && !isStreaming ? (
            <div className="empty-state">
              <h2>Upload & Ask Anything</h2>
              <p>
                Attach a resume (PDF, CSV, Markdown), README (.md), or portfolio files,
                provide a GitHub profile link to evaluate, or ask any career question.
              </p>
            </div>
          ) : (
            <>
              {messages.map((msg, idx) => {
                const isOutOfCtx = msg.role === "assistant" && isOutOfContextMessage(msg.content);
                const displayContent = msg.role === "assistant" ? cleanMessageContent(msg.content) : msg.content;
                return (
                  <div key={idx} className={`message-row ${msg.role} ${isOutOfCtx ? "out-of-context-row" : ""}`}>
                    <div className="avatar">
                      {msg.role === "user" ? <User size={16} /> : <Bot size={16} />}
                    </div>
                    <div className="message-bubble">
                      {msg.role === "assistant" ? (
                        <div className="message-content markdown-body">
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={markdownComponents}
                          >
                            {displayContent}
                          </ReactMarkdown>

                          {isOutOfCtx && (
                            <div className="out-of-context-cta">
                              <div className="out-of-context-divider" />
                              <button
                                type="button"
                                className="start-new-chat-pill-btn"
                                onClick={handleNewChat}
                                title="Start a fresh conversation"
                              >
                                <Plus size={15} />
                                <span>Start a New Chat</span>
                              </button>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="message-content">{msg.content}</div>
                      )}

                      {/* Message metadata & actions bar */}
                      <div className="message-footer">
                        {msg.role === "assistant" && msg.lastTokenDuration ? (
                          <div
                            className="last-token-time"
                            title={`Time from user input to last token: ${msg.lastTokenDuration}s`}
                          >
                            <Clock size={12} />
                            <span>Time taken: {msg.lastTokenDuration}s</span>
                          </div>
                        ) : (
                          <span className="footer-spacer" />
                        )}

                        <div className="message-actions">
                          <button
                            type="button"
                            className="msg-action-btn"
                            title={copiedIdx === idx ? "Copied to clipboard!" : "Copy text"}
                            onClick={() => handleCopy(cleanMessageContent(msg.content), idx)}
                          >
                            {copiedIdx === idx ? (
                              <>
                                <Check size={12} className="copy-success-icon" />
                                <span>Copied</span>
                              </>
                            ) : (
                              <>
                                <Copy size={12} />
                                <span>Copy</span>
                              </>
                            )}
                          </button>

                          {msg.role === "assistant" && (
                            <button
                              type="button"
                              className="msg-action-btn"
                              title="Regenerate this response"
                              onClick={() => handleRegenerate(idx)}
                              disabled={isStreaming}
                            >
                              <RotateCcw size={12} />
                              <span>Regenerate</span>
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}

              {/* Streaming token bubble */}
              {isStreaming && (
                <div className={`message-row assistant streaming ${isOutOfContextMessage(streamingContent) ? "out-of-context-row" : ""}`}>
                  <div className="avatar">
                    <Bot size={16} />
                  </div>
                  <div className="message-bubble">
                    <div className="message-content markdown-body">
                      {streamingContent ? (
                        <>
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={markdownComponents}
                          >
                            {cleanMessageContent(streamingContent)}
                          </ReactMarkdown>
                          <span className="cursor-blink" />

                          {isOutOfContextMessage(streamingContent) && (
                            <div className="out-of-context-cta">
                              <div className="out-of-context-divider" />
                              <button
                                type="button"
                                className="start-new-chat-pill-btn"
                                onClick={handleNewChat}
                                title="Start a fresh conversation"
                              >
                                <Plus size={15} />
                                <span>Start a New Chat</span>
                              </button>
                            </div>
                          )}
                        </>
                      ) : (
                        <div className="thinking-row">
                          <span className="thinking-text">
                            Analyzing context & generating...
                          </span>
                          <span className="cursor-blink" />
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Quick Regenerate button below the conversation */}
              {canRegenerateLast && (
                <div className="bottom-regenerate-wrapper">
                  <button
                    type="button"
                    className="regenerate-bottom-btn"
                    onClick={() => handleRegenerate()}
                    disabled={isStreaming}
                    title="Regenerate the latest response"
                  >
                    <RotateCcw size={13} />
                    <span>Regenerate response</span>
                  </button>
                </div>
              )}
            </>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Composer with Attachment support */}
        <form className="composer-wrapper" onSubmit={handleSend}>
          <div className="composer-box">
            {/* Hidden file input */}
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.csv,.md,.markdown,.txt,.json,.yaml,.yml,.rst,.html,.htm,.py,.log"
              style={{ display: "none" }}
              onChange={handleFileUpload}
            />

            <button
              type="button"
              className="attach-btn"
              title="Upload PDF, CSV, README (.md), YAML, or text files to chat memory"
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading || isStreaming}
            >
              {isUploading ? (
                <Loader2 size={18} className="spin-icon" />
              ) : (
                <Paperclip size={18} />
              )}
            </button>

            <textarea
              ref={textareaRef}
              rows={2}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                isUploading
                  ? "Indexing document..."
                  : "Ask a question, paste a GitHub link, or upload a resume / README... (Enter to send)"
              }
              disabled={isStreaming}
            />

            <div className="composer-actions">
              {isStreaming ? (
                <button
                  type="button"
                  className="action-btn stop-btn"
                  onClick={handleStop}
                  title="Stop generating"
                >
                  <Square size={14} fill="currentColor" />
                  <span>Stop</span>
                </button>
              ) : (
                <button
                  type="submit"
                  className="action-btn send-btn"
                  disabled={!input.trim() || isUploading}
                  title="Send message"
                >
                  <ArrowUp size={16} />
                </button>
              )}
            </div>
          </div>
        </form>
      </main>
    </div>
  );
}
