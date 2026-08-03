(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.WorkAssistantMessageFormatter = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function normalizeLanguage(language) {
    const value = String(language || "text").trim().toLowerCase();
    const aliases = {
      shell: "bash", sh: "bash", zsh: "bash", cmd: "batch",
      ps: "powershell", ps1: "powershell", py: "python",
      js: "javascript", ts: "typescript", yml: "yaml",
    };
    return aliases[value] || value.replace(/[^a-z0-9_+#.-]/g, "") || "text";
  }

  function toolLanguage(toolName) {
    const name = String(toolName || "").toLowerCase();
    if (/bash|shell|terminal/.test(name)) return "bash";
    if (/powershell|pwsh/.test(name)) return "powershell";
    if (/python/.test(name)) return "python";
    if (/json|api|http/.test(name)) return "json";
    return "text";
  }

  function parseToolFields(body) {
    const fields = [];
    let current = null;
    for (const line of String(body || "").replace(/^\s+|\s+$/g, "").split(/\r?\n/)) {
      const match = line.match(/^([A-Za-z_][A-Za-z0-9_ -]{0,40}):\s*(.*)$/);
      if (match) {
        current = { key: match[1].trim(), value: match[2] };
        fields.push(current);
      } else if (current) {
        current.value += "\n" + line;
      } else if (line.trim()) {
        current = { key: "content", value: line };
        fields.push(current);
      }
    }
    return fields;
  }

  function renderCodeBlock(code, language, title) {
    const normalized = normalizeLanguage(language);
    const label = title || normalized;
    return `<div class="content-code-block language-${normalized}">` +
      `<div class="content-code-header"><span>${escapeHtml(label)}</span></div>` +
      `<pre><code>${escapeHtml(String(code || "").replace(/^\n|\n$/g, ""))}</code></pre>` +
      `</div>`;
  }

  function renderToolCall(toolName, body) {
    const name = String(toolName || "Tool").trim();
    const fields = parseToolFields(body);
    const commandField = fields.find(field => /^(command|cmd|script)$/i.test(field.key));
    const descriptionField = fields.find(field => /^description$/i.test(field.key));
    const remaining = fields.filter(field => field !== commandField && field !== descriptionField);
    const fallbackContent = fields.length === 1 && fields[0].key === "content" ? fields[0].value : "";

    let details = "";
    if (descriptionField && descriptionField.value.trim()) {
      details += `<p class="tool-description">${escapeHtml(descriptionField.value.trim())}</p>`;
    }
    if (remaining.length) {
      details += '<dl class="tool-properties">' + remaining.map(field =>
        `<div><dt>${escapeHtml(field.key)}</dt><dd>${escapeHtml(field.value)}</dd></div>`
      ).join("") + "</dl>";
    }
    const code = commandField ? commandField.value : fallbackContent;
    if (code) details += renderCodeBlock(code, toolLanguage(name), toolLanguage(name));

    return `<section class="external-tool-card tool-call-card">` +
      `<header class="external-tool-header"><span class="tool-terminal-icon">&gt;_</span>` +
      `<strong>${escapeHtml(name)}</strong><span>工具调用</span></header>` +
      `<div class="external-tool-body">${details || renderCodeBlock(body, toolLanguage(name))}</div>` +
      `</section>`;
  }

  function renderToolResult(status, body) {
    const normalizedStatus = String(status || "result").trim().toLowerCase();
    const isError = /error|failed|failure|rejected|denied/.test(normalizedStatus) ||
      /rejected|doesn't want to proceed|permission denied|失败|拒绝/i.test(body);
    const isSuccess = /success|ok|completed/.test(normalizedStatus) && !isError;
    const stateClass = isError ? "tool-result-error" : isSuccess ? "tool-result-success" : "tool-result-neutral";
    const stateLabel = isError ? "失败" : isSuccess ? "成功" : "结果";
    return `<section class="external-tool-card tool-result-card ${stateClass}">` +
      `<header class="external-tool-header"><span class="tool-result-dot"></span>` +
      `<strong>工具结果</strong><span>${stateLabel}</span></header>` +
      `<div class="external-tool-result-text">${escapeHtml(String(body || "").trim())}</div>` +
      `</section>`;
  }

  function renderInline(value) {
    let text = value;
    const protectedParts = [];
    const protect = html => {
      const index = protectedParts.push(html) - 1;
      return `\uE100${index}\uE101`;
    };

    text = text.replace(/`([^`\n]+)`/g, (_, code) =>
      protect(`<code class="inline-code">${code}</code>`)
    );
    text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, (_, label, url) =>
      protect(`<a href="${url}" target="_blank" rel="noopener noreferrer">${label}</a>`)
    );
    text = text.replace(/(^|[\s(])(https?:\/\/[^\s<]+)/g, (_, prefix, url) => {
      const punctuation = url.match(/[.,;:!?，。；：！？]+$/)?.[0] || "";
      const cleanUrl = punctuation ? url.slice(0, -punctuation.length) : url;
      return prefix + protect(`<a href="${cleanUrl}" target="_blank" rel="noopener noreferrer">${cleanUrl}</a>`) + punctuation;
    });
    text = text
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/__([^_]+)__/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
      .replace(/\[✓\]/g, '<span class="step-done">[✓]</span>')
      .replace(/\[→\]/g, '<span class="step-active">[→]</span>')
      .replace(/\[○\]/g, '<span class="step-pending">[○]</span>')
      .replace(/█+░*/g, match => `<span class="progress-bar">${match}</span>`);

    return text.replace(/\uE100(\d+)\uE101/g, (_, index) => protectedParts[Number(index)] || "");
  }

  function renderMarkdownText(escapedText, blockParts) {
    const lines = escapedText.split("\n");
    const html = [];
    let paragraph = [];
    let listType = null;
    let listItems = [];

    const flushParagraph = () => {
      if (!paragraph.length) return;
      html.push(`<p>${paragraph.map(renderInline).join("<br>")}</p>`);
      paragraph = [];
    };
    const flushList = () => {
      if (!listType) return;
      html.push(`<${listType}>${listItems.map(item => `<li>${renderInline(item)}</li>`).join("")}</${listType}>`);
      listType = null;
      listItems = [];
    };

    for (const line of lines) {
      const blockMatch = line.trim().match(/^\uE000(\d+)\uE001$/);
      if (blockMatch) {
        flushParagraph();
        flushList();
        html.push(blockParts[Number(blockMatch[1])] || "");
        continue;
      }
      if (!line.trim()) {
        flushParagraph();
        flushList();
        continue;
      }
      const heading = line.match(/^(#{1,4})\s+(.+)$/);
      if (heading) {
        flushParagraph();
        flushList();
        const level = heading[1].length + 1;
        html.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
        continue;
      }
      const unordered = line.match(/^\s*[-*+]\s+(.+)$/);
      const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
      if (unordered || ordered) {
        flushParagraph();
        const nextType = unordered ? "ul" : "ol";
        if (listType && listType !== nextType) flushList();
        listType = nextType;
        listItems.push((unordered || ordered)[1]);
        continue;
      }
      const quote = line.match(/^\s*&gt;\s?(.*)$/);
      if (quote) {
        flushParagraph();
        flushList();
        html.push(`<blockquote>${renderInline(quote[1])}</blockquote>`);
        continue;
      }
      if (/^\s*(---|___|\*\*\*)\s*$/.test(line)) {
        flushParagraph();
        flushList();
        html.push("<hr>");
        continue;
      }
      if (/^\s*!\s+/.test(line)) {
        flushParagraph();
        flushList();
        html.push(`<div class="content-warning">${renderInline(line.replace(/^\s*!\s+/, ""))}</div>`);
        continue;
      }
      flushList();
      paragraph.push(line);
    }
    flushParagraph();
    flushList();
    return html.join("");
  }

  function format(value) {
    let source = String(value ?? "").replace(/\r\n?/g, "\n");
    const blockParts = [];
    const stash = html => {
      const index = blockParts.push(html) - 1;
      return `\n\uE000${index}\uE001\n`;
    };

    source = source.replace(
      /\[external_agent_tool_call:\s*([^\]]+)\]\s*([\s\S]*?)\s*\[\/external_agent_tool_call\]/gi,
      (_, toolName, body) => stash(renderToolCall(toolName, body))
    );
    source = source.replace(
      /\[external_agent_tool_result(?::\s*([^\]]+))?\]\s*([\s\S]*?)\s*\[\/external_agent_tool_result\]/gi,
      (_, status, body) => stash(renderToolResult(status, body))
    );
    source = source.replace(/```([^\n`]*)\n([\s\S]*?)```/g, (_, language, code) =>
      stash(renderCodeBlock(code, language || "text"))
    );

    const body = renderMarkdownText(escapeHtml(source), blockParts);
    return `<div class="rich-message">${body}</div>`;
  }

  return { escapeHtml, format, parseToolFields };
});
