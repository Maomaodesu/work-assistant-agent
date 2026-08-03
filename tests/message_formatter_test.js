"use strict";

const assert = require("assert");
const formatter = require("../static/message_formatter.js");

const bashExample = `重新编译并重启：
[external_agent_tool_call: Bash]
description: Recompile
command: JAVA_HOME="/c/Program Files/Java/jdk-17" PATH="$JAVA_HOME/bin:$PATH" mvn compile -q 2>&1 && echo "COMPILE OK"
[/external_agent_tool_call]
[external_agent_tool_result: error]
The user doesn't want to proceed with this tool use. The tool use was rejected.
[/external_agent_tool_result]`;

const bashHtml = formatter.format(bashExample);
assert(bashHtml.includes("tool-call-card"));
assert(bashHtml.includes("language-bash"));
assert(bashHtml.includes("Recompile"));
assert(bashHtml.includes("JAVA_HOME"));
assert(bashHtml.includes("tool-result-error"));
assert(bashHtml.includes("重新编译并重启"));

const readHtml = formatter.format(`[external_agent_tool_call: Read]
file_path: C:/workspace/demo.py
offset: 20
limit: 50
[/external_agent_tool_call]`);
assert(readHtml.includes("tool-properties"));
assert(readHtml.includes("file_path"));
assert(readHtml.includes("C:/workspace/demo.py"));

const markdownHtml = formatter.format(`# 标题

- 第一项
- 第二项

> 引用

\`inline\`

\`\`\`python
print("ok")
\`\`\``);
assert(markdownHtml.includes("<h2>标题</h2>"));
assert(markdownHtml.includes("<ul><li>第一项</li><li>第二项</li></ul>"));
assert(markdownHtml.includes("<blockquote>引用</blockquote>"));
assert(markdownHtml.includes('class="inline-code"'));
assert(markdownHtml.includes("language-python"));

const unsafeHtml = formatter.format(`<img src=x onerror=alert(1)>
[external_agent_tool_result: error]
<script>alert(1)</script>
[/external_agent_tool_result]`);
assert(!unsafeHtml.includes("<img"));
assert(!unsafeHtml.includes("<script>"));
assert(unsafeHtml.includes("&lt;img"));
assert(unsafeHtml.includes("&lt;script&gt;"));

console.log("MESSAGE_FORMATTER_TEST_OK");
