/* global d3 */

const convListEl = document.getElementById("convList");
const messagesEl = document.getElementById("messages");
const emptyHint = document.getElementById("emptyHint");
const chatTitleEl = document.getElementById("chatTitle");
const questionEl = document.getElementById("question");
const imageEl = document.getElementById("image");
const fileNameEl = document.getElementById("fileName");
const sendBtn = document.getElementById("sendBtn");
const graphDetail = document.getElementById("graphDetail");

let currentConvId = null;
let running = false;
let conversations = [];
let lastGraph = null;

const nodeColors = {
  question: "#22d3ee",
  subq: "#f59e0b",
  candidate: "#60a5fa",
  candidate_selected: "#34d399",
  pred: "#a78bfa",
};

imageEl.addEventListener("change", () => {
  const f = imageEl.files[0];
  fileNameEl.textContent = f ? f.name : "添加图片";
});

async function api(url, options = {}) {
  const res = await fetch(url, {
    credentials: "same-origin",
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (res.status === 401) {
    window.location.href = "/login";
    throw new Error("请先登录");
  }
  if (!res.ok) throw new Error(data.error || "请求失败");
  return data;
}

function clearMessages() {
  messagesEl.querySelectorAll(".msg-row").forEach((el) => el.remove());
  emptyHint.hidden = false;
}

function appendMessage(role, text, imageUrl = null, graph = null) {
  emptyHint.hidden = true;
  const row = document.createElement("div");
  row.className = `msg-row ${role === "user" ? "user" : "bot"}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "我" : "AI";

  const body = document.createElement("div");
  body.className = "msg-body";
  const textEl = document.createElement("div");
  textEl.className = "text";
  textEl.textContent = text;
  body.appendChild(textEl);

  if (imageUrl) {
    const img = document.createElement("img");
    img.src = imageUrl;
    img.alt = "question image";
    body.appendChild(img);
  }

  if (role === "assistant" && graph) {
    row.dataset.hasGraph = "1";
    row.style.cursor = "pointer";
    row.title = "点击查看推理图";
    row.addEventListener("click", () => renderGraph(graph));
  }

  row.appendChild(avatar);
  row.appendChild(body);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderConvList() {
  convListEl.innerHTML = "";
  conversations.forEach((c) => {
    const item = document.createElement("div");
    item.className = "conv-item" + (c.id === currentConvId ? " active" : "");
    const title = document.createElement("span");
    title.className = "conv-title";
    title.textContent = c.title || "新对话";
    const del = document.createElement("button");
    del.className = "conv-del";
    del.textContent = "×";
    del.onclick = async (e) => {
      e.stopPropagation();
      if (!confirm("删除该对话？")) return;
      await api(`/api/conversations/${c.id}`, { method: "DELETE" });
      if (currentConvId === c.id) {
        currentConvId = null;
        clearMessages();
        chatTitleEl.textContent = "新对话";
      }
      await loadConversations();
    };
    item.onclick = () => selectConversation(c.id, c.title);
    item.appendChild(title);
    item.appendChild(del);
    convListEl.appendChild(item);
  });
}

async function loadConversations() {
  const data = await api("/api/conversations");
  conversations = data.conversations || [];
  renderConvList();
}

async function selectConversation(id, title) {
  currentConvId = id;
  chatTitleEl.textContent = title || "对话";
  renderConvList();
  clearMessages();

  const data = await api(`/api/conversations/${id}/messages`);
  (data.messages || []).forEach((m) => {
    const isUser = m.role === "user";
    appendMessage(
      isUser ? "user" : "assistant",
      m.content || m.answer || "",
      m.image_url || null,
      !isUser ? m.graph : null
    );
  });
}

async function createConversation() {
  const data = await api("/api/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "新对话" }),
  });
  currentConvId = data.conversation.id;
  chatTitleEl.textContent = "新对话";
  clearMessages();
  await loadConversations();
  renderConvList();
}

async function ensureConversation() {
  if (currentConvId) return currentConvId;
  await createConversation();
  return currentConvId;
}

function setLoading(flag) {
  running = flag;
  sendBtn.disabled = flag;
  sendBtn.textContent = flag ? "推理中…" : "发送";
}

function renderGraph(graph) {
  if (!graph || !graph.nodes) {
    graphDetail.textContent = "无可视化数据";
    return;
  }
  const svg = d3.select("#graph");
  svg.selectAll("*").remove();
  graphDetail.textContent = "点击节点查看详情";

  const width = svg.node().clientWidth || 320;
  const height = svg.node().clientHeight || 400;
  const nodes = graph.nodes.map((d) => ({ ...d }));
  const links = graph.links.map((d) => ({ ...d }));

  const sim = d3
    .forceSimulation(nodes)
    .force("link", d3.forceLink(links).id((d) => d.id).distance(80))
    .force("charge", d3.forceManyBody().strength(-300))
    .force("center", d3.forceCenter(width / 2, height / 2));

  const link = svg
    .append("g")
    .attr("stroke", "#334155")
    .selectAll("line")
    .data(links)
    .join("line");

  const node = svg
    .append("g")
    .selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("r", (d) => (d.type && d.type.includes("candidate") ? 10 : 14))
    .attr("fill", (d) => nodeColors[d.type] || "#94a3b8")
    .call(
      d3
        .drag()
        .on("start", (event, d) => {
          if (!event.active) sim.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on("drag", (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on("end", (event, d) => {
          if (!event.active) sim.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        })
    )
    .on("click", (_, d) => {
      graphDetail.textContent = JSON.stringify(
        { id: d.id, label: d.label, type: d.type, detail: d.detail },
        null,
        2
      );
    });

  sim.on("tick", () => {
    link
      .attr("x1", (d) => d.source.x)
      .attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x)
      .attr("y2", (d) => d.target.y);
    node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
  });
}

async function sendMessage() {
  if (running) return;
  const question = questionEl.value.trim();
  const file = imageEl.files[0];
  if (!question || !file) {
    alert("请同时上传图片并输入问题");
    return;
  }
  await ensureConversation();
  const previewUrl = URL.createObjectURL(file);
  appendMessage("user", question, previewUrl);
  questionEl.value = "";
  imageEl.value = "";
  fileNameEl.textContent = "添加图片";
  setLoading(true);

  try {
    const form = new FormData();
    form.append("question", question);
    form.append("image", file);

    const data = await api(`/api/conversations/${currentConvId}/ask`, {
      method: "POST",
      body: form,
    });

    const answerText = data.answer || "（无回答）";
    lastGraph = data.graph || null;
    appendMessage("assistant", answerText, data.image_url, data.graph);
    renderGraph(data.graph);
    await loadConversations();
    const conv = conversations.find((c) => c.id === currentConvId);
    if (conv) chatTitleEl.textContent = conv.title;
  } catch (err) {
    appendMessage("assistant", "推理失败：" + err.message, null, null);
  } finally {
    setLoading(false);
  }
}

document.getElementById("newChatBtn").onclick = createConversation;
document.getElementById("logoutBtn").onclick = async () => {
  await api("/api/auth/logout", { method: "POST" });
  window.location.href = "/login";
};
sendBtn.onclick = sendMessage;
questionEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

loadConversations().catch(() => {
  window.location.href = "/login";
});
