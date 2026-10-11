# Vision QA Portal（用户版在线问答）

独立于 `chat_vqa_webapp`（7861）的完整服务，带注册登录、历史会话与 GPT 风格界面。

| 服务 | 端口 | 说明 |
|------|------|------|
| chat_vqa_webapp | **7861** | 原版在线问答（无登录，保持不变） |
| **chat_vqa_portal** | **7862** | 新版：登录 + MongoDB + 历史记录 |

## 功能

- 用户注册 / 登录（用户名、密码、邮箱）
- **MongoDB** 文档库存储用户与问答记录（无 MongoDB 时自动降级为本地 **Mongita** JSON 存储）
- GPT 风格界面：左侧会话列表、中间对话、右侧 D3 推理图
- 查看历史询问、新建对话

## 启动

```bash
cd /root/autodl-tmp/vision_qa_platform/agent_workspace/chat_vqa_portal
/root/miniconda3/envs/llava_env/bin/pip install -r requirements.txt

# 可选：使用真实 MongoDB
# export MONGO_URI=mongodb://127.0.0.1:27017

/root/miniconda3/envs/llava_env/bin/python app.py
```

浏览器访问：`http://<服务器IP>:7862`

或使用统一脚本（若已更新 webapps_ctl.sh）：

```bash
cd /root/autodl-tmp/vision_qa_platform/agent_workspace
./webapps_ctl.sh start-portal
```

## 数据库

- 默认尝试连接 `mongodb://127.0.0.1:27017`，库名 `vision_qa_portal`
- 若 MongoDB 未启动，数据写入 `mongo_data/`（Mongita，仍为文档型非结构化存储）

集合说明：

| 集合 | 字段 |
|------|------|
| users | username, email, password_hash, created_at |
| conversations | user_id, title, created_at, updated_at |
| messages | conversation_id, role, content, image_path, answer, graph, trace, … |

## 环境变量

- `PORTAL_PORT`：端口，默认 `7862`
- `MONGO_URI`：MongoDB 连接串
- `PORTAL_SECRET_KEY`：Flask 会话密钥（生产环境请修改）
