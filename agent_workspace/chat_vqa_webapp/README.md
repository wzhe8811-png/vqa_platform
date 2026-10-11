# ChatGPT式 VQA 可视化界面（新服务）

该服务支持：

- 上传图片 + 输入问题（聊天式）
- 后端使用四模块推理（分解器/生成器/评估器/搜索器）
- 前端 D3 动态渲染推理图
- 点击节点查看 thought / answer / score 等详细信息

## 启动

```bash
cd /root/autodl-tmp/vision_qa_platform/agent_workspace/chat_vqa_webapp
/root/miniconda3/envs/llava_env/bin/python app.py
```

默认地址：`http://127.0.0.1:7861`

说明：你之前的旧可视化服务（7860）可继续保留，不受影响。
