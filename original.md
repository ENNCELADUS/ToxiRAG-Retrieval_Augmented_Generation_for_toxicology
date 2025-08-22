# RAG Agent for Liver Cancer Toxicity — Planning & Prediction (with Paper-Based KB)
1. What app I'm going to create: A **RAG-supported AI agent** that assists researchers in **planning and predicting animal experiments for liver cancer toxicity studies**.
It works with **paper-based data** that strictly follow a fixed Markdown form (below), and answers researcher queries by retrieving **evidence-backed snippets** from the local knowledge base and reasoning with top LLMs.

All ingested documents **must** conform to this template exactly.

```markdown
#  论文标题: 
（来源：p.? / Title / DOI 页面）

---

##  实验小鼠1信息
- 品系:
- 分组数及每组数量:
- 总数:
- 性别: 
- 体重: 
- 周龄: 
（来源：p.? / Methods / Animals / Table ?）

##  实验小鼠2信息（如适用）
- 品系:
- 分组数及每组数量:
- 总数:
- 性别: 
- 体重: 
- 周龄: 
（来源：p.?）

##  实验小鼠3信息（如适用）
- 品系:
- 分组数及每组数量:
- 总数:
- 性别: 
- 体重: 
- 周龄: 
（来源：p.?）

##  实验小鼠4信息（如适用）
- 品系:
- 分组数及每组数量:
- 总数:
- 性别: 
- 体重: 
- 周龄: 
（来源：p.?）

##  实验大鼠信息（如适用）
- 品系:
- 分组数及每组数量:
- 总数:
- 性别: 
- 体重: 
- 周龄: 
（来源：p.?）

---

##  细胞种类（如有）
- 细胞名称:
- 接种方式: 
- 接种量: 
（来源：p.? / Cell line / Inoculation）

---

## 肿瘤模型信息（如有）
- 模型类型： 
- 肿瘤类型： 
- 接种位置： 
- 成瘤时间（天）： 
- 给药开始时间（成瘤后第几天）： 
- 成瘤总天数： 
（来源：p.? / Figure ? / Table ?）

---

## 实验时间线简表
| 时间点 | 操作内容           |
|--------|--------------------|
| Day 0  | 接种肿瘤/建模处理   |
| Day X  | 成瘤完成/模型建立成功 |
| Day X+Y| 开始给药           |
| Day Z  | 终点采样/处死动物  |
（将 X、Y、Z 用文中真实时间替换；未知则写“未说明”。来源：p.?）

---

##  实验分组与给药
- 分组名称及数量:
- 给药方式:
- 药物剂量与频率:
- 给药周期:
（确保与下方所有数据表格分组一致；来源：p.? / Table ?）

---

##  数据记录表格（如有的话，全部都要）

### 肿瘤体积变化（mm³）
| 分组 | Day … | Day … | Day … | … |
|------|-------|-------|-------|---|
（包含文中出现的所有时间点；若为图像估读请注明。来源：p.? / Fig.?）

### 小鼠体重变化（g）
| 分组 | Day … | Day … | Day … | … |
|------|-------|-------|-------|---|
（来源：p.?）

### 肿瘤质量与抑瘤率
| 分组 | 肿瘤质量（g） | 抑瘤率（%） |
|------|----------------|--------------|
（来源：p.? / Table ?；如仅给均值±SD，照录；如只给箱线图，估读并标注）

---

##  病理检测（HE/IHC/TUNEL等）
- 检测组织：
- 染色方法：
- 阳性结果表现：
（来源：p.? / Figure ? / 方法节）

---

##  机制研究结果
简要总结分子/细胞机制（如信号通路、凋亡蛋白等），**只可依据文中表述与数据**；若仅有示意图则写“未说明（仅示意图）”。（来源：p.?）

### 机制检测数据表（（如有的话，全部都要））
| 指标 | 对照组 | 药物组 | P值 |
|------|--------|--------|------|
（可按多组扩展列或改为长表：指标/分组/数值/P；来源：p.?）

---

##  其他检测（（如有的话，全部都要））

### 免疫器官质量与指数
| 分组 | 脾重（mg） | 胸腺重（mg） | 脾指数 | 胸腺指数 |
|------|------------|---------------|---------|-----------|
（来源：p.?）

### 血常规
| 指标 | 单位 | 对照组 | 药物组 |
|------|------|--------|--------|
（若多组，多列展开或长表；来源：p.?）

### 肝功能
| 指标 | 单位 | 对照组 | 药物组 |
|------|------|--------|--------|
（来源：p.?）

### 肾功能
| 指标 | 单位 | 对照组 | 药物组 |
|------|------|--------|--------|
（来源：p.?）

---

##  研究结论
总结药效、机制和研究价值（严格来自论文结论/讨论，不得发挥）。（来源：p.?）

---

##  关键词
`#动物实验` `#肿瘤模型` `#药效评价` `#机制研究` `#实验时间线`
```

2. Help me think through how to break this into iterative pieces and write a plan.md

3. Repository layout

```
.
├─ app/            # Web/API app and UI
├─ api/            # (optional, not implement now)
├─ ingest/
├─ retriever/      # Hybrid retrieval (vector + keyword), evidence packing
├─ llm/            # LLM orchestration with agno (OpenAI + Gemini, tools)
├─ scripts/        # CLI utilities (ingest, reindex, eval runs)
├─ data/           # /md for raw papers; /fixtures for tests; /samples
├─ docker/         # (optional, not implement now)
├─ tests/          # Unit tests per component
├─ eval/           # E2E evaluation configs & golden answers
└─ README.md
```

4. Requirements (one by one)

* Use `OPENAI_API_KEY` and `GOOGLE_API_KEY` to access **gpt-5** and **gemini-2.5-pro** as the core LLMs.
* Can use **local Markdown data** (format **exactly** as shown) as the **knowledge base**.
* Use `OPENAI_EMBED_MODEL` as the embed model to **take one or more Markdown/TXT files** that strictly follow the template, **parse them without任何外部查询**, **normalize** fields (units, enums), **chunk** into retrievable units, **embed**, and **upsert** to your vector store.
* Given a user query, **construct sub-queries**, perform **混合检索（向量 + 关键词）** over the knowledge base, and assemble an **evidence pack** (with numbering and metadata) for LLM consumption.
* Use **LanceDB** as the database to store the uploaded MD data and vector embeddings.
* Use the **agno** package to handle OpenAI API, Gemini API, and **reasoning-tools** to help LLM thinking. Refer to:

  * [https://github.com/agno-agi/agno/blob/main/cookbook/reasoning/tools/gemini\_reasoning\_tools.py](https://github.com/agno-agi/agno/blob/main/cookbook/reasoning/tools/gemini_reasoning_tools.py)
  * [https://github.com/agno-agi/agno/blob/main/cookbook/reasoning/tools/openai\_reasoning\_tools.py](https://github.com/agno-agi/agno/blob/main/cookbook/reasoning/tools/openai_reasoning_tools.py)
  * [https://github.com/agno-agi/agno/blob/main/cookbook/reasoning/tools/knowledge\_tools.py](https://github.com/agno-agi/agno/blob/main/cookbook/reasoning/tools/knowledge_tools.py)
  * [https://docs.agno.com/reasoning/reasoning-tools](https://docs.agno.com/reasoning/reasoning-tools)
* Components include: `scripts`, `app`, `data`, `docker`, `ingest`, `llm`, `retriever`, `tests`, `eval`, `api`.
* **Add unit tests for every component**, and an **end-to-end test**.
* Use **git** for version control with **descriptive commits**.
* Already create a conda env named `toxirag` that has consisted all the packages in `requirements.txt`, use this env for the project.
* Has `.env` consisting the necessary api keys.

Check off items in the plan as we accomplish them as a todo list. If you have open questions that require my input, add those in the plan as well.