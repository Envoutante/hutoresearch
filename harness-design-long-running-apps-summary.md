# Harness Design for Long-Running Apps — 摘要

> 原文：Anthropic Engineering Blog, *Harness design for long-running application development* (Mar 24, 2026)  
> 作者：Prithvi Rajasekaran (Anthropic Labs)

---

## 1. 核心问题：为什么 naive 长时任务会失败

长时 agentic 任务有两个典型失败模式：

| 问题 | 表现 | 解决方法 |
|------|------|----------|
| **上下文衰减 / Context anxiety** | 随着 context window 填满，模型逐渐失去连贯性；或在接近上下文上限时提前结束工作 | **Context Reset**：定期清空上下文、启动新 agent session，通过**结构化 handoff artifact** 传递状态 |
| **自我评估偏差** | 模型评估自己产出时过于宽容（尤其设计等主观任务），even 有客观测试时也会漏 bug | **分离 Generator 与 Evaluator**：让独立的 evaluator 代理来批评 generator 的工作 |

> 注：Claude Sonnet 4.5 的 context anxiety 很强，仅靠 compaction（压缩历史）不够，必须做 context reset。Opus 4.6 显著改善了这一行为，可支持连续 2h+ 的 session，从而可去掉强制 reset。

---

## 2. 推荐架构：Planner → Generator ↔ Evaluator

文章最终采用的三代理架构：

- **Planner**：将简单 prompt（1-4 句话）扩展为完整产品 spec，定义交付物和高级技术方向，避免过早指定实现细节导致错误级联。
- **Generator**：按 sprint 逐步实现功能，每轮完成后自评并产出可运行版本。
- **Evaluator**：通过 **Playwright MCP** 与实时代码/页面交互，按明确标准评分、写详细 critique，**将反馈流回 Generator 作为下一轮输入**。

### 关键流程
1. **Sprint Contract**：Generator 和 Evaluator 在写代码前就“完成标准”达成一致（ bridging high-level spec ↔ testable implementation）。
2. **Generator 执行 sprint** → 产出代码与状态文件。
3. **Evaluator 被激活** → 实际测试、评分、指出问题。
4. **反馈回流 Generator** → 启动下一轮改进。

> 前端设计实验中，这个循环跑了 **5–15 次迭代**，Evaluator 会实际浏览页面、截图、仔细研究后再评估，**完整运行长达 4 小时**。

---

## 3. 长时任务的关键工程实践

### (a) 结构化产物传递（Structured Artifacts）
代理之间通过**文件**通信，而不是全靠对话历史。这是 context reset 后能无缝衔接的前提：
- 状态文件包含：已完成工作、当前代码状态、下一轮计划。
- 产物足够自包含，新 agent 启动后可直接读取并恢复上下文。

### (b) 任务分解为 Tractable Chunks
复杂构建被拆分为 sprints（如 16 个功能分 10 个 sprints）。每块足够小，能被一个 agent session 可靠完成。

### (c) 独立 Evaluator 的 Skepticism 调优
独立 evaluator 虽然也是 LLM，但**单独调优让它变得 skeptical** 比让 generator 自我批评容易得多。校准方法：
- 在 prompt 中给出 few-shot 评分示例。
- 反复阅读 evaluator 日志，找到与人类判断偏离的案例，再修正 QA prompt。

---

## 4. 模型演进对 Harness 的影响（Opus 4.6）

**Harness 的复杂度不是固定的，应随模型能力动态调整：**

- **Opus 4.5**：需要 sprint 分解 + 每轮中间 QA，否则长任务会偏离。
- **Opus 4.6**：可连续运行 2 小时以上而不需要 sprint 分解；Evaluator 改为在**最后做一次单遍检查**即可，只在 generator 能力不足的边缘部分才需要中间 QA。

> **Evaluator 不是固定开销**，而是"当任务超出当前模型 solo 可靠边界时才值得投入"。

---

## 5. 对“实验结束后唤醒 Agent”场景的落地建议

如果你想实现“实验结束后唤醒 agent 分析并设计下一轮”，文章推荐的模式是：

1. **不要让 agent 空等或 sleep 轮询**。
2. **Generator agent 启动实验后退出/挂起**。
3. 实验结束时，由外部 harness（如你的脚本）**重新唤醒 Evaluator agent**。
4. Evaluator 读取结构化产物（结果文件、日志、metrics），进行分析并输出下一轮实验设计。
5. **Harness 再唤醒 Generator**，传入 Evaluator 的分析，开始下一轮。

### 在 Claude Code 中的实现方式
- 实验脚本结束时调用：
  ```bash
  claude -p "请分析 ./results/exp_042.json 并设计下一组实验参数，然后更新 experiments.yaml。"
  ```
- 或使用内置的 **`:skill autoresearch`**，它正是为“自主迭代实验循环”设计的。

---

## 6. 关键引用

> "Every component in a harness encodes an assumption about what the model can't do on its own, and those assumptions are worth stress testing... they can quickly go stale as models improve."

> "The space of interesting harness combinations doesn't shrink as models improve. Instead, it moves."
