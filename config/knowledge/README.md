# Knowledge Base

角色专属知识库。每个 `.md` 文件包含特定领域的检查清单、模式库、反模式、修复方法。

## 扩展方式

### 添加新知识库

1. 在此目录创建 `<name>.md`
2. 在 `config/agents.yaml` 中对应角色的 `knowledge:` 列表中引用
3. 多个角色可共享同一个知识库文件
4. **无需修改任何代码**

### 添加新角色

1. 在 `config/agents.yaml` 的 `roles:` 下添加新条目
2. 指定 `name`, `description`, `tools`, `knowledge`, `prompt`
3. delegate 工具会自动发现新角色
4. 更新 delegate 工具的 `enum` 列表（`src/tools/isolation.py`）

### 知识库编写规范

- 使用 Markdown 表格列出模式和修复方法
- 用 `CRITICAL / HIGH / MEDIUM / LOW` 标注严重级别
- 包含具体代码示例（好的和坏的对比）
- 保持每个文件 < 500 行，便于 LLM 处理

## 文件索引（17 个知识库）

| 文件 | 领域 | 引用角色 |
|---|---|---|
| **通用** |||
| `architect.md` | 系统设计、ADR、反模式、可扩展性规划 | architect |
| `code_review.md` | 通用代码审查清单（4 级严重度）| code_reviewer |
| `security.md` | OWASP Top 10、密钥检测、紧急响应 | security_auditor, guardian |
| `tdd.md` | TDD 方法论、8 种必测边界、覆盖率 | tdd_guide |
| `e2e.md` | E2E 测试、POM 模式、Flaky 处理 | tdd_guide |
| `build_errors.md` | 多语言构建错误通用修复流程 | build_resolver |
| `refactor.md` | 死代码清理、重复检测、安全删除流程 | coder |
| `database.md` | SQL 优化、索引策略、RLS、Schema 设计 | analyst |
| **语言专项** |||
| `python.md` | 安全、Pythonic 写法、Django/FastAPI/Flask | coder, code_reviewer, build_resolver |
| `typescript.md` | 类型安全、异步正确性、React/Next.js/Node.js | coder, code_reviewer, build_resolver |
| `go.md` | 错误处理、并发、goroutine 安全、惯用法 | coder, code_reviewer, build_resolver |
| `java.md` | Spring Boot 分层、JPA、工作流状态机 | coder, code_reviewer, build_resolver |
| `rust.md` | 所有权/生命周期、unsafe 审计、borrow checker | coder, code_reviewer, build_resolver |
| `cpp.md` | RAII、Rule of Five、内存安全、CMake | coder, code_reviewer, build_resolver |
| `kotlin.md` | 协程安全、Jetpack Compose、Android 安全 | coder, code_reviewer, build_resolver |
| `flutter.md` | Widget 组合、状态管理、资源生命周期 | coder, code_reviewer |
| `pytorch.md` | CUDA 错误、Shape 调试、AMP、梯度检查点 | coder, build_resolver |
