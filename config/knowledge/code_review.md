# Code Review Knowledge Base

## Review Checklist (ordered by severity)

### CRITICAL — Must fix before merge

- [ ] **SQL Injection**: String concatenation/f-strings in SQL queries
- [ ] **Command Injection**: User input passed to shell commands (subprocess, os.system, exec)
- [ ] **XSS**: Unescaped user content rendered in HTML (innerHTML, dangerouslySetInnerHTML)
- [ ] **Auth Bypass**: Missing authentication or authorization checks on endpoints
- [ ] **Hardcoded Secrets**: API keys, passwords, tokens in source code
- [ ] **Path Traversal**: User-controlled file paths without sanitization (../../etc/passwd)
- [ ] **Insecure Deserialization**: pickle.loads, yaml.load (without SafeLoader), eval() on user data
- [ ] **SSRF**: User-controlled URLs in server-side HTTP requests

### HIGH — Should fix before merge

- [ ] **Missing Error Handling**: Bare except, swallowed exceptions, no error propagation
- [ ] **Race Conditions**: Shared mutable state without synchronization
- [ ] **Resource Leaks**: Unclosed files, connections, cursors (missing context managers / try-finally)
- [ ] **N+1 Queries**: Loop of individual DB queries instead of batch/join
- [ ] **Missing Input Validation**: No validation at system boundaries (API endpoints, CLI args)
- [ ] **Unbounded Operations**: No pagination, no limits on query results, no timeouts on external calls
- [ ] **Breaking API Changes**: Removing fields, changing types without versioning

### MEDIUM — Fix in follow-up

- [ ] **Dead Code**: Unreachable code, unused imports/variables, commented-out code blocks
- [ ] **Missing Tests**: New logic with no test coverage
- [ ] **Performance**: Unnecessary copies, O(n²) where O(n) is possible, missing indexes
- [ ] **Logging**: Sensitive data in logs, missing error context, wrong log levels

### LOW — Nice to have

- [ ] **Naming**: Unclear variable/function names, inconsistent naming conventions
- [ ] **Complexity**: Functions >50 lines, deeply nested logic (>3 levels)
- [ ] **Documentation**: Missing docstrings on public APIs, outdated comments

## AI-Generated Code Review Addendum

When reviewing AI-generated code, additionally check:
- [ ] **Behavioral Regression**: Does the change accidentally alter existing behavior?
- [ ] **Trust Boundary**: Is AI-generated code properly sandboxed from privileged operations?
- [ ] **Architecture Drift**: Does the change follow the project's established patterns?
- [ ] **Fabricated APIs**: Does the code call functions/methods that don't actually exist?
- [ ] **Over-Engineering**: Is there unnecessary abstraction or complexity?

## Finding Report Format

```
[SEVERITY] file:line — Description
  Problem: What's wrong
  Impact: What could happen
  Fix: How to fix it
  Confidence: N%
```
