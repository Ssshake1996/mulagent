# Refactoring & Dead Code Cleanup Knowledge Base

## Workflow

```
1. ANALYZE — Run detection tools, categorize by risk
2. VERIFY  — Grep ALL references (including dynamic imports, reflection, tests)
3. REMOVE  — One category at a time, test after each batch
4. CONSOLIDATE — Merge duplicates, extract common patterns
```

## Risk Classification

| Risk | Criteria | Action |
|---|---|---|
| **SAFE** | Unused imports, unreachable code, no-op functions | Delete directly |
| **CAREFUL** | Unused exports, unused parameters, dead feature flags | Verify no external consumers, then delete |
| **RISKY** | Unused files, unused dependencies, duplicate implementations | Deep verification needed — may be used dynamically |

## Detection Tools by Language

### JavaScript/TypeScript
```bash
# Unused exports and files
npx knip                          # Comprehensive: exports, files, deps, types
npx ts-prune                      # TypeScript unused exports only

# Unused dependencies
npx depcheck                      # Check package.json vs actual imports

# Dead code in bundle
npx webpack-bundle-analyzer       # Visualize what's in the bundle
```

### Python
```bash
# Unused imports
ruff check --select F401 src/     # Fast, reliable

# Unused code (functions, classes, variables)
vulture src/                      # Finds unused code
# WARNING: vulture has false positives on framework code (Django views, etc.)

# Unused dependencies
pip-extra-reqs --requirements-file=requirements.txt src/
```

### Go
```bash
# Unused code
go vet ./...                      # Built-in dead code detection
staticcheck ./...                 # U1000: unused code

# Unused dependencies
go mod tidy                       # Remove unused, add missing
```

### Rust
```bash
# Unused code
cargo clippy -- -W dead_code      # Warn on dead code

# Unused dependencies
cargo-udeps                       # Find unused Cargo.toml deps
```

## Verification Checklist

Before deleting ANYTHING, check ALL of these:

- [ ] **Static references**: `grep -r "function_name" --include="*.{ts,js,py,go}" .`
- [ ] **Dynamic imports**: `grep -r "import(" .` — might load lazily
- [ ] **Reflection / string-based access**: `getattr(obj, "name")`, `object["key"]`
- [ ] **Configuration files**: Check YAML, JSON, env files for references
- [ ] **Tests**: Don't delete test helpers that other tests import
- [ ] **Documentation**: Update docs if removing public API
- [ ] **External consumers**: Is this a library? Check if other projects depend on it
- [ ] **Feature flags**: Is the code behind a flag that might be enabled later?

## Safe Removal Order

Remove in this order (least risky first):

1. **Unused imports** — Always safe
2. **Unreachable code** — After `return`, after `if True`, impossible branches
3. **Unused local variables** — Verify no side effects in assignment
4. **Unused private functions** — Only reachable from same file
5. **Unused exported functions** — Verify no external consumers
6. **Unused files** — Verify no dynamic imports
7. **Unused dependencies** — Verify no transitive usage
8. **Duplicate implementations** — Pick the better one, update all callers

## Duplicate Detection

### Code Duplication Patterns

| Type | Example | Consolidation |
|---|---|---|
| Identical functions | Same code in 2+ files | Extract to shared module |
| Similar functions | Same logic, different parameters | Parameterize: add arguments |
| Copy-paste with mutations | 90% same, 10% different | Extract template, inject differences |
| Parallel hierarchies | `UserService.save()` + `UserRepo.save()` + `UserAPI.save()` all similar | Single abstraction with adapter pattern |

### How to Consolidate

```python
# BEFORE: duplicated in 3 files
def format_user_display(user):
    return f"{user.name} ({user.email})"

# AFTER: single shared function
# utils/formatting.py
def format_display(name: str, detail: str) -> str:
    return f"{name} ({detail})"
```

## Safety Rules

- **Never remove during active feature development** — wait for feature to stabilize
- **Never remove before a deploy** — do cleanup in its own PR, not mixed with features
- **One category per PR** — don't mix "remove unused deps" with "remove dead functions"
- **Test after each batch** — run full test suite, check build, check runtime
- **Git commit before each batch** — easy rollback if something breaks
- **Don't remove TODO/FIXME comments** — these are task markers, not dead code
- **Don't remove commented-out code in active development** — ask the author first

## Consolidation Metrics

| Metric | Good Target |
|---|---|
| Duplicate code ratio | <5% (measured by tool like PMD-CPD, jscpd) |
| Unused exports | 0 (everything exported should be imported somewhere) |
| Unused dependencies | 0 |
| Files with no imports | Review — might be entry points or might be dead |
