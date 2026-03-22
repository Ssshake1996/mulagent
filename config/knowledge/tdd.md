# TDD Knowledge Base

## Red-Green-Refactor Cycle (Strict)

```
┌─────────┐     ┌─────────┐     ┌──────────┐
│  RED    │ ──→ │  GREEN  │ ──→ │ REFACTOR │ ──→ (repeat)
│ (write  │     │ (make   │     │ (clean   │
│  test)  │     │  pass)  │     │  up)     │
└─────────┘     └─────────┘     └──────────┘
```

1. **RED**: Write ONE failing test for the next piece of behavior
2. **Run tests** — the new test MUST fail (proves test is meaningful)
3. **GREEN**: Write the MINIMUM code to make it pass — no more
4. **Run tests** — ALL tests must pass
5. **REFACTOR**: Improve code quality without changing behavior
6. **Run tests** — ALL tests must still pass

## 8 Mandatory Edge Case Categories

Every function/method should be tested against ALL applicable categories:

| # | Category | Examples |
|---|---|---|
| 1 | Null/None/undefined | `None`, `null`, `undefined`, missing keys |
| 2 | Empty collections | `[]`, `{}`, `""`, `set()` |
| 3 | Invalid types | String where int expected, object where array expected |
| 4 | Boundary values | `0`, `-1`, `MAX_INT`, `MIN_INT`, empty string `""` |
| 5 | Error paths | Network timeout, file not found, permission denied |
| 6 | Concurrent access | Race conditions, deadlocks, data corruption |
| 7 | Large data | 10K+ items, 1MB+ strings, deeply nested structures |
| 8 | Special characters | Unicode (中文, émojis 🎉), SQL chars (`'; DROP`), HTML (`<script>`) |

## Test Quality Anti-Patterns

| Anti-Pattern | Problem | Fix |
|---|---|---|
| Testing implementation | Test breaks when refactoring | Test behavior/output, not internal calls |
| Shared test state | Tests pass alone, fail together | setUp/tearDown, isolated fixtures |
| Weak assertions | `assert result is not None` | Assert specific expected values |
| Mock everything | Tests pass but real code fails | Use real objects where feasible; mock only external services |
| Test per line | Redundant tests, slow suite | Test per behavior/scenario |
| Flaky tests | Intermittent failures | Fix root cause or quarantine with `@pytest.mark.skip(reason="flaky: ...")` |

## Coverage Targets

| Type | Minimum | Ideal |
|---|---|---|
| Unit tests | 80% line coverage | 90%+ branch coverage |
| Integration tests | Critical paths covered | All API endpoints tested |
| E2E tests | Happy paths | Happy + top 3 error paths |

## Test Organization

```
tests/
├── unit/           # Fast, isolated, no external deps
│   ├── test_*.py   # One test file per source module
├── integration/    # Tests with real DB, API, etc.
│   ├── test_*.py
├── e2e/            # Full system tests
│   ├── test_*.py
├── conftest.py     # Shared fixtures
└── factories.py    # Test data factories
```

## Framework-Specific Tips

### Python (pytest)
- Use `@pytest.fixture` for setup, `conftest.py` for shared fixtures
- Use `@pytest.mark.parametrize` for data-driven tests
- Use `pytest-cov` for coverage: `pytest --cov=src --cov-report=term-missing`
- Use `pytest-asyncio` for async tests: `@pytest.mark.asyncio`

### JavaScript/TypeScript (Jest/Vitest)
- Use `describe/it` for organization, `beforeEach/afterEach` for setup
- Use `jest.mock()` sparingly — prefer dependency injection
- Use `--coverage` flag for coverage report

### Go
- Table-driven tests with `[]struct{ name, input, want }`
- Use `testify/assert` for readable assertions
- Use `go test -race` to detect race conditions
- Use `go test -cover` for coverage
