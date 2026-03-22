# Python Knowledge Base

## Security Patterns

| Pattern | Risk | Fix |
|---|---|---|
| `f"SELECT * FROM {table}"` | SQL Injection (CRITICAL) | Use parameterized queries: `cursor.execute("SELECT * FROM ?", (table,))` |
| `os.system(user_input)` | Command Injection (CRITICAL) | Use `subprocess.run([...], shell=False)` with list args |
| `eval(user_input)` | Code Execution (CRITICAL) | Use `ast.literal_eval()` or JSON parsing |
| `pickle.loads(untrusted)` | Deserialization (CRITICAL) | Use JSON or MessagePack instead |
| `yaml.load(data)` | Code Execution (CRITICAL) | Use `yaml.safe_load(data)` |
| `password = "hardcoded"` | Secrets in Code (CRITICAL) | Use environment variables or secret manager |
| `hashlib.md5(password)` | Weak Crypto (HIGH) | Use `bcrypt.hashpw()` or `argon2` |
| `except:` (bare) | Swallows all errors (HIGH) | Catch specific exceptions: `except ValueError:` |
| `except Exception as e: pass` | Silent failure (HIGH) | At minimum: `logger.exception("...")` |

## Pythonic Patterns

| Anti-Pattern | Pythonic Alternative |
|---|---|
| `if len(lst) > 0:` | `if lst:` |
| `for i in range(len(lst)):` | `for item in lst:` or `for i, item in enumerate(lst):` |
| `if type(x) == int:` | `if isinstance(x, int):` |
| `d = {}; for k,v in items: d[k] = v` | `d = {k: v for k, v in items}` |
| Magic numbers: `if status == 3:` | Use Enum: `if status == Status.COMPLETED:` |
| `def f(x, lst=[]):` | `def f(x, lst=None): lst = lst or []` |
| `try: d["key"] except KeyError:` | `d.get("key", default)` |
| Manual file handling | `with open(path) as f:` (context manager) |

## Type Hints

```python
# Good: explicit annotations on public APIs
def process_users(users: list[User], *, active_only: bool = True) -> dict[str, int]:
    ...

# Avoid: overuse of Any
def process(data: Any) -> Any:  # BAD — loses all type safety
    ...

# Use Optional for nullable
from typing import Optional
def find_user(user_id: int) -> Optional[User]:  # or User | None (3.10+)
    ...
```

## Framework-Specific Checks

### Django
- [ ] Use `select_related()` / `prefetch_related()` to avoid N+1 queries
- [ ] Use `@transaction.atomic` for multi-write operations
- [ ] Set `ALLOWED_HOSTS` in production
- [ ] Use `csrf_protect` decorator or middleware
- [ ] Use `QuerySet.only()` / `defer()` for large models

### FastAPI
- [ ] Use `Depends()` for dependency injection (not global state)
- [ ] Use Pydantic models for request/response validation
- [ ] Don't use blocking I/O in async endpoints (use `run_in_executor`)
- [ ] Configure CORS properly (not `allow_origins=["*"]`)
- [ ] Use `BackgroundTasks` for fire-and-forget work

### Flask
- [ ] Register error handlers (`@app.errorhandler`)
- [ ] Use `flask-wtf` for CSRF protection
- [ ] Use `flask-sqlalchemy` with proper session management
- [ ] Set `SECRET_KEY` from environment, not hardcoded

## Testing

```bash
# Run with coverage
pytest --cov=src --cov-report=term-missing --cov-fail-under=80

# Type checking
mypy src/ --strict

# Linting
ruff check src/
ruff format src/

# Security scan
bandit -r src/
```

## Common Build Errors

| Error | Quick Fix |
|---|---|
| `ModuleNotFoundError: No module named 'X'` | `pip install X` or check `PYTHONPATH` |
| `ImportError: cannot import name 'Y'` | Circular import → restructure, or name doesn't exist |
| `SyntaxError: f-string expression part cannot include a backslash` | Extract to variable first |
| `TypeError: X() got an unexpected keyword argument 'Y'` | API changed → check docs/changelog |
| `AttributeError: module 'X' has no attribute 'Y'` | Wrong version installed, or name changed |
