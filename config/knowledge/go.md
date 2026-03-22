# Go Knowledge Base

## Security Patterns

| Pattern | Risk | Fix |
|---|---|---|
| `fmt.Sprintf("SELECT * FROM %s", input)` | SQL Injection (CRITICAL) | Use `db.Query("SELECT * FROM ?", input)` |
| `exec.Command("sh", "-c", userInput)` | Command Injection (CRITICAL) | Use `exec.Command(binary, args...)` without shell |
| `os.Open(userPath)` | Path Traversal (CRITICAL) | Use `filepath.Clean()`, check prefix |
| `http.Get(userURL)` | SSRF (CRITICAL) | Validate URL against allowlist |
| `race condition on shared map` | Data Corruption (CRITICAL) | Use `sync.Mutex` or `sync.Map` |
| Hardcoded secrets | Secrets in Code (CRITICAL) | Use environment variables or vault |
| `InsecureSkipVerify: true` | TLS Bypass (HIGH) | Remove; fix certificate instead |

## Error Handling

| Anti-Pattern | Fix |
|---|---|
| `result, _ := doSomething()` | Handle error: `result, err := doSomething(); if err != nil { ... }` |
| `if err != nil { return err }` (no context) | `return fmt.Errorf("doing X: %w", err)` |
| `panic()` for recoverable errors | Return error instead; panic only for programmer bugs |
| Error string starts with uppercase | Start with lowercase: `fmt.Errorf("failed to connect: %w", err)` |
| `errors.New` for existing error | Use `fmt.Errorf` with `%w` for wrapping |

## Concurrency

| Anti-Pattern | Problem | Fix |
|---|---|---|
| Goroutine without shutdown signal | Goroutine leak | Pass `context.Context`, select on `ctx.Done()` |
| Unbuffered channel in producer/consumer | Deadlock risk | Use buffered channel or select with default |
| `go func() { ... }()` without WaitGroup | No way to wait for completion | Use `sync.WaitGroup` or `errgroup.Group` |
| Shared map access from goroutines | Race condition (crashes) | Use `sync.RWMutex` or `sync.Map` |
| `sync.Mutex` not released (no defer) | Deadlock | Always `mu.Lock(); defer mu.Unlock()` |
| `time.Sleep` in goroutine | Unreliable, wastes resources | Use `time.Ticker` or `context.WithTimeout` |

## Idiomatic Go

| Anti-Pattern | Idiomatic |
|---|---|
| Getter method `GetName()` | Just `Name()` (Go convention) |
| `if x == true` | `if x` |
| `if err == nil { return result, nil } else { return nil, err }` | Early return: `if err != nil { return nil, err }; return result, nil` |
| Interface with 10+ methods | Split into small interfaces (1-3 methods) |
| Package name `utils`, `helpers`, `common` | Name by what it does: `auth`, `cache`, `metrics` |
| `interface{}` everywhere | Use generics (Go 1.18+) or specific types |

## Testing

```bash
# Run with race detector
go test -race ./...

# Coverage
go test -coverprofile=cover.out ./...
go tool cover -html=cover.out

# Linting
golangci-lint run ./...

# Security scan
govulncheck ./...
```

## Common Build Errors

| Error | Quick Fix |
|---|---|
| `undefined: X` | Add import, or check unexported (lowercase first letter) |
| `cannot use X (type T) as type U` | Add type conversion: `U(x)` |
| `X does not implement interface Y` | Add missing method with correct signature |
| `import cycle not allowed` | Move shared types to separate package, use interfaces |
| `declared and not used` | Remove unused variable/import, or use `_` |
| `missing return at end of function` | Add return for all code paths |
| `go.sum mismatch` | `go mod tidy` then `go mod verify` |
