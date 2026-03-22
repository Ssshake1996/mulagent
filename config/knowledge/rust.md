# Rust Knowledge Base

## Safety Patterns

| Pattern | Risk | Fix |
|---|---|---|
| `.unwrap()` / `.expect()` in production | Panic/crash (CRITICAL) | Use `?` operator, or match/if-let |
| `unsafe` without `// SAFETY:` comment | Undocumented invariant (CRITICAL) | Document why unsafe is needed and what invariants must hold |
| `format!("SELECT {}", input)` | SQL Injection (CRITICAL) | Use parameterized query (sqlx `query!`) |
| `Command::new("sh").arg("-c").arg(input)` | Command Injection (CRITICAL) | Use `Command::new(binary).args(&[...])` without shell |
| Hardcoded secrets | Secrets in Code (CRITICAL) | Use `std::env::var()` or secret manager |
| Raw pointer dereference without check | Use-after-free (CRITICAL) | Minimize unsafe; verify lifetime guarantees |
| `unsafe impl Send/Sync` | Data race (CRITICAL) | Only if you can prove thread-safety |

## Error Handling

| Anti-Pattern | Fix |
|---|---|
| `.unwrap()` on Result | Use `?`, `.unwrap_or()`, `.unwrap_or_else()`, or match |
| `Box<dyn Error>` in library code | Define specific error enum with `thiserror` |
| Missing `.context()` on error propagation | Add context: `.context("failed to open config")?` (anyhow) |
| `panic!()` for recoverable errors | Return `Result<T, E>` instead |
| Silencing `#[must_use]` return | Handle the returned value or explicitly drop |

## Ownership & Lifetimes

| Anti-Pattern | Fix |
|---|---|
| Unnecessary `.clone()` | Use references `&T` or `Cow<T>` |
| `String` parameter when `&str` suffices | Accept `&str` or `impl AsRef<str>` |
| `Vec<T>` parameter when `&[T]` suffices | Accept `&[T]` slice reference |
| Lifetime over-annotation `<'a, 'b, 'c>` | Let compiler elide lifetimes where possible |
| `Rc<RefCell<T>>` everywhere | Consider redesigning data flow; often indicates bad architecture |

## Concurrency

| Anti-Pattern | Problem | Fix |
|---|---|---|
| `std::thread::sleep` in async code | Blocks executor thread | Use `tokio::time::sleep` |
| Unbounded `mpsc::channel()` | Memory exhaustion | Use bounded channel: `mpsc::channel(capacity)` |
| `Mutex` poisoning ignored | Data corruption | Handle `PoisonError` or use `parking_lot::Mutex` |
| Missing `Send + Sync` bounds on generic async | Compiler error in async context | Add bounds: `T: Send + Sync + 'static` |
| Deadlock from nested locks | Lock ordering violation | Always acquire locks in consistent order |
| `.await` while holding `MutexGuard` | Deadlock in async | Drop guard before .await: `{ let _g = lock.lock(); } do_async.await` |

## Idiomatic Rust

| Anti-Pattern | Idiomatic |
|---|---|
| `if option.is_some() { option.unwrap() }` | `if let Some(val) = option { ... }` |
| `match x { A => ..., B => ..., _ => {} }` | Ensure exhaustive match; avoid catch-all `_` when possible |
| `for i in 0..vec.len() { vec[i] }` | `for item in &vec { ... }` or `vec.iter()` |
| `let mut s = String::new(); s.push_str(a); s.push_str(b);` | `format!("{a}{b}")` or `[a, b].concat()` |
| `Box<dyn Fn()>` when concrete type known | Use `impl Fn()` or generic parameter |
| Manual `Drop` implementation | Prefer RAII wrappers from std/crates |

## Testing

```bash
# Build + test
cargo test

# With coverage (cargo-tarpaulin)
cargo tarpaulin --out Html

# Linting
cargo clippy -- -D warnings

# Format check
cargo fmt -- --check

# Security audit
cargo audit
cargo deny check
```

## Common Build Errors

| Error | Quick Fix |
|---|---|
| `cannot borrow X as mutable` | Clone the value, or restructure borrows to avoid aliasing |
| `X does not live long enough` | Return owned type, use `Arc<T>`, or adjust lifetime bounds |
| `cannot move out of borrowed content` | Clone, or use `&T` instead of `T` |
| `trait bound X: Y not satisfied` | `#[derive(Y)]` or manual `impl Y for X` |
| `unresolved import` | Add to `Cargo.toml [dependencies]`, check feature flags |
| `type mismatch: expected T, found U` | Add `.into()`, explicit conversion, or fix the type |
| `async fn is not Send` | Box non-Send futures, or restructure to avoid non-Send types across .await |
| `conflicting implementations` | Use newtype pattern to avoid orphan rules |
