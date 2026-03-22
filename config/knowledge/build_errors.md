# Build Error Resolution Knowledge Base

## Universal Resolution Workflow

```
1. Run build → collect ALL errors
2. Categorize errors (type, import, dep, config)
3. Fix in priority order: build-blocking → type errors → warnings
4. Re-run build → verify fix
5. Run tests → verify no regressions
6. STOP after 3 failed attempts on same error
```

## Critical Rules
- **Minimal diff**: <5% of affected file changed
- **No refactoring**: Only fix the error, nothing else
- **No silent suppression**: Don't add `// @ts-ignore`, `# type: ignore`, `#pragma GCC diagnostic ignored`
- **Preserve behavior**: Fix must not change runtime behavior

## Common Error Patterns by Language

### Python
| Error | Fix |
|---|---|
| `ModuleNotFoundError` | `pip install <package>` or fix `PYTHONPATH` |
| `ImportError: cannot import name` | Check circular imports, verify export exists |
| `SyntaxError` | Check Python version compatibility, missing f-string prefix |
| `TypeError: missing required argument` | Check function signature changed, update callers |
| `AttributeError: has no attribute` | Check typos, verify class/module API |
| `IndentationError` | Fix tabs vs spaces, check mixed indentation |

### TypeScript / JavaScript
| Error | Fix |
|---|---|
| `TS2304: Cannot find name` | Add import or install `@types/<pkg>` |
| `TS2322: Type X not assignable to Y` | Fix type mismatch, add type assertion if safe |
| `TS2345: Argument of type X not assignable` | Check function signature, add type narrowing |
| `TS7006: Parameter implicitly has 'any'` | Add explicit type annotation |
| `TS2532: Object is possibly undefined` | Add null check or optional chaining `?.` |
| `TS1005: ';' expected` | Syntax error, check for missing brackets/parens |
| `Module not found` | `npm install <pkg>`, check tsconfig paths |

### Go
| Error | Fix |
|---|---|
| `undefined: X` | Add import, check unexported name (lowercase) |
| `cannot use X as type Y` | Fix type conversion, check interface implementation |
| `X does not implement Y` | Add missing method to satisfy interface |
| `import cycle` | Break cycle with interface, move shared types to separate package |
| `unused import/variable` | Remove or use `_` for intentionally unused |
| `missing return` | Add return statement for all code paths |

### Java
| Error | Fix |
|---|---|
| `cannot find symbol` | Add import, check classpath, verify dependency in pom.xml/build.gradle |
| `incompatible types` | Fix type mismatch, add cast if safe |
| `does not override abstract method` | Implement all interface/abstract methods |
| `package does not exist` | Add dependency to pom.xml or build.gradle |
| `annotation processing` (Lombok) | Ensure annotation processor configured in build tool |

### Rust
| Error | Fix |
|---|---|
| `cannot borrow as mutable` | Clone value, restructure borrows, use interior mutability |
| `value does not live long enough` | Return owned type, extend lifetime, use `'static` or `Arc` |
| `cannot move out of` | Clone, or use reference instead |
| `trait bound not satisfied` | Implement trait, add `#[derive]`, add bound to generic |
| `unresolved import` | Add to `Cargo.toml`, check `use` path, verify feature flag |
| `type mismatch` | Check expected vs actual, add `.into()` or explicit conversion |

### C/C++
| Error | Fix |
|---|---|
| `undefined reference` | Check linker flags, add source file to build, verify library linked |
| `no matching function` | Check overload resolution, argument types |
| `undeclared identifier` | Add `#include`, forward declaration, or check scope |
| `multiple definition` | Move definition to .cpp, use `inline`, or fix include guards |
| `implicit conversion` | Add explicit cast |

## Dependency Resolution

### Python (pip/poetry)
```bash
pip install -r requirements.txt  # Install all
pip install --upgrade <pkg>      # Upgrade specific
pip check                         # Verify consistency
```

### Node.js (npm/yarn/pnpm)
```bash
npm install           # Install all
npm ls --all          # Dependency tree
npm audit             # Security check
npm dedupe            # Reduce duplicates
```

### Go
```bash
go mod tidy           # Clean up go.mod
go mod why <pkg>      # Why is this dependency needed?
go get -u <pkg>       # Update specific
go clean -modcache    # Nuclear option: clear cache
```

### Rust (Cargo)
```bash
cargo update          # Update within semver ranges
cargo tree -d         # Find duplicate dependencies
cargo check           # Fast type-check without full build
```

### Java (Maven/Gradle)
```bash
mvn dependency:tree   # Full dependency tree
mvn dependency:analyze # Find unused/undeclared deps
./gradlew dependencies # Gradle equivalent
```
