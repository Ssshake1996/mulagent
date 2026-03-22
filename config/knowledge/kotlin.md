# Kotlin & Android Knowledge Base

## Security Patterns

| Pattern | Risk | Fix |
|---|---|---|
| Exported Activity/Service without permission | Unauthorized Access (CRITICAL) | Set `android:exported="false"` or add permission check |
| `SharedPreferences` for sensitive data | Data Leak (CRITICAL) | Use `EncryptedSharedPreferences` or Android Keystore |
| `WebView.settings.javaScriptEnabled = true` + `addJavascriptInterface` | XSS/RCE (CRITICAL) | Limit JS interface, validate URLs, use `@JavascriptInterface` annotation |
| Cleartext HTTP traffic | MITM (HIGH) | Enforce HTTPS in `network_security_config.xml` |
| Logging sensitive data: `Log.d(TAG, "token=$token")` | Data Leak (HIGH) | Never log tokens/passwords; use ProGuard to strip logs in release |
| Insecure random: `java.util.Random()` | Weak RNG (HIGH) | Use `java.security.SecureRandom()` |
| SQL query via string concatenation | SQL Injection (CRITICAL) | Use Room `@Query` with parameters, or `rawQuery` with `selectionArgs` |
| Hardcoded API keys | Secrets in Code (CRITICAL) | Use `BuildConfig` field from `local.properties` or secret manager |

## Coroutines & Flows

| Anti-Pattern | Problem | Fix |
|---|---|---|
| `GlobalScope.launch { }` | Leaks — no lifecycle cancellation | Use `viewModelScope`, `lifecycleScope`, or structured scope |
| `catch (e: CancellationException)` | Breaks cancellation propagation | Rethrow: `catch (e: Exception) { if (e is CancellationException) throw e }` |
| Heavy work without `withContext(Dispatchers.IO)` | Blocks main thread | Wrap IO/CPU work: `withContext(Dispatchers.IO) { ... }` |
| `MutableStateFlow` with mutable data class | State mutation without emission | Use immutable data classes: `data class State(val items: List<Item>)` |
| `flow.collect` without `WhileSubscribed` in ViewModel | Flow stays active when UI gone | Use `stateIn(scope, SharingStarted.WhileSubscribed(5000), initial)` |
| `launch` without error handling | Silent crash | Add `CoroutineExceptionHandler` or `try/catch` inside launch |
| Nested `withContext` with same dispatcher | Unnecessary context switch | Remove inner `withContext` if dispatcher matches |

## Jetpack Compose

| Anti-Pattern | Problem | Fix |
|---|---|---|
| Unstable parameters in Composable | Skipped recomposition fails, unnecessary recomposition | Use `@Immutable` or `@Stable` annotations, use `List` → `ImmutableList` |
| Side effects outside `LaunchedEffect` | Runs on every recomposition | Wrap in `LaunchedEffect(key) { }` |
| `NavController` passed deep into composables | Tight coupling to navigation | Pass lambda callbacks: `onNavigate: (Route) -> Unit` |
| Missing `key` in `LazyColumn` items | Wrong item recomposition | `items(list, key = { it.id }) { ... }` |
| `remember` without all keys | Stale cached value | `remember(dep1, dep2) { compute() }` |
| `mutableStateOf` inside Composable without `remember` | Resets on recomposition | `val state = remember { mutableStateOf(initial) }` |
| Large build() >80 lines | Hard to read, slow preview | Extract sub-composables |
| `StatefulWidget` when stateless works | Unnecessary complexity | Hoist state: stateless composable + state holder |

## Kotlin Idioms

| Anti-Pattern | Idiomatic Kotlin |
|---|---|
| `if (x != null) { x.doSomething() }` | `x?.doSomething()` |
| `if (x != null) x else default` | `x ?: default` |
| `when(x) { ... else -> throw }` | Exhaustive `when` on sealed class (no else needed) |
| `for (i in 0 until list.size)` | `for (item in list)` or `list.forEach { }` |
| `val result = ArrayList<T>(); for (...) result.add(...)` | `val result = list.map { ... }` |
| `object : Callback { override fun on... }` | Lambda: `setCallback { result -> ... }` |
| `companion object { const val TAG = "..." }` for DI | Use Hilt `@Inject` or Koin modules |
| `String.format("Hello %s", name)` | String template: `"Hello $name"` |
| Nullable Boolean: `if (flag == true)` | OK in Kotlin, but prefer non-null: `val flag: Boolean = false` |

## Architecture Patterns

### Clean Architecture Layers
```
┌─────────────────────┐
│   Presentation      │  ← Composables, ViewModels
│   (UI Layer)        │     Depends on: Domain
├─────────────────────┤
│   Domain            │  ← Use Cases, Entities, Repository Interfaces
│   (Business Logic)  │     NO framework imports
├─────────────────────┤
│   Data              │  ← Repository Implementations, API, DB
│   (Infrastructure)  │     Depends on: Domain
└─────────────────────┘
```

### Violations to Detect
| Violation | Symptom | Fix |
|---|---|---|
| Domain imports framework | `import android.*` in domain module | Extract to interface in domain, implement in data |
| Data layer leaks to UI | ViewModel returns Room Entity | Map to domain model or UI model |
| Business logic in ViewModel | Complex calculations in ViewModel | Extract to UseCase class |
| Circular dependency between modules | Module A imports B, B imports A | Move shared types to common module |

## Gradle / Build

### Common Build Errors
| Error | Fix |
|---|---|
| `Unresolved reference: X` | Add import, check dependency in `build.gradle.kts` |
| `Type mismatch: inferred type X but Y expected` | Fix type, add explicit conversion |
| `No applicable candidates` | Check function overloads, argument types |
| `Smart cast impossible` | Check for mutability: use `val` or local copy |
| `Non-exhaustive when` | Add missing branches or `else ->` |
| `Suspend function called outside coroutine` | Wrap in `launch { }`, `async { }`, or make caller `suspend` |
| `Internal visibility` | Class/function is `internal`, can't access from other module |
| `Conflicting declarations` | Rename, or check import conflicts |
| Dependency resolution failure | `./gradlew dependencies`, check version catalogs |
| Detekt violations | `./gradlew detekt`, fix or suppress with `@Suppress` + reason |

### Diagnostic Tools
```bash
# Build
./gradlew assembleDebug

# Tests
./gradlew test
./gradlew connectedAndroidTest

# Static analysis
./gradlew detekt
./gradlew lint

# Dependency analysis
./gradlew dependencies --configuration releaseRuntimeClasspath
```
