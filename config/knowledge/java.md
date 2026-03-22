# Java Knowledge Base

## Security Patterns

| Pattern | Risk | Fix |
|---|---|---|
| `"SELECT * FROM " + input` | SQL Injection (CRITICAL) | Use PreparedStatement with `?` parameters |
| `Runtime.exec(userInput)` | Command Injection (CRITICAL) | Use ProcessBuilder with argument list |
| `ScriptEngine.eval(userInput)` | Code Injection (CRITICAL) | Never eval user input; use safe parser |
| Path traversal via `new File(userPath)` | Path Traversal (CRITICAL) | Canonicalize path, verify prefix |
| Hardcoded `password = "..."` | Secrets in Code (CRITICAL) | Use environment variables or vault |
| `PII in log.info()` | Data Leak (HIGH) | Mask/redact PII before logging |
| `@CrossOrigin(origins = "*")` | CORS Misconfiguration (HIGH) | Restrict to specific origins |
| Missing `@Valid` on request body | Input Validation Bypass (HIGH) | Add `@Valid` on `@RequestBody` parameters |
| CSRF disabled without reason | CSRF vulnerability (HIGH) | Enable CSRF or document exception |

## Spring Boot Architecture

| Anti-Pattern | Fix |
|---|---|
| `@Autowired` field injection | Constructor injection: `private final UserService userService;` |
| Business logic in `@Controller` | Move to `@Service` layer |
| `@Transactional` on Controller | Move to Service layer methods |
| Entity returned directly in response | Use DTO/Response class |
| `@Component` / `@Service` without interface | Define interface for testability |
| Circular dependency between services | Refactor; use events or break cycle with interface |

## JPA / Database

| Anti-Pattern | Problem | Fix |
|---|---|---|
| `FetchType.EAGER` on collections | N+1 queries, loads entire graph | Use `FetchType.LAZY` + `@EntityGraph` when needed |
| Unbounded `findAll()` | Memory explosion | Always paginate: `Pageable` parameter |
| Missing `@Modifying` on update query | Query doesn't execute | Add `@Modifying` and `@Transactional` |
| `CascadeType.ALL` on parent | Accidental deletes/updates | Use specific: `CascadeType.PERSIST, MERGE` |
| No index on frequently filtered columns | Slow queries | Add `@Index` or database migration |
| `toString()` on lazy entity | LazyInitializationException | Exclude lazy fields from toString |

## Workflow / State Machine Patterns

| Anti-Pattern | Fix |
|---|---|
| No idempotency key on mutation | Add idempotency key, check before processing |
| Illegal state transition not guarded | Use state machine (Spring Statemachine) or guard in service |
| Non-atomic compensation | Wrap in transaction, or use Saga with compensation log |
| Missing retry with jitter | Add exponential backoff with jitter: `delay * (1 + random)` |
| No dead-letter handling | Send failed events to DLQ, alert and retry manually |

## Testing

```bash
# Maven
mvn verify
mvn test -Dtest=ClassName#methodName

# Gradle
./gradlew test
./gradlew test --tests "ClassName.methodName"

# Coverage (JaCoCo)
mvn jacoco:report

# Static analysis
mvn checkstyle:check
mvn spotbugs:check

# Dependency security
mvn org.owasp:dependency-check-maven:check
```

## Common Build Errors

| Error | Quick Fix |
|---|---|
| `cannot find symbol` | Add import, check dependency in pom.xml/build.gradle |
| `incompatible types` | Add explicit cast or fix type mismatch |
| `does not override abstract method` | Implement all interface/abstract methods |
| `package does not exist` | Add dependency to build file |
| Lombok not processing | Ensure `lombok` + `maven-compiler-plugin` annotation processing configured |
| MapStruct not generating | Add `mapstruct-processor` to annotation processors |
| Circular dependency (Spring) | Use `@Lazy`, refactor, or use events |
| `java.lang.NoSuchMethodError` | Version conflict — check `mvn dependency:tree -Dverbose` |
