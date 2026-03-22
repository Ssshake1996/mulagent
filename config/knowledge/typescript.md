# TypeScript / JavaScript Knowledge Base

## Security Patterns

| Pattern | Risk | Fix |
|---|---|---|
| `eval(userInput)` | Code Execution (CRITICAL) | Use JSON.parse() or purpose-built parser |
| `new Function(userInput)` | Code Execution (CRITICAL) | Avoid; use safe evaluation libraries |
| `innerHTML = userInput` | XSS (CRITICAL) | Use `textContent` or DOMPurify |
| `dangerouslySetInnerHTML` | XSS (CRITICAL) | Sanitize with DOMPurify first |
| `` `SELECT * FROM ${table}` `` | SQL Injection (CRITICAL) | Use parameterized queries |
| `child_process.exec(userInput)` | Command Injection (CRITICAL) | Use `execFile` with argument array |
| `Object.assign({}, userInput)` | Prototype Pollution (HIGH) | Validate keys, use `Object.create(null)` |
| `JSON.parse(untrusted)` without validation | Type confusion (HIGH) | Validate with Zod/Yup schema after parsing |

## Type Safety

| Anti-Pattern | Fix |
|---|---|
| `any` without justification | Use `unknown` and narrow, or define proper type |
| `x as SomeType` (unsafe cast) | Use type guard: `if ('field' in x)` |
| `x!` (non-null assertion) | Add proper null check: `if (x != null)` |
| Relaxed tsconfig (`strict: false`) | Enable `strict: true` and fix errors |
| `// @ts-ignore` | Fix the type error; if truly unfixable, use `// @ts-expect-error` with reason |

## Async Correctness

| Anti-Pattern | Problem | Fix |
|---|---|---|
| `array.forEach(async ...)` | Promises ignored | Use `for...of` with `await`, or `Promise.all(arr.map(...))` |
| Sequential awaits for independent work | Unnecessary slowdown | `const [a, b] = await Promise.all([fetchA(), fetchB()])` |
| Missing `.catch()` on Promise | Unhandled rejection | Add `.catch()` or use try/catch with await |
| `async` function without `await` | Returns wrapped Promise unnecessarily | Remove `async` or add missing `await` |
| Missing error handling in Promise.all | One failure rejects all | Use `Promise.allSettled()` when partial results OK |

## React / Next.js Patterns

| Anti-Pattern | Fix |
|---|---|
| Missing dependency array in useEffect | Add ALL dependencies used inside the effect |
| State mutation: `state.push(item)` | `setState([...state, item])` |
| Index as key in dynamic list | Use stable unique ID: `key={item.id}` |
| useEffect for derived state | Use `useMemo` or compute during render |
| Prop drilling through 5+ levels | Use Context, Zustand, or component composition |
| `useEffect(() => { fetchData() }, [])` | Use React Query / SWR for data fetching |
| Server/client boundary leak (Next.js) | Mark with `"use client"` / `"use server"` properly |

## Node.js Backend

| Anti-Pattern | Fix |
|---|---|
| `fs.readFileSync` in request handler | Use `fs.promises.readFile` (async) |
| `process.env.X` without validation | Validate at startup with `zod` or `envalid` |
| Missing input validation at boundaries | Use Zod/Joi/class-validator on all external input |
| `throw new Error("...")` without context | Include context: `throw new Error(\`User ${id} not found\`)` |
| Express without helmet/cors | `app.use(helmet())`, configure CORS properly |
| No rate limiting on API | Use `express-rate-limit` or API gateway |

## Testing

```bash
# Jest
npx jest --coverage --coverageThreshold='{"global":{"branches":80,"functions":80}}'

# Vitest
npx vitest --coverage

# Type checking
npx tsc --noEmit

# Linting
npx eslint src/ --ext .ts,.tsx
```

## Common Build Errors

| Error | Quick Fix |
|---|---|
| `TS2304: Cannot find name 'X'` | Add import; install `@types/X` if needed |
| `TS2322: Type X not assignable to type Y` | Fix type mismatch or add proper conversion |
| `TS2345: Argument type mismatch` | Check function signature, add type narrowing |
| `TS7006: Parameter implicitly has 'any'` | Add type annotation |
| `TS2532: Object is possibly 'undefined'` | Add `?.` or null check |
| `Module not found` | `npm install <pkg>`, check tsconfig `paths` |
| `SyntaxError: Unexpected token` | Check Node.js version, add babel/swc transform |
