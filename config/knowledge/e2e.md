# E2E Testing Knowledge Base

## Test Planning

### Identify Critical User Journeys (by risk)
1. **Revenue-critical**: Checkout, payment, subscription flows
2. **Auth-critical**: Login, registration, password reset, SSO
3. **Data-critical**: CRUD operations, data import/export, file upload
4. **Core UX**: Main feature flows (search, create, edit, share)

Prioritize: test the paths that, if broken, would page someone at 3am.

## Page Object Model (POM)

```typescript
// GOOD: Encapsulate page interactions
class LoginPage {
  private readonly page: Page;

  constructor(page: Page) { this.page = page; }

  async navigate() { await this.page.goto('/login'); }
  async fillEmail(email: string) { await this.page.fill('[data-testid="email"]', email); }
  async fillPassword(pwd: string) { await this.page.fill('[data-testid="password"]', pwd); }
  async submit() { await this.page.click('[data-testid="submit"]'); }
  async getError() { return this.page.textContent('[data-testid="error"]'); }

  async login(email: string, password: string) {
    await this.navigate();
    await this.fillEmail(email);
    await this.fillPassword(password);
    await this.submit();
  }
}

// Usage in test
test('successful login', async ({ page }) => {
  const loginPage = new LoginPage(page);
  await loginPage.login('user@test.com', 'password123');
  await expect(page).toHaveURL('/dashboard');
});
```

## Locator Strategy (priority order)

| Priority | Strategy | Example | When |
|---|---|---|---|
| 1 | `data-testid` | `[data-testid="submit"]` | Best — stable, decoupled from UI |
| 2 | Semantic role | `getByRole('button', { name: 'Submit' })` | Good — accessibility-friendly |
| 3 | Label text | `getByLabel('Email')` | Good for form fields |
| 4 | Placeholder | `getByPlaceholder('Enter email')` | OK for inputs |
| 5 | CSS class | `.btn-primary` | Avoid — breaks on styling changes |
| 6 | XPath | `//div[@class="..."]` | Last resort |

**Never use**: auto-generated IDs, DOM position (`nth-child`), text content that changes with i18n.

## Wait Strategy

| Anti-Pattern | Problem | Fix |
|---|---|---|
| `await page.waitForTimeout(3000)` | Flaky, slow | Wait for condition: `await page.waitForSelector(...)` |
| `sleep(5)` between steps | Wastes time, still flaky | `await expect(element).toBeVisible()` |
| No wait before assertion | Race condition | Use auto-waiting assertions: `await expect(...)` |

### Correct Wait Patterns
```typescript
// Wait for navigation
await page.waitForURL('/dashboard');

// Wait for network idle
await page.waitForLoadState('networkidle');

// Wait for specific API response
const response = await page.waitForResponse('**/api/users');
expect(response.status()).toBe(200);

// Wait for element state
await expect(page.locator('[data-testid="result"]')).toBeVisible();
await expect(page.locator('[data-testid="spinner"]')).toBeHidden();
```

## Flaky Test Handling

### Detection
Run each test 3-5 times. If it fails inconsistently, it's flaky.
```bash
# Playwright: run 5 times
npx playwright test --repeat-each=5 tests/checkout.spec.ts
```

### Quarantine Protocol
1. Mark as flaky: `test.fixme('flaky: intermittent timeout on CI')`
2. Track in issue tracker with `flaky-test` label
3. Investigate root cause (usually: timing, shared state, external dependency)
4. Fix and un-quarantine
5. **Never delete** a flaky test — it's telling you something

### Common Flaky Causes
| Cause | Fix |
|---|---|
| Race condition with animations | Wait for animation end, or disable animations in test |
| Shared test data | Isolate: each test creates its own data |
| External API dependency | Mock external APIs, or use record/replay |
| Timezone-dependent logic | Fix test timezone: `TZ=UTC` |
| Order-dependent tests | Each test must be independently runnable |

## Test Data Management

```typescript
// GOOD: Factory pattern for test data
function createTestUser(overrides = {}) {
  return {
    email: `test-${Date.now()}@example.com`,
    password: 'TestPassword123!',
    name: 'Test User',
    ...overrides,
  };
}

// GOOD: Cleanup after each test
test.afterEach(async ({ page }) => {
  // Delete test-created resources
  await api.cleanup();
});
```

## Test Organization

```
tests/
├── e2e/
│   ├── auth/
│   │   ├── login.spec.ts
│   │   ├── register.spec.ts
│   │   └── password-reset.spec.ts
│   ├── checkout/
│   │   ├── cart.spec.ts
│   │   └── payment.spec.ts
│   ├── pages/              # Page Object Models
│   │   ├── login.page.ts
│   │   └── checkout.page.ts
│   ├── fixtures/           # Test data factories
│   │   └── users.ts
│   └── helpers/            # Shared utilities
│       └── auth.ts
```

## Success Metrics

| Metric | Target |
|---|---|
| Critical journeys passing | 100% |
| Overall pass rate | >95% |
| Flaky test rate | <5% |
| Total suite duration | <10 minutes |
| Test count per critical journey | 3-5 scenarios (happy + top errors) |

## Playwright Configuration

```typescript
// playwright.config.ts
export default defineConfig({
  testDir: './tests/e2e',
  retries: process.env.CI ? 2 : 0,     // Retry on CI only
  timeout: 30_000,                       // 30s per test
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',            // Trace on retry for debugging
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  webServer: {
    command: 'npm run dev',
    port: 3000,
    reuseExistingServer: !process.env.CI,
  },
});
```
