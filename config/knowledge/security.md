# Security Knowledge Base

## OWASP Top 10 (2021) Checklist

### A01: Broken Access Control
- [ ] Every endpoint has authentication check
- [ ] Authorization enforced (not just authentication)
- [ ] CORS configured restrictively (not `*`)
- [ ] Directory listing disabled
- [ ] JWT tokens validated (expiry, signature, issuer)
- [ ] ID references checked for ownership (IDOR prevention)

### A02: Cryptographic Failures
- [ ] No MD5/SHA1 for passwords (use bcrypt/scrypt/argon2)
- [ ] TLS 1.2+ enforced for all connections
- [ ] Encryption at rest for sensitive data
- [ ] No hardcoded encryption keys
- [ ] Secure random number generation (not math.random)

### A03: Injection
- [ ] Parameterized queries for all SQL (no string concatenation)
- [ ] Command injection prevention (no shell=True with user input)
- [ ] LDAP injection prevention
- [ ] Template injection prevention (Jinja2 autoescape)
- [ ] NoSQL injection prevention (MongoDB operator injection)

### A04: Insecure Design
- [ ] Rate limiting on authentication endpoints
- [ ] Account lockout after failed attempts
- [ ] Business logic validated server-side (not just client)
- [ ] Sensitive operations require re-authentication

### A05: Security Misconfiguration
- [ ] Debug mode disabled in production
- [ ] Default credentials changed
- [ ] Unnecessary features/ports disabled
- [ ] Security headers set (CSP, X-Frame-Options, HSTS)
- [ ] Error messages don't leak internal details

### A06: Vulnerable Components
- [ ] Dependencies updated (no known CVEs)
- [ ] Dependency lock file committed
- [ ] Minimal dependencies (no unused packages)

### A07: Authentication Failures
- [ ] Strong password policy enforced
- [ ] Multi-factor authentication available
- [ ] Session timeout configured
- [ ] Session invalidation on logout/password change

### A08: Data Integrity Failures
- [ ] CI/CD pipeline integrity (signed commits, protected branches)
- [ ] No auto-update from untrusted sources
- [ ] Serialization validation (no pickle/yaml.load from untrusted)

### A09: Logging & Monitoring
- [ ] Authentication events logged
- [ ] Authorization failures logged
- [ ] No sensitive data in logs (passwords, tokens, PII)
- [ ] Log tampering prevention

### A10: SSRF
- [ ] URL validation for server-side requests
- [ ] Allowlist for external service URLs
- [ ] Internal network access blocked for user-provided URLs
- [ ] DNS rebinding prevention

## Secret Detection Patterns

| Type | Pattern | Example |
|---|---|---|
| AWS Access Key | `AKIA[A-Z0-9]{16}` | AKIAIOSFODNN7EXAMPLE |
| AWS Secret Key | 40-char base64 after `aws_secret` | wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE |
| GitHub Token | `ghp_[a-zA-Z0-9]{36}` | ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx |
| GitLab Token | `glpat-[a-zA-Z0-9_-]{20}` | glpat-xxxxxxxxxxxxxxxxxxxx |
| Slack Token | `xoxb-` or `xoxp-` | xoxb-xxxx-xxxx-xxxx |
| Google API Key | `AIza[0-9A-Za-z_-]{35}` | AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxx |
| Private Key | `-----BEGIN (RSA )?PRIVATE KEY-----` | |
| JWT | `eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+` | |
| Connection String | `(mongodb\|mysql\|postgres\|redis)://.*:.*@` | postgres://user:pass@host |

## Emergency Response Protocol

When a CRITICAL security issue is found:
1. **Document**: Exact location, reproduction steps, potential impact
2. **Immediate Fix**: Provide the smallest possible fix
3. **Rotate Secrets**: If any credential was exposed, it MUST be rotated
4. **Verify**: Confirm the fix actually closes the vulnerability
5. **Audit**: Check for similar patterns elsewhere in the codebase
