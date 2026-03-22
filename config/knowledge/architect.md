# Architecture Knowledge Base

## Anti-Patterns to Detect

| Anti-Pattern | Symptom | Fix |
|---|---|---|
| Big Ball of Mud | No clear module boundaries, everything depends on everything | Introduce bounded contexts, define clear interfaces |
| God Object | Single class/module with 500+ lines doing everything | Split by responsibility (SRP) |
| Golden Hammer | Using same technology for every problem | Evaluate fit per use case |
| Tight Coupling | Changing one module forces changes in 5+ others | Dependency inversion, interface-based design |
| Distributed Monolith | Microservices that must deploy together | Merge back or properly decouple via events |
| Premature Optimization | Complex caching/sharding before proving it's needed | Measure first, optimize bottlenecks only |
| Feature Creep Architecture | Architecture designed for hypothetical future requirements | YAGNI — design for current + 1 step ahead |

## Design Patterns by Domain

### Frontend
- **Component Composition**: Small, focused components composed together
- **Container/Presenter**: Separate data fetching from rendering
- **Code Splitting**: Lazy load routes and heavy components
- **State Colocation**: Keep state as close to where it's used as possible

### Backend
- **Repository Pattern**: Abstract data access behind interfaces
- **Service Layer**: Business logic separate from HTTP/transport layer
- **CQRS**: Separate read and write models for complex domains
- **Event-Driven**: Decouple services via events for eventual consistency
- **Circuit Breaker**: Protect against cascading failures in distributed systems

### Data
- **Event Sourcing**: Store events, not state — full audit trail
- **Cache-Aside**: Application manages cache population and invalidation
- **Eventual Consistency**: Accept temporary inconsistency for availability
- **Saga Pattern**: Manage distributed transactions via compensating actions

## Scalability Planning Framework

| Scale | Users | Key Concerns | Typical Architecture |
|---|---|---|---|
| Startup | <10K | Simplicity, speed of iteration | Monolith, single DB |
| Growth | 10K-100K | Performance, reliability | Load balancer, read replicas, caching |
| Scale | 100K-1M | Horizontal scaling, resilience | Microservices, message queues, CDN |
| Enterprise | 1M-10M | Multi-region, compliance, cost | Multi-region, event-driven, data sharding |

## Non-Functional Requirements Checklist
- [ ] **Latency**: P50, P95, P99 targets defined?
- [ ] **Throughput**: Peak RPS estimated? Growth factor accounted?
- [ ] **Availability**: SLA target (99.9%? 99.99%)?
- [ ] **Durability**: Data loss tolerance? Backup strategy?
- [ ] **Security**: Auth model? Encryption at rest/transit? Audit logging?
- [ ] **Observability**: Structured logging? Distributed tracing? Alerting?
- [ ] **Cost**: Infrastructure cost estimate? Cost per user?
- [ ] **Compliance**: GDPR/CCPA? Data residency requirements?
