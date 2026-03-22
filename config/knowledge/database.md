# Database Knowledge Base

## Query Performance

### Always EXPLAIN ANALYZE
Before optimizing, understand what's happening:
```sql
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) SELECT ...;
```

Key metrics to check:
- **Seq Scan on large table** → Needs index
- **Nested Loop with high row count** → Consider Hash Join (add index)
- **Sort** with high cost → Add index for ORDER BY columns
- **Rows estimated vs actual** diverge → Run `ANALYZE table_name`

### Index Strategy

| Scenario | Index Type |
|---|---|
| Equality filter (`WHERE status = 'active'`) | B-tree (default) |
| Range filter (`WHERE created_at > ...`) | B-tree |
| Full-text search | GIN with tsvector |
| JSON field queries | GIN on jsonb column |
| Geospatial queries | GiST |
| Composite filters (`WHERE a = ? AND b > ?`) | Composite: `(a, b)` — equality columns first |
| Low-cardinality filter + sort | Composite: `(status, created_at)` |

### N+1 Query Detection
```
# BAD: N+1
for user in users:
    orders = db.query("SELECT * FROM orders WHERE user_id = ?", user.id)

# GOOD: batch
orders = db.query("SELECT * FROM orders WHERE user_id IN (?)", user_ids)
orders_by_user = group_by(orders, 'user_id')
```

## Schema Design

### Type Selection
| Use Case | Recommended Type | Avoid |
|---|---|---|
| Primary key | `bigint` (auto-increment) or `uuid` | `int` (runs out), `varchar` PK |
| Text fields | `text` | `varchar(255)` (arbitrary limit) |
| Timestamps | `timestamptz` | `timestamp` (no timezone) |
| Money/decimal | `numeric(precision, scale)` | `float`/`double` (rounding errors) |
| Boolean | `boolean` | `int` (0/1) |
| Enums | `text` + CHECK constraint | DB-level ENUM (hard to modify) |
| JSON data | `jsonb` | `json` (no indexing), `text` |

### Naming Conventions
- Table names: `lowercase_snake_case`, plural (`users`, `order_items`)
- Column names: `lowercase_snake_case` (`created_at`, `user_id`)
- Foreign keys: `<referenced_table>_id` (`user_id`, `order_id`)
- Indexes: `idx_<table>_<columns>` (`idx_users_email`)

## Security (PostgreSQL/Supabase)

### Row Level Security (RLS)
```sql
-- Enable RLS
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Policy: users can only see their own documents
CREATE POLICY "users_own_documents" ON documents
  FOR ALL
  USING (user_id = (SELECT auth.uid()));

-- IMPORTANT: Index the policy column for performance
CREATE INDEX idx_documents_user_id ON documents(user_id);
```

### Least Privilege
```sql
-- Application role should NOT be superuser
GRANT SELECT, INSERT, UPDATE ON users TO app_role;
-- NEVER: GRANT ALL ON ALL TABLES TO app_role;

-- Sensitive columns
REVOKE SELECT (password_hash) ON users FROM app_role;
```

## Anti-Patterns

| Anti-Pattern | Problem | Fix |
|---|---|---|
| `SELECT *` | Fetches unnecessary data, breaks on schema change | List specific columns |
| `varchar(255)` everywhere | Arbitrary limit, wastes catalog space | Use `text` + CHECK constraint if needed |
| `timestamp` without `tz` | Timezone ambiguity | Use `timestamptz` |
| Random UUID as PK | Poor index locality, B-tree fragmentation | Use `uuid_generate_v7()` (time-sorted) or `bigint` |
| `OFFSET` pagination | Scans and discards rows | Use cursor/keyset pagination: `WHERE id > last_id LIMIT N` |
| Unparameterized queries | SQL injection | Always use `$1, $2` parameters |
| Missing `NOT NULL` | Null handling bugs | Add `NOT NULL` with sensible defaults |
| No foreign keys | Orphaned data | Add `REFERENCES` + `ON DELETE` policy |
| `ON DELETE CASCADE` everywhere | Accidental mass deletion | Use `ON DELETE RESTRICT` unless cascade is intended |

## Migration Best Practices

1. **One change per migration**: Don't mix schema + data changes
2. **Backward compatible**: New columns should be nullable or have defaults
3. **No downtime**: Avoid `ALTER TABLE ... ADD COLUMN ... NOT NULL` without default
4. **Index concurrently**: `CREATE INDEX CONCURRENTLY` to avoid locking
5. **Test rollback**: Every migration should have a working rollback script
