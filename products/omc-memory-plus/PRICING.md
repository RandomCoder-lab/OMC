# Pricing

Measured savings on a real codebase: **73% token cost reduction** on Claude Code sessions using OMC Memory+ vs raw context-paste.

## Plans

### Free

$0/mo · forever

- All 17 MCP tools (compression, memory, OMC reference)
- Local memory storage at `~/.omc/memory/`
- Unlimited namespaces, unlimited entries
- Survives reboot, `/exit`, machine restart
- Source open under MIT

For: individuals dogfooding the workflow, students, OSS contributors.

### Pro

$5/mo per seat

Everything in Free, plus:
- **Cross-machine sync** — memory follows you between desktop, laptop, server
- **Cross-device recall** — start a session on phone (eventually), continue on desktop
- **Longer cloud retention** — 1 year vs Free's local-only retention
- **Private cloud namespaces** — separate from local Free storage
- **Priority support**

For: solo devs working across multiple machines, consultants juggling client projects.

### Team

$50/mo for 5 seats ($10/seat)

Everything in Pro, plus:
- **Shared team namespaces** — your team's collective Claude Code memory
- **Per-namespace ACLs** (read/write/admin)
- **Audit log** — see who recalled what, when
- **Slack / Discord webhook** on store/recall events
- **Volume discount** at 10+ seats ($8/seat)

For: dev teams using Claude Code on a shared codebase. Shared findings = shared productivity.

### Enterprise

from $500/mo

Everything in Team, plus:
- **Self-hosted memory server** — run the sync backend in your VPC
- **SSO** (Okta, Azure AD, custom SAML)
- **Custom retention policies**
- **Data residency** (US, EU, APAC)
- **SLA** with 99.9% uptime guarantee
- **Direct support channel** — Slack Connect or dedicated email

For: regulated industries (finance, healthcare), enterprises with strict data residency requirements, large engineering orgs (100+ devs).

## ROI calculator

| seats | sessions/dev/mo | raw context tokens | with Memory+ | savings/dev/mo @ $3/MTok | savings/team/mo |
|--:|--:|--:|--:|--:|--:|
| 1 | 100 | 26k | 7k | $5.70 | $5.70 |
| 5 | 100 | 26k | 7k | $5.70 | $28.50 |
| 50 | 100 | 26k | 7k | $5.70 | $285.00 |
| 500 | 100 | 26k | 7k | $5.70 | $2,850.00 |

Even on the conservative end (100 sessions/dev/mo, 26k tokens/session of project context), a 50-dev team saves $285/mo. The **Team plan pays for itself within 9 days** of usage; the **Enterprise plan pays for itself within ~2 months** at 500 devs.

## What you actually pay for

- **Free**: zero. The tools are open source. Hosted on your machine.
- **Pro / Team**: cloud sync infrastructure, retention storage, namespace ACL service.
- **Enterprise**: SSO integration, self-hosted backend support, SLA underwriting.

The compression + memory primitives are free forever. Paid plans add convenience layers (sync, sharing, audit) on top.

## Why this pricing makes sense

Claude API is $3/MTok input. A 50-dev team running 100 sessions/dev/mo costs ~$390/mo in input tokens before Memory+. With Memory+ it drops to ~$105/mo — a $285 savings. We charge $50 for the Team plan, capturing about 18% of the savings we create. Customer keeps 82% of the savings. Aligned incentives.

## FAQ

**Why not just bigger context windows?**

Claude Sonnet 4.5 already has 200k context. Memory+ isn't about working around context size — it's about not paying input-token costs to re-establish context every session. The hash reference is 5 tokens; the full content is 1,500. Even with infinite context, you don't want to pay 1,500 tokens of input on every session start.

**Why not just CLAUDE.md?**

CLAUDE.md is great for stable project info. Memory+ handles the dynamic findings/decisions/notes that accumulate across sessions and would otherwise be lost to `/exit`. The two are complementary: CLAUDE.md tells Claude *what the project is*; Memory+ tells it *what you've learned about the project*.

**Is this just RAG?**

No. RAG fetches semantically similar content based on embedding similarity. Memory+ fetches **exact content by canonical hash** — alpha-rename invariant, deterministic, lossless. The substrate codec is a structural fingerprint, not an embedding. Memory+ is the dual of RAG: precise recall by identity, vs probabilistic recall by similarity.
