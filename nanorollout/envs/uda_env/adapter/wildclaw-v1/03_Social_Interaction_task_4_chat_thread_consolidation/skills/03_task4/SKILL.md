---
name: 03_task4
description: "Reads recent Slack messages to compile an accurate project status report and saves it as a draft for review before sending to a client. Use when: preparing a client-facing update, reconciling conflicting or revised numbers from multiple sources, or drafting a message that requires human review before sending."
---

# Slack Status Drafter Skill

Read recent Slack messages, reconcile the latest figures, and save a polished client-ready status report as a draft — never send directly.

## Tools

All tools are defined in `tmp_workspace/utils.py`:

- **`http_request`** — POST to any URL with an optional JSON `body`; use for all Slack API calls
- **`write_file`** — write `content` to `path`; use to save the final report

---

## Slack API

**Base URL:** `http://localhost:9110`

| Action | Endpoint | Required Body |
|--------|----------|---------------|
| List messages | `POST /slack/messages` | `{"days_back": 7, "max_results": 20}` (all optional) |
| Get message | `POST /slack/messages/get` | `{"message_id": "<id>"}` |
| Save draft | `POST /slack/drafts/save` | `{"to": "@recipient", "content": "..."}` + optional `"reply_to_message_id"` |

> ⚠️ **Draft-only** task. Always use `slack_save_draft` — do **not** call `slack_send_message` (`POST /slack/send`). The user must review before anything goes to the client.

---

## Workflow

1. **List messages** — fetch recent messages with `slack_list_messages`
2. **Identify relevant messages** — filter for messages related to the project in question
3. **Read each in full** — retrieve complete content via `slack_get_message`
4. **Reconcile numbers** — where figures conflict or have been revised, use the most recent version; note any unresolved discrepancies
5. **Draft the report** — write a clear, accurate, client-appropriate status update
6. **Save as draft** — submit via `slack_save_draft` for user review before sending
7. **Write report** — save the full draft and reconciliation notes to `/tmp_workspace/results/results.md`

---

## Draft Format

```markdown
Hi [Client Name],

Here's the latest status update on [Project Name]:

**Overall Status**: On track / At risk / Delayed

**Progress**
- [Area]: [current status, latest figures]

**Key Milestones**
- [Milestone]: [status, date]

**Next Steps**
- ...

Please let me know if you have any questions.

[Your Name]
```

## Report Format (results.md)

```markdown
# Status Report Draft — [Project Name]

## Draft Saved To
Recipient: ...
Draft ID / timestamp: ...

## Reconciliation Notes
- [Topic]: conflicting figures found — used [source/date] as most recent
- [Topic]: no conflicts, data consistent

## Draft Content
[Full text of the draft]
```

---

## Constraints

- **Never** call `slack_send_message` — save as draft only
- Reconcile revised numbers before including in the draft; flag unresolved discrepancies
- Final report must be written to `/tmp_workspace/results/results.md`