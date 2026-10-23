# AI Concierge Agent — Implementation Notes

## Architecture

The concierge agent uses a Python tool-calling loop compatible with the Anthropic Messages API. It uses explicit Python logic for safety-critical decisions rather than relying on the LLM alone. Booking requests are checked, member and ticket details are retrieved through tools, and complete booking proposals are prepared for human approval before vendor contact.

The agent uses an allowlisted tool set for member lookup, ticket management, vendor discovery, proposal creation, vendor messaging, and human escalation. Outbound messages use the house booking identity rather than the member's private details. Vendor confirmations are checked against the requested booking details. Ambiguous requests, changed confirmations, requests for private information, and uncertain delivery outcomes are escalated.

## Reliability and safety

- Deduplicate incoming messages and booking proposals.
- Require explicit human approval before initial vendor contact.
- Use idempotency keys to prevent duplicate sends.
- Treat transport timeouts as uncertain delivery, not proof of failure.
- Send at most one reminder to an unresponsive vendor before escalation.
- Escalate cancellation requests after vendor contact so a human can verify the cancellation.
- Keep member contact and payment details out of vendor messages.
- Test duplicate approval, incorrect confirmations, silent vendors, and requests for private information.

## Scaling to 1,000 members and 50 bookings per day

Fifty bookings per day is a modest workload. For production, run the agent as a stateless service behind a queue, persist tickets, proposals, approvals, and message IDs in a transactional database, and use a durable outbox plus provider-supported idempotency for outbound messages. Add per-member access controls, encrypted secrets, audit logs, retries with backoff, monitoring, and rate limits. Use scheduled workers for reminders and escalation deadlines. These changes would improve durability across restarts and support concurrent requests.

## Classifier placement and validation

`classifier/predict.py` provides a lightweight rule-based baseline with the interface `predict(texts: list[str]) -> list[str]`. It returns `request` or `not_request` and favors recall so potentially actionable messages are less likely to be missed. It is a preliminary routing aid, not a trained model or a safety boundary. The agent must still apply request validation, privacy checks, and human approval.

Validate it on representative booking requests, cancellations, ambiguous messages, ordinary chatter, and multilingual/Hinglish examples. Track false negatives and false positives separately, prioritize reducing false negatives, and keep human review for uncertain cases.

## Limitations and unfinished work

The classifier is rule-based and needs a larger labeled evaluation set. Production deployment still needs persistent state, durable job processing, stronger concurrency controls, operational monitoring, and end-to-end testing against real vendor integrations. Cancellation completion must be verified with the vendor rather than inferred from an internal ticket status.

## AI-tool usage

AI assistance was used to help review implementation logic, identify edge cases, draft tests, and prepare documentation. The code and scenario results should be reviewed and tested by the developer before submission.