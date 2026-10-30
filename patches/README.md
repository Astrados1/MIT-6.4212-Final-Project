# Patches to upstream code

Each change we make to an upstream repository lives here as a patch, and nowhere else.
`scripts/setup_upstream.sh` applies them to fresh checkouts. `scripts/record_provenance.sh` and every run
record show whether they are applied. Each patch is a deviation from the upstream baseline and must also be
listed under "Behavior-changing modifications" in `docs/baseline_report.md`.

| patch | repo | why | behavior change |
|---|---|---|---|
| `tiptop/0001-gemini-robotics-er-2.patch` | tiptop @ `d8f5afd` | `gemini-robotics-er-1.6-preview`, the default in the pinned code, was shut down by Google on Aug 31, 2026. Google's documented replacement is `gemini-robotics-er-2-preview`. | Yes: a different perception model. The thinking setting also changes from `thinking_budget=0` (thinking off, documented for ER 1.6) to `thinking_level=MINIMAL`, the lowest level documented for ER 2. Google does not document a budget range for ER 2, and a request cannot combine budget and level. |

Sources: [Gemini deprecations](https://ai.google.dev/gemini-api/docs/deprecations),
[Robotics ER overview](https://ai.google.dev/gemini-api/docs/robotics-overview),
[thinking levels](https://ai.google.dev/gemini-api/docs/thinking). The locked `google-genai` 1.70.0 has
`types.ThinkingLevel.MINIMAL`.

Not verified against the live API yet (no key). If ER 2 rejects `MINIMAL`, record the error and the setting
actually used, update the patch, and note the change in `docs/baseline_report.md`.
