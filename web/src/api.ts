// Types mirror the Pydantic models in app/schemas.py. `label` is the same
// three-value enum the backend enforces - the UI never invents a status string.

export type Label = "DUPLICATE" | "NOT_DUPLICATE" | "UNSURE";
export type DecisionKind = "approve" | "reject" | "override";
export type ReviewStatus = "awaiting_review" | "finalised";

export interface EvidenceItem {
  tool: string;
  observation: string;
}

export interface Verdict {
  label: Label;
  confidence: number;
  rationale: string;
  evidence: EvidenceItem[];
}

export interface TraceEvent {
  seq: number;
  ts: string;
  kind: string;
  detail: Record<string, unknown>;
}

export interface InvestigationSummary {
  investigation_id: string;
  pair_id: string;
  case_a: string;
  case_b: string;
  account_a: string;
  account_b: string;
  subject_a: string;
  subject_b: string;
  cheap_score: number;
  status: ReviewStatus;
  draft_label: Label | null;
  draft_confidence: number | null;
  final_label: Label | null;
  created_at: string;
}

export interface InvestigationDetail extends InvestigationSummary {
  candidate_reasons: string[];
  verdict: Verdict | null;
  steps_used: number;
  tool_calls_used: number;
  llm_mode: string;
  injection_flags: unknown[];
  human_decision: Record<string, unknown> | null;
  trace: TraceEvent[];
}

export interface DecisionPayload {
  decision: DecisionKind;
  reviewer: string;
  note: string;
  override_label?: Label;
}

/** The backend answers 4xx with {detail: ...}; surface that rather than a bare
 *  status code, because the human gate's refusals (409 on a second decision,
 *  422 on a missing note) are the interesting responses, not edge cases. */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) {
        message =
          typeof body.detail === "string"
            ? body.detail
            : JSON.stringify(body.detail);
      }
    } catch {
      /* non-JSON error body; keep the status line */
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export const listInvestigations = (status: ReviewStatus) =>
  request<InvestigationSummary[]>(`/investigations?status=${status}`);

export const getInvestigation = (id: string) =>
  request<InvestigationDetail>(`/investigations/${id}`);

export const decide = (id: string, payload: DecisionPayload) =>
  request<Record<string, unknown>>(`/investigations/${id}/decision`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
