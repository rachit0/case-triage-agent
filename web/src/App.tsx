import { useCallback, useEffect, useState } from "react";
import {
  decide,
  getInvestigation,
  listInvestigations,
  type DecisionKind,
  type InvestigationDetail,
  type InvestigationSummary,
  type Label,
  type ReviewStatus,
  type TraceEvent,
} from "./api";

const LABELS: Label[] = ["DUPLICATE", "NOT_DUPLICATE", "UNSURE"];

function labelClass(label: string | null | undefined): string {
  return `tag tag-${(label ?? "none").toLowerCase()}`;
}

/** One trace row. The detail blob is deliberately shown verbatim: the point of
 *  the audit trail is that a reviewer sees what the agent actually saw, not a
 *  prettified summary of it. */
function TraceRow({ event }: { event: TraceEvent }) {
  const [open, setOpen] = useState(false);
  const summary =
    (event.detail.tool as string) ??
    (event.detail.rule as string) ??
    (event.detail.decision as string) ??
    "";
  return (
    <li className={`trace trace-${event.kind}`}>
      <button className="trace-head" onClick={() => setOpen((v) => !v)}>
        <span className="seq">#{event.seq}</span>
        <span className="kind">{event.kind}</span>
        {summary && <span className="summary">{summary}</span>}
        <span className="chev">{open ? "−" : "+"}</span>
      </button>
      {open && <pre>{JSON.stringify(event.detail, null, 2)}</pre>}
    </li>
  );
}

function DecisionForm({
  detail,
  onDone,
}: {
  detail: InvestigationDetail;
  onDone: () => void;
}) {
  const [reviewer, setReviewer] = useState("rachit");
  const [note, setNote] = useState("");
  const [overrideLabel, setOverrideLabel] = useState<Label>("NOT_DUPLICATE");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(kind: DecisionKind) {
    setBusy(true);
    setError(null);
    try {
      await decide(detail.investigation_id, {
        decision: kind,
        reviewer,
        note,
        ...(kind === "override" ? { override_label: overrideLabel } : {}),
      });
      onDone();
    } catch (e) {
      // The backend rejects reject/override without a note, and 409s a second
      // decision. Showing that verbatim is the clearest proof the gate is
      // enforced server-side and not by this form.
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="decide">
      <h3>Your decision</h3>
      <div className="row">
        <label>
          Reviewer
          <input value={reviewer} onChange={(e) => setReviewer(e.target.value)} />
        </label>
        <label className="grow">
          Note <span className="hint">(required to reject or override)</span>
          <input
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="why you agree or disagree with the agent"
          />
        </label>
      </div>
      <div className="row">
        <button
          className="btn approve"
          disabled={busy || !reviewer.trim()}
          onClick={() => submit("approve")}
        >
          Approve
        </button>
        <button
          className="btn reject"
          disabled={busy || !reviewer.trim()}
          onClick={() => submit("reject")}
        >
          Reject
        </button>
        <span className="or">or override to</span>
        <select
          value={overrideLabel}
          onChange={(e) => setOverrideLabel(e.target.value as Label)}
        >
          {LABELS.map((l) => (
            <option key={l} value={l}>
              {l}
            </option>
          ))}
        </select>
        <button
          className="btn override"
          disabled={busy || !reviewer.trim()}
          onClick={() => submit("override")}
        >
          Override
        </button>
      </div>
      {error && <p className="error">Backend refused: {error}</p>}
    </div>
  );
}

function Detail({
  id,
  onDecided,
}: {
  id: string;
  onDecided: () => void;
}) {
  const [detail, setDetail] = useState<InvestigationDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setError(null);
    getInvestigation(id).then(setDetail).catch((e) => setError(String(e)));
  }, [id]);

  useEffect(load, [load]);

  if (error) return <p className="error">{error}</p>;
  if (!detail) return <p className="muted">Loading…</p>;

  const v = detail.verdict;
  return (
    <div className="detail">
      <header>
        <h2>{detail.pair_id}</h2>
        <span className={labelClass(detail.status)}>{detail.status}</span>
      </header>

      <div className="cases">
        <div className="case">
          <h4>{detail.case_a}</h4>
          <p className="acct">{detail.account_a}</p>
          <p>{detail.subject_a}</p>
        </div>
        <div className="case">
          <h4>{detail.case_b}</h4>
          <p className="acct">{detail.account_b}</p>
          <p>{detail.subject_b}</p>
        </div>
      </div>

      <p className="muted small">
        pre-filter score {detail.cheap_score.toFixed(2)} ·{" "}
        {detail.candidate_reasons.join(", ")}
      </p>

      {detail.injection_flags.length > 0 && (
        <p className="warn">
          ⚠ Instruction-like text was found in this customer's case text. It was
          recorded as data and never obeyed.
        </p>
      )}

      {v && (
        <section className="verdict">
          <h3>
            Agent recommendation{" "}
            <span className={labelClass(v.label)}>{v.label}</span>
            <span className="conf">confidence {v.confidence.toFixed(2)}</span>
          </h3>
          <p>{v.rationale}</p>
          <h4>Evidence</h4>
          {v.evidence.length === 0 ? (
            <p className="muted">
              No citable evidence survived the guards — this is why it is UNSURE.
            </p>
          ) : (
            <ul className="evidence">
              {v.evidence.map((e, i) => (
                <li key={i}>
                  <code>{e.tool}</code>
                  <span>{e.observation}</span>
                </li>
              ))}
            </ul>
          )}
          <p className="muted small">
            {detail.steps_used} steps · {detail.tool_calls_used} tool calls ·{" "}
            {detail.llm_mode}
          </p>
        </section>
      )}

      {detail.human_decision ? (
        <section className="decided">
          <h3>Human decision recorded</h3>
          <pre>{JSON.stringify(detail.human_decision, null, 2)}</pre>
        </section>
      ) : (
        <DecisionForm
          detail={detail}
          onDone={() => {
            load();
            onDecided();
          }}
        />
      )}

      <section>
        <h3>Audit trail ({detail.trace.length} events)</h3>
        <ul className="tracelist">
          {detail.trace.map((e) => (
            <TraceRow key={e.seq} event={e} />
          ))}
        </ul>
      </section>
    </div>
  );
}

export default function App() {
  const [status, setStatus] = useState<ReviewStatus>("awaiting_review");
  const [items, setItems] = useState<InvestigationSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    setError(null);
    listInvestigations(status)
      .then((rows) => {
        setItems(rows);
        setSelected((cur) =>
          cur && rows.some((r) => r.investigation_id === cur)
            ? cur
            : (rows[0]?.investigation_id ?? null),
        );
      })
      .catch((e) =>
        setError(
          `${e}. Is the API running?  uvicorn app.api:app --reload`,
        ),
      );
  }, [status]);

  useEffect(refresh, [refresh]);

  return (
    <div className="app">
      <aside>
        <h1>Review inbox</h1>
        <div className="tabs">
          {(["awaiting_review", "finalised"] as ReviewStatus[]).map((s) => (
            <button
              key={s}
              className={s === status ? "on" : ""}
              onClick={() => setStatus(s)}
            >
              {s.replace("_", " ")}
            </button>
          ))}
        </div>
        {error && <p className="error">{error}</p>}
        {!error && items.length === 0 && (
          <p className="muted">
            Nothing here. Seed the queue with{" "}
            <code>python -m scripts.run_batch --limit 12</code>
          </p>
        )}
        <ul className="queue">
          {items.map((it) => (
            <li key={it.investigation_id}>
              <button
                className={it.investigation_id === selected ? "on" : ""}
                onClick={() => setSelected(it.investigation_id)}
              >
                <span className={labelClass(it.final_label ?? it.draft_label)}>
                  {it.final_label ?? it.draft_label}
                </span>
                <strong>{it.account_a}</strong>
                <em>{it.subject_a}</em>
                <span className="muted small">{it.pair_id}</span>
              </button>
            </li>
          ))}
        </ul>
      </aside>
      <main>
        {selected ? (
          <Detail id={selected} onDecided={refresh} />
        ) : (
          <p className="muted">Select an investigation.</p>
        )}
      </main>
    </div>
  );
}
