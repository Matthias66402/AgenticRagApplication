import type { TrainTripPlan } from "../api/types";
import { formatDate } from "../lib/labels";
import { JourneyCard, TransitNote } from "./JourneyCard";
import { LinkedText } from "./LinkedText";

/** Reisekarte einer reinen Bahnfahrt: gewählte Verbindung(en) statt Karte, Etappen und Exporten. */
export function TrainTripCard({ plan }: { plan: TrainTripPlan }) {
  const outbound = plan.outbound_journey;
  return (
    <article className="plan">
      <header>
        <div className="plan-title">
          <h2>{plan.title}</h2>
        </div>
        <div className="chips">
          <span className="chip">🚆 Bahnfahrt</span>
          <span className="chip">📅 {formatDate(outbound.departure.slice(0, 10))}</span>
          {plan.return_journey && <span className="chip">↔ mit Rückfahrt</span>}
        </div>
        <p>
          <LinkedText text={plan.summary} />
        </p>
      </header>

      <section>
        <h3>Verbindung</h3>
        <div className="journeys">
          <JourneyCard label="🚆 Hinfahrt" journey={outbound} />
          {plan.return_journey && <JourneyCard label="🚆 Rückfahrt" journey={plan.return_journey} />}
        </div>
        <TransitNote />
      </section>
    </article>
  );
}
