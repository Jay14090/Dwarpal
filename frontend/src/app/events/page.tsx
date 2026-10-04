"use client";

import { CircleDot } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { EventRow } from "@/components/event-row";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select } from "@/components/ui/input";
import { api, type EventItem, type PlateItem } from "@/lib/api";
import { useLiveFeed, type FeedMessage } from "@/lib/use-live-feed";
import { Badge } from "@/components/ui/badge";
import { fmtTime } from "@/lib/utils";

export default function EventsPage() {
  const [events, setEvents] = useState<EventItem[]>([]);
  const [plates, setPlates] = useState<PlateItem[]>([]);
  const [rule, setRule] = useState("");
  const [openOnly, setOpenOnly] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [version, setVersion] = useState(0);
  const load = useCallback(() => setVersion((v) => v + 1), []);
  useEffect(() => {
    let stale = false;
    const qs = new URLSearchParams({ limit: "200" });
    if (rule) qs.set("rule", rule);
    if (openOnly) qs.set("unacknowledged", "true");
    Promise.all([api<EventItem[]>(`/events?${qs}`), api<PlateItem[]>("/plates?limit=50")])
      .then(([e, p]) => {
        if (stale) return;
        setEvents(e);
        setPlates(p);
        setError(null);
      })
      .catch((e) => !stale && setError((e as Error).message));
    return () => {
      stale = true;
    };
  }, [rule, openOnly, version]);

  const onMsg = useCallback(
    (m: FeedMessage) => {
      if (m.kind === "event" && (!rule || m.data.rule === rule)) setEvents((p) => [m.data, ...p].slice(0, 500));
      if (m.kind === "plate") setPlates((p) => [m.data, ...p].slice(0, 100));
    },
    [rule],
  );
  const connected = useLiveFeed(onMsg);

  const rules = useMemo(() => Array.from(new Set(events.map((e) => e.rule))).sort(), [events]);

  const ack = async (id: number) => {
    try {
      const upd = await api<EventItem>(`/events/${id}/ack`, { method: "POST" });
      setEvents((p) => (openOnly ? p.filter((e) => e.id !== id) : p.map((e) => (e.id === id ? upd : e))));
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <div className="flex flex-col gap-4 xl:flex-row">
      <Card className="min-w-0 flex-1">
        <CardHeader className="flex-row flex-wrap items-center justify-between gap-2">
          <CardTitle className="flex items-center gap-2">
            Events
            <CircleDot className={connected ? "size-3 text-resident" : "size-3 text-unknown"} />
          </CardTitle>
          <div className="flex items-center gap-2">
            <Select value={rule} onChange={(e) => setRule(e.target.value)} className="w-56">
              <option value="">All rules</option>
              {rules.map((r) => (
                <option key={r}>{r}</option>
              ))}
            </Select>
            <label className="flex items-center gap-1 text-xs">
              <input type="checkbox" checked={openOnly} onChange={(e) => setOpenOnly(e.target.checked)} /> Open only
            </label>
            <Button size="sm" variant="outline" onClick={load}>
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent className="grid gap-2">
          {error && <p className="text-sm text-unknown">{error}</p>}
          {events.length === 0 && <p className="text-sm text-muted-foreground">No events.</p>}
          {events.map((ev, i) => (
            <EventRow key={`${ev.id ?? "x"}-${i}`} ev={ev} onAck={ack} />
          ))}
        </CardContent>
      </Card>
      <Card className="w-full xl:w-96">
        <CardHeader>
          <CardTitle>Plate reads</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-1.5 text-xs">
          {plates.length === 0 && <p className="text-muted-foreground">No plate reads.</p>}
          {plates.map((p, i) => (
            <div key={`${p.id}-${i}`} className="flex items-center gap-2 rounded border bg-background/40 px-2 py-1.5">
              <span className="font-mono text-sm font-semibold">{p.plate}</span>
              <Badge variant={p.status === "registered" ? "resident" : p.status === "likely_registered" ? "medium" : "unknown"}>
                {p.status.replace("_", " ")}
              </Badge>
              <span className="ml-auto text-muted-foreground">{p.camera_id}</span>
              <span className="font-mono text-muted-foreground">{fmtTime(p.ts)}</span>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
