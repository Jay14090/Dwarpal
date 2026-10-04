"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { api, type Metric } from "@/lib/api";
import { pct } from "@/lib/utils";

type MetricsResponse = { headline: Metric[]; reports: Record<string, unknown> };

function value(m: Metric): string {
  if (m.value == null) return "not measured";
  if (m.key === "alerts_once") return String(m.value);
  return pct(m.value);
}

export default function MetricsPage() {
  const [data, setData] = useState<MetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);

  useEffect(() => {
    api<MetricsResponse>("/metrics")
      .then(setData)
      .catch((e) => setError((e as Error).message));
  }, []);

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-4">
      <div>
        <h1 className="text-lg font-semibold">Metrics</h1>
        <p className="text-sm text-muted-foreground">
          Every number comes from an evaluation report in <code className="font-mono">data/eval/</code>, computed from a real run. Missing
          numbers show the command that measures them.
        </p>
      </div>
      {error && <p className="text-sm text-unknown">{error}</p>}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {data?.headline.map((m) => (
          <Card key={m.key}>
            <CardHeader>
              <CardDescription>{m.label}</CardDescription>
              <CardTitle className={m.value == null ? "text-base text-muted-foreground" : "font-mono text-3xl"}>{value(m)}</CardTitle>
            </CardHeader>
            <CardContent className="grid gap-1 text-xs text-muted-foreground">
              {m.detail && <span>{m.detail}</span>}
              {m.source ? (
                <span className="font-mono">{m.source}</span>
              ) : (
                <span>
                  run <code className="font-mono text-foreground">{m.how}</code>
                </span>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
      {data && (
        <Card>
          <CardHeader>
            <CardTitle>Reports ({Object.keys(data.reports).length})</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-2">
            {Object.entries(data.reports).map(([name, rep]) => (
              <div key={name} className="rounded-md border">
                <button className="w-full px-3 py-2 text-left font-mono text-sm hover:bg-accent" onClick={() => setOpen(open === name ? null : name)}>
                  {name}
                </button>
                {open === name && (
                  <pre className="max-h-96 overflow-auto border-t bg-background/60 p-3 text-xs">{JSON.stringify(rep, null, 1)}</pre>
                )}
              </div>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
