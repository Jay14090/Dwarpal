"use client";

import { Car, Eye, Film, Search as SearchIcon, X } from "lucide-react";
import { useState } from "react";
import { Badge, RoleBadge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { api, apiUrl, protectedImage, type PlateResult, type SearchFilter, type SearchResponse, type TrackResult } from "@/lib/api";
import { isAdmin, useActor } from "@/lib/use-actor";
import { fmtTime } from "@/lib/utils";

const EXAMPLES = [
  "guy in a green t-shirt, around six foot, near the gate after 9 pm",
  "all unknown people today",
  "when did TN09AB1234 enter?",
  "person in a blue jacket and black pants in the gym",
  "unregistered cars at the main gate today",
];

function FilterChips({ f }: { f: SearchFilter }) {
  const chips: string[] = [`entity: ${f.entity}`];
  if (f.roles.length) chips.push(`roles: ${f.roles.join(", ")}`);
  if (f.upper_color) chips.push(`upper: ${f.upper_color}`);
  if (f.lower_color) chips.push(`lower: ${f.lower_color}`);
  if (f.height_cm) chips.push(`height: ${f.height_cm.min ?? "…"}–${f.height_cm.max ?? "…"} cm`);
  if (f.cameras.length) chips.push(`cameras: ${f.cameras.join(", ")}`);
  if (f.zones.length) chips.push(`zones: ${f.zones.join(", ")}`);
  if (f.time_range) chips.push(`time: ${fmtTime(f.time_range.from)} → ${fmtTime(f.time_range.to)}`);
  if (f.plate) chips.push(`plate: ${f.plate}`);
  if (f.free_text) chips.push(`appearance: “${f.free_text}”`);
  return (
    <div className="flex flex-wrap gap-1.5">
      {chips.map((c) => (
        <Badge key={c} variant="secondary" className="font-mono">
          {c}
        </Badge>
      ))}
    </div>
  );
}

function TrackCard({ r, onPlay }: { r: TrackResult; onPlay: (url: string) => void }) {
  const admin = isAdmin(useActor());
  const [raw, setRaw] = useState<string | null>(null);
  const unblur = async () => {
    try {
      setRaw(await protectedImage(`/tracks/${r.track_id}/thumb.jpg?unblur=true`));
    } catch (e) {
      alert((e as Error).message);
    }
  };
  return (
    <Card className="overflow-hidden">
      <div className="relative flex aspect-[3/4] items-center justify-center bg-black">
        {r.thumb_url ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={raw ?? apiUrl(r.thumb_url)} alt="" className="size-full object-contain" />
        ) : (
          <span className="text-xs text-muted-foreground">no thumbnail</span>
        )}
        {r.score != null && (
          <span className="absolute right-1 top-1 rounded bg-black/70 px-1.5 font-mono text-[10px]">{r.score.toFixed(3)}</span>
        )}
      </div>
      <CardContent className="grid gap-1 p-2 text-xs">
        <div className="flex items-center justify-between gap-1">
          <RoleBadge role={r.role} />
          <span className="font-mono text-muted-foreground">{r.global_id != null ? `#${r.global_id}` : `t${r.track_id}`}</span>
        </div>
        <div className="font-mono">{r.camera_id}</div>
        <div className="text-muted-foreground">{fmtTime(r.start_ts)}</div>
        <div className="text-muted-foreground">
          {r.upper_color ?? "?"} / {r.lower_color ?? "?"}
          {r.height_cm != null && ` · ${Math.round(r.height_cm)}±${Math.round(r.height_err_cm ?? 0)} cm`}
        </div>
        <div className="flex gap-1">
          {r.clip_url && (
            <Button size="sm" variant="outline" className="flex-1" onClick={() => onPlay(r.clip_url!)}>
              <Film /> Clip
            </Button>
          )}
          {admin && r.thumb_url && !raw && ["unknown", "pending"].includes(r.role) && (
            <Button size="sm" variant="outline" title="Show the face (audited)" onClick={unblur}>
              <Eye />
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function PlateRow({ r }: { r: PlateResult }) {
  const variant = r.status === "registered" ? "resident" : r.status === "likely_registered" ? "medium" : "unknown";
  return (
    <div className="flex items-center gap-3 rounded-md border bg-card p-2 text-sm">
      <div className="flex h-12 w-28 items-center justify-center overflow-hidden rounded bg-black">
        {r.thumb_url ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={apiUrl(r.thumb_url)} alt="" className="h-full object-contain" />
        ) : (
          <Car className="size-5 text-muted-foreground" />
        )}
      </div>
      <span className="font-mono text-base font-semibold">{r.plate}</span>
      <Badge variant={variant}>{r.status.replace("_", " ")}</Badge>
      <span className="text-muted-foreground">{r.camera_id}</span>
      <span className="ml-auto font-mono text-xs text-muted-foreground">{fmtTime(r.ts)}</span>
    </div>
  );
}

export default function SearchPage() {
  const [q, setQ] = useState("");
  const [res, setRes] = useState<SearchResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [clip, setClip] = useState<string | null>(null);

  const run = async (query: string) => {
    if (!query.trim()) return;
    setQ(query);
    setBusy(true);
    setError(null);
    try {
      setRes(await api<SearchResponse>("/search", { method: "POST", body: JSON.stringify({ query, limit: 48 }) }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const tracks = res?.results.filter((r): r is TrackResult => r.kind === "track") ?? [];
  const plates = res?.results.filter((r): r is PlateResult => r.kind === "plate_read") ?? [];

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-4">
      <h1 className="text-lg font-semibold">Search footage</h1>
      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          run(q);
        }}
      >
        <Input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="e.g. unknown person in a red jacket near the gate after 9 pm"
          className="h-10 text-base"
        />
        <Button type="submit" size="lg" disabled={busy}>
          <SearchIcon /> {busy ? "Searching…" : "Search"}
        </Button>
      </form>
      <div className="flex flex-wrap gap-2">
        {EXAMPLES.map((ex) => (
          <button key={ex} onClick={() => run(ex)} className="rounded-full border px-3 py-1 text-xs text-muted-foreground hover:bg-accent">
            {ex}
          </button>
        ))}
      </div>
      {error && <p className="rounded border border-unknown/40 bg-unknown/10 p-2 text-sm text-unknown">{error}</p>}
      {res && (
        <div className="grid gap-2">
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <span>
              {res.results.length} result{res.results.length === 1 ? "" : "s"} · parsed by {res.parsed_by} · ranked by {res.ranked_by}
              {res.plate_match && res.plate_match !== "all" && ` · plate match: ${res.plate_match}`}
            </span>
          </div>
          <FilterChips f={res.filter} />
          {res.relaxed.length > 0 && (
            <p className="text-xs text-yellow-400">
              No exact colour match; showing the closest by appearance (relaxed: {res.relaxed.join(", ")}).
            </p>
          )}
        </div>
      )}
      {tracks.length > 0 && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
          {tracks.map((r) => (
            <TrackCard key={r.track_id} r={r} onPlay={(u) => setClip(u)} />
          ))}
        </div>
      )}
      {plates.length > 0 && (
        <div className="grid gap-2">
          {plates.map((r) => (
            <PlateRow key={r.plate_read_id} r={r} />
          ))}
        </div>
      )}
      {res && res.results.length === 0 && <p className="text-sm text-muted-foreground">Nothing matched.</p>}
      {clip && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4" onClick={() => setClip(null)}>
          <div className="relative w-full max-w-4xl" onClick={(e) => e.stopPropagation()}>
            <button className="absolute -top-9 right-0 text-muted-foreground hover:text-foreground" onClick={() => setClip(null)}>
              <X />
            </button>
            <video src={apiUrl(clip)} controls autoPlay className="w-full rounded-lg bg-black" />
          </div>
        </div>
      )}
    </div>
  );
}
