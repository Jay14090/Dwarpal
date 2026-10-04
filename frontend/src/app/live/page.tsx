"use client";

import { Camera, CircleDot, Eye, UserPlus } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { EventRow } from "@/components/event-row";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input, Label, Select } from "@/components/ui/input";
import { api, apiUrl, type CameraInfo, type EventItem } from "@/lib/api";
import { useLiveFeed, type FeedMessage } from "@/lib/use-live-feed";
import { isAdmin, useActor } from "@/lib/use-actor";

type FrameMeta = { tracks: { label: string; role: string | null; global_id: number | null }[] };

function CameraTile({ cam }: { cam: CameraInfo }) {
  const [counts, setCounts] = useState<Record<string, number>>({});
  const admin = isAdmin(useActor());
  const unblur = async () => {
    const reason = prompt(`Reason for showing unknown faces on ${cam.name} for 60 s (written to the audit log):`);
    if (!reason) return;
    try {
      await api("/privacy/unblur-stream", { method: "POST", body: JSON.stringify({ camera_id: cam.id, seconds: 60, reason }) });
    } catch (e) {
      alert((e as Error).message);
    }
  };
  useEffect(() => {
    const t = setInterval(async () => {
      try {
        const m = await api<FrameMeta>(`/cameras/${cam.id}/frame.json`);
        const c: Record<string, number> = {};
        for (const tr of m.tracks) {
          const k = tr.label === "person" ? (tr.role ?? "pending") : "vehicle";
          c[k] = (c[k] ?? 0) + 1;
        }
        setCounts(c);
      } catch {}
    }, 1000);
    return () => clearInterval(t);
  }, [cam.id]);
  const s = cam.stats as { mode?: string; process_fps?: number; infer_ms?: number } | null;
  return (
    <Card className="overflow-hidden">
      <div className="relative aspect-video bg-black">
        {/* MJPEG stream with role-coloured boxes drawn by the backend */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={apiUrl(cam.stream_url)} alt={cam.name} className="size-full object-contain" />
        <div className="absolute left-2 top-2 flex items-center gap-1 rounded bg-black/60 px-2 py-0.5 font-mono text-[11px]">
          <CircleDot className={cam.online ? "size-3 text-resident" : "size-3 text-unknown"} />
          {cam.online ? "LIVE" : "OFFLINE"}
        </div>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 text-xs">
        <div>
          <div className="font-semibold">{cam.name}</div>
          <div className="font-mono text-muted-foreground">
            {cam.id} · {s?.mode ?? cam.run_mode}
            {s?.process_fps != null && ` · ${s.process_fps} fps`}
            {s?.infer_ms ? ` · ${s.infer_ms} ms` : ""}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-1">
          {admin && (
            <button onClick={unblur} title="Unblur faces for 60 s (audited)" className="rounded border p-1 text-muted-foreground hover:bg-accent">
              <Eye className="size-3.5" />
            </button>
          )}
          {(["resident", "staff", "unknown", "pending"] as const).map((r) =>
            counts[r] ? (
              <Badge key={r} variant={r}>
                {counts[r]} {r}
              </Badge>
            ) : null,
          )}
          {counts.vehicle ? <Badge variant="secondary">{counts.vehicle} vehicle</Badge> : null}
        </div>
      </div>
    </Card>
  );
}

function EnrollPanel({ cameras }: { cameras: CameraInfo[] }) {
  const live = cameras.filter((c) => c.enabled && c.run_mode === "realtime");
  const [form, setForm] = useState({ display_name: "", unit: "", role: "resident", consent: false, camera_id: "webcam" });
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  // fall back to the first realtime camera when the chosen one is not running
  const cameraId = live.some((c) => c.id === form.camera_id) ? form.camera_id : (live[0]?.id ?? form.camera_id);

  const submit = async () => {
    setBusy(true);
    setMsg({ ok: true, text: "Capturing… look at the camera and turn your head slightly." });
    try {
      const r = await api<{ id: number; face_shots?: number; body_shots?: number }>("/enroll/capture", {
        method: "POST",
        body: JSON.stringify({ ...form, camera_id: cameraId, unit: form.unit || null }),
      });
      setMsg({ ok: true, text: `Enrolled #${r.id} (${r.face_shots ?? 0} face, ${r.body_shots ?? 0} body shots). Step back in.` });
      setForm((f) => ({ ...f, display_name: "", unit: "", consent: false }));
    } catch (e) {
      setMsg({ ok: false, text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <UserPlus className="size-4" /> Enroll from camera
        </CardTitle>
        <CardDescription>3–5 shots in about 10 s. Requires the person&apos;s consent.</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-2">
        <div className="grid grid-cols-2 gap-2">
          <div className="col-span-2 grid gap-1">
            <Label htmlFor="name">Name</Label>
            <Input id="name" value={form.display_name} onChange={(e) => setForm({ ...form, display_name: e.target.value })} />
          </div>
          <div className="grid gap-1">
            <Label htmlFor="unit">Unit</Label>
            <Input id="unit" placeholder="B-402" value={form.unit} onChange={(e) => setForm({ ...form, unit: e.target.value })} />
          </div>
          <div className="grid gap-1">
            <Label htmlFor="role">Role</Label>
            <Select id="role" value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })}>
              <option value="resident">Resident</option>
              <option value="staff">Staff</option>
            </Select>
          </div>
          <div className="col-span-2 grid gap-1">
            <Label htmlFor="cam">Camera</Label>
            <Select id="cam" value={cameraId} onChange={(e) => setForm({ ...form, camera_id: e.target.value })}>
              {live.length === 0 && <option value="webcam">webcam (not running)</option>}
              {live.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </Select>
          </div>
        </div>
        <label className="flex items-center gap-2 text-xs">
          <input type="checkbox" checked={form.consent} onChange={(e) => setForm({ ...form, consent: e.target.checked })} />
          The person consents to face and body enrollment
        </label>
        <Button onClick={submit} disabled={busy || !form.display_name || !form.consent}>
          <Camera /> {busy ? "Capturing…" : "Capture & enroll"}
        </Button>
        {msg && <p className={msg.ok ? "text-xs text-muted-foreground" : "text-xs text-unknown"}>{msg.text}</p>}
      </CardContent>
    </Card>
  );
}

export default function LivePage() {
  const [cameras, setCameras] = useState<CameraInfo[]>([]);
  const [events, setEvents] = useState<EventItem[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        setCameras(await api<CameraInfo[]>("/cameras"));
        setError(null);
      } catch (e) {
        setError(`Backend unreachable: ${(e as Error).message}`);
      }
    };
    load();
    api<EventItem[]>("/events?limit=30").then(setEvents).catch(() => {});
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, []);

  const onMsg = useCallback((m: FeedMessage) => {
    if (m.kind === "event") setEvents((prev) => [m.data, ...prev].slice(0, 100));
  }, []);
  const connected = useLiveFeed(onMsg);
  const enabled = cameras.filter((c) => c.enabled);

  return (
    <div className="flex flex-col gap-4 xl:flex-row">
      <section className="min-w-0 flex-1">
        <div className="mb-3 flex items-center justify-between">
          <h1 className="text-lg font-semibold">Live cameras</h1>
          <div className="flex gap-2 text-xs">
            {(["resident", "staff", "unknown", "pending"] as const).map((r) => (
              <Badge key={r} variant={r}>
                {r}
              </Badge>
            ))}
          </div>
        </div>
        {error && <p className="mb-3 rounded border border-unknown/40 bg-unknown/10 p-2 text-sm text-unknown">{error}</p>}
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2 2xl:grid-cols-3">
          {enabled.map((c) => (
            <CameraTile key={c.id} cam={c} />
          ))}
        </div>
        {!error && enabled.length === 0 && <p className="text-sm text-muted-foreground">No enabled cameras in config/cameras.yaml.</p>}
      </section>
      <aside className="flex w-full flex-col gap-4 xl:w-96">
        <EnrollPanel cameras={cameras} />
        <Card className="flex min-h-0 flex-1 flex-col">
          <CardHeader className="flex-row items-center justify-between">
            <CardTitle>Event feed</CardTitle>
            <span className="flex items-center gap-1 text-xs text-muted-foreground">
              <CircleDot className={connected ? "size-3 text-resident" : "size-3 text-unknown"} />
              {connected ? "live" : "reconnecting"}
            </span>
          </CardHeader>
          <CardContent className="flex max-h-[60vh] flex-col gap-2 overflow-y-auto">
            {events.length === 0 && <p className="text-xs text-muted-foreground">No events yet.</p>}
            {events.map((ev, i) => (
              <EventRow key={`${ev.id ?? "x"}-${i}`} ev={ev} compact />
            ))}
          </CardContent>
        </Card>
      </aside>
    </div>
  );
}
