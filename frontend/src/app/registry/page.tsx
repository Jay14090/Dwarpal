"use client";

import { Car, ScrollText, Trash2, Upload, User } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Badge, RoleBadge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input, Label, Select } from "@/components/ui/input";
import { api, apiUrl, type Vehicle } from "@/lib/api";
import { isAdmin, useActor } from "@/lib/use-actor";
import { fmtTime } from "@/lib/utils";

type Person = {
  id: number;
  role: string;
  display_name: string;
  unit: string | null;
  consent: boolean;
  face_samples: number;
  body_samples: number;
};

function Avatar({ id }: { id: number }) {
  const [ok, setOk] = useState(true);
  return (
    <div className="flex size-12 shrink-0 items-center justify-center overflow-hidden rounded-md bg-muted">
      {ok ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={apiUrl(`/people/${id}/thumb.jpg`)} alt="" className="size-full object-cover" onError={() => setOk(false)} />
      ) : (
        <User className="size-5 text-muted-foreground" />
      )}
    </div>
  );
}

function UploadEnroll({ onDone }: { onDone: () => void }) {
  const [name, setName] = useState("");
  const [unit, setUnit] = useState("");
  const [role, setRole] = useState("resident");
  const [consent, setConsent] = useState(false);
  const [files, setFiles] = useState<FileList | null>(null);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    if (!files?.length) return;
    const fd = new FormData();
    fd.set("display_name", name);
    fd.set("role", role);
    fd.set("consent", String(consent));
    if (unit) fd.set("unit", unit);
    Array.from(files).forEach((f) => fd.append("images", f));
    setBusy(true);
    try {
      const r = await api<{ id: number; face_shots: number; body_shots: number }>("/enroll/upload", { method: "POST", body: fd });
      setMsg({ ok: true, text: `Enrolled #${r.id}: ${r.face_shots} face, ${r.body_shots} body samples.` });
      setName("");
      setUnit("");
      setConsent(false);
      setFiles(null);
      onDone();
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
          <Upload className="size-4" /> Enroll from photos
        </CardTitle>
        <CardDescription>3–5 clear photos. For live capture use the panel on the Live page.</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-2">
        <div className="grid gap-1">
          <Label>Name</Label>
          <Input value={name} onChange={(e) => setName(e.target.value)} />
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div className="grid gap-1">
            <Label>Unit</Label>
            <Input value={unit} onChange={(e) => setUnit(e.target.value)} placeholder="B-402" />
          </div>
          <div className="grid gap-1">
            <Label>Role</Label>
            <Select value={role} onChange={(e) => setRole(e.target.value)}>
              <option value="resident">Resident</option>
              <option value="staff">Staff</option>
            </Select>
          </div>
        </div>
        <Input type="file" accept="image/*" multiple onChange={(e) => setFiles(e.target.files)} className="h-auto py-1.5" />
        <label className="flex items-center gap-2 text-xs">
          <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
          The person consents to face and body enrollment
        </label>
        <Button onClick={submit} disabled={busy || !name || !consent || !files?.length}>
          {busy ? "Enrolling…" : "Enroll"}
        </Button>
        {msg && <p className={msg.ok ? "text-xs text-muted-foreground" : "text-xs text-unknown"}>{msg.text}</p>}
      </CardContent>
    </Card>
  );
}

type Audit = { id: number; ts: string; actor: string; action: string; target: string; details: Record<string, unknown> };

function PrivacyCard() {
  const admin = isAdmin(useActor());
  const [rows, setRows] = useState<Audit[] | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [version, setVersion] = useState(0);
  useEffect(() => {
    if (!admin) return;
    let stale = false;
    api<Audit[]>("/audit?limit=50")
      .then((r) => !stale && setRows(r))
      .catch((e) => !stale && setMsg((e as Error).message));
    return () => {
      stale = true;
    };
  }, [admin, version]);
  const retention = async () => {
    try {
      const r = await api<{ tracks: number; embeddings: number; files: number }>("/privacy/retention", { method: "POST" });
      setMsg(`Retention: ${r.tracks} tracks, ${r.embeddings} embeddings, ${r.files} files removed.`);
      setVersion((v) => v + 1);
    } catch (e) {
      setMsg((e as Error).message);
    }
  };
  return (
    <Card className="xl:col-span-3">
      <CardHeader className="flex-row items-center justify-between">
        <div>
          <CardTitle className="flex items-center gap-2">
            <ScrollText className="size-4" /> Privacy & audit log
          </CardTitle>
          <CardDescription>
            Unknown faces are blurred everywhere. Unknown-person data is deleted after the retention window. Every unblur, enrollment and
            acknowledgement is logged.
          </CardDescription>
        </div>
        {admin && (
          <Button size="sm" variant="outline" onClick={retention}>
            Run retention now
          </Button>
        )}
      </CardHeader>
      <CardContent className="grid gap-1 text-xs">
        {!admin && <p className="text-muted-foreground">Switch to the admin actor (bottom of the menu) to view the audit log.</p>}
        {msg && <p className="text-muted-foreground">{msg}</p>}
        {admin &&
          rows?.map((a) => (
            <div key={a.id} className="grid grid-cols-[10rem_6rem_10rem_1fr] gap-2 border-b py-1 font-mono last:border-0">
              <span className="text-muted-foreground">{fmtTime(a.ts)}</span>
              <span>{a.actor}</span>
              <span>{a.action}</span>
              <span className="truncate text-muted-foreground">
                {a.target} {Object.keys(a.details ?? {}).length ? JSON.stringify(a.details) : ""}
              </span>
            </div>
          ))}
      </CardContent>
    </Card>
  );
}

export default function RegistryPage() {
  const [people, setPeople] = useState<Person[]>([]);
  const [vehicles, setVehicles] = useState<Vehicle[]>([]);
  const [plate, setPlate] = useState("");
  const [vtype, setVtype] = useState("car");
  const [owner, setOwner] = useState("");
  const [error, setError] = useState<string | null>(null);

  const [version, setVersion] = useState(0);
  const load = useCallback(() => setVersion((v) => v + 1), []);
  useEffect(() => {
    let stale = false;
    Promise.all([api<Person[]>("/people"), api<Vehicle[]>("/vehicles")])
      .then(([p, v]) => {
        if (stale) return;
        setPeople(p);
        setVehicles(v);
        setError(null);
      })
      .catch((e) => !stale && setError((e as Error).message));
    return () => {
      stale = true;
    };
  }, [version]);

  const act = async (fn: () => Promise<unknown>) => {
    try {
      await fn();
      load();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const names = Object.fromEntries(people.map((p) => [p.id, p.display_name]));

  return (
    <div className="grid gap-4 xl:grid-cols-[1fr_1fr_20rem]">
      <Card>
        <CardHeader>
          <CardTitle>People ({people.length})</CardTitle>
          <CardDescription>Enrolled with consent. Names are never shown on the live view, only roles.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-2">
          {error && <p className="text-sm text-unknown">{error}</p>}
          {people.length === 0 && <p className="text-sm text-muted-foreground">Nobody enrolled yet.</p>}
          {people.map((p) => (
            <div key={p.id} className="flex items-center gap-3 rounded-md border bg-background/40 p-2">
              <Avatar id={p.id} />
              <div className="min-w-0 flex-1 text-sm">
                <div className="flex items-center gap-2">
                  <span className="truncate font-medium">{p.display_name}</span>
                  <RoleBadge role={p.role} />
                </div>
                <div className="text-xs text-muted-foreground">
                  #{p.id} {p.unit && `· ${p.unit}`} · {p.face_samples} face / {p.body_samples} body
                  {!p.consent && <Badge variant="unknown" className="ml-2">no consent</Badge>}
                </div>
              </div>
              <Button
                size="icon"
                variant="ghost"
                title="Delete person and all their embeddings"
                onClick={() => confirm(`Delete ${p.display_name} and all their embeddings?`) && act(() => api(`/people/${p.id}`, { method: "DELETE" }))}
              >
                <Trash2 />
              </Button>
            </div>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Vehicles ({vehicles.length})</CardTitle>
          <CardDescription>Exact plate → registered; one character off → likely registered (verify).</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-2">
          <form
            className="grid grid-cols-[1fr_6rem] gap-2 sm:grid-cols-[1fr_6rem_8rem_auto]"
            onSubmit={(e) => {
              e.preventDefault();
              act(() =>
                api("/vehicles", {
                  method: "POST",
                  body: JSON.stringify({ plate, vehicle_type: vtype, owner_person_id: owner ? Number(owner) : null }),
                }),
              ).then(() => setPlate(""));
            }}
          >
            <Input value={plate} onChange={(e) => setPlate(e.target.value)} placeholder="TN09AB1234" className="font-mono uppercase" />
            <Select value={vtype} onChange={(e) => setVtype(e.target.value)}>
              <option>car</option>
              <option>motorcycle</option>
              <option>truck</option>
              <option>bus</option>
            </Select>
            <Select value={owner} onChange={(e) => setOwner(e.target.value)}>
              <option value="">No owner</option>
              {people.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.display_name}
                </option>
              ))}
            </Select>
            <Button type="submit" disabled={plate.length < 4}>
              Add
            </Button>
          </form>
          {vehicles.map((v) => (
            <div key={v.id} className="flex items-center gap-3 rounded-md border bg-background/40 px-3 py-2 text-sm">
              <Car className="size-4 text-muted-foreground" />
              <span className="font-mono font-semibold">{v.plate}</span>
              <span className="text-xs text-muted-foreground">{v.vehicle_type}</span>
              {v.owner_person_id != null && <span className="text-xs">{names[v.owner_person_id] ?? `#${v.owner_person_id}`}</span>}
              <Button size="icon" variant="ghost" className="ml-auto" onClick={() => act(() => api(`/vehicles/${v.id}`, { method: "DELETE" }))}>
                <Trash2 />
              </Button>
            </div>
          ))}
        </CardContent>
      </Card>

      <UploadEnroll onDone={load} />
      <PrivacyCard />
    </div>
  );
}
