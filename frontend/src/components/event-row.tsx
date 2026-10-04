import { Car, User } from "lucide-react";
import { Badge, SeverityBadge, RoleBadge } from "@/components/ui/badge";
import { apiUrl, type EventItem } from "@/lib/api";
import { cn } from "@/lib/utils";

export function eventTime(ts: string | number): string {
  const d = typeof ts === "number" ? new Date(ts * 1000) : new Date(ts);
  return d.toLocaleTimeString(undefined, { hour12: false });
}

export function EventRow({ ev, compact, onAck }: { ev: EventItem; compact?: boolean; onAck?: (id: number) => void }) {
  const p = ev.payload ?? {};
  const isPlate = ev.plate_read_id != null || "plate" in p;
  const thumb = ev.thumb_url ?? (ev.id != null && (p.thumb || (ev as { has_thumb?: boolean }).has_thumb) ? `/events/${ev.id}/thumb.jpg` : null);
  return (
    <div className={cn("flex gap-3 rounded-md border bg-background/40 p-2", ev.acknowledged && "opacity-50")}>
      <div className="flex size-14 shrink-0 items-center justify-center overflow-hidden rounded bg-muted">
        {thumb ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={apiUrl(thumb)} alt="" className="size-full object-cover" />
        ) : isPlate ? (
          <Car className="size-5 text-muted-foreground" />
        ) : (
          <User className="size-5 text-muted-foreground" />
        )}
      </div>
      <div className="min-w-0 flex-1 text-xs">
        <div className="flex flex-wrap items-center gap-1.5">
          <SeverityBadge severity={ev.severity} />
          <span className="truncate font-mono text-[13px] font-semibold">{ev.rule}</span>
        </div>
        <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-muted-foreground">
          <span className="font-mono">{eventTime(ev.ts)}</span>
          <span>{ev.camera_id}</span>
          {typeof p.zone === "string" && <Badge variant="outline">{p.zone}</Badge>}
          {ev.global_id != null && <span className="font-mono">#{ev.global_id}</span>}
          {typeof p.role === "string" && <RoleBadge role={p.role} />}
          {typeof p.plate === "string" && <span className="font-mono text-foreground">{p.plate}</span>}
          {!compact && typeof p.dwell_s === "number" && p.dwell_s > 0 && <span>{p.dwell_s}s in zone</span>}
        </div>
      </div>
      {onAck && ev.id != null && !ev.acknowledged && (
        <button onClick={() => onAck(ev.id!)} className="self-start rounded border px-2 py-1 text-xs hover:bg-accent">
          Ack
        </button>
      )}
    </div>
  );
}
