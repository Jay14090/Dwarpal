// Thin typed client for the Dwarpal FastAPI backend (browser-side; CORS allows the dev origin).

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export function apiUrl(path: string | null | undefined): string | undefined {
  return path ? `${API_URL}${path}` : undefined;
}

export function wsUrl(path: string): string {
  return API_URL.replace(/^http/, "ws") + path;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

// Who is acting (sent as X-Actor and written to the audit log). Demo-grade identity, not authentication:
// actors listed in privacy.admin_actors (default "admin") may unblur faces.
const ACTOR_KEY = "dwarpal.actor";

export function getActor(): string {
  try {
    return (typeof window !== "undefined" && window.localStorage.getItem(ACTOR_KEY)) || "operator";
  } catch {
    return "operator";
  }
}

export function setActor(actor: string): void {
  try {
    window.localStorage.setItem(ACTOR_KEY, actor || "operator");
    window.dispatchEvent(new Event("dwarpal-actor"));
  } catch {}
}

/** Fetch an image that needs the X-Actor header (admin unblur) and return an object URL. */
export async function protectedImage(path: string): Promise<string> {
  const res = await fetch(`${API_URL}${path}`, { headers: { "X-Actor": getActor() }, cache: "no-store" });
  if (!res.ok) throw new ApiError(res.status, res.status === 403 ? "admin only" : res.statusText);
  return URL.createObjectURL(await res.blob());
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (init?.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");
  headers.set("X-Actor", getActor());
  const res = await fetch(`${API_URL}${path}`, { ...init, headers, cache: "no-store" });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const body = await res.json();
      msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {}
    throw new ApiError(res.status, msg);
  }
  return res.json() as Promise<T>;
}

export type Zone = { name: string; restricted: boolean; polygon: [number, number][] };

export type CameraInfo = {
  id: string;
  name: string;
  enabled: boolean;
  run_mode: "realtime" | "cached";
  source_type: string;
  online: boolean;
  zones: Zone[];
  stats: { fps?: number; processed_fps?: number; infer_ms?: number; [k: string]: unknown } | null;
  stream_url: string;
};

export type EventItem = {
  id: number | null;
  rule: string;
  rule_type?: string | null;
  severity: string;
  camera_id: string | null;
  ts: string | number;
  global_id: number | null;
  plate_read_id: number | null;
  payload: Record<string, unknown>;
  acknowledged?: boolean;
  thumb_url?: string | null;
};

export type PlateItem = {
  id: number;
  plate: string;
  status: string;
  camera_id: string;
  ts: string | number;
  confidence: number;
  thumb_url?: string | null;
  [k: string]: unknown;
};

export type Person = {
  id: number;
  role: string;
  display_name: string;
  unit: string | null;
  consent: boolean;
  thumb_url?: string | null;
  [k: string]: unknown;
};

export type Vehicle = { id: number; plate: string; vehicle_type: string | null; owner_person_id: number | null };

export type SearchFilter = {
  entity: "person" | "vehicle";
  roles: string[];
  upper_color: string | null;
  lower_color: string | null;
  height_cm: { min: number | null; max: number | null } | null;
  cameras: string[];
  zones: string[];
  time_range: { from: string | null; to: string | null } | null;
  plate: string | null;
  free_text: string | null;
};

export type TrackResult = {
  kind: "track";
  track_id: number;
  global_id: number | null;
  camera_id: string;
  start_ts: string;
  end_ts: string;
  role: string;
  upper_color: string | null;
  lower_color: string | null;
  height_cm: number | null;
  height_err_cm: number | null;
  zones: string[];
  score: number | null;
  thumb_url: string | null;
  clip_url: string | null;
};

export type PlateResult = {
  kind: "plate_read";
  plate_read_id: number;
  plate: string;
  status: string;
  camera_id: string;
  ts: string;
  confidence: number;
  thumb_url: string | null;
};

export type SearchResponse = {
  query: string;
  filter: SearchFilter;
  parsed_by: string;
  results: (TrackResult | PlateResult)[];
  relaxed: string[];
  ranked_by: string;
  plate_match?: string;
};

export type Metric = {
  key: string;
  label: string;
  value: number | null;
  source: string | null;
  how: string;
  detail: string;
};
