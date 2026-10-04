import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** ISO string or epoch seconds (live WebSocket payloads) -> local date/time. */
export function fmtTime(t: string | number | null | undefined): string {
  if (t == null || t === "") return "—";
  const d = typeof t === "number" ? new Date(t * 1000) : new Date(t);
  return d.toLocaleString(undefined, { dateStyle: "short", timeStyle: "medium" });
}

export function pct(v: number | null | undefined): string {
  return v == null ? "—" : `${(100 * v).toFixed(1)}%`;
}
