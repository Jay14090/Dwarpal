"use client";

import { useSyncExternalStore } from "react";
import { getActor } from "@/lib/api";

function subscribe(cb: () => void) {
  window.addEventListener("dwarpal-actor", cb);
  window.addEventListener("storage", cb);
  return () => {
    window.removeEventListener("dwarpal-actor", cb);
    window.removeEventListener("storage", cb);
  };
}

/** Current actor name (localStorage), re-rendering when it changes. */
export function useActor(): string {
  return useSyncExternalStore(subscribe, getActor, () => "operator");
}

export function isAdmin(actor: string): boolean {
  return actor === "admin";
}
