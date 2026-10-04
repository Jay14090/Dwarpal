"use client";

import { useEffect, useRef, useState } from "react";
import { wsUrl, type EventItem, type PlateItem } from "@/lib/api";

export type FeedMessage = { kind: "event"; data: EventItem } | { kind: "plate"; data: PlateItem };

/** Live events/plates from /ws/events, reconnecting with backoff. */
export function useLiveFeed(onMessage?: (m: FeedMessage) => void) {
  const [connected, setConnected] = useState(false);
  const cb = useRef(onMessage);
  useEffect(() => {
    cb.current = onMessage;
  }, [onMessage]);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let retry = 1000;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let closed = false;
    const connect = () => {
      ws = new WebSocket(wsUrl("/ws/events"));
      ws.onopen = () => {
        setConnected(true);
        retry = 1000;
      };
      ws.onmessage = (e) => {
        try {
          cb.current?.(JSON.parse(e.data) as FeedMessage);
        } catch {}
      };
      ws.onclose = () => {
        setConnected(false);
        if (!closed) timer = setTimeout(connect, (retry = Math.min(retry * 2, 15000)));
      };
    };
    connect();
    return () => {
      closed = true;
      clearTimeout(timer);
      ws?.close();
    };
  }, []);
  return connected;
}
