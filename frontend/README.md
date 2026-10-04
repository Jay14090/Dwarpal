# Dwarpal dashboard

Next.js (App Router) + TypeScript + Tailwind, with shadcn/ui-style components (`src/components/ui`, `components.json`).

| Page | What it does |
|---|---|
| `/live` | MJPEG grid (role-coloured boxes drawn by the backend), per-camera role counts, the live event feed (WebSocket), and enrollment from the webcam |
| `/search` | Plain-English search → parsed filter chips, thumbnails, and clips |
| `/events` | Alert history with thumbnails, filter, acknowledge (audited), live plate reads |
| `/registry` | Enrolled people (consent), enroll from photos, the vehicle registry |
| `/metrics` | Numbers from `data/eval/*.json` reports only; missing ones show the command that measures them |

```bash
make web-install     # once
make dev             # API on :8000 (other terminal)
make web             # dashboard on http://localhost:3000
```

The API URL is `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`; see `.env.example`).
