# Demo script (about 8 minutes)

## Before the audience arrives
1. In **WSL2**: `make demo` (first time: `make setup && make data-meva && make index && make index-db`).
   Wait for `ready: open http://localhost:3000/live`.
2. In **Windows PowerShell**, from the repo folder: `.\scripts\webcam_publish.ps1` (`-List` picks a camera).
   The *Lobby webcam* tile turns LIVE.
3. Browser: `http://localhost:3000/live`, full screen. Leave the actor (bottom-left) on **operator**.
4. Optional: register the plate of a car that will appear on the gate camera (`/registry` → Vehicles).
   Pre-enroll one "staff" volunteer so a blue box shows up.
5. Check `make events-replay` once; `/metrics` then shows the once-per-incident result.

The four replayed cameras are MEVA *Main gate drive* (G336), *Resident parking* (G328), *Clubhouse cafe* (G421)
and *Clubhouse gym* (G299, a restricted zone, so unknown visitors there raise alerts). People on the two outdoor
cameras are only 30–47 px tall, too small to identify, so they stay *pending*. The indoor cameras show the
identity and alert flow.

## 1. Live view (1 min)
- "Five cameras: four replayed CCTV cameras from a real campus (MEVA) and this webcam, live."
- Box colours: **green** resident, **blue** staff, **red** unknown, **grey** pending (still deciding).
  No names on the live view, only roles.
- Point at a person walking between cameras: the `#id` stays the same (cross-camera global ID).
- Heads of unknown and pending people are **blurred** on the stream.

## 2. The live moment: unknown → resident (2 min)
1. A volunteer steps in front of the webcam. Their box goes grey (*pending*), then after about 3 s of good views **red**
   (*unknown*). The webcam zone `lobby` is restricted, so an **unknown_in_restricted_zone** alert appears in the
   event feed with a blurred thumbnail.
2. In *Enroll from camera*: name, unit (e.g. `B-402`), role *Resident*, tick **consent** → *Capture & enroll*.
   The volunteer looks at the camera and turns their head slightly. That's about 6 s and 3–5 shots.
3. The volunteer steps out and back in. The box turns **green** (*resident*) and their face is no longer blurred.
   No alert this time.
- Talking point: without consent the enroll button stays disabled and the API refuses (422).

## 3. Search in plain English (2 min)
On `/search`, click or type:
- `all unknown people today`: returns the volunteer's earlier unknown track (now shown as resident, because roles
  are current) and any others.
- `person in a blue jacket and black pants in the gym`: the filter chips show what was understood (colours,
  zone, appearance text). Results are ranked by CLIP similarity.
- `man in a white shirt sitting at a table`: note the "relaxed" notice. The colour estimate said gray, but CLIP
  still finds him first.
- `when did TN09AB1234 enter?`: a plate search (exact, then one-character-off matches).
- Click **Clip** on a result: a short clip with unknown faces blurred.

## 4. Alerts and accountability (1.5 min)
- `/events`: alerts with thumbnails, severity, zone, dwell time. Each one fired **once per incident**, not every
  frame; a person staying 10 minutes produces one alert, plus a loitering alert after 60 s.
- Click **Ack**: it's written to the audit log with the actor.
- Switch the actor to **admin** (bottom-left). An eye button appears on search results and live tiles. Unblur
  one thumbnail, then show `/registry` → *Privacy & audit log*: the unblur is recorded with who and when.
  A live-stream unblur asks for a reason and expires after 60 s.
- *Run retention now*: unknown-person embeddings, thumbnails and clips older than 7 days are deleted (audited).

## 5. Numbers (1 min)
`/metrics` shows only measured numbers, each pointing at its report in `data/eval/`. Anything not measured yet
says so and shows the command to run (`make eval`, `make plates-eval`). Nothing is typed in by hand.

## If something goes wrong
- A tile is OFFLINE: the engine is still starting (the first start loads models), or the webcam publisher isn't
  running.
- No boxes on the webcam: check the GPU (`/health` → `device`). On CPU the webcam runs at a few fps.
- Search returns nothing: `make index-db` hasn't been run since the last `make index`.
