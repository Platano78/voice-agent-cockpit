# Shadow Monarch — Immersive Talking-Avatar Theme (port spec)

**Status:** SPEC — ready to implement · **Owner ruling:** B2 (general viewport-takeover avatar capability), FULL IMMERSIVE.
**Repo:** `~/project/voice-agent-cockpit` · implement under `webclient/` only.
**Product vision:** this is the prototype for **premium, user-pickable themes** — build it as a reusable capability, not a one-off.
**Do NOT commit — the orchestrator commits after independent verification. Do NOT touch other themes/avatars' behavior.**

This ports the verified look-dev demo `webclient/theme-demos/4-threed/index.html` (a giant Shadow Monarch
face in a Three.js void with floating System panels, a shadow-army particle field, cinematic camera, and a
`uMouthOpen` lip-sync shader) into the real client as a new immersive avatar that lip-syncs to real AI speech.

## Verified context (from recon — trust these file:line anchors)
- **Theme contract is CSS-only** (`webclient/themes/THEME-SPEC.md:1-13`): a theme may ONLY inject a `<link>`; it may not touch DOM/JS. Therefore immersion is NOT a theme — it is an avatar-module capability (this spec). Do not implement immersion via CSS-theme hacks.
- **Avatar module contract** (identical in `avatar/avatar2d.mjs` & `avatar/avatar.mjs`): `export async function initAvatarX(mountNode, fallbackNode, hooks) → { start(), stop(), destroy() }`. `hooks.getPlayCtx()` → shared `AudioContext`; `hooks.setTap(gainNode)` → registers a gain node the playback path fans into. On load failure: set `fallbackNode` text + `.hidden=false`, `mountNode.hidden=true`, return no-op stub.
- **Dispatch:** `loadAvatarModule()` (`index.html:2460-2497`) branches on `entry.kind`; `switchAvatar` (`index.html:2504-2520`) does `destroy()` + `replaceChildren()` on a kind change. `AVATAR_REGISTRY` at `index.html:2427-2441`.
- **HeadAudio tap pattern to reuse verbatim:** `avatar2d.mjs:150-171` (`getPlayCtx` → `audioWorklet.addModule(HEADWORKLET_URL)` → `new HeadAudio` → `loadModel` → `onvalue` writes `visemeVals` → build own `GainNode` → `hooks.setTap`). Viseme→scalar reduction: `OPEN_SCALE` map + per-frame `max(visemeVals[k]*scale)` (`avatar2d.mjs:26-30,113-118`). This scalar is exactly the demo shader's `uMouthOpen`.
- **AudioWorklet degrade-gracefully** (`avatar2d.mjs:178-193`): HeadAudio failure is non-fatal — `catch { ha=null; console.warn }`, face still renders motionless. REQUIRED in the new module.
- **Three.js is vendored as ESM:** `avatar/vendor/three/three.module.js`. The demo uses CDN r128 UMD (global `THREE`). **Port the scene to ESM `import * as THREE from "../vendor/three/three.module.js"` and adapt any r128 API that changed in the vendored version.** Do NOT add a CDN dependency to the product.
- **Layout:** `#contentGrid` (`index.html:219-276`) = mobile flex → desktop 3-col grid `280px minmax(0,640px) 280px`. Avatar renders in `#avatarPane`/`#avatarMount` (`index.html:304-333`), a desktop-only fixed **640×300** box. Rails: `#historyRail` (:1210), `#sessionRail` (:1284), PTT controls in `<main>`.
- **Assets:** face = `themes/assets/sololeveling/monarch-character.png` (720×1170, fully opaque, dark-blue baked bg — key it in-shader by LUMINANCE, not chroma). Demo shader constants already mirror `avatar2d.mjs` (`SPLIT_FRAC 0.565`, 6% slide) — reuse them.

## Slices (implement in order; each is independently verifiable)

### Slice 1 — Kill the purple (trivial, independent, do first)
`themes/sololeveling.css:30` `--border: rgba(139,92,246,0.32)` and `:34` `--accent-2: #8b5cf6` are violet. Swap BOTH to a cyan family consistent with the theme's `swatch` (`#22d9ff`) — e.g. `--border: rgba(34,217,255,0.32)`, `--accent-2: #22d9ff` (or a slightly deeper cyan `#0fb5d6` for the secondary if two cyans read better). These fan out to ~38 usages via custom properties, so this 2-line change recolors the whole theme. Change nothing else in that file.
- **Gate:** load the client, switch to Shadow Monarch theme, confirm zero purple/violet remains on panel borders, brackets, PTT glow, quest/gate fills. Other themes unchanged (they define their own `--accent-2`).

### Slice 2 — New immersive avatar module `avatar/avatarMonarch3d.mjs`
Port the tier-4 demo scene into a module implementing the contract. It MUST:
- `import * as THREE from "../vendor/three/three.module.js"` (ESM, vendored — adapt r128 APIs as needed).
- Render into `mountNode` a `<canvas>` with the Three.js scene from `theme-demos/4-threed/index.html`: the Monarch face plane (load `../themes/assets/sololeveling/monarch-character.png` via `THREE.TextureLoader`; luminance-key the dark-blue bg in the fragment shader; cyan grade; hot red eyes), the floating System panels, the shadow-army particle field, cinematic camera drift, the `uMouthOpen` lip-sync shader (reuse the demo's split=0.565 / 6%-slide / feather geometry).
- **Lip-sync from REAL speech:** copy `avatar2d.mjs`'s `setupHeadAudio()` verbatim (getPlayCtx → addModule → new HeadAudio → loadModel → onvalue). Reduce visemes to a scalar with the SAME `OPEN_SCALE` map + per-frame max, and feed that scalar into the shader `uMouthOpen` uniform each frame (replace the demo's synthetic envelope). Keep the synthetic envelope ONLY as the fallback when HeadAudio is unavailable, and add a real `.catch` degrade path (motionless face, `console.warn`, never fatal).
- Contract: `initAvatarMonarch3d(mountNode, fallbackNode, hooks) → { start(), stop(), destroy() }`. `start/stop` gate the rAF loop (+ `visibilitychange`, same fn ref). `destroy()` = full teardown: stop rAF, remove listener, dispose Three resources (geometry/material/texture/renderer), disconnect ONLY the audio nodes this module created (never the shared `playCtx`), remove canvas, delete any `window.__*` it installed. On WebGL/texture/HeadAudio init failure → fallback stub per contract.
- **Declare the immersive capability** so slice 4 can promote it: return an object that also carries `immersive: true` (i.e. `{ start, stop, destroy, immersive: true }`), OR install `window.__avatarImmersive = true` on init and delete it in `destroy()`. Pick the return-flag approach (cleaner, no globals) unless the dispatch site can't read it.
- **Cap pixelRatio (≤2), handle resize, honor `prefers-reduced-motion`** (freeze camera/particles/mouth to a calm frame).

### Slice 3 — Registry + dispatch wiring (`index.html`)
- Add an `AVATAR_REGISTRY` entry: `{ id: "monarch3d", name: "Shadow Monarch (Cinematic)", kind: "3d-immersive" }`.
- In `loadAvatarModule()`, add a branch for `kind === "3d-immersive"` → `import("./avatar/avatarMonarch3d.mjs")` → `mod.initAvatarMonarch3d(els.avatarMount, els.avatarFallback, { getPlayCtx, setTap })`. Reuse the exact `getPlayCtx`/`setTap` hooks the other branches pass.
- Ensure `switchAvatar`'s teardown path calls the new module's `destroy()` on switch-away (kind change already triggers destroy — verify the immersive-mode DOM promotion from slice 4 is also reverted here).

### Slice 4 — B2 viewport-takeover capability (additive, opt-in, reusable)
Generalize "an avatar can take over the viewport" — NOT a Shadow-Monarch special case. When `loadAvatarModule()` mounts a module whose init result reports `immersive: true`:
- Add a body/root class (e.g. `document.documentElement.classList.add("avatar-immersive")`) that, via CSS, promotes `#avatarMount`'s canvas to a **full-viewport fixed background layer** (`position:fixed; inset:0; z-index:0`) and floats `#historyRail`, `#sessionRail`, `<main>`'s cards, and the PTT controls ABOVE it (raise their z-index, make their backgrounds translucent so the Monarch shows through). Keep everything readable (backdrop-blur / scrim on the text panels).
- On `destroy()`/switch to a non-immersive avatar, REMOVE the class and fully restore the normal 3-col grid + 640×300 pane. No residual state.
- **Strictly additive & scoped:** gate ALL immersive CSS behind `.avatar-immersive` so the default layout (every other avatar/theme) is byte-for-byte unchanged when the class is absent. Do not edit the shared `#contentGrid`/`#avatarPane` base rules except to add `.avatar-immersive`-scoped overrides.
- Mobile: out of scope for this slice (desktop has the avatar pane; mobile has none). Note it as a follow-up; do not break mobile — the class simply has no mobile effect.

## HARD GATES (run all; paste real output; any failure = fix before reporting)
1. **Other avatars/themes intact:** switch through 2-3 other avatars (a 3D GLB, the 2D) and 2 other themes (editorial, chainsawman) — normal 3-col layout, 640×300 pane, no immersive residue, no console errors. This is the top regression risk (slice 4 must be scoped).
2. **Immersive mount:** select "Shadow Monarch (Cinematic)" → Monarch fills the viewport, rails float readable on top, no horizontal scroll, no console errors.
3. **Lip-sync:** with real playback audio, the face's mouth tracks speech (or, if AudioWorklet unavailable on this origin, degrades to a motionless face with a single `console.warn`, NOT a crash). Verify the tap is parallel (audio still audible, not rerouted).
4. **Teardown:** switch away → full layout restored, Three resources disposed (no WebGL context leak across a few switches), no orphaned rAF/audio nodes.
5. **Purple gone** (slice 1 gate).
6. **Reduced-motion:** `prefers-reduced-motion` → calm frozen frame, not blank.
7. If a lint/build step exists (`package.json` scripts), run it clean.

## Report format
Per slice: what changed (file:line), gate command outputs verbatim, honest notes on anything degraded or uncertain (especially r128→vendored-Three API adaptations, and the immersive CSS scoping). The orchestrator will independently screenshot-verify over the served client, review the diff for the slice-4 scoping and the audio-tap wiring, and only then commit.

---

## Slice 5 — Mobile parity for the immersive avatar (added 2026-07-08)

**Problem:** the avatar is desktop-only — `#avatarPane { display: none }` is the BASE (mobile) rule (`index.html:227`), shown only inside `@media (min-width:1024px)`. So NO avatar renders on phones. The immersive Monarch (portrait 720×1170 face) is a great fit for a phone screen; make it work below 1024px.

**Scope (all additive & scoped — desktop behavior must stay byte-identical):**
1. **Show the immersive avatar on mobile.** Add mobile (base / `<1024px`) CSS, gated behind `.avatar-immersive`, that promotes `#avatarPane`/`#avatarMount` to a full-viewport fixed background canvas on phones (mirror the desktop immersive treatment but for a narrow portrait viewport). Do NOT show non-immersive avatars on mobile — keep their current hidden behavior; only the `.avatar-immersive` case reveals the pane on mobile.
2. **Float the existing mobile content over it.** On mobile the single-column content (`#transcript`, `#assistantText`, `#ptt`, and a compact session summary) should float above the Monarch with translucent `rgba(14,15,32,0.78)` + `backdrop-filter: blur` backgrounds so it stays readable. History/session rails can stay hidden on mobile as they are today.
3. **Portrait framing in `avatarMonarch3d.mjs`.** The scene was composed for landscape 16:9. On a portrait/narrow aspect, adjust so the FACE fills nicely (e.g. widen camera FOV or pull the face plane so it fills the portrait frame; the 720×1170 texture suits portrait). Detect aspect at resize; do NOT hardcode desktop dims.
4. **De-clutter on mobile.** The floating 3D System side-panels crowd a phone — hide or minimize them below ~1024px (keep the face + the DOM content overlay). Keep the lip-sync mouth + red eyes + particle ambiance.

**Gates:** at 390×844 and 414×896 (portrait phone) — Monarch fills the screen, content overlay readable, no horizontal scroll, no console errors; switching to a non-immersive avatar restores the normal mobile single-column layout with the avatar hidden again; **desktop (≥1024px) is visually unchanged** (regression check — screenshot at 1280 before/after). Reduced-motion still freezes.
