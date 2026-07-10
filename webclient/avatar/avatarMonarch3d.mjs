// Shadow Monarch immersive 3D avatar — port of theme-demos/4-threed into the
// real client. Full viewport takeover: Monarch face, System panels, shadow-army
// particles, cinematic camera. Lip-syncs to real AI speech via HeadAudio.
//
// Contract: initAvatarMonarch3d(mountNode, fallbackNode, hooks)
//   → { start(), stop(), destroy(), immersive: true }
//
// Lifecycle: start/stop gate the rAF loop. destroy() = full teardown (dispose
// Three resources, disconnect audio, remove canvas). HeadAudio failure is
// non-fatal — the face renders motionless at rest.

import * as THREE from "./vendor/three/three.module.js";
import { HeadAudio } from "./vendor/headaudio/headaudio.mjs";

const MONARCH_TEXTURE_URL = "./themes/assets/sololeveling/monarch-character.png";
const HEADWORKLET_URL = "./avatar/vendor/headaudio/headworklet.mjs";
const HEADAUDIO_MODEL_URL = "./avatar/vendor/headaudio/model-en-mixed.bin";

// Mirror avatar2d.mjs constants so the mouth geometry stays identical
const SPLIT_FRAC = 0.565;
const REST_OPEN = 0.15;

// Per-viseme mouth-open contribution (identical to avatar2d.mjs)
const OPEN_SCALE = {
  viseme_aa: 1, viseme_O: 1, viseme_U: 0.9, viseme_E: 0.7, viseme_I: 0.65,
  viseme_nn: 0.5, viseme_RR: 0.5, viseme_CH: 0.5, viseme_PP: 0.2, viseme_FF: 0.25,
  viseme_SS: 0.3, viseme_TH: 0.3, viseme_DD: 0.35, viseme_kk: 0.35, viseme_sil: 0,
};

// Monarch fragment shader (port from demo, verbatim shader logic)
const monarchVertexShader = [
  "varying vec2 vUv;",
  "void main() {",
  "  vUv = uv;",
  "  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);",
  "}"
].join("\n");

const monarchFragmentShader = [
  "uniform sampler2D map;",
  "uniform float uOpacity;",
  "uniform float uTime;",
  "uniform float uMouthOpen;",
  "uniform float uSplitFrac;",
  "varying vec2 vUv;",
  "void main() {",
  // Lip-sync mouth in UV space
  "  float vSplit = 1.0 - uSplitFrac;",
  "  float offV = uMouthOpen * 0.12;",
  "  vec2 mc = vec2((vUv.x - 0.5) / 0.30, (vUv.y - vSplit) / 0.10);",
  "  float mask = 1.0 - smoothstep(0.55, 1.0, length(mc));",
  "  vec2 uvS = vUv;",
  "  if (vUv.y < vSplit) uvS.y = vUv.y + offV * mask;",
  "  vec4 tex = mix(texture2D(map, vUv), texture2D(map, uvS), mask);",
  // Dark interior in the opened gap
  "  float gapLo = vSplit - offV;",
  "  float gap = step(gapLo, vUv.y) * step(vUv.y, vSplit) * mask;",
  "  float gpos = clamp((vSplit - vUv.y) / max(offV, 1e-4), 0.0, 1.0);",
  "  vec3 interiorCol = mix(vec3(0.102, 0.016, 0.024), vec3(0.020, 0.008, 0.012), gpos);",
  "  tex.rgb = mix(tex.rgb, interiorCol, gap);",
  // Luminance-based alpha (dark blue bg melts into void)
  "  float lum = dot(tex.rgb, vec3(0.299, 0.587, 0.114));",
  "  float a = smoothstep(0.05, 0.26, lum);",
  "  a = mix(a, 1.0, 0.22);",
  "  a = mix(a, 1.0, gap);",
  // Soft edge vignette
  "  float edge = smoothstep(0.0, 0.20, vUv.x) * smoothstep(1.0, 0.78, vUv.x);",
  "  edge *= smoothstep(0.0, 0.16, vUv.y) * smoothstep(1.0, 0.80, vUv.y);",
  "  a *= edge * uOpacity;",
  // Color grade: cyan lean, hot red eyes, teeth/eye bloom
  "  vec3 col = tex.rgb;",
  "  float redness = clamp(tex.r - max(tex.g, tex.b), 0.0, 1.0);",
  "  col = mix(col, col * vec3(0.62, 1.02, 1.28), 0.38);",
  "  col += vec3(1.7, 0.16, 0.10) * redness * 2.4;",
  "  col += vec3(0.14, 0.44, 0.60) * smoothstep(0.55, 0.96, lum);",
  // Slow spectral breathing
  "  a *= 0.90 + 0.10 * sin(uTime * 1.1);",
  "  gl_FragColor = vec4(col, clamp(a, 0.0, 1.0));",
  "}"
].join("\n");

// Synthetic mouth envelope for when HeadAudio is unavailable
function synthOpen(t) {
  const syl = Math.abs(
    Math.sin(t * 6.2832 * 3.6) * 0.6 +
    Math.sin(t * 6.2832 * 6.3 + 1.7) * 0.4
  );
  let word = 0.5 + 0.5 * Math.sin(t * 6.2832 * 0.85 + 0.4);
  word *= word;
  const jitter = 0.82 + 0.18 * Math.sin(t * 11.0 + 2.0);
  return Math.min(0.9, syl * word * jitter * 1.15);
}

// Canvas-drawn texture helpers (from demo)
function makeSpriteTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  const grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grd.addColorStop(0.0, "rgba(220,247,255,1)");
  grd.addColorStop(0.25, "rgba(143,227,255,0.85)");
  grd.addColorStop(0.6, "rgba(34,217,255,0.35)");
  grd.addColorStop(1.0, "rgba(34,217,255,0)");
  g.fillStyle = grd;
  g.fillRect(0, 0, 64, 64);
  const t = new THREE.CanvasTexture(c);
  t.needsUpdate = true;
  return t;
}

function makeGlowTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const g = c.getContext("2d");
  const grd = g.createRadialGradient(128, 128, 0, 128, 128, 128);
  grd.addColorStop(0.0, "rgba(60,210,255,0.55)");
  grd.addColorStop(0.4, "rgba(34,217,255,0.28)");
  grd.addColorStop(1.0, "rgba(34,217,255,0)");
  g.fillStyle = grd;
  g.fillRect(0, 0, 256, 256);
  return new THREE.CanvasTexture(c);
}

function makePanelTexture(header, lines) {
  const W = 512, H = 320;
  const c = document.createElement("canvas");
  c.width = W; c.height = H;
  const g = c.getContext("2d");
  // panel fill
  g.fillStyle = "rgba(9,16,26,0.82)";
  g.fillRect(0, 0, W, H);
  const grd = g.createLinearGradient(0, 0, 0, H);
  grd.addColorStop(0, "rgba(20,70,110,0.22)");
  grd.addColorStop(1, "rgba(6,8,16,0.0)");
  g.fillStyle = grd;
  g.fillRect(0, 0, W, H);
  // border
  g.strokeStyle = "rgba(34,217,255,0.9)";
  g.lineWidth = 3;
  g.strokeRect(6, 6, W - 12, H - 12);
  // corner brackets
  g.strokeStyle = "rgba(189,243,255,1)";
  g.lineWidth = 5;
  const b = 34, m = 14;
  function corner(x, y, dx, dy) {
    g.beginPath();
    g.moveTo(x + dx * b, y);
    g.lineTo(x, y);
    g.lineTo(x, y + dy * b);
    g.stroke();
  }
  corner(m, m, 1, 1);
  corner(W - m, m, -1, 1);
  corner(m, H - m, 1, -1);
  corner(W - m, H - m, -1, -1);
  // header bar
  g.fillStyle = "rgba(34,217,255,0.16)";
  g.fillRect(14, 14, W - 28, 54);
  g.fillStyle = "#bdf3ff";
  g.font = "700 30px Georgia, serif";
  g.textBaseline = "middle";
  g.shadowColor = "rgba(34,217,255,0.8)";
  g.shadowBlur = 14;
  g.fillText(header, 30, 42);
  g.shadowBlur = 0;
  // monospace body lines
  g.font = "20px 'Courier New', monospace";
  for (let i = 0; i < lines.length; i++) {
    g.fillStyle = i === 0 ? "#8fe3ff" : "rgba(143,227,255,0.72)";
    g.fillText(lines[i], 30, 110 + i * 40);
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export async function initAvatarMonarch3d(mountNode, fallbackNode, hooks) {
  let running = false;
  let rafHandle = null;
  let userExpanded = true;

  // Three.js scene state
  let renderer = null;
  let ro = null;
  let scene = null;
  let camera = null;
  let clock = null;
  let canvas = null;

  // Scene objects
  let portrait = null;
  let monarchMat = null;
  let monarchTex = null;
  let backGlow = null;
  let shaft = null;
  let panels = [];
  let particles = null;

  // Lip-sync
  let ha = null;
  let tapNode = null;
  let curOpen = REST_OPEN;
  const visemeVals = {};
  let textureLoaded = false;

  // Reduced motion
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ── Helpers ──────────────────────────────────────────────────────────

  function showFallback(message) {
    if (mountNode) mountNode.hidden = true;
    if (fallbackNode) {
      fallbackNode.textContent = message;
      fallbackNode.hidden = false;
    }
  }

  function applyRunState() {
    const shouldRun = userExpanded && document.visibilityState === "visible";
    if (shouldRun && !running) startRunning();
    else if (!shouldRun && running) stopRunning();
  }

  // ── Scene init ───────────────────────────────────────────────────────

  async function buildScene() {
    canvas = document.createElement("canvas");
    canvas.style.cssText = "display:block;width:100%;height:100%";
    if (mountNode) mountNode.appendChild(canvas);

    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(mountNode.clientWidth || 640, mountNode.clientHeight || 300, false);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setClearColor(0x05050c, 1);

    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x05050c, 0.055);

    camera = new THREE.PerspectiveCamera(48, mountNode.clientWidth / mountNode.clientHeight, 0.1, 260);
    camera.position.set(0, 1.4, 8.2);
    camera.lookAt(0, 0.7, 0);

    clock = new THREE.Clock();

    // Grid floor
    const grid = new THREE.GridHelper(120, 90, 0x22d9ff, 0x123448);
    grid.position.y = -2.2;
    grid.material.transparent = true;
    grid.material.opacity = 0.5;
    scene.add(grid);

    // Dark floor plane
    const floorMat = new THREE.MeshBasicMaterial({ color: 0x060810 });
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(300, 300), floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -2.22;
    scene.add(floor);

    // Ground glow pool
    const glowTex = makeGlowTexture();
    const pool = new THREE.Mesh(
      new THREE.PlaneGeometry(9, 9),
      new THREE.MeshBasicMaterial({
        map: glowTex, transparent: true, blending: THREE.AdditiveBlending,
        depthWrite: false, opacity: 0.85
      })
    );
    pool.rotation.x = -Math.PI / 2;
    pool.position.y = -2.18;
    scene.add(pool);

    // Shaft of cyan light
    shaft = new THREE.Mesh(
      new THREE.ConeGeometry(2.6, 9, 40, 1, true),
      new THREE.MeshBasicMaterial({
        map: glowTex, color: 0x22d9ff, transparent: true,
        blending: THREE.AdditiveBlending, depthWrite: false,
        side: THREE.DoubleSide, opacity: 0.16
      })
    );
    shaft.position.set(0, 2.3, -0.2);
    scene.add(shaft);

    // Backing glow
    backGlow = new THREE.Mesh(
      new THREE.PlaneGeometry(8, 10),
      new THREE.MeshBasicMaterial({
        map: glowTex, color: 0x2ec6ff, transparent: true,
        blending: THREE.AdditiveBlending, depthWrite: false, opacity: 0.55
      })
    );
    backGlow.position.set(0, 1.6, -2.4);
    scene.add(backGlow);

    // ── The Shadow Monarch portrait ──
    const mH = 8.6, mW = mH * (720 / 1170);
    monarchMat = new THREE.ShaderMaterial({
      uniforms: {
        map: { value: null },
        uOpacity: { value: 0.0 },
        uTime: { value: 0.0 },
        uMouthOpen: { value: REST_OPEN },
        uSplitFrac: { value: SPLIT_FRAC }
      },
      transparent: true,
      depthWrite: false,
      blending: THREE.NormalBlending,
      side: THREE.DoubleSide,
      vertexShader: monarchVertexShader,
      fragmentShader: monarchFragmentShader
    });
    portrait = new THREE.Mesh(new THREE.PlaneGeometry(mW, mH), monarchMat);
    portrait.position.set(0, 1.2, -1.6);
    scene.add(portrait);

    // Load the Monarch texture
    monarchTex = new THREE.TextureLoader().load(
      MONARCH_TEXTURE_URL,
      (t) => {
        t.colorSpace = THREE.LinearSRGBColorSpace;
        monarchMat.uniforms.map.value = t;
        monarchMat.uniforms.uOpacity.value = 1.0;
        monarchMat.needsUpdate = true;
        textureLoaded = true;
      },
      undefined,
      () => {
        // Texture load failure — scene continues with a dark placeholder
        console.warn("[avatarMonarch3d] Monarch texture failed to load");
      }
    );

    // ── Floating System panels ──
    const panelDefs = [
      {
        header: "\u27e1 QUEST \u27e3",
        lines: ["GATE: RED / S-RANK", "OBJECTIVE ... CLEARED", "REWARD: +1 SHADOW", "STATUS >> COMPLETE"],
        pos: [-3.5, 1.7, -0.4], rot: 0.28, scale: 1.0
      },
      {
        header: "\u27e1 SHADOW \u27e3",
        lines: ["EXTRACT: IGRIS", "GRADE ...... MARSHAL", "LOYALTY .... 100%", "ARISE >> READY"],
        pos: [3.6, 0.9, -0.7], rot: -0.3, scale: 1.05
      },
      {
        header: "\u27e1 STATUS \u27e3",
        lines: ["STR 214  AGI 190", "PER 176  VIT 168", "INT 158", "TITLE: MONARCH"],
        pos: [3.1, 2.5, -1.6], rot: -0.22, scale: 0.82
      }
    ];
    for (const d of panelDefs) {
      const tex = makePanelTexture(d.header, d.lines);
      const pw = 2.0 * d.scale, ph = 1.25 * d.scale;
      const mat = new THREE.MeshBasicMaterial({
        map: tex, transparent: true, depthWrite: false,
        side: THREE.DoubleSide, opacity: 0.92
      });
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(pw, ph), mat);
      mesh.position.set(d.pos[0], d.pos[1], d.pos[2]);
      mesh.userData = { baseY: d.pos[1], phase: Math.random() * 6.28, amp: 0.12 + Math.random() * 0.08 };
      scene.add(mesh);
      panels.push(mesh);
    }

    // ── Shadow army particles ──
    const COUNT = REDUCED ? 1400 : 3600;
    const pos = new Float32Array(COUNT * 3);
    const swirl = {
      theta: new Float32Array(COUNT),
      rad: new Float32Array(COUNT),
      y: new Float32Array(COUNT),
      spd: new Float32Array(COUNT),
      bob: new Float32Array(COUNT)
    };
    for (let i = 0; i < COUNT; i++) {
      const r = 2.4 + Math.pow(Math.random(), 0.6) * 15;
      const th = Math.random() * Math.PI * 2;
      const yy = -2.1 + Math.random() * 8.5;
      swirl.theta[i] = th;
      swirl.rad[i] = r;
      swirl.y[i] = yy;
      swirl.spd[i] = (0.04 + 0.14 / (r * 0.35)) * (Math.random() < 0.5 ? 1 : 0.8);
      swirl.bob[i] = Math.random() * 6.28;
      pos[i * 3] = Math.cos(th) * r;
      pos[i * 3 + 1] = yy;
      pos[i * 3 + 2] = Math.sin(th) * r;
    }
    const pGeo = new THREE.BufferGeometry();
    pGeo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    const pMat = new THREE.PointsMaterial({
      size: 0.16, map: makeSpriteTexture(), color: 0x6fdcff,
      transparent: true, blending: THREE.AdditiveBlending,
      depthWrite: false, sizeAttenuation: true, opacity: 0.9
    });
    particles = new THREE.Points(pGeo, pMat);
    particles.userData = { swirl, count: COUNT, arr: pos };
    scene.add(particles);

    // Resize handler
    window.addEventListener("resize", onResize);
    // The immersive pane starts display:none on mobile and only gets its real size
    // once the .avatar-immersive class lands AFTER init — a window-resize never fires
    // for that, so observe the mount directly to re-size when it becomes visible.
    if (typeof ResizeObserver !== "undefined") { ro = new ResizeObserver(() => onResize()); ro.observe(mountNode); }
    updatePanelVisibility();

    // Reduced motion: render one static frame
    if (REDUCED) {
      camera.position.set(1.8, 1.6, 8.0);
      camera.lookAt(0, 1.1, 0);
      renderer.render(scene, camera);
    } else {
      startRunning();
    }
  }

  function onResize() {
    if (!renderer) return;
    const w = mountNode.clientWidth;
    const h = mountNode.clientHeight;
    if (!w || !h) return;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(w, h, false);
    updatePanelVisibility();
  }

  // Mobile detection: below 1024px, hide side-panels and reframe for portrait
  function updatePanelVisibility() {
    const w = mountNode.clientWidth;
    const isMobile = w < 1024;
    for (const p of panels) {
      p.visible = !isMobile;
    }
    // Portrait framing: pull camera closer so the 720×1170 face fills a narrow screen
    if (camera.aspect < 1) {
      camera.position.set(0, 0.8, 7);
      camera.lookAt(0, 0.8, 0);
    } else {
      camera.position.set(0, 1.4, 8.2);
      camera.lookAt(0, 0.9, 0);
    }
  }

  // ── Animation loop ──────────────────────────────────────────────────

  function animate() {
    rafHandle = requestAnimationFrame(animate);
    const t = clock.getElapsedTime();
    const isMobile = mountNode.clientWidth < 1024;

    // Cinematic camera drift (portrait mode: closer framing, lower y)
    const ang = Math.sin(t * 0.08) * 0.42;
    const portraitRad = 3.0;
    const landscapeRad = 8.2;
    const r = isMobile ? portraitRad : landscapeRad;
    camera.position.x = Math.sin(ang) * r;
    camera.position.z = Math.cos(ang) * r + (isMobile ? 7 : 0);
    camera.position.y = isMobile
      ? 0.8 + Math.sin(t * 0.16) * 0.25
      : 1.35 + Math.sin(t * 0.16) * 0.35;
    camera.lookAt(0, isMobile ? 0.8 : 0.9, 0);

    // Portrait: update shader uniforms
    if (portrait && monarchMat && monarchMat.uniforms) {
      monarchMat.uniforms.uTime.value = t;
      // Mouth target: viseme-driven from HeadAudio when the agent is actually
      // speaking; otherwise fall back to the synthetic envelope so the Monarch
      // stays visibly alive (real visemes dominate the moment speech arrives).
      let maxOpen = 0;
      for (const [key, scale] of Object.entries(OPEN_SCALE)) {
        const v = (visemeVals[key] || 0) * scale;
        if (v > maxOpen) maxOpen = v;
      }
      // When HeadAudio is available, drive purely from real visemes so the mouth
      // moves ONLY while the agent speaks and rests silent otherwise. Only when
      // HeadAudio can't run at all (e.g. a non-secure http origin) fall back to the
      // synthetic envelope so the face isn't completely dead.
      const target = ha ? maxOpen : synthOpen(t);
      // Smooth to avoid snaps
      curOpen += (target - curOpen) * 0.35;
      monarchMat.uniforms.uMouthOpen.value = curOpen;
      // Subtle parallax yaw so gaze tracks camera
      portrait.rotation.y = Math.atan2(
        camera.position.x - portrait.position.x,
        camera.position.z - portrait.position.z
      ) * 0.12;
    }

    // Back glow follows camera
    if (backGlow) backGlow.quaternion.copy(camera.quaternion);

    // Panels: face camera + bob
    for (let i = 0; i < panels.length; i++) {
      const p = panels[i];
      p.quaternion.copy(camera.quaternion);
      p.position.y = p.userData.baseY + Math.sin(t * 0.9 + p.userData.phase) * p.userData.amp;
    }

    // Shaft shimmer
    if (shaft) shaft.material.opacity = 0.13 + Math.sin(t * 0.7) * 0.04;

    // Swirl particles
    if (particles) {
      const s = particles.userData.swirl, arr = particles.userData.arr, n = particles.userData.count;
      for (let k = 0; k < n; k++) {
        s.theta[k] += s.spd[k] * 0.016;
        const r = s.rad[k];
        arr[k * 3] = Math.cos(s.theta[k]) * r;
        arr[k * 3 + 2] = Math.sin(s.theta[k]) * r;
        arr[k * 3 + 1] = s.y[k] + Math.sin(t * 0.6 + s.bob[k]) * 0.35;
      }
      particles.geometry.attributes.position.needsUpdate = true;
      particles.rotation.y = t * 0.01;
    }

    renderer.render(scene, camera);
  }

  function startRunning() {
    running = true;
    ha?.start();
    if (REDUCED) {
      // Static frame, no loop
      renderer.render(scene, camera);
    } else {
      rafHandle = requestAnimationFrame(animate);
    }
  }

  function stopRunning() {
    running = false;
    if (rafHandle !== null) { cancelAnimationFrame(rafHandle); rafHandle = null; }
    ha?.stop();
  }

  // ── HeadAudio setup (verbatim pattern from avatar2d.mjs) ─────────────

  async function setupHeadAudio() {
    const audioCtx = hooks.getPlayCtx();
    try { await audioCtx.audioWorklet.addModule(HEADWORKLET_URL); }
    catch (e) { /* processor may already be registered */ }
    ha = new HeadAudio(audioCtx, {
      processorOptions: { visemeEventsEnabled: true },
      parameterData: { vadGateActiveDb: -40, vadGateInactiveDb: -60 },
    });
    await ha.loadModel(HEADAUDIO_MODEL_URL);
    ha.onvalue = (key, value) => {
      visemeVals[key] = value;
    };
    // Pure analysis sink — gain node the playback path connects into
    tapNode = audioCtx.createGain();
    tapNode.gain.value = 1;
    tapNode.connect(ha);
    hooks.setTap(tapNode);
  }

  // ── Teardown ─────────────────────────────────────────────────────────

  function destroy() {
    userExpanded = false;
    stopRunning();
    document.removeEventListener("visibilitychange", applyRunState);
    window.removeEventListener("resize", onResize);
    try { ro?.disconnect(); } catch {} ro = null;

    // Disconnect audio nodes this module created
    try { ha?.disconnect(); } catch {}
    try { tapNode?.disconnect(); } catch {}

    // Dispose Three.js resources
    if (monarchTex) monarchTex.dispose();
    if (monarchMat) monarchMat.dispose();
    if (portrait) portrait.geometry.dispose();
    for (const p of panels) {
      p.geometry.dispose();
      p.material.map?.dispose();
      p.material.dispose();
    }
    panels = [];
    if (particles) {
      particles.geometry.dispose();
      particles.material.map?.dispose();
      particles.material.dispose();
    }
    if (backGlow) { backGlow.geometry.dispose(); backGlow.material.map?.dispose(); backGlow.material.dispose(); }
    if (shaft) { shaft.geometry.dispose(); shaft.material.map?.dispose(); shaft.material.dispose(); }
    if (renderer) renderer.dispose();

    // Remove canvas from DOM
    if (canvas) canvas.remove();

    // Null out all references
    renderer = null; scene = null; camera = null; clock = null;
    canvas = null; portrait = null; monarchMat = null; monarchTex = null;
    backGlow = null; shaft = null; particles = null;
    ha = null; tapNode = null;
  }

  // ── Init ─────────────────────────────────────────────────────────────

  try {
    await buildScene();
  } catch (e) {
    showFallback("Avatar unavailable: WebGL or scene init failed");
    return { start() {}, stop() {}, destroy() {}, immersive: true };
  }

  // HeadAudio is optional — degrade to motionless face if unavailable
  try {
    await setupHeadAudio();
  } catch (e) {
    ha = null;
    console.warn("[avatarMonarch3d] lip-sync unavailable:", e.message);
  }

  document.addEventListener("visibilitychange", applyRunState);

  return {
    start() { userExpanded = true; applyRunState(); },
    stop() { userExpanded = false; applyRunState(); },
    destroy,
    immersive: true,
  };
}
