import * as THREE from "three";
import { OrbitControls } from "/vendor/OrbitControls.js";

const pictures = { left: null, right: null, front: null };
const flips = {
  left: { h: false, v: false },
  right: { h: false, v: false },
  front: { h: false, v: false },
};
const crops = { left: null, right: null, front: null };
const imageAspects = { left: null, right: null, front: null };
const cropCustom = { left: false, right: false, front: false };
let orientation = "vertical";
let pictureAspect = null;
let lockedAspect = 200 / 112.5;
let framed = false;
let framedSize = "";
let ticket = 0;
let inflight = null;
let lastMeta = null;
let magnetOutline = null;

const status = document.getElementById("status");
const canvas = document.getElementById("view");
const viewport = document.getElementById("viewport");
const exportButton = document.getElementById("export-stl");

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NoToneMapping;
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0xe7e1d6, 1);

const scene = new THREE.Scene();
scene.add(new THREE.AmbientLight(0xffffff, 0.82));
const key = new THREE.DirectionalLight(0xffffff, 0.55);
key.position.set(-2, 3, 1);
scene.add(key);
const fill = new THREE.DirectionalLight(0xffffff, 0.28);
fill.position.set(2, 2, -1);
scene.add(fill);
const grid = new THREE.GridHelper(200, 20, 0xc4bfb4, 0xd5cfc3);
grid.position.y = -0.05;
scene.add(grid);

const plate = new THREE.Group();
scene.add(plate);

const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 5000);
camera.position.set(50, 60, 80);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.target.set(0, 1, 0);

window.__lentic = {
  camera: () => camera.position.toArray(),
  target: () => controls.target.toArray(),
  meshes: () => plate.children.length,
  status: () => status.textContent,
  magnetPlaces: () => (lastMeta && lastMeta.magnets ? lastMeta.magnets.places : null),
  magnetLines: () => {
    if (!magnetOutline) return null;
    const positions = magnetOutline.geometry.getAttribute("position");
    return { visible: magnetOutline.visible, count: positions ? positions.count : 0 };
  },
};

function resize() {
  const rect = viewport.getBoundingClientRect();
  const width = Math.max(rect.width, 1);
  const height = Math.max(rect.height, 1);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}
resize();
window.addEventListener("resize", resize);

function animate() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
animate();

function numberValue(id) {
  const value = Number(document.getElementById(id).value);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error("Every size and resolution needs a positive number of millimeters.");
  }
  return value;
}

function millimetersOrZero(id, message) {
  const value = Number(document.getElementById(id).value);
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(message);
  }
  return value;
}

function formData() {
  const data = new FormData();
  data.set("width", String(numberValue("width")));
  data.set("height", String(numberValue("height")));
  data.set("base", String(numberValue("base")));
  data.set("ridge_height", String(numberValue("ridge")));
  data.set("pitch", String(numberValue("pitch")));
  data.set("row", String(numberValue("row")));
  data.set("max_colors", String(numberValue("max-colors")));
  data.set("dither", document.getElementById("dither").checked ? "1" : "0");
  data.set("base_color", document.getElementById("base-color").value);
  data.set("orientation", orientation);
  data.set("nozzle", String(numberValue("nozzle")));
  data.set("crest_line", document.getElementById("crest-line").checked ? "1" : "0");
  if (document.getElementById("crest-line").checked) {
    data.set("crest_color", document.getElementById("crest-color").value);
    data.set("crest_width", String(numberValue("crest-width")));
    data.set("crest_height", String(numberValue("crest-height")));
  }
  data.set("magnets", document.getElementById("magnets").checked ? "1" : "0");
  if (document.getElementById("magnets").checked) {
    const shape = document.getElementById("magnet-shape").value;
    data.set("magnet_count", String(numberValue("magnet-count")));
    data.set("magnet_shape", shape);
    data.set("magnet_below", String(millimetersOrZero("magnet-below", "Thickness below the magnets needs zero or more millimeters.")));
    data.set("magnet_above", String(numberValue("magnet-above")));
    data.set("magnet_arrange", document.getElementById("magnet-arrange").value);
    const edge = Number(document.getElementById("magnet-edge").value);
    if (!Number.isFinite(edge) || edge < 0) {
      throw new Error("Distance from the edge needs zero or more millimeters.");
    }
    data.set("magnet_edge", String(edge));
    if (shape === "rect") {
      data.set("magnet_width", String(numberValue("magnet-width")));
      data.set("magnet_length", String(numberValue("magnet-length")));
      data.set("magnet_thickness", String(numberValue("magnet-rect-thickness")));
      data.set("magnet_turn", document.getElementById("magnet-turn").checked ? "1" : "0");
    } else {
      data.set("magnet_diameter", String(numberValue("magnet-diameter")));
      data.set("magnet_thickness", String(numberValue("magnet-thickness")));
    }
  }
  if (document.getElementById("use-spools").checked && spools.length) {
    data.set("palette", spools.join(","));
  } else if (document.body.dataset.palette) {
    data.set("palette", document.body.dataset.palette);
  }
  const first = orientation === "horizontal" ? "top" : "left";
  const second = orientation === "horizontal" ? "bottom" : "right";
  data.set(first, pictures.left, pictures.left.name);
  data.set(second, pictures.right, pictures.right.name);
  const firstCrop = cropValue("left");
  const secondCrop = cropValue("right");
  if (firstCrop) data.set(`crop_${first}`, firstCrop);
  if (secondCrop) data.set(`crop_${second}`, secondCrop);
  writeFlip(data, first, "left");
  writeFlip(data, second, "right");
  if (pictures.front) {
    data.set("front", pictures.front, pictures.front.name);
    const frontCrop = cropValue("front");
    if (frontCrop) data.set("crop_front", frontCrop);
    writeFlip(data, "front", "front");
  }
  return data;
}

function writeFlip(data, field, slot) {
  data.set(`flip_h_${field}`, flips[slot].h ? "1" : "0");
  data.set(`flip_v_${field}`, flips[slot].v ? "1" : "0");
}

function showFlip(slot) {
  const thumb = document.getElementById(`thumb-${slot}`);
  thumb.classList.toggle("flip-h", flips[slot].h);
  thumb.classList.toggle("flip-v", flips[slot].v);
  document.querySelectorAll(`[data-flip="${slot}"]`).forEach((button) => {
    const on = button.dataset.axis === "h" ? flips[slot].h : flips[slot].v;
    button.setAttribute("aria-pressed", on ? "true" : "false");
  });
}

function cropValue(slot) {
  const crop = crops[slot];
  if (!crop) return "";
  return [crop.x, crop.y, crop.w, crop.h].map((value) => value.toFixed(5)).join(",");
}

function schedule() {
  window.clearTimeout(schedule.timer);
  schedule.timer = window.setTimeout(preview, 250);
}

function setBusy(on) {
  document.getElementById("busy").hidden = !on;
}

async function preview() {
  exportButton.disabled = true;
  if (!pictures.left || !pictures.right) {
    setBusy(false);
    status.textContent = idleStatus();
    clearPlate();
    return;
  }
  let data;
  try {
    data = formData();
  } catch (error) {
    setBusy(false);
    status.textContent = error.message;
    return;
  }
  const id = ++ticket;
  if (inflight) inflight.abort();
  const controller = new AbortController();
  inflight = controller;
  status.textContent = "Building the plate...";
  setBusy(true);
  try {
    const response = await fetch("/api/preview", { method: "POST", body: data, signal: controller.signal });
    const type = response.headers.get("content-type") || "";
    if (!response.ok || type.includes("json")) {
      const payload = await response.json();
      throw new Error(payload.error || "The plate could not be built.");
    }
    const buffer = await response.arrayBuffer();
    if (id !== ticket) return;
    showPlate(buffer);
  } catch (error) {
    if (error.name === "AbortError" || id !== ticket) return;
    status.textContent = error.message;
  } finally {
    if (id === ticket) setBusy(false);
  }
}

function showPlate(buffer) {
  const view = new DataView(buffer);
  const metaLength = view.getUint32(0, true);
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, metaLength)));
  const dataStart = 4 + metaLength + ((4 - (metaLength % 4)) % 4);
  const floats = new Float32Array(buffer.slice(dataStart));
  clearPlate();
  for (const part of meta.parts) {
    const positions = floats.slice(part.offset, part.offset + part.count);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const material = new THREE.MeshLambertMaterial({ color: part.hex, flatShading: true });
    plate.add(new THREE.Mesh(geometry, material));
  }
  renderSwatches(meta.parts);
  lastMeta = meta;
  showMagnetLines(meta);
  exportButton.disabled = plate.children.length === 0;
  status.textContent =
    `${meta.width_mm.toFixed(1)} x ${meta.height_mm.toFixed(1)} x ${meta.depth_mm.toFixed(1)} mm, ` +
    `${meta.ridges} x ${meta.rows} picture at ${meta.pitch_mm.toFixed(2)} mm ridges and ${meta.row_mm.toFixed(2)} mm rows.`;
  if (meta.magnets) {
    if (meta.magnets.below_mm === 0) {
      status.textContent += " Glue the magnets in from the back after the print.";
    } else {
      status.textContent += ` Pause at ${meta.magnets.pause_mm.toFixed(1)} mm to drop in the magnets.`;
    }
  }
  const sizeKey = `${meta.width_mm.toFixed(2)}x${meta.height_mm.toFixed(2)}`;
  if (!framed || sizeKey !== framedSize) frame(framed);
  framed = true;
  framedSize = sizeKey;
}

function clearPlate() {
  magnetOutline = null;
  for (const mesh of [...plate.children]) {
    plate.remove(mesh);
    mesh.geometry.dispose();
    mesh.material.dispose();
  }
  document.getElementById("swatches").replaceChildren();
}

function showMagnetLines(meta) {
  if (magnetOutline) {
    plate.remove(magnetOutline);
    magnetOutline.geometry.dispose();
    magnetOutline.material.dispose();
    magnetOutline = null;
  }
  const magnets = meta && meta.magnets;
  if (!magnets || !magnets.places || !document.getElementById("show-magnets").checked) return;
  const positions = pocketPositions(meta);
  if (!positions.length) return;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({ color: 0x0f766e, depthTest: false });
  magnetOutline = new THREE.LineSegments(geometry, material);
  magnetOutline.userData.magnets = true;
  magnetOutline.renderOrder = 3;
  plate.add(magnetOutline);
}

function pocketPositions(meta) {
  const magnets = meta.magnets;
  const width = meta.width_mm;
  const height = meta.height_mm;
  const positions = [];
  for (const place of magnets.places) {
    const cx = place[0];
    const cy = place[1];
    if (magnets.shape === "round") {
      const radius = magnets.hole_w / 2;
      const steps = 32;
      for (let index = 0; index < steps; index += 1) {
        const a0 = (index / steps) * Math.PI * 2;
        const a1 = ((index + 1) / steps) * Math.PI * 2;
        const x0 = cx + radius * Math.cos(a0);
        const y0 = cy + radius * Math.sin(a0);
        const x1 = cx + radius * Math.cos(a1);
        const y1 = cy + radius * Math.sin(a1);
        pushEdge(positions, width, height, x0, y0, magnets.z0, x1, y1, magnets.z0);
        pushEdge(positions, width, height, x0, y0, magnets.z1, x1, y1, magnets.z1);
        if (index % 8 === 0) pushEdge(positions, width, height, x0, y0, magnets.z0, x0, y0, magnets.z1);
      }
    } else {
      const halfW = magnets.hole_w / 2;
      const halfH = magnets.hole_h / 2;
      const corners = [
        [cx - halfW, cy - halfH],
        [cx + halfW, cy - halfH],
        [cx + halfW, cy + halfH],
        [cx - halfW, cy + halfH],
      ];
      for (const z of [magnets.z0, magnets.z1]) {
        for (let index = 0; index < 4; index += 1) {
          const start = corners[index];
          const end = corners[(index + 1) % 4];
          pushEdge(positions, width, height, start[0], start[1], z, end[0], end[1], z);
        }
      }
      for (const corner of corners) {
        pushEdge(positions, width, height, corner[0], corner[1], magnets.z0, corner[0], corner[1], magnets.z1);
      }
    }
  }
  return positions;
}

function pushEdge(positions, width, height, x0, y0, z0, x1, y1, z1) {
  positions.push(x0 - width / 2, z0, -(y0 - height / 2));
  positions.push(x1 - width / 2, z1, -(y1 - height / 2));
}

function renderSwatches(parts) {
  const list = document.getElementById("swatches");
  list.replaceChildren();
  for (const part of parts) {
    const item = document.createElement("li");
    const chip = document.createElement("span");
    chip.className = "swatch";
    chip.style.background = part.hex;
    const label = part.role === "base" ? `base ${part.name || ""}` : (part.name || part.role);
    item.append(chip, document.createTextNode(`${label} ${part.hex}`));
    list.append(item);
  }
}

function alignOrbit(up) {
  camera.up.copy(up);
  controls._quat.setFromUnitVectors(camera.up, new THREE.Vector3(0, 1, 0));
  controls._quatInverse.copy(controls._quat).invert();
}

function frame(keepAngle) {
  const box = new THREE.Box3().setFromObject(plate);
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const span = Math.max(size.x, size.y, size.z, 1);
  const offset = camera.position.clone().sub(controls.target);
  const fit = span * 1.7;
  controls.target.copy(center);
  if (!keepAngle) alignOrbit(new THREE.Vector3(0, 1, 0));
  if (keepAngle && offset.lengthSq() > 1e-4) {
    const distance = Math.max(offset.length(), fit);
    camera.position.copy(center).addScaledVector(offset.normalize(), distance);
  } else {
    camera.position.set(center.x - span * 1.15, center.y + span * 0.9, center.z + span * 0.4);
  }
  camera.near = Math.max(span / 200, 0.01);
  camera.far = Math.max(span * 40, camera.position.distanceTo(center) * 8);
  camera.updateProjectionMatrix();
  controls.update();
}

function lookFrom(direction) {
  const box = new THREE.Box3().setFromObject(plate);
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const span = Math.max(size.x, size.y, size.z, 1);
  const distance = span * 2.4;
  const lift = Math.cos(Math.PI / 4) * distance;
  const side = Math.sin(Math.PI / 4) * distance;
  controls.target.copy(center);
  // The picture's top is -Z. Left and right views stand that edge upright.
  if (direction === "left" || direction === "right") alignOrbit(new THREE.Vector3(0, 0, -1));
  else alignOrbit(new THREE.Vector3(0, 1, 0));
  if (direction === "left") camera.position.set(center.x - side, center.y + lift, center.z);
  else if (direction === "right") camera.position.set(center.x + side, center.y + lift, center.z);
  else if (direction === "top") camera.position.set(center.x, center.y + lift, center.z - side);
  else camera.position.set(center.x, center.y + lift, center.z + side);
  camera.near = Math.max(span / 200, 0.01);
  camera.far = Math.max(span * 40, distance * 8);
  camera.updateProjectionMatrix();
  controls.update();
}

function bindUpload(slot) {
  const input = document.getElementById(`file-${slot}`);
  document.getElementById(`pick-${slot}`).addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    const file = input.files && input.files[0];
    if (!file) return;
    delete document.body.dataset.palette;
    setPicture(slot, file);
  });
}

function setPicture(slot, file) {
  pictures[slot] = file;
  flips[slot] = { h: false, v: false };
  showFlip(slot);
  const chosen = document.querySelector(`#slot-${slot} .chosen`);
  chosen.hidden = false;
  document.getElementById(`name-${slot}`).textContent = file.name;
  const thumb = document.getElementById(`thumb-${slot}`);
  if (thumb.dataset.url) URL.revokeObjectURL(thumb.dataset.url);
  const url = URL.createObjectURL(file);
  thumb.dataset.url = url;
  thumb.src = url;
  cropCustom[slot] = false;
  measurePicture(slot, file);
}

function millimeters(value) {
  return (Math.round(Math.max(value, 5) * 10) / 10).toFixed(1);
}

function idleStatus() {
  return orientation === "horizontal"
    ? "Upload a top picture and a bottom picture."
    : "Upload a left picture and a right picture.";
}

const LINES_PER_SLOPE = 5;

function formatMm(value) {
  return String(Math.round(value * 1000) / 1000);
}

function readMm(id) {
  const value = Number(document.getElementById(id).value);
  return Number.isFinite(value) ? value : 0;
}

function nozzlePreset() {
  const nozzle = Number(document.getElementById("nozzle").value);
  const images = pictures.front ? 3 : 2;
  const pitch = nozzle * LINES_PER_SLOPE * images;
  return {
    nozzle,
    pitch,
    row: nozzle,
    ridge: pitch * 0.8,
    crestWidth: nozzle,
    crestHeight: nozzle / 2,
  };
}

function applyNozzleDefaults() {
  const preset = nozzlePreset();
  document.getElementById("pitch").value = formatMm(preset.pitch);
  document.getElementById("row").value = formatMm(preset.row);
  document.getElementById("ridge").value = formatMm(preset.ridge);
  document.getElementById("crest-width").value = formatMm(preset.crestWidth);
  document.getElementById("crest-height").value = formatMm(preset.crestHeight);
  writeResolutionHint();
}

function writeResolutionHint() {
  const nozzle = readMm("nozzle");
  const pitch = readMm("pitch");
  const ridge = readMm("ridge");
  const row = readMm("row");
  const crestWidth = readMm("crest-width");
  const crestHeight = readMm("crest-height");
  const slope = pictures.front ? 0.25 : 0.5;
  const lines = nozzle > 0 ? (pitch * slope) / nozzle : 0;
  const flat = nozzle * 1.25;
  const lead = orientation === "horizontal"
    ? "Each row of the picture is one ridge. A larger plate keeps this pitch, so a tall plate holds far more detail."
    : "Each column is one ridge and each row is one band along it. A larger plate keeps this pitch, so it holds more of the picture.";
  document.getElementById("resolution-hint").textContent =
    `${lead} This plate uses a ${formatMm(pitch)} mm pitch, a ${formatMm(ridge)} mm ridge, and ${formatMm(row)} mm rows. ` +
    `Each slope is ${formatMm(lines)} lines of the ${formatMm(nozzle)} mm nozzle. ` +
    `The tip is cut flat ${formatMm(flat)} mm in from each side. ` +
    `The crest line, if you turn it on, is ${formatMm(crestWidth)} mm wide and ${formatMm(crestHeight)} mm tall.`;
}

function applyOrientation() {
  const horizontal = orientation === "horizontal";
  document.getElementById("orient-vertical").setAttribute("aria-pressed", horizontal ? "false" : "true");
  document.getElementById("orient-horizontal").setAttribute("aria-pressed", horizontal ? "true" : "false");
  document.getElementById("pick-left").textContent = horizontal ? "Upload top picture" : "Upload left picture";
  document.getElementById("pick-right").textContent = horizontal ? "Upload bottom picture" : "Upload right picture";
  document.getElementById("lead").textContent = horizontal
    ? "Two pictures, one plate. Looking from above shows the top picture. Looking from below shows the bottom picture."
    : "Two pictures, one plate. The left slopes show the first picture and the right slopes show the second.";
  document.getElementById("view-first").textContent = horizontal ? "View from Top" : "View from Left";
  document.getElementById("view-second").textContent = horizontal ? "View from Bottom" : "View from Right";
  document.getElementById("orient-hint").textContent = horizontal
    ? "Ridges run sideways. Tip the plate up or down to switch pictures, like a Clean / Dirty magnet."
    : "Ridges run up and down. Tip the plate left or right to switch pictures.";
  writeResolutionHint();
  document.querySelector('[data-text="left"]').placeholder = horizontal ? "Clean" : "Hello";
  document.querySelector('[data-text="right"]').placeholder = horizontal ? "Dirty" : "Goodbye";
  const statusText = status.textContent;
  if (statusText.startsWith("Upload a ")) status.textContent = idleStatus();
}

function syncSize(source) {
  if (!lockedAspect || !document.getElementById("lock-aspect").checked) return;
  const widthInput = document.getElementById("width");
  const heightInput = document.getElementById("height");
  if (source === "width") {
    const width = Number(widthInput.value);
    if (width > 0) heightInput.value = millimeters(width / lockedAspect);
  } else {
    const height = Number(heightInput.value);
    if (height > 0) widthInput.value = millimeters(height * lockedAspect);
  }
}

function plateAspect() {
  const width = Number(document.getElementById("width").value);
  const height = Number(document.getElementById("height").value);
  return width > 0 && height > 0 ? width / height : 1;
}

function maxCrop(imageAspect, frameAspect) {
  if (imageAspect > frameAspect) {
    const width = frameAspect / imageAspect;
    return { x: (1 - width) / 2, y: 0, w: width, h: 1 };
  }
  const height = Math.min(1, imageAspect / frameAspect);
  return { x: 0, y: (1 - height) / 2, w: 1, h: height };
}

function placeCrop(slot) {
  const box = document.querySelector(`#crop-${slot} .crop-box`);
  const crop = crops[slot];
  if (!box || !crop) return;
  box.style.left = `${crop.x * 100}%`;
  box.style.top = `${crop.y * 100}%`;
  box.style.width = `${crop.w * 100}%`;
  box.style.height = `${crop.h * 100}%`;
}

function layoutCrops() {
  const frame = plateAspect();
  for (const slot of ["left", "right", "front"]) {
    if (!imageAspects[slot]) continue;
    crops[slot] = cropCustom[slot] && crops[slot]
      ? refitCrop(crops[slot], imageAspects[slot], frame)
      : maxCrop(imageAspects[slot], frame);
    placeCrop(slot);
  }
}

function refitCrop(crop, imageAspect, frameAspect) {
  const ratio = frameAspect / imageAspect;
  const centerX = crop.x + crop.w / 2;
  const centerY = crop.y + crop.h / 2;
  let width = Math.min(centerX, 1 - centerX) * 2;
  let height = width / ratio;
  const maxHeight = Math.min(centerY, 1 - centerY) * 2;
  if (height > maxHeight) {
    height = maxHeight;
    width = height * ratio;
  }
  width = Math.max(0.05, Math.min(width, 1));
  height = width / ratio;
  return clampCrop(centerX - width / 2, centerY - height / 2, width, height, ratio);
}

function clampCrop(x, y, width, height, ratio) {
  width = Math.max(0.05, Math.min(width, 1));
  height = width / ratio;
  if (height > 1) {
    height = 1;
    width = height * ratio;
  }
  x = Math.min(Math.max(x, 0), 1 - width);
  y = Math.min(Math.max(y, 0), 1 - height);
  return { x, y, w: width, h: height };
}

async function measurePicture(slot, file) {
  const image = new Image();
  image.src = document.getElementById(`thumb-${slot}`).src;
  try {
    await image.decode();
  } catch (_error) {
    schedule();
    return;
  }
  if (!image.naturalWidth || !image.naturalHeight) {
    schedule();
    return;
  }
  imageAspects[slot] = image.naturalWidth / image.naturalHeight;
  document.getElementById(`crop-${slot}`).style.aspectRatio = `${image.naturalWidth} / ${image.naturalHeight}`;
  if (slot === "left") {
    pictureAspect = imageAspects.left;
    if (document.getElementById("lock-aspect").checked) {
      lockedAspect = pictureAspect;
      syncSize("width");
    }
  }
  crops[slot] = maxCrop(imageAspects[slot], plateAspect());
  placeCrop(slot);
  writeResolutionHint();
  schedule();
  void file;
}

function bindCrop(slot) {
  const stage = document.getElementById(`crop-${slot}`);
  const box = stage.querySelector(".crop-box");
  box.addEventListener("pointerdown", (event) => {
    if (!crops[slot]) return;
    event.preventDefault();
    const handle = event.target.dataset.handle || "move";
    const bounds = stage.getBoundingClientRect();
    const start = { x: event.clientX, y: event.clientY, crop: { ...crops[slot] } };
    box.setPointerCapture(event.pointerId);
    const move = (ev) => {
      const dx = (ev.clientX - start.x) / Math.max(bounds.width, 1);
      const dy = (ev.clientY - start.y) / Math.max(bounds.height, 1);
      const ratio = plateAspect() / imageAspects[slot];
      crops[slot] = handle === "move"
        ? clampCrop(start.crop.x + dx, start.crop.y + dy, start.crop.w, start.crop.h, ratio)
        : resizeCrop(start.crop, handle, dx, dy, ratio);
      cropCustom[slot] = true;
      placeCrop(slot);
    };
    const stop = () => {
      box.removeEventListener("pointermove", move);
      box.removeEventListener("pointerup", stop);
      box.removeEventListener("pointercancel", stop);
      schedule();
    };
    box.addEventListener("pointermove", move);
    box.addEventListener("pointerup", stop);
    box.addEventListener("pointercancel", stop);
  });
}

function resizeCrop(start, handle, dx, dy, ratio) {
  const east = handle.includes("e");
  const south = handle.includes("s");
  const anchorX = east ? start.x : start.x + start.w;
  const anchorY = south ? start.y : start.y + start.h;
  let width = Math.max(0.05, east ? start.w + dx : start.w - dx);
  let height = Math.max(0.05, south ? start.h + dy : start.h - dy);
  if (Math.abs(dx) >= Math.abs(dy)) height = width / ratio;
  else width = height * ratio;
  const x = east ? anchorX : anchorX - width;
  const y = south ? anchorY : anchorY - height;
  return clampCrop(x, y, width, height, ratio);
}

document.querySelectorAll("[data-flip]").forEach((button) => {
  button.addEventListener("click", () => {
    const slot = button.dataset.flip;
    const axis = button.dataset.axis;
    flips[slot][axis] = !flips[slot][axis];
    showFlip(slot);
    schedule();
  });
});

document.querySelectorAll("[data-clear]").forEach((button) => {
  button.addEventListener("click", () => {
    const slot = button.dataset.clear;
    pictures[slot] = null;
    flips[slot] = { h: false, v: false };
    showFlip(slot);
    crops[slot] = null;
    imageAspects[slot] = null;
    cropCustom[slot] = false;
    document.getElementById(`file-${slot}`).value = "";
    const thumb = document.getElementById(`thumb-${slot}`);
    if (thumb.dataset.url) {
      URL.revokeObjectURL(thumb.dataset.url);
      delete thumb.dataset.url;
    }
    thumb.removeAttribute("src");
    document.querySelector(`#slot-${slot} .chosen`).hidden = true;
    writeResolutionHint();
    schedule();
  });
});

["base", "ridge", "pitch", "row", "max-colors", "base-color"].forEach((id) => {
  document.getElementById(id).addEventListener("input", () => {
    if (id === "ridge" || id === "pitch" || id === "row") writeResolutionHint();
    schedule();
  });
});
document.getElementById("base").addEventListener("input", () => {
  if (!document.getElementById("base").disabled) plainBase = document.getElementById("base").value;
});
document.getElementById("width").addEventListener("input", () => {
  syncSize("width");
  layoutCrops();
  schedule();
});
document.getElementById("height").addEventListener("input", () => {
  syncSize("height");
  layoutCrops();
  schedule();
});
document.querySelectorAll("[data-reset-crop]").forEach((button) => {
  button.addEventListener("click", () => {
    const slot = button.dataset.resetCrop;
    if (!imageAspects[slot]) return;
    cropCustom[slot] = false;
    crops[slot] = maxCrop(imageAspects[slot], plateAspect());
    placeCrop(slot);
    schedule();
  });
});
document.getElementById("lock-aspect").addEventListener("change", () => {
  const locked = document.getElementById("lock-aspect").checked;
  if (!locked) return;
  if (pictureAspect) {
    lockedAspect = pictureAspect;
    syncSize("width");
    schedule();
    return;
  }
  const width = Number(document.getElementById("width").value);
  const height = Number(document.getElementById("height").value);
  if (width > 0 && height > 0) lockedAspect = width / height;
});
document.getElementById("orient-vertical").addEventListener("click", () => {
  if (orientation === "vertical") return;
  orientation = "vertical";
  applyOrientation();
  schedule();
});
document.getElementById("orient-horizontal").addEventListener("click", () => {
  if (orientation === "horizontal") return;
  orientation = "horizontal";
  applyOrientation();
  schedule();
});
document.getElementById("dither").addEventListener("change", schedule);
document.getElementById("nozzle").addEventListener("change", () => {
  applyNozzleDefaults();
  schedule();
});

document.getElementById("sample").addEventListener("click", async () => {
  document.getElementById("dither").checked = false;
  document.getElementById("use-spools").checked = false;
  setSpools(false);
  document.body.dataset.palette = "#ffd60a,#e63946,#0d1b2a,#4cc9f0";
  const [left, right] = await Promise.all([
    fetch("/sample/left.png").then((response) => response.blob()),
    fetch("/sample/right.png").then((response) => response.blob()),
  ]);
  setPicture("left", new File([left], "sample-left.png", { type: "image/png" }));
  setPicture("right", new File([right], "sample-right.png", { type: "image/png" }));
});

document.getElementById("reset-view").addEventListener("click", () => {
  frame(false);
  framed = true;
});
document.getElementById("view-first").addEventListener("click", () => {
  lookFrom(orientation === "horizontal" ? "top" : "left");
});
document.getElementById("view-second").addEventListener("click", () => {
  lookFrom(orientation === "horizontal" ? "bottom" : "right");
});

function syncCrest() {
  document.getElementById("crest-fields").hidden = !document.getElementById("crest-line").checked;
}

document.getElementById("crest-line").addEventListener("change", () => {
  syncCrest();
  schedule();
});
document.getElementById("crest-color").addEventListener("input", schedule);
["crest-width", "crest-height"].forEach((id) => {
  document.getElementById(id).addEventListener("input", () => {
    writeResolutionHint();
    schedule();
  });
});

exportButton.addEventListener("click", async () => {
  if (!pictures.left || !pictures.right) return;
  exportButton.disabled = true;
  status.textContent = "Writing the slicer file...";
  try {
    const response = await fetch("/api/export", { method: "POST", body: formData() });
    const type = response.headers.get("content-type") || "";
    if (!response.ok || type.includes("json")) {
      const payload = await response.json();
      throw new Error(payload.error || "Export failed.");
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "lentic-plate.3mf";
    link.click();
    URL.revokeObjectURL(url);
    status.textContent = "Saved lentic-plate.3mf. One part per color, already assembled.";
  } catch (error) {
    status.textContent = error.message;
  } finally {
    exportButton.disabled = plate.children.length === 0;
  }
});

const FONTS = [
  { family: "Oswald", weight: 700 },
  { family: "Nunito", weight: 700 },
  { family: "Libre Baskerville", weight: 700 },
  { family: "Inconsolata", weight: 700 },
];
const starterSpools = ["#111111", "#f7f4ee", "#c0392b", "#1d4e89"];
let spools = starterSpools.slice();

function renderSpools() {
  const list = document.getElementById("spool-list");
  list.replaceChildren();
  spools.forEach((hex, index) => {
    const row = document.createElement("div");
    row.className = "spool-row";
    const input = document.createElement("input");
    input.type = "color";
    input.value = hex;
    input.setAttribute("aria-label", `Filament color ${index + 1}`);
    input.addEventListener("input", () => {
      spools[index] = input.value;
      if (document.getElementById("use-spools").checked) schedule();
    });
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "secondary";
    remove.textContent = "\u00d7";
    remove.setAttribute("aria-label", `Remove filament color ${index + 1}`);
    remove.disabled = spools.length <= 1;
    remove.addEventListener("click", () => {
      spools.splice(index, 1);
      renderSpools();
      if (document.getElementById("use-spools").checked) schedule();
    });
    row.append(input, remove);
    list.append(row);
  });
}

function setSpools(on) {
  document.getElementById("spools").hidden = !on;
  document.getElementById("max-colors").disabled = on;
}

async function makeTextPicture(slot) {
  const text = document.querySelector(`[data-text="${slot}"]`).value.trim();
  if (!text) {
    status.textContent = "Type some text first.";
    return;
  }
  const family = document.querySelector(`[data-font="${slot}"]`).value;
  const font = FONTS.find((item) => item.family === family) || FONTS[0];
  const ink = document.querySelector(`[data-ink="${slot}"]`).value;
  const paper = document.querySelector(`[data-paper="${slot}"]`).value;
  const width = 1600;
  const height = Math.max(200, Math.round(width / Math.max(plateAspect(), 0.2)));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  const spec = `${font.weight} 200px "${font.family}"`;
  await document.fonts.load(spec);
  context.fillStyle = paper;
  context.fillRect(0, 0, width, height);
  context.fillStyle = ink;
  context.textAlign = "center";
  context.textBaseline = "middle";
  let size = Math.floor(Math.min(height * 0.62, 420));
  const maxWidth = width * 0.9;
  context.font = `${font.weight} ${size}px "${font.family}"`;
  while (size > 24 && context.measureText(text).width > maxWidth) {
    size -= 4;
    context.font = `${font.weight} ${size}px "${font.family}"`;
  }
  context.fillText(text, width / 2, height / 2);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  delete document.body.dataset.palette;
  setPicture(slot, new File([blob], `${slot}-text.png`, { type: "image/png" }));
}

for (const select of document.querySelectorAll("[data-font]")) {
  for (const font of FONTS) {
    const option = document.createElement("option");
    option.value = font.family;
    option.textContent = font.family;
    option.style.fontFamily = `"${font.family}", sans-serif`;
    select.append(option);
  }
}
document.querySelectorAll("[data-make-text]").forEach((button) => {
  button.addEventListener("click", () => makeTextPicture(button.dataset.makeText));
});
document.getElementById("use-spools").addEventListener("change", () => {
  setSpools(document.getElementById("use-spools").checked);
  schedule();
});
document.getElementById("add-spool").addEventListener("click", () => {
  if (spools.length >= 8) {
    status.textContent = "Eight filaments is the limit.";
    return;
  }
  spools.push("#888888");
  renderSpools();
  if (document.getElementById("use-spools").checked) schedule();
});

bindUpload("left");
bindUpload("right");
bindUpload("front");
bindCrop("left");
bindCrop("right");
bindCrop("front");
let plainBase = document.getElementById("base").value;

function magnetBodyThickness() {
  const shape = document.getElementById("magnet-shape").value;
  const id = shape === "rect" ? "magnet-rect-thickness" : "magnet-thickness";
  return Number(document.getElementById(id).value);
}

function syncMagnets() {
  const on = document.getElementById("magnets").checked;
  const rectangular = document.getElementById("magnet-shape").value === "rect";
  document.getElementById("magnet-fields").hidden = !on;
  document.getElementById("magnet-round").hidden = rectangular;
  document.getElementById("magnet-rect").hidden = !rectangular;
  document.getElementById("magnet-turn-label").hidden = !rectangular;
  const base = document.getElementById("base");
  if (!on) {
    if (base.disabled) {
      base.disabled = false;
      base.value = plainBase;
    }
    return;
  }
  if (!base.disabled) plainBase = base.value;
  base.disabled = true;
  const total = Number(document.getElementById("magnet-below").value) + magnetBodyThickness() + Number(document.getElementById("magnet-above").value);
  if (Number.isFinite(total) && total > 0) base.value = (Math.round(total * 10) / 10).toFixed(1);
}

document.getElementById("magnets").addEventListener("change", () => {
  syncMagnets();
  schedule();
});
document.getElementById("magnet-shape").addEventListener("change", () => {
  syncMagnets();
  schedule();
});
document.getElementById("magnet-arrange").addEventListener("change", schedule);
document.getElementById("magnet-edge").addEventListener("input", schedule);
document.getElementById("magnet-turn").addEventListener("change", schedule);
document.getElementById("show-magnets").addEventListener("change", () => showMagnetLines(lastMeta));
["magnet-count", "magnet-diameter", "magnet-thickness", "magnet-width", "magnet-length", "magnet-rect-thickness", "magnet-below", "magnet-above"].forEach((id) => {
  document.getElementById(id).addEventListener("input", () => {
    syncMagnets();
    schedule();
  });
});

renderSpools();
applyOrientation();
syncMagnets();
syncCrest();
