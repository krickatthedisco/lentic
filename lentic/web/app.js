import * as THREE from "three";
import { OrbitControls } from "/vendor/OrbitControls.js";

const pictures = { left: null, right: null, front: null };
const crops = { left: null, right: null, front: null };
const imageAspects = { left: null, right: null, front: null };
const cropCustom = { left: false, right: false, front: false };
let orientation = "vertical";
let pictureAspect = null;
let lockedAspect = 80 / 45;
let framed = false;
let framedSize = "";
let ticket = 0;
let inflight = null;

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
  meshes: () => plate.children.length,
  status: () => status.textContent,
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
  if (document.body.dataset.palette) data.set("palette", document.body.dataset.palette);
  const first = orientation === "horizontal" ? "top" : "left";
  const second = orientation === "horizontal" ? "bottom" : "right";
  data.set(first, pictures.left, pictures.left.name);
  data.set(second, pictures.right, pictures.right.name);
  const firstCrop = cropValue("left");
  const secondCrop = cropValue("right");
  if (firstCrop) data.set(`crop_${first}`, firstCrop);
  if (secondCrop) data.set(`crop_${second}`, secondCrop);
  if (pictures.front) {
    data.set("front", pictures.front, pictures.front.name);
    const frontCrop = cropValue("front");
    if (frontCrop) data.set("crop_front", frontCrop);
  }
  return data;
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

async function preview() {
  exportButton.disabled = true;
  if (!pictures.left || !pictures.right) {
    status.textContent = idleStatus();
    clearPlate();
    return;
  }
  let data;
  try {
    data = formData();
  } catch (error) {
    status.textContent = error.message;
    return;
  }
  const id = ++ticket;
  if (inflight) inflight.abort();
  const controller = new AbortController();
  inflight = controller;
  status.textContent = "Building the plate...";
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
  exportButton.disabled = plate.children.length === 0;
  status.textContent =
    `${meta.width_mm.toFixed(1)} x ${meta.height_mm.toFixed(1)} x ${meta.depth_mm.toFixed(1)} mm, ` +
    `${meta.ridges} x ${meta.rows} picture at ${meta.pitch_mm.toFixed(2)} mm ridges and ${meta.row_mm.toFixed(2)} mm rows.`;
  const sizeKey = `${meta.width_mm.toFixed(2)}x${meta.height_mm.toFixed(2)}`;
  if (!framed || sizeKey !== framedSize) frame(framed);
  framed = true;
  framedSize = sizeKey;
}

function clearPlate() {
  for (const mesh of [...plate.children]) {
    plate.remove(mesh);
    mesh.geometry.dispose();
    mesh.material.dispose();
  }
  document.getElementById("swatches").replaceChildren();
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

function frame(keepAngle) {
  const box = new THREE.Box3().setFromObject(plate);
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const span = Math.max(size.x, size.y, size.z, 1);
  const offset = camera.position.clone().sub(controls.target);
  const fit = span * 1.7;
  controls.target.copy(center);
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

function applyOrientation() {
  const horizontal = orientation === "horizontal";
  document.getElementById("orient-vertical").setAttribute("aria-pressed", horizontal ? "false" : "true");
  document.getElementById("orient-horizontal").setAttribute("aria-pressed", horizontal ? "true" : "false");
  document.getElementById("pick-left").textContent = horizontal ? "Upload top picture" : "Upload left picture";
  document.getElementById("pick-right").textContent = horizontal ? "Upload bottom picture" : "Upload right picture";
  document.getElementById("lead").textContent = horizontal
    ? "Two pictures, one plate. Looking from above shows the top picture. Looking from below shows the bottom picture."
    : "Two pictures, one plate. The left slopes show the first picture and the right slopes show the second.";
  document.getElementById("orient-hint").textContent = horizontal
    ? "Ridges run sideways. Tip the plate up or down to switch pictures, like a Clean / Dirty magnet."
    : "Ridges run up and down. Tip the plate left or right to switch pictures.";
  document.getElementById("resolution-hint").textContent = horizontal
    ? "Each row of the picture is one ridge. A larger plate keeps this pitch, so a tall plate holds far more detail. A 0.4 mm nozzle starts at 0.8 mm pitch and 0.4 mm rows."
    : "Each column is one ridge and each row is one band along it. A larger plate keeps this pitch, so 300 mm across holds far more of the picture than 80 mm. A 0.4 mm nozzle starts at 0.8 mm pitch and 0.4 mm rows. Raise the pitch toward 1.6 mm if a slope comes out too thin.";
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

document.querySelectorAll("[data-clear]").forEach((button) => {
  button.addEventListener("click", () => {
    const slot = button.dataset.clear;
    pictures[slot] = null;
    crops[slot] = null;
    imageAspects[slot] = null;
    cropCustom[slot] = false;
    document.getElementById(`file-${slot}`).value = "";
    document.querySelector(`#slot-${slot} .chosen`).hidden = true;
    schedule();
  });
});

["base", "ridge", "pitch", "row", "max-colors", "base-color"].forEach((id) => {
  document.getElementById(id).addEventListener("input", schedule);
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

document.getElementById("sample").addEventListener("click", async () => {
  document.getElementById("dither").checked = false;
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

exportButton.addEventListener("click", async () => {
  if (!pictures.left || !pictures.right) return;
  exportButton.disabled = true;
  status.textContent = "Writing the multi-part STL...";
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
    link.download = "lentic-plate.stl";
    link.click();
    URL.revokeObjectURL(url);
    status.textContent = "Saved lentic-plate.stl. Each solid is one filament.";
  } catch (error) {
    status.textContent = error.message;
  } finally {
    exportButton.disabled = plate.children.length === 0;
  }
});

bindUpload("left");
bindUpload("right");
bindUpload("front");
bindCrop("left");
bindCrop("right");
bindCrop("front");
