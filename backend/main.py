import gc, threading, time, cv2, logging, numpy as np, torch
from collections import deque
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel

from detector import (
    load_yolo, load_threat_model, load_vlm,
    smolvlm_infer, run_vlm_threat, run_weapons, draw_weapons,
    state, state_lock, threat_classes, yolo_edge_classes,
    ProximityTracker, push_alert, pad_crop,
    RED_HOLD_SEC, RED_CONFIDENCE, vlm_abort,
    DEFAULT_PROXIMITY_PROMPT, DEFAULT_COUNT_CHANGE_PROMPT,
    DEFAULT_WEAPON_PROMPT, DEFAULT_SCENE_PROMPT,
    telegram_notifier, SUPPORTED_MODELS,
    get_scene_prompt, get_proximity_prompt, get_count_change_prompt,
    get_person_prompt, get_temporal_scene_prompt, get_temporal_count_prompt,
)
from notifications import MEDIA_ROOT, create_alert_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True)

STREAM_W             = 640
STREAM_H             = 480
STREAM_FPS           = 30
NEW_PERSON_COOL      = 30.0
VLM_THREAD_TIMEOUT   = 20.0
WEAPON_MIN_FRAMES    = 3
WEAPON_SCAN_INTERVAL = 3        # run threat model every Nth frame

TRIGGER_COOLDOWN = {            # minimum seconds between same-type trigger fires
    "weapon":       8.0,
    "proximity":    5.0,
    "count_change": 3.0,
}

# ── MJPEG client tracking ─────────────────────────────────────────────────────
_mjpeg_clients    = 0
_mjpeg_clients_lk = threading.Lock()
_new_frame_event  = threading.Event()


class TelegramConfigPayload(BaseModel):
    enabled: bool | None = None
    bot_token: str | None = None
    chat_id: str | None = None

class VlmLoadPayload(BaseModel):
    model_key: str
    quantization: str = "4bit"

app = FastAPI(title="CCTV Surveillance API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/media", StaticFiles(directory=str(MEDIA_ROOT)), name="media")

# ── Load models ───────────────────────────────────────────────────────────────
log.info("=" * 50)
yolo_model               = load_yolo()
threat_model             = load_threat_model()
vlm_model, vlm_processor = load_vlm()

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85)
    yolo_model.to("cuda"); yolo_model.model.half()
    if threat_model:
        threat_model.to("cuda"); threat_model.model.half()
    log.info(f"[GPU] {torch.cuda.get_device_name(0)}")
else:
    log.warning("[GPU] No CUDA")
log.info("=" * 50)

# ── Engine ────────────────────────────────────────────────────────────────────
engine = {
    "running":    False,
    "source":     None,
    "thread":     None,
    "frame":      None,
    "frame_lock": threading.Lock(),
}

# ── VLM Priority Task Manager ─────────────────────────────────────────────────
# "trigger" tasks (weapon/proximity/count_change) preempt "passive" tasks.
# Triggers never interrupt other triggers — first one wins.
_vlm_task = {
    "thread": None,
    "type":   "passive",   # "passive" | "trigger"
    "lock":   threading.Lock(),
}

def launch_vlm(task_type: str, fn, args) -> bool:
    """
    Launch VLM task with priority.
    - trigger preempts passive (aborts it via vlm_abort event)
    - passive skips if anything is running
    - trigger skips if another trigger is running
    Returns True if launched.
    """
    with _vlm_task["lock"]:
        t      = _vlm_task["thread"]
        active = t is not None and t.is_alive()

        if active:
            if task_type == "trigger" and _vlm_task["type"] == "passive":
                log.info("[VLM] Aborting passive task — trigger incoming")
                vlm_abort.set()
                t.join(timeout=2.0)
                vlm_abort.clear()
                # fall through to start trigger
            else:
                return False  # skip passive-on-anything or trigger-on-trigger

        def _run():
            try:
                fn(*args)
            except Exception as e:
                log.warning(f"[VLM task] {e}")

        new_t = threading.Thread(target=_run, daemon=True)
        new_t._start_time    = time.time()
        _vlm_task["thread"]  = new_t
        _vlm_task["type"]    = task_type
        vlm_abort.clear()
        new_t.start()
        return True


def vlm_running() -> bool:
    t = _vlm_task["thread"]
    return t is not None and t.is_alive()


# ── VRAM ──────────────────────────────────────────────────────────────────────
def get_vram_pct() -> float:
    try:
        return (torch.cuda.memory_reserved() /
                torch.cuda.get_device_properties(0).total_memory * 100)
    except Exception:
        return 0.0

def offload_vlm():
    global vlm_model
    if vlm_model is None: return
    try:
        vlm_model = vlm_model.to("cpu")
        torch.cuda.empty_cache()
        log.info("[VLM] Offloaded to CPU")
    except Exception as e:
        log.warning(f"[VLM] Offload: {e}")

def reload_vlm():
    global vlm_model
    if vlm_model is None: return
    try:
        vlm_model = vlm_model.to("cuda")
        torch.cuda.synchronize()
        log.info("[VLM] Reloaded to GPU")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        log.error("[VLM] OOM reload — staying CPU")
    except Exception as e:
        log.warning(f"[VLM] Reload: {e}")

def load_vlm_with_config(model_key: str, quantization: str):
    global vlm_model, vlm_processor
    # Abort any running inference and wait for it to finish
    vlm_abort.set()
    t = _vlm_task["thread"]
    if t and t.is_alive():
        t.join(timeout=3.0)
    vlm_abort.clear()

    with state_lock:
        state["vlm_enabled"]       = False
        state["scene_description"] = ""

    # Free old model from GPU memory before loading the new one
    old_model     = vlm_model
    vlm_model     = None
    vlm_processor = None
    if old_model is not None:
        try:
            old_model.cpu()
        except Exception:
            pass
        del old_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    new_model, new_proc = load_vlm(model_key, quantization)
    vlm_model     = new_model
    vlm_processor = new_proc

    with state_lock:
        state["mode_switching"] = False
        if vlm_model is not None:
            state["vlm_enabled"] = True


# ── Engine ────────────────────────────────────────────────────────────────────
def run_engine(source):
    prox                = ProximityTracker()
    described_ids       = {}
    prev_count          = -1
    prev_yolo_on        = False
    frame_count         = 0
    weapon_consecutive  = 0
    _last_weapon_result = ([], None, None)
    trigger_last_fired: dict = {}

    try:    src = int(source)
    except: src = source

    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  STREAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_H)
        cap.set(cv2.CAP_PROP_FPS,          STREAM_FPS)

    if not cap.isOpened():
        log.error(f"Cannot open: {source}")
        engine["running"] = False
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120: fps = 30.0
    delay = 1.0 / fps
    recent_frames = deque(maxlen=max(24, min(90, int(fps * 3))))
    with state_lock:
        state["source_fps"] = round(fps, 2)
    log.info(f"Stream: {source} @ {fps:.0f}fps")

    # ── VLM callbacks ─────────────────────────────────────────────────────────
    def build_context(snapshot_frame, trigger_type):
        return create_alert_context(
            snapshot_frame=snapshot_frame,
            clip_frames=list(recent_frames),
            clip_fps=max(6.0, min(fps, 12.0)),
            trigger_type=trigger_type,
            source=str(source),
        )

    def do_weapon_vlm(crop, prompt, alert_context):
        r = run_vlm_threat(crop, vlm_model, vlm_processor, prompt)
        if r.get("threat") and r.get("confidence") in RED_CONFIDENCE:
            push_alert("RED", f"Weapon confirmed: {r.get('type')}", r, alert_context=alert_context)
        else:
            push_alert("YELLOW", "Weapon detected - unconfirmed by VLM", r, alert_context=alert_context)

    def do_proximity_vlm(crop, prompt, alert_context):
        desc = smolvlm_infer(crop, prompt or get_proximity_prompt(),
                             vlm_model, vlm_processor, max_tokens=60)
        if not desc:
            push_alert("YELLOW", "Sustained contact detected", alert_context=alert_context)
            return
        d = desc.upper()
        threat = "THREATENING" in d or any(
            w in desc.lower() for w in ["assault", "attack", "fight", "stab", "armed"]
        )
        vlm_result = {
            "description": desc,
            "type": "interaction_threat" if threat else "interaction_notice",
            "confidence": "medium" if threat else "low",
        }
        if threat:
            push_alert("RED", f"Proximity threat: {desc[:80]}", vlm_result, alert_context=alert_context)
        else:
            push_alert("YELLOW", f"Sustained contact: {desc[:80]}", vlm_result, alert_context=alert_context)
        with state_lock:
            state["scene_description"] = desc

    def do_count_change_vlm(crop, prompt, previous_count, current_count, alert_context):
        if prompt:
            final_prompt = prompt
        else:
            with state_lock:
                last_desc = state["scene_description"]
            final_prompt = (
                get_temporal_count_prompt(last_desc, previous_count, current_count)
                if last_desc else get_count_change_prompt()
            )
        desc = smolvlm_infer(crop, final_prompt, vlm_model, vlm_processor, max_tokens=100)
        if desc:
            with state_lock:
                state["scene_description"] = desc
            log.info(f"[COUNT VLM] {desc}")
            push_alert(
                "YELLOW",
                f"Person count: {previous_count}->{current_count}",
                {
                    "description": desc,
                    "type": "count_change",
                    "confidence": "low",
                },
                alert_context=alert_context,
            )
        else:
            push_alert(
                "YELLOW",
                f"Person count: {previous_count}->{current_count}",
                alert_context=alert_context,
            )

    def do_scene_vlm(crop, prompt):
        if prompt:
            final_prompt = prompt
        else:
            with state_lock:
                last_desc = state["scene_description"]
            final_prompt = (
                get_temporal_scene_prompt(last_desc) if last_desc
                else get_scene_prompt()
            )
        desc = smolvlm_infer(crop, final_prompt, vlm_model, vlm_processor, max_tokens=100)
        if desc:
            with state_lock:
                state["scene_description"] = desc

    # ── Main loop ─────────────────────────────────────────────────────────────
    while engine["running"]:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            if isinstance(src, str) and not str(src).startswith("rtsp"):
                # Local video file — loop
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            # Live source (camera index or RTSP) — attempt reconnect
            log.warning(f"[ENGINE] Stream lost ({source}) — reconnecting in 2s")
            cap.release()
            time.sleep(2.0)
            cap = cv2.VideoCapture(src)
            if isinstance(src, int):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  STREAM_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_H)
                cap.set(cv2.CAP_PROP_FPS,          STREAM_FPS)
            if not cap.isOpened():
                log.error(f"[ENGINE] Reconnect failed — stopping")
                break
            log.info(f"[ENGINE] Reconnected: {source}")
            continue

        try:
            frame_count += 1
            now = time.time()
            recent_frames.append(frame)   # raw — prepared lazily at alert time

            with state_lock:
                yolo_on  = state["yolo_enabled"]
                vlm_on   = state["vlm_enabled"]
                vlm_ivl  = state["vlm_interval"]
                last_vlm = state["last_vlm_time"]
                switching= state["mode_switching"]
                prompts  = dict(state["trigger_prompts"])

            # ── RAW mode ──────────────────────────────────────────────────────
            if not yolo_on and not vlm_on:
                with _mjpeg_clients_lk:
                    has_clients = _mjpeg_clients > 0
                if has_clients:
                    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    with engine["frame_lock"]:
                        engine["frame"] = buf.tobytes()
                    _new_frame_event.set()
                time.sleep(max(0, delay-(time.time()-t0))); continue

            # ── VLM thread watchdog ────────────────────────────────────────────
            t_obj = _vlm_task["thread"]
            if (t_obj and t_obj.is_alive()
                    and hasattr(t_obj, "_start_time")
                    and now - t_obj._start_time > VLM_THREAD_TIMEOUT):
                log.warning("[VLM] Watchdog: thread hung >20s — aborting")
                vlm_abort.set()
                t_obj.join(timeout=1.0)
                vlm_abort.clear()

            # ── YOLO tracking ─────────────────────────────────────────────────
            annotated    = frame
            people_boxes = {}
            class_counts = {}
            weapon_dets  = []
            edge_dets    = []   # axe/scissor/crowbar from yolo26n
            yolo_trigger = None

            # Detect YOLO toggle-off → reset count baseline
            if prev_yolo_on and not yolo_on:
                prev_count = -1
            prev_yolo_on = yolo_on

            if yolo_on:
                tr = yolo_model.track(
                    frame, persist=True, tracker="bytetrack.yaml",
                    conf=0.4, imgsz=640, verbose=False,
                )
                annotated = tr[0].plot()

                # Batch all GPU→CPU tensor transfers in one pass
                boxes_obj = tr[0].boxes
                if boxes_obj.id is not None:
                    ids_t   = boxes_obj.id.int().cpu().tolist()
                    cls_t   = boxes_obj.cls.int().cpu().tolist()
                    confs_t = boxes_obj.conf.cpu().tolist()
                    xyxy_t  = boxes_obj.xyxy.cpu().numpy()
                else:
                    ids_t = cls_t = confs_t = []
                    xyxy_t = []

                for tid, cid, conf_val, xyxy in zip(ids_t, cls_t, confs_t, xyxy_t):
                    name = yolo_model.names[cid]
                    class_counts[name] = class_counts.get(name, 0) + 1

                    if cid == 0:
                        people_boxes[tid] = xyxy

                    if cid in yolo_edge_classes and conf_val >= 0.60:
                        x1,y1,x2,y2 = map(int, xyxy)
                        edge_dets.append({
                            "label":      yolo_edge_classes[cid],
                            "confidence": round(conf_val, 2),
                            "bbox":       [x1,y1,x2,y2],
                        })

                # Prune stale track IDs from description cache
                for stale in [t for t in described_ids if t not in people_boxes]:
                    del described_ids[stale]

                with state_lock:
                    state["person_count"] = len(people_boxes)

                # ── Weapon detection (sampled every WEAPON_SCAN_INTERVAL frames) ──
                if frame_count % WEAPON_SCAN_INTERVAL == 0:
                    _last_weapon_result = run_weapons(threat_model, frame, edge_dets)
                w_dets, w_trigger, w_crop = _last_weapon_result
                if w_dets:
                    weapon_consecutive += 1
                else:
                    weapon_consecutive  = 0

                if weapon_consecutive >= WEAPON_MIN_FRAMES and w_trigger:
                    weapon_dets  = w_dets
                    yolo_trigger = ("weapon", w_trigger, w_crop)
                    annotated    = draw_weapons(annotated, weapon_dets)

                with state_lock:
                    state["weapon_detections"] = weapon_dets

                # ── Proximity trigger ─────────────────────────────────────────
                if yolo_trigger is None:
                    pr = prox.update(people_boxes)
                    if pr:
                        pair_ids, mb = pr
                        crop = pad_crop(frame, mb)
                        yolo_trigger = (
                            "proximity",
                            f"Sustained contact — IDs {pair_ids}",
                            crop,
                        )

                # ── Count change trigger ──────────────────────────────────────
                cur_count = len(people_boxes)
                count_cooldown = TRIGGER_COOLDOWN.get("count_change", 3.0)
                if (cur_count != prev_count and prev_count != -1
                        and now - trigger_last_fired.get("count_change", 0.0) >= count_cooldown):
                    trigger_last_fired["count_change"] = now
                    count_context = build_context(frame.copy(), "count_change")
                    log.info(f"[COUNT] {prev_count} → {cur_count}")
                    if vlm_on and vlm_model:
                        with state_lock:
                            state["last_vlm_time"] = now
                        launched = launch_vlm(
                            "trigger", do_count_change_vlm,
                            (
                                frame.copy(),
                                prompts.get("count_change", ""),
                                prev_count,
                                cur_count,
                                count_context,
                            )
                        )
                        if not launched:
                            log.info("[COUNT] VLM busy — alert without description")
                            push_alert("YELLOW", f"Person count: {prev_count}->{cur_count}", alert_context=count_context)
                    else:
                        push_alert("YELLOW", f"Person count: {prev_count}->{cur_count}", alert_context=count_context)
                prev_count = cur_count

            # ── VLM trigger dispatch ──────────────────────────────────────────
            with state_lock:
                cur_alert = state["alert"]
                last_red  = state["last_red_time"]
                last_vlm  = state["last_vlm_time"]

            if yolo_trigger:
                kind, reason, crop = yolo_trigger
                # Per-trigger cooldown — prevents oscillating detections from spamming VLM
                cooldown = TRIGGER_COOLDOWN.get(kind, 5.0)
                if now - trigger_last_fired.get(kind, 0.0) < cooldown:
                    yolo_trigger = None
                else:
                    trigger_last_fired[kind] = now

            if yolo_trigger:
                kind, reason, crop = yolo_trigger
                alert_context = build_context(
                    crop.copy() if crop is not None and crop.size > 0 else frame.copy(),
                    kind,
                )
                launched = False
                if vlm_on and vlm_model:
                    with state_lock:
                        state["last_vlm_time"] = now
                    if kind == "weapon":
                        launched = launch_vlm(
                            "trigger", do_weapon_vlm,
                            (crop.copy(), prompts.get("weapon", ""), alert_context)
                        )
                    elif kind == "proximity":
                        launched = launch_vlm(
                            "trigger", do_proximity_vlm,
                            (crop.copy(), prompts.get("proximity", ""), alert_context)
                        )
                if cur_alert == "CLEAR" and not launched:
                    push_alert("YELLOW", reason, alert_context=alert_context)

            # ── Passive scene VLM (interval-based) ────────────────────────────
            if (vlm_on and vlm_model
                    and now - last_vlm >= vlm_ivl
                    and not yolo_trigger
                    and get_vram_pct() < 75):
                with state_lock:
                    state["last_vlm_time"] = now
                launch_vlm("passive", do_scene_vlm, (frame.copy(), ""))

            # ── Person description (VLM + YOLO both on) ───────────────────────
            if yolo_on and vlm_on and vlm_model:
                for tid, xyxy in people_boxes.items():
                    cx = int((xyxy[0]+xyxy[2])/2/80)
                    cy = int((xyxy[1]+xyxy[3])/2/80)
                    ph = cx*1000+cy
                    le = described_ids.get(tid, {"time":0,"ph":-1})
                    if (le["time"]==0
                            or (now-le["time"]>NEW_PERSON_COOL and le["ph"]!=ph)):
                        described_ids[tid] = {"time":now,"ph":ph}
                        crop = pad_crop(frame, xyxy, 40)
                        if crop.size > 0:
                            def _person_desc(t_id=tid, c=crop.copy()):
                                desc = smolvlm_infer(
                                    c,
                                    get_person_prompt(),
                                    vlm_model, vlm_processor, max_tokens=80
                                )
                                if desc:
                                    with state_lock:
                                        state["person_log"].append({
                                            "time":        datetime.now().strftime("%H:%M:%S"),
                                            "track_id":    t_id,
                                            "description": desc,
                                        })
                                        state["person_log"] = state["person_log"][-50:]
                            launch_vlm("passive", _person_desc, ())

            # ── Clear stale alert ─────────────────────────────────────────────
            with state_lock:
                cur_alert = state["alert"]
                last_red  = state["last_red_time"]
            if (cur_alert != "CLEAR"
                    and not yolo_trigger
                    and now - last_red >= RED_HOLD_SEC
                    and not vlm_running()):
                push_alert("CLEAR", "")

            # ── Detection summary ──────────────────────────────────────────────
            w_names = [d["label"] for d in weapon_dets]
            p_str   = f"{len(people_boxes)} person(s)" if people_boxes else ""
            w_str   = f"⚠️ {', '.join(w_names)}"      if w_names    else ""
            o_str   = ", ".join(f"{v}× {k}" for k,v in class_counts.items()
                                if k != "person" and k not in w_names)
            with state_lock:
                state["detection_summary"] = (
                    " | ".join(p for p in [p_str,w_str,o_str] if p)
                    or ("Streaming…" if not yolo_on else "Nothing detected")
                )

            # ── Overlay ────────────────────────────────────────────────────────
            with _mjpeg_clients_lk:
                has_clients = _mjpeg_clients > 0
            if has_clients:
                _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                with engine["frame_lock"]:
                    engine["frame"] = buf.tobytes()
                _new_frame_event.set()

        except torch.cuda.OutOfMemoryError:
            log.error(f"OOM frame #{frame_count}")
            torch.cuda.empty_cache(); torch.cuda.synchronize(); time.sleep(0.5)
        except Exception as e:
            log.error(f"Engine #{frame_count}: {e}")

        time.sleep(max(0, delay-(time.time()-t0)))

    cap.release()
    engine["running"] = False
    with state_lock:
        state["weapon_detections"] = []
        state["source_fps"]        = 0.0
    push_alert("CLEAR", "")
    log.info("[ENGINE] Stopped")


# ── MJPEG ─────────────────────────────────────────────────────────────────────
def mjpeg_gen():
    global _mjpeg_clients
    with _mjpeg_clients_lk:
        _mjpeg_clients += 1
    try:
        while True:
            _new_frame_event.wait(timeout=0.1)
            _new_frame_event.clear()
            with engine["frame_lock"]:
                f = engine["frame"]
            if f:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
    finally:
        with _mjpeg_clients_lk:
            _mjpeg_clients -= 1


def _stop():
    engine["running"] = False
    vlm_abort.set()
    if engine["thread"] and engine["thread"].is_alive():
        engine["thread"].join(timeout=4)
    t_vlm = _vlm_task["thread"]
    if t_vlm and t_vlm.is_alive():
        t_vlm.join(timeout=3)
    vlm_abort.clear()
    with state_lock:
        state.update({
            "alert":"CLEAR","reason":"","person_count":0,
            "weapon_detections":[],"source_fps":0.0,
            "detection_summary":"","scene_description":"",
            "alert_id":"","alert_snapshot_url":"","alert_clip_url":"",
            "alert_trigger_type":"","alert_telegram_status":"disabled",
            "alert_telegram_error":"","vlm_description":"","threat_type":"none",
        })
    with engine["frame_lock"]:
        engine["frame"] = None

def _start(source):
    _stop()
    engine["source"]  = source
    engine["running"] = True
    engine["thread"]  = threading.Thread(target=run_engine,
                                         args=(source,), daemon=True)
    engine["thread"].start()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(mjpeg_gen(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/start/camera")
def start_camera(index: int = 0):
    _start(index); return {"status":"started","source":f"camera:{index}"}

@app.post("/start/video")
async def start_video(file: UploadFile = File(...)):
    p = UPLOAD_DIR / file.filename
    p.write_bytes(await file.read())
    _start(str(p)); return {"status":"started","source":file.filename}

@app.post("/start/path")
def start_path(path: str):
    _start(path); return {"status":"started","source":path}

@app.post("/stop")
def stop():
    _stop(); return {"status":"stopped"}

@app.get("/status")
def get_status():
    with state_lock:
        return {
            "running":           engine["running"],
            "source":            str(engine["source"]),
            "alert":             state["alert"],
            "reason":            state["reason"],
            "description":       state["vlm_description"],
            "threat_type":       state["threat_type"],
            "alert_id":          state["alert_id"],
            "alert_snapshot_url":state["alert_snapshot_url"],
            "alert_clip_url":    state["alert_clip_url"],
            "alert_trigger_type":state["alert_trigger_type"],
            "alert_telegram_status": state["alert_telegram_status"],
            "alert_telegram_error":  state["alert_telegram_error"],
            "scene_description": state["scene_description"],
            "detection_summary": state["detection_summary"],
            "person_count":      state["person_count"],
            "weapon_detections": state["weapon_detections"],
            "source_fps":        state["source_fps"],
            "yolo_enabled":      state["yolo_enabled"],
            "vlm_enabled":       state["vlm_enabled"],
            "vlm_interval":      state["vlm_interval"],
            "mode_switching":    state["mode_switching"],
            "vlm_model_key":     state.get("vlm_model_key", "smolvlm_2b"),
            "vlm_quantization":  state.get("vlm_quantization", "4bit"),
            "trigger_prompts":   dict(state["trigger_prompts"]),
            "vram_pct":          round(get_vram_pct(), 1),
            "telegram":          telegram_notifier.get_public_config(),
        }

@app.get("/alerts")
def get_alerts():
    with state_lock:
        return [
            {k: v for k, v in alert.items() if k not in {"snapshot_path", "clip_path"}}
            for alert in state["alert_log"]
        ]

@app.get("/persons")
def get_persons():
    with state_lock: return state["person_log"]

@app.get("/vram")
def get_vram():
    try:
        p = torch.cuda.get_device_properties(0)
        t = p.total_memory / 1024**3
        return {
            "gpu_name":     p.name,
            "total_gb":     round(t, 2),
            "allocated_gb": round(torch.cuda.memory_allocated()/1024**3, 2),
            "reserved_gb":  round(torch.cuda.memory_reserved()/1024**3,  2),
            "free_gb":      round(t - torch.cuda.memory_reserved()/1024**3, 2),
            "usage_pct":    round(get_vram_pct(), 1),
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/weapon_classes")
def get_weapon_classes():
    return {"threat_model": list(threat_classes.values()),
            "yolo_edge":    list(yolo_edge_classes.values())}

@app.get("/telegram/config")
def get_telegram_config():
    return telegram_notifier.get_public_config()

@app.post("/telegram/config")
def update_telegram_config(payload: TelegramConfigPayload):
    return telegram_notifier.update_config(
        enabled=payload.enabled,
        bot_token=payload.bot_token,
        chat_id=payload.chat_id,
    )

@app.post("/telegram/test")
def send_telegram_test():
    try:
        return {"status": "sent", "config": telegram_notifier.send_test()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

# ── YOLO toggle ────────────────────────────────────────────────────────────────
@app.post("/yolo/enable")
def yolo_enable():
    with state_lock: state["yolo_enabled"] = True
    return {"yolo_enabled": True}

@app.post("/yolo/disable")
def yolo_disable():
    with state_lock:
        state["yolo_enabled"]      = False
        state["weapon_detections"] = []
        state["person_count"]      = 0
        state["detection_summary"] = ""
    return {"yolo_enabled": False}

# ── VLM toggle ─────────────────────────────────────────────────────────────────
@app.post("/vlm/enable")
def vlm_enable():
    with state_lock:
        if state["mode_switching"]:
            return {"error": "Already switching"}
        state["mode_switching"] = True
    def do():
        reload_vlm()
        with state_lock:
            state["vlm_enabled"]    = True
            state["mode_switching"] = False
    threading.Thread(target=do, daemon=True).start()
    return {"vlm_enabled": True, "mode_switching": True}

@app.post("/vlm/disable")
def vlm_disable():
    vlm_abort.set()
    with state_lock:
        state["vlm_enabled"]       = False
        state["scene_description"] = ""
    time.sleep(0.2)
    vlm_abort.clear()
    offload_vlm()
    return {"vlm_enabled": False}

# ── VLM interval (2–30 s) ─────────────────────────────────────────────────────
@app.post("/vlm/interval")
def set_interval(seconds: float):
    seconds = max(1.0, min(seconds, 30.0))
    with state_lock: state["vlm_interval"] = seconds
    return {"vlm_interval": seconds}

# ── VLM model selection ────────────────────────────────────────────────────────
@app.get("/vlm/models")
def get_vlm_models():
    with state_lock:
        current_key   = state.get("vlm_model_key", "smolvlm_2b")
        current_quant = state.get("vlm_quantization", "4bit")
    return {
        "models":               SUPPORTED_MODELS,
        "current_model_key":    current_key,
        "current_quantization": current_quant,
    }

@app.post("/vlm/load")
def load_vlm_model(payload: VlmLoadPayload):
    if payload.model_key not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model key: {payload.model_key}")
    if payload.quantization not in ("4bit", "8bit", "fp16"):
        raise HTTPException(status_code=400, detail="quantization must be 4bit, 8bit, or fp16")
    with state_lock:
        if state["mode_switching"]:
            return {"error": "Already loading a model"}
        state["mode_switching"] = True
    threading.Thread(
        target=load_vlm_with_config,
        args=(payload.model_key, payload.quantization),
        daemon=True,
    ).start()
    return {
        "status":       "loading",
        "model_key":    payload.model_key,
        "quantization": payload.quantization,
    }

# ── Trigger prompts ────────────────────────────────────────────────────────────
@app.get("/trigger_prompts")
def get_trigger_prompts():
    with state_lock: return dict(state["trigger_prompts"])

@app.post("/trigger_prompts/{trigger_type}")
def set_trigger_prompt(trigger_type: str, prompt: str = ""):
    if trigger_type not in ("proximity", "count_change", "weapon"):
        return {"error": "trigger_type must be proximity | count_change | weapon"}
    with state_lock:
        state["trigger_prompts"][trigger_type] = prompt.strip()
    return {"trigger_type": trigger_type, "prompt": prompt.strip()}

@app.delete("/trigger_prompts/{trigger_type}")
def clear_trigger_prompt(trigger_type: str):
    if trigger_type not in ("proximity", "count_change", "weapon"):
        return {"error": "Invalid trigger type"}
    with state_lock:
        state["trigger_prompts"][trigger_type] = ""
    return {"trigger_type": trigger_type, "prompt": ""}


# ── Serve built frontend (must be last — catches all unmatched paths) ─────────
_FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
