import cv2, time, json, threading, logging, uuid
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
from pathlib import Path
from huggingface_hub import hf_hub_download

from notifications import TelegramNotifier, save_alert_artifacts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH = "yolo26n.pt"
VLM_MODEL_ID    = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

SUPPORTED_MODELS = {
    "smolvlm_2b": {
        "id":     "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "name":   "SmolVLM2 2.2B",
        "family": "smolvlm",
        "vram_gb": {"4bit": 3, "8bit": 5, "fp16": 5},
    },
    "qwen_3b": {
        "id":     "Qwen/Qwen2.5-VL-3B-Instruct",
        "name":   "Qwen2.5-VL 3B",
        "family": "qwen",
        "vram_gb": {"4bit": 3, "8bit": 5, "fp16": 7},
    },
    "qwen_7b": {
        "id":     "Qwen/Qwen2.5-VL-7B-Instruct",
        "name":   "Qwen2.5-VL 7B",
        "family": "qwen",
        "vram_gb": {"4bit": 5, "8bit": 8, "fp16": 15},
    },
}

_vlm_model_key    = "smolvlm_2b"
_vlm_model_family = "smolvlm"

THREAT_REPO     = "Subh775/Threat-Detection-YOLOv8n"
THREAT_FILE     = "weights/best.pt"

# Gun (cls 1) removed — too many false positives on furniture/chairs
# Knife (cls 4) only from threat model
# Axe / crowbar / scissors scanned from yolo26n class names
ACTIVE_THREATS  = {"knife"}
YOLO_EDGE_NAMES = {"axe", "crowbar", "scissors", "scissor", "blade", "machete"}

PROXIMITY_DURATION = 2.5
RED_HOLD_SEC       = 6.0
RED_CONFIDENCE     = {"medium", "high"}

WEAPON_COLORS = {
    "knife":   (0,   0,   255),   # red
    "axe":     (0,  128,  255),   # orange
    "crowbar": (0,  200,  150),   # teal
    "scissors":(200, 0,   200),   # magenta
}

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    "alert":             "CLEAR",
    "reason":            "",
    "vlm_description":   "",
    "threat_type":       "none",
    "alert_id":          "",
    "alert_snapshot_url":"",
    "alert_clip_url":    "",
    "alert_trigger_type":"",
    "alert_telegram_status":"disabled",
    "alert_telegram_error":"",
    "last_vlm_time":     0.0,
    "last_red_time":     0.0,
    "alert_log":         [],
    "scene_description": "",
    "detection_summary": "",
    "weapon_detections": [],
    "source_fps":        0.0,
    "person_log":        [],
    "person_count":      0,
    "yolo_enabled":      False,
    "vlm_enabled":       False,
    "vlm_interval":      10.0,       # passive scene interval 2–30s
    "mode_switching":    False,
    "vlm_model_key":     "smolvlm_2b",
    "vlm_quantization":  "4bit",
    "trigger_prompts": {             # per-trigger customizable prompts
        "proximity":    "",
        "count_change": "",
        "weapon":       "",
    },
}
state_lock = threading.Lock()
telegram_notifier = TelegramNotifier()

# Filled after model load
threat_classes: dict = {}    # from threat model  {cls_id: label}
yolo_edge_classes: dict = {} # from yolo26n names  {cls_id: label}

# ── VLM abort event (used by StoppingCriteria) ────────────────────────────────
vlm_abort = threading.Event()

# StoppingCriteria defined at module level — imported lazily so no crash if
# transformers is not installed (vlm_model will be None anyway)
_abort_sc = None
def _init_abort_criteria():
    global _abort_sc
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
        class _AbortCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return vlm_abort.is_set()
        _abort_sc = StoppingCriteriaList([_AbortCriteria()])
        log.info("[VLM] AbortStopCriteria ready")
    except Exception as e:
        log.warning(f"[VLM] StoppingCriteria unavailable: {e}")
        _abort_sc = None


# ── Model loaders ─────────────────────────────────────────────────────────────
def load_yolo():
    log.info("Loading YOLO26n...")
    model = YOLO(YOLO_MODEL_PATH)
    # Scan class names for edge-case weapons (axe, crowbar, scissors…)
    global yolo_edge_classes
    yolo_edge_classes = {
        cid: name
        for cid, name in model.names.items()
        if name.lower() in YOLO_EDGE_NAMES
    }
    if yolo_edge_classes:
        log.info(f"[YOLO] Edge weapon classes found: {yolo_edge_classes}")
    else:
        log.info("[YOLO] No axe/crowbar/scissors in yolo26n — only knife from threat model")
    log.info(f"[YOLO] Ready — {len(model.names)} classes")
    return model


def load_threat_model():
    global threat_classes
    try:
        log.info(f"[THREAT] Downloading {THREAT_REPO}...")
        path  = hf_hub_download(repo_id=THREAT_REPO, filename=THREAT_FILE)
        model = YOLO(path)
        # Only keep classes in ACTIVE_THREATS (knife only — gun excluded)
        threat_classes = {
            cid: name
            for cid, name in model.names.items()
            if name.lower() in ACTIVE_THREATS
        }
        if not threat_classes:
            threat_classes = {4: "knife"}  # known ID fallback
        log.info(f"[THREAT] Loaded. Active: {threat_classes} | Skipped: gun/explosive/grenade")
        return model
    except Exception as e:
        log.warning(f"[THREAT] Download failed: {e} — fallback YOLOv8n COCO")
        model          = YOLO("yolov8n.pt")
        threat_classes = {49: "knife"}
        return model


def load_vlm(model_key: str = "smolvlm_2b", quantization: str = "4bit"):
    global _vlm_model_key, _vlm_model_family
    cfg = SUPPORTED_MODELS.get(model_key)
    if cfg is None:
        log.warning(f"[VLM] Unknown model key '{model_key}' — using smolvlm_2b")
        model_key = "smolvlm_2b"
        cfg       = SUPPORTED_MODELS["smolvlm_2b"]

    model_id = cfg["id"]
    family   = cfg["family"]
    log.info(f"[VLM] Loading {cfg['name']} ({quantization})...")

    try:
        from transformers import AutoProcessor, BitsAndBytesConfig

        if quantization == "4bit":
            bnb      = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            extra_kw = {}
        elif quantization == "8bit":
            bnb      = BitsAndBytesConfig(load_in_8bit=True)
            extra_kw = {}
        else:  # fp16
            bnb      = None
            extra_kw = {"torch_dtype": torch.float16}

        load_kw = {"device_map": "cuda"}
        if bnb is not None:
            load_kw["quantization_config"] = bnb
        load_kw.update(extra_kw)

        proc = AutoProcessor.from_pretrained(model_id)

        if family == "smolvlm":
            from transformers import AutoModelForImageTextToText
            mdl = AutoModelForImageTextToText.from_pretrained(
                model_id, _attn_implementation="eager", **load_kw
            )
        elif family == "qwen":
            from transformers import Qwen2_5_VLForConditionalGeneration
            mdl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, **load_kw
            )
        else:
            log.warning(f"[VLM] Unknown family '{family}'")
            return None, None

        mdl.eval()
        _init_abort_criteria()

        _vlm_model_key    = model_key
        _vlm_model_family = family
        with state_lock:
            state["vlm_model_key"]    = model_key
            state["vlm_quantization"] = quantization

        log.info(f"[VLM] {cfg['name']} ({quantization}) ready")
        return mdl, proc
    except Exception as e:
        log.warning(f"[VLM] Load failed ({e}) — YOLO-only mode")
        return None, None


# ── Default prompts (SmolVLM family) ─────────────────────────────────────────
DEFAULT_SCENE_PROMPT = (
    "You are a surveillance security analyst monitoring a CCTV feed. "
    "Describe what you see, then end with one of: SAFE / SUSPICIOUS / THREATENING. "
    "One to two sentences."
)
DEFAULT_PROXIMITY_PROMPT = (
    "You are a surveillance analyst. Two people are in close physical contact. "
    "Is this SAFE, SUSPICIOUS, or THREATENING? Explain briefly in one sentence."
)
DEFAULT_COUNT_CHANGE_PROMPT = (
    "You are a surveillance analyst. The number of people in the scene just changed. "
    "Describe who entered or left and what they are doing now. One sentence."
)
DEFAULT_WEAPON_PROMPT = (
    "You are a surveillance analyst. A bladed object is visible in this frame. "
    "Reply with HELD_THREATENING, HELD_SAFE, or NOT_HELD. "
    "Then briefly describe what you observe."
)

# ── Qwen-specific prompts (structured, more detailed) ─────────────────────────
QWEN_SCENE_PROMPT = (
    "You are a security camera AI monitoring a CCTV feed. "
    "Describe how many people are present, what each is doing, "
    "and any suspicious behavior or unusual objects. "
    "End with SAFE, SUSPICIOUS, or THREATENING. 1-2 sentences."
)
QWEN_PROXIMITY_PROMPT = (
    "You are a security camera AI. Two people are in close physical contact. "
    "Classify as SAFE, SUSPICIOUS, or THREATENING. "
    "Then describe specifically what you observe about their interaction in one sentence."
)
QWEN_COUNT_CHANGE_PROMPT = (
    "You are a security camera AI. The number of people in the scene just changed. "
    "Describe who entered or left the frame, what direction they moved, "
    "and what everyone in the scene is doing now. One sentence."
)
QWEN_WEAPON_PROMPT = (
    "You are a security camera AI. A bladed weapon or dangerous object is visible. "
    "Reply with HELD_THREATENING, HELD_SAFE, or NOT_HELD. "
    "Then describe: what the object is, who is holding it, and their posture."
)


# ── Model-aware prompt selectors ──────────────────────────────────────────────
def get_scene_prompt() -> str:
    return QWEN_SCENE_PROMPT if _vlm_model_family == "qwen" else DEFAULT_SCENE_PROMPT

def get_proximity_prompt() -> str:
    return QWEN_PROXIMITY_PROMPT if _vlm_model_family == "qwen" else DEFAULT_PROXIMITY_PROMPT

def get_count_change_prompt() -> str:
    return QWEN_COUNT_CHANGE_PROMPT if _vlm_model_family == "qwen" else DEFAULT_COUNT_CHANGE_PROMPT

def get_weapon_prompt() -> str:
    return QWEN_WEAPON_PROMPT if _vlm_model_family == "qwen" else DEFAULT_WEAPON_PROMPT

def get_person_prompt() -> str:
    if _vlm_model_family == "qwen":
        return (
            "Describe this person: their clothing, posture, current action, "
            "and anything suspicious or notable. One sentence."
        )
    return (
        "Describe this person: what they are wearing, what they are doing, "
        "and anything notable about their behavior. One sentence."
    )

def get_temporal_scene_prompt(last_desc: str) -> str:
    last_desc = last_desc[:150].strip()
    if _vlm_model_family == "qwen":
        return (
            f"You are a security camera AI. Previous observation: '{last_desc}'. "
            "What is happening now? Describe any changes and end with SAFE, SUSPICIOUS, or THREATENING. 1-2 sentences."
        )
    return (
        f"Previous: '{last_desc}'. "
        "What is happening now? Note any changes. End with SAFE, SUSPICIOUS, or THREATENING. 1-2 sentences."
    )

def get_temporal_count_prompt(last_desc: str, prev_count: int, cur_count: int) -> str:
    last_desc = last_desc[:120].strip()
    direction = "entered" if cur_count > prev_count else "left"
    if _vlm_model_family == "qwen":
        return (
            f"You are a security camera AI. Previously: '{last_desc}'. "
            f"The person count changed from {prev_count} to {cur_count} — someone {direction}. "
            "Describe what is happening now. One sentence."
        )
    return (
        f"Previous: '{last_desc}'. Count {prev_count}→{cur_count}, someone {direction}. "
        "What is happening now? One sentence."
    )


# ── VLM inference ─────────────────────────────────────────────────────────────
def _qwen_infer(img: Image.Image, prompt: str,
                vlm_model, processor, max_tokens: int = 80,
                deterministic: bool = False) -> str:
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt}
    ]}]
    tp  = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = processor(text=[tp], images=[img], return_tensors="pt").to(vlm_model.device)

    if deterministic:
        gen_kwargs = dict(do_sample=False, max_new_tokens=max_tokens)
    else:
        gen_kwargs = dict(do_sample=True, temperature=0.3,
                          top_p=0.9, max_new_tokens=max_tokens)
    if _abort_sc is not None:
        gen_kwargs["stopping_criteria"] = _abort_sc

    input_len = inp["input_ids"].shape[1]
    with torch.inference_mode():
        out = vlm_model.generate(**inp, **gen_kwargs)
    del inp

    if vlm_abort.is_set():
        log.info("[VLM] Generation aborted — discarding result")
        return ""

    generated_ids = out[:, input_len:]
    res = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    lines = [l.strip() for l in res.split("\n") if l.strip()]
    return lines[0] if lines else res


def smolvlm_infer(crop_bgr: np.ndarray, prompt: str,
                  vlm_model, processor, max_tokens: int = 80,
                  deterministic: bool = False) -> str:
    if vlm_abort.is_set():
        return ""
    if vlm_model is None or processor is None:
        return ""
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 8 or crop_bgr.shape[1] < 8:
        return ""
    try:
        h, w = crop_bgr.shape[:2]

        if _vlm_model_family == "qwen":
            # Qwen2.5-VL supports dynamic high resolution — give it up to 1024px
            scale = 1024 / max(h, w)
            if scale < 1.0:
                crop_bgr = cv2.resize(crop_bgr, (int(w*scale), int(h*scale)),
                                      interpolation=cv2.INTER_AREA)
            img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            return _qwen_infer(img, prompt, vlm_model, processor, max_tokens, deterministic)

        # SmolVLM is trained at 512px — keep original limit
        scale = 512 / max(h, w)
        if scale < 1.0:
            crop_bgr = cv2.resize(crop_bgr, (int(w*scale), int(h*scale)),
                                  interpolation=cv2.INTER_AREA)
        img  = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}
        ]}]
        tp  = processor.apply_chat_template(msgs, add_generation_prompt=True)
        inp = processor(text=tp, images=[img], return_tensors="pt")\
              .to(vlm_model.device, dtype=torch.bfloat16)

        if deterministic:
            gen_kwargs = dict(do_sample=False, max_new_tokens=max_tokens)
        else:
            gen_kwargs = dict(do_sample=True, temperature=0.3,
                              top_p=0.9, max_new_tokens=max_tokens)
        if _abort_sc is not None:
            gen_kwargs["stopping_criteria"] = _abort_sc

        input_len = inp["input_ids"].shape[1]
        with torch.inference_mode():
            out = vlm_model.generate(**inp, **gen_kwargs)

        del inp

        # Discard result if we were aborted mid-generation
        if vlm_abort.is_set():
            log.info("[VLM] Generation aborted — discarding result")
            return ""

        # Decode only the newly generated tokens — avoids prompt contamination
        generated_ids = out[:, input_len:]
        res = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        lines = [l.strip() for l in res.split("\n") if l.strip()]
        return lines[0] if lines else res

    except torch.cuda.OutOfMemoryError:
        log.error("[VLM] OOM — clearing VRAM")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return ""
    except Exception as e:
        log.warning(f"[VLM] Infer error: {e}")
        return ""


def _classify_weapon_text(raw: str) -> dict:
    """Multi-stage classifier: JSON → structured token → semantic fallback."""
    try:
        js = raw[raw.rfind("{"):raw.rfind("}")+1]
        if js:
            parsed = json.loads(js)
            if "description" not in parsed:
                parsed["description"] = raw[:120]
            return parsed
    except Exception:
        pass

    r = raw.lower()
    # Structured token from improved prompt
    if "held_threatening" in r:
        return {"threat": True,  "type": "weapon_held",      "confidence": "high",   "description": raw[:120]}
    if "held_safe" in r:
        return {"threat": False, "type": "weapon_unattended", "confidence": "medium", "description": raw[:120]}
    if "not_held" in r:
        return {"threat": False, "type": "none",              "confidence": "medium", "description": raw[:120]}

    # Semantic fallback
    threat_words = {"threatening", "attack", "assault", "stab", "wield", "armed", "danger", "brandish"}
    safe_words   = {"resting", "table", "ground", "unattended", "not holding", "no one", "empty"}
    threat_score = sum(1 for w in threat_words if w in r)
    safe_score   = sum(1 for w in safe_words   if w in r)
    threat = threat_score > safe_score
    return {
        "threat":      threat,
        "type":        "weapon_held" if threat else "none",
        "confidence":  "medium" if threat_score >= 2 else "low",
        "description": raw[:120],
    }


def run_vlm_threat(crop_bgr: np.ndarray, vlm_model, processor,
                   custom_prompt: str = "") -> dict:
    prompt = custom_prompt or get_weapon_prompt()
    raw    = smolvlm_infer(crop_bgr, prompt, vlm_model, processor,
                           max_tokens=60, deterministic=True)
    if not raw:
        return {"threat": False, "type": "none",
                "confidence": "low", "description": "Aborted or no response"}
    return _classify_weapon_text(raw)


# ── Weapon detection ──────────────────────────────────────────────────────────
def run_weapons(threat_model: YOLO, frame: np.ndarray,
                yolo_extra_boxes: list = None) -> tuple:
    """
    Run threat model (knife only) + optionally pass axe/scissor detections
    already found in yolo_extra_boxes from the main YOLO pass.
    Returns: (detections, trigger_reason, trigger_crop)
    """
    dets       = []
    fh, fw     = frame.shape[:2]
    frame_area = fh * fw

    # ── Threat model (knife) ──────────────────────────────────────────────────
    if threat_model is not None:
        results = threat_model(frame, conf=0.60, imgsz=640, verbose=False)
        for box in results[0].boxes:
            cid = int(box.cls[0])
            if cid not in threat_classes:
                continue
            conf  = float(box.conf[0])
            label = threat_classes[cid]
            xyxy  = box.xyxy[0].cpu().numpy()
            x1,y1,x2,y2 = map(int, xyxy)
            bw, bh = x2-x1, y2-y1
            area   = bw*bh
            aspect = bw / max(bh, 1)
            if area < frame_area*0.015:   continue
            if label == "knife" and not (1.2 < aspect < 8.0): continue
            if bw > fw*0.85 or bh > fh*0.85:  continue
            dets.append({"label": label, "confidence": round(conf,2),
                         "bbox": [x1,y1,x2,y2]})
            log.info(f"[WEAPON] {label} @ {int(conf*100)}%")

    # ── Edge weapons from main YOLO (axe/crowbar/scissors) ───────────────────
    if yolo_extra_boxes:
        dets.extend(yolo_extra_boxes)

    trigger = None
    crop    = None
    if dets:
        best    = max(dets, key=lambda d: d["confidence"])
        trigger = f"Weapon detected: {best['label']} ({int(best['confidence']*100)}%)"
        b       = best["bbox"]
        pad     = 60
        crop    = frame[max(0,b[1]-pad):min(fh,b[3]+pad),
                        max(0,b[0]-pad):min(fw,b[2]+pad)]
    return dets, trigger, crop


def draw_weapons(frame: np.ndarray, dets: list) -> np.ndarray:
    for d in dets:
        x1,y1,x2,y2 = d["bbox"]
        lbl   = f"{d['label']} {int(d['confidence']*100)}%"
        color = WEAPON_COLORS.get(d["label"], (0, 0, 255))
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1,y1-th-8), (x1+tw+6,y1), color, -1)
        cv2.putText(frame, lbl, (x1+3,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return frame


# ── Utilities ─────────────────────────────────────────────────────────────────
def people_close(b1, b2) -> bool:
    c1   = ((b1[0]+b1[2])/2, (b1[1]+b1[3])/2)
    c2   = ((b2[0]+b2[2])/2, (b2[1]+b2[3])/2)
    dist = np.hypot(c1[0]-c2[0], c1[1]-c2[1])
    avg_h = ((b1[3]-b1[1]) + (b2[3]-b2[1])) / 2
    return dist < avg_h * 0.7


def merged_bbox(b1, b2) -> list:
    return [min(b1[0],b2[0]), min(b1[1],b2[1]),
            max(b1[2],b2[2]), max(b1[3],b2[3])]


def pad_crop(frame: np.ndarray, box, pad: int = 60) -> np.ndarray:
    h, w = frame.shape[:2]
    return frame[max(0,int(box[1])-pad):min(h,int(box[3])+pad),
                 max(0,int(box[0])-pad):min(w,int(box[2])+pad)]


def _update_alert_delivery(alert_id: str, status: str, error: str = ""):
    with state_lock:
        for entry in reversed(state["alert_log"]):
            if entry.get("id") == alert_id:
                entry["telegram_status"] = status
                entry["telegram_error"] = error
                entry["telegram_updated_at"] = datetime.now().strftime("%H:%M:%S")
                break

        if state.get("alert_id") == alert_id:
            state["alert_telegram_status"] = status
            state["alert_telegram_error"] = error


def push_alert(
    alert: str,
    reason: str,
    vlm_result: dict = None,
    alert_context: dict | None = None,
):
    alert_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    media = save_alert_artifacts(alert_id, alert_context)
    telegram_cfg = telegram_notifier.get_public_config()
    telegram_status = (
        "pending"
        if alert != "CLEAR" and telegram_cfg["enabled"] and telegram_cfg["configured"]
        else ("disabled" if not telegram_cfg["enabled"] else "not_configured")
    )
    vlm_description = ""
    threat_type = "none"
    if vlm_result:
        vlm_description = vlm_result.get("description", "")
        threat_type = vlm_result.get("type", "none")

    with state_lock:
        state["alert"]  = alert
        state["reason"] = reason
        state["vlm_description"] = vlm_description
        state["threat_type"]     = threat_type
        state["alert_id"] = alert_id if alert != "CLEAR" else ""
        state["alert_snapshot_url"] = media.get("snapshot_url", "") if alert != "CLEAR" else ""
        state["alert_clip_url"] = media.get("clip_url", "") if alert != "CLEAR" else ""
        state["alert_trigger_type"] = (alert_context or {}).get("trigger_type", "") if alert != "CLEAR" else ""
        state["alert_telegram_status"] = telegram_status if alert != "CLEAR" else "disabled"
        state["alert_telegram_error"] = ""
        if alert == "RED":
            state["last_red_time"] = time.time()
        if alert != "CLEAR":
            entry = {
                "id": alert_id,
                "time": datetime.now().strftime("%H:%M:%S"),
                "alert": alert,
                "reason": reason,
                "vlm": vlm_description,
                "threat_type": threat_type,
                "trigger_type": (alert_context or {}).get("trigger_type", ""),
                "source": (alert_context or {}).get("source", ""),
                "snapshot_url": media.get("snapshot_url", ""),
                "clip_url": media.get("clip_url", ""),
                "snapshot_path": media.get("snapshot_path", ""),
                "clip_path": media.get("clip_path", ""),
                "telegram_status": telegram_status,
                "telegram_error": "",
            }
            state["alert_log"].append(entry)
            state["alert_log"] = state["alert_log"][-100:]
    log.info(f"[{alert}] {reason}")

    if alert != "CLEAR":
        latest_entry = {
            "id": alert_id,
            "time": datetime.now().strftime("%H:%M:%S"),
            "alert": alert,
            "reason": reason,
            "vlm": vlm_description,
            "threat_type": threat_type,
            "trigger_type": (alert_context or {}).get("trigger_type", ""),
            "source": (alert_context or {}).get("source", ""),
            "snapshot_path": media.get("snapshot_path", ""),
            "clip_path": media.get("clip_path", ""),
        }
        telegram_notifier.queue_alert(latest_entry, _update_alert_delivery)


class ProximityTracker:
    def __init__(self):
        self.pair_since: dict = {}

    def update(self, boxes: dict):
        ids, now  = list(boxes.keys()), time.time()
        active    = set()
        result    = None
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                a, b = ids[i], ids[j]
                pair = (min(a,b), max(a,b))
                if people_close(boxes[a], boxes[b]):
                    active.add(pair)
                    self.pair_since.setdefault(pair, now)
                    if now - self.pair_since[pair] >= PROXIMITY_DURATION:
                        result = (pair, merged_bbox(boxes[a], boxes[b]))
        for p in list(self.pair_since):
            if p not in active:
                del self.pair_since[p]
        return result
