import logging
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

MEDIA_ROOT = Path("media")
ALERT_MEDIA_DIR = MEDIA_ROOT / "alerts"
ALERT_MEDIA_DIR.mkdir(parents=True, exist_ok=True)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def mask_secret(value: str, keep: int = 4) -> str:
    if not value:
        return ""
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}{'*' * max(4, len(value) - keep * 2)}{value[-keep:]}"


def prepare_media_frame(frame: np.ndarray | None, max_width: int = 640) -> np.ndarray | None:
    if frame is None or not hasattr(frame, "size") or frame.size == 0:
        return None

    prepared = frame.copy()
    if prepared.ndim == 2:
        prepared = cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR)

    height, width = prepared.shape[:2]
    if width > max_width:
        scale = max_width / float(width)
        prepared = cv2.resize(
            prepared,
            (max(2, int(width * scale)), max(2, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    out_h, out_w = prepared.shape[:2]
    if out_w % 2:
        prepared = prepared[:, :-1]
    if out_h % 2:
        prepared = prepared[:-1, :]

    return prepared


def create_alert_context(
    *,
    snapshot_frame: np.ndarray | None = None,
    clip_frames: list[np.ndarray] | None = None,
    clip_fps: float = 10.0,
    trigger_type: str = "",
    source: str = "",
) -> dict:
    snapshot = prepare_media_frame(snapshot_frame)
    prepared_clip = []
    for frame in clip_frames or []:
        item = prepare_media_frame(frame)
        if item is not None:
            prepared_clip.append(item)

    return {
        "snapshot_frame": snapshot,
        "clip_frames": prepared_clip,
        "clip_fps": max(1.0, float(clip_fps or 10.0)),
        "trigger_type": trigger_type,
        "source": source,
    }


def save_alert_artifacts(alert_id: str, context: dict | None) -> dict:
    if not context:
        return {}

    timestamp = datetime.now().strftime("%Y%m%d")
    target_dir = ALERT_MEDIA_DIR / timestamp
    target_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    snapshot = context.get("snapshot_frame")
    clip_frames = context.get("clip_frames") or []

    if snapshot is not None:
        snapshot_path = target_dir / f"{alert_id}_snapshot.jpg"
        ok, encoded = cv2.imencode(
            ".jpg",
            snapshot,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
        if ok:
            snapshot_path.write_bytes(encoded.tobytes())
            files["snapshot_path"] = str(snapshot_path)
            files["snapshot_url"] = f"/media/{snapshot_path.relative_to(MEDIA_ROOT).as_posix()}"

    if clip_frames:
        base_h, base_w = clip_frames[0].shape[:2]
        clip_path = target_dir / f"{alert_id}_clip.mp4"
        writer = cv2.VideoWriter(
            str(clip_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(context.get("clip_fps", 10.0)),
            (base_w, base_h),
        )
        try:
            if writer.isOpened():
                for frame in clip_frames:
                    if frame.shape[:2] != (base_h, base_w):
                        frame = cv2.resize(frame, (base_w, base_h), interpolation=cv2.INTER_AREA)
                    writer.write(frame)
        finally:
            writer.release()

        if clip_path.exists() and clip_path.stat().st_size > 0:
            files["clip_path"] = str(clip_path)
            files["clip_url"] = f"/media/{clip_path.relative_to(MEDIA_ROOT).as_posix()}"
        elif clip_path.exists():
            clip_path.unlink(missing_ok=True)

    return files


class TelegramNotifier:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session = requests.Session()
        self._config = {
            "enabled": _env_flag("TELEGRAM_ENABLED", False),
            "bot_token": os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            "chat_id": os.getenv("TELEGRAM_CHAT_ID", "").strip(),
            "last_error": "",
            "last_success_at": "",
        }

    def get_public_config(self) -> dict:
        with self._lock:
            bot_token = self._config["bot_token"]
            chat_id = self._config["chat_id"]
            return {
                "enabled": self._config["enabled"],
                "configured": bool(bot_token and chat_id),
                "bot_token_masked": mask_secret(bot_token),
                "chat_id": chat_id,
                "last_error": self._config["last_error"],
                "last_success_at": self._config["last_success_at"],
            }

    def update_config(
        self,
        *,
        enabled: bool | None = None,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ) -> dict:
        with self._lock:
            if enabled is not None:
                self._config["enabled"] = bool(enabled)
            if bot_token is not None and bot_token.strip():
                self._config["bot_token"] = bot_token.strip()
            if chat_id is not None and chat_id.strip():
                self._config["chat_id"] = chat_id.strip()
            self._config["last_error"] = ""
        return self.get_public_config()

    def queue_alert(self, entry: dict, update_entry: Callable[[str, str, str], None]) -> None:
        thread = threading.Thread(
            target=self._send_alert,
            args=(dict(entry), update_entry),
            daemon=True,
        )
        thread.start()

    def send_test(self) -> dict:
        self._send_message("Telegram test message from CCTV Surveillance.")
        with self._lock:
            self._config["last_error"] = ""
            self._config["last_success_at"] = datetime.now().isoformat(timespec="seconds")
        return self.get_public_config()

    def _send_alert(self, entry: dict, update_entry: Callable[[str, str, str], None]) -> None:
        alert_id = entry["id"]
        try:
            with self._lock:
                enabled = self._config["enabled"]
                configured = bool(self._config["bot_token"] and self._config["chat_id"])

            if not enabled:
                update_entry(alert_id, "disabled", "")
                return
            if not configured:
                update_entry(alert_id, "not_configured", "Telegram bot token/chat ID missing.")
                return

            caption = self._build_caption(entry, limit=900)
            short_caption = f"{entry.get('alert', 'Alert')} clip | {entry.get('time', '')}".strip(" |")

            snapshot_path = entry.get("snapshot_path")
            clip_path = entry.get("clip_path")

            if snapshot_path and Path(snapshot_path).exists():
                with Path(snapshot_path).open("rb") as photo_file:
                    self._post(
                        "sendPhoto",
                        data={"caption": caption},
                        files={"photo": photo_file},
                    )
            else:
                self._send_message(caption)

            if clip_path and Path(clip_path).exists():
                try:
                    with Path(clip_path).open("rb") as video_file:
                        self._post(
                            "sendVideo",
                            data={"caption": short_caption, "supports_streaming": "true"},
                            files={"video": video_file},
                        )
                except Exception:
                    with Path(clip_path).open("rb") as clip_file:
                        self._post(
                            "sendDocument",
                            data={"caption": short_caption},
                            files={"document": clip_file},
                        )

            with self._lock:
                self._config["last_error"] = ""
                self._config["last_success_at"] = datetime.now().isoformat(timespec="seconds")
            update_entry(alert_id, "sent", "")
        except Exception as exc:
            message = str(exc)
            with self._lock:
                self._config["last_error"] = message
            update_entry(alert_id, "error", message)
            log.warning("[Telegram] %s", message)

    def _build_caption(self, entry: dict, limit: int = 900) -> str:
        lines = [
            "CCTV Alert",
            f"Level: {entry.get('alert', 'UNKNOWN')}",
            f"Time: {entry.get('time', '')}",
            f"Reason: {entry.get('reason', '')}",
        ]
        if entry.get("trigger_type"):
            lines.append(f"Trigger: {entry['trigger_type']}")
        if entry.get("threat_type") and entry["threat_type"] != "none":
            lines.append(f"Threat: {entry['threat_type']}")
        if entry.get("vlm"):
            lines.append(f"Description: {entry['vlm']}")
        if entry.get("source"):
            lines.append(f"Source: {entry['source']}")

        caption = "\n".join(line for line in lines if line.strip())
        return caption[:limit].rstrip()

    def _send_message(self, text: str) -> None:
        self._post("sendMessage", data={"text": text[:3500]})

    def _post(self, method: str, *, data: dict | None = None, files: dict | None = None) -> dict:
        with self._lock:
            token = self._config["bot_token"]
            chat_id = self._config["chat_id"]

        if not token or not chat_id:
            raise RuntimeError("Telegram bot token/chat ID missing.")

        payload = {"chat_id": chat_id}
        if data:
            payload.update(data)

        response = self._session.post(
            f"https://api.telegram.org/bot{token}/{method}",
            data=payload,
            files=files,
            timeout=30,
        )
        response.raise_for_status()

        body = response.json()
        if not body.get("ok"):
            raise RuntimeError(body.get("description") or "Telegram API request failed.")
        return body

