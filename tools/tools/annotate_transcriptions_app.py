#!/usr/bin/env python3
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


APP_TITLE = "Transcription Annotator (segmented_dataset.xlsx)"


def _tk_dialogs_supported() -> bool:
    try:
        import tkinter  # noqa: F401
        from tkinter import filedialog  # noqa: F401
        return True
    except Exception:
        return False


def _tk_dialog(kind: str, **kwargs: Any) -> str:
    """
    Use native OS dialogs (runs on the Streamlit server machine).
    kind: 'open_file' | 'open_dir' | 'save_file'
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass

        if kind == "open_file":
            return str(filedialog.askopenfilename(**kwargs) or "")
        if kind == "open_dir":
            return str(filedialog.askdirectory(**kwargs) or "")
        if kind == "save_file":
            return str(filedialog.asksaveasfilename(**kwargs) or "")
        raise ValueError(f"Unknown dialog kind: {kind}")
    finally:
        try:
            root.destroy()
        except Exception:
            pass


@dataclass(frozen=True)
class LoadedDataset:
    df: pd.DataFrame
    source_label: str  # for display only


def _is_probably_path(value: str) -> bool:
    v = (value or "").strip()
    if not v:
        return False
    if v.startswith(("/", "\\")):
        return True
    # Windows drive prefix like C:\... or C:/...
    return len(v) >= 2 and v[1] == ":"


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x)


def _read_excel_from_upload(upload) -> LoadedDataset:
    data = upload.getvalue()
    df = pd.read_excel(BytesIO(data), engine="openpyxl")
    return LoadedDataset(df=df, source_label=f"upload:{upload.name}")


def _read_excel_from_path(path: Path) -> LoadedDataset:
    df = pd.read_excel(path, engine="openpyxl")
    return LoadedDataset(df=df, source_label=str(path))


def _default_column(columns: List[str], preferred: List[str]) -> str:
    cols_lower = {c.lower(): c for c in columns}
    for p in preferred:
        if p.lower() in cols_lower:
            return cols_lower[p.lower()]
    return columns[0] if columns else ""


def _valid_selected_column(current_value: str, columns: List[str], preferred: List[str]) -> str:
    if current_value in columns:
        return current_value
    return _default_column(columns, preferred)


_INT_RE = re.compile(r"\b\d+\b")


def _normalize_numbers_arabic(text: str) -> Tuple[str, Optional[str]]:
    """
    Replaces standalone integers (e.g., '123') with Arabic words.
    Returns (new_text, error_message_if_any).
    """
    text = text or ""
    try:
        from num2words import num2words  # type: ignore
    except Exception as e:
        return text, f"num2words not available: {e}"

    def repl(m: re.Match[str]) -> str:
        s = m.group(0)
        try:
            n = int(s)
        except Exception:
            return s
        try:
            return str(num2words(n, lang="ar"))
        except Exception:
            return s

    return _INT_RE.sub(repl, text), None


def _write_excel_safely(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Temp file + replace to reduce partial writes; if locked, write a timestamped alternative.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    try:
        df.to_excel(tmp, index=False, engine="openpyxl")
        tmp.replace(output_path)
        return output_path
    except PermissionError:
        ts = time.strftime("%Y%m%d_%H%M%S")
        alt = output_path.with_name(f"{output_path.stem}_{ts}{output_path.suffix}")
        df.to_excel(alt, index=False, engine="openpyxl")
        return alt
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def _candidate_audio_paths(audio_root: Path, bucket: str, audio_value: str) -> List[Path]:
    """
    Returns candidate paths for a given audio_value.
    audio_root is expected to be .../segmented_audio (but we keep it flexible).
    """
    audio_value = (audio_value or "").strip().strip('"').strip("'")
    if not audio_value:
        return []

    p_raw = Path(audio_value)
    candidates: List[Path] = []

    # 1) Absolute path as-is.
    if _is_probably_path(audio_value) and p_raw.is_absolute():
        candidates.append(p_raw)
        return candidates

    # Normalize separators without forcing POSIX/Windows semantics
    audio_value_norm = audio_value.replace("\\", "/")
    p_norm = Path(audio_value_norm)

    # 2) audio_root / bucket / audio_value (typical for segmented_audio/A/<file>)
    if bucket:
        candidates.append((audio_root / bucket / p_norm))

    # 3) audio_root / audio_value (if column already omits bucket)
    candidates.append(audio_root / p_norm)

    # 4) audio_root.parent / audio_value (if column contains 'segmented_audio/A/..')
    candidates.append(audio_root.parent / p_norm)

    # 5) If audio_value includes dirs, also try only its basename under root/bucket
    basename = p_norm.name
    if basename and basename != str(p_norm):
        if bucket:
            candidates.append(audio_root / bucket / basename)
        candidates.append(audio_root / basename)

    # Dedup while preserving order
    seen: set[str] = set()
    out: List[Path] = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _resolve_audio_path(audio_root: Path, bucket: str, audio_value: str) -> Tuple[Optional[Path], List[Path]]:
    cands = _candidate_audio_paths(audio_root, bucket, audio_value)
    for p in cands:
        try:
            if p.exists() and p.is_file():
                return p, cands
        except OSError:
            continue
    return None, cands


def _read_audio_bytes(path: Path) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        return path.read_bytes(), None
    except Exception as e:
        return None, repr(e)


def _mime_for_audio_name(name_or_path: str) -> str:
    suffix = Path(name_or_path).suffix.lower()
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix == ".m4a":
        return "audio/mp4"
    if suffix == ".ogg":
        return "audio/ogg"
    if suffix == ".flac":
        return "audio/flac"
    return "audio/*"


def _uploaded_relpath(uploaded_file: Any) -> str:
    """
    Best-effort: return a relative path for uploaded files.
    Directory uploads may provide webkitRelativePath/relative_path; otherwise we fall back to name.
    """
    for attr in ("webkitRelativePath", "webkit_relative_path", "relative_path", "relativePath", "path"):
        try:
            v = getattr(uploaded_file, attr, None)
            if isinstance(v, str) and v.strip():
                return v.strip().replace("\\", "/").lstrip("/")
        except Exception:
            pass
    try:
        v = str(getattr(uploaded_file, "name", "") or "")
        return v.replace("\\", "/").lstrip("/")
    except Exception:
        return ""


def _build_uploaded_audio_index(files: List[Any]) -> Dict[str, Any]:
    """
    Map multiple keys -> UploadedFile for robust lookup.
    Keys include:
      - full relative path (if available)
      - path with leading 'segmented_audio/' stripped
      - basename
      - path with leading '<bucket>/' stripped
    """
    idx: Dict[str, Any] = {}
    for f in files:
        rel = _uploaded_relpath(f)
        if not rel:
            continue
        rel_norm = rel.replace("\\", "/").lstrip("/")

        keys: List[str] = []
        keys.append(rel_norm)
        if rel_norm.lower().startswith("segmented_audio/"):
            keys.append(rel_norm[len("segmented_audio/") :])

        parts = [p for p in rel_norm.split("/") if p]
        if parts:
            keys.append(parts[-1])  # basename
            if len(parts) >= 2 and parts[0] in {"A", "B", "C", "D", "E"}:
                keys.append("/".join(parts[1:]))  # strip bucket

        for k in keys:
            if k and k not in idx:
                idx[k] = f
    return idx


def _buckets_from_uploads(files: List[Any]) -> List[str]:
    base = ["A", "B", "C", "D", "E"]
    found: set[str] = set()
    for f in files:
        rel = _uploaded_relpath(f)
        if not rel:
            continue
        parts = [p for p in rel.replace("\\", "/").split("/") if p]
        if not parts:
            continue
        # Handle either "A/..." or "segmented_audio/A/..."
        if parts[0] in base:
            found.add(parts[0])
        elif len(parts) >= 2 and parts[0].lower() == "segmented_audio" and parts[1] in base:
            found.add(parts[1])
    present = [b for b in base if b in found]
    return present or base


def _resolve_uploaded_audio(
    uploaded_idx: Dict[str, Any], bucket: str, audio_value: str
) -> Optional[Any]:
    audio_value = (audio_value or "").strip().strip('"').strip("'")
    if not audio_value:
        return None

    audio_norm = audio_value.replace("\\", "/").lstrip("/")
    p = Path(audio_norm)
    basename = p.name

    candidates: List[str] = []
    # Common manifest values include: "segmented_audio/A/foo.flac" or "A/foo.flac" or "foo.flac"
    candidates.append(audio_norm)
    if audio_norm.lower().startswith("segmented_audio/"):
        candidates.append(audio_norm[len("segmented_audio/") :])
    if bucket:
        candidates.append(f"{bucket}/{audio_norm}")
        candidates.append(f"{bucket}/{basename}")
        candidates.append(f"segmented_audio/{bucket}/{basename}")
    candidates.append(basename)

    for c in candidates:
        f = uploaded_idx.get(c)
        if f is not None:
            return f
    return None


def _list_bucket_options(audio_root: Path) -> List[str]:
    # Requirement: pick from A–E, but also show what exists if different.
    base = ["A", "B", "C", "D", "E"]
    try:
        if audio_root.exists():
            existing = sorted([p.name for p in audio_root.iterdir() if p.is_dir()])
        else:
            existing = []
    except Exception:
        existing = []

    # Prefer A–E order, then any extra folders.
    extra = [x for x in existing if x not in base]
    present = [x for x in base if x in existing] or base
    return present + extra


def _set_state_default(key: str, value: Any) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    _set_state_default("row_idx", 0)
    _set_state_default("autosave", True)
    _set_state_default("audio_root", "")
    _set_state_default("bucket", "A")
    _set_state_default("output_path", "")
    _set_state_default("dataset_mode", "path")
    _set_state_default("excel_path", "")
    _set_state_default("transcription_col", "")
    _set_state_default("audio_col", "")
    _set_state_default("excel_source_label", "")
    _set_state_default("dataset_id", "")
    _set_state_default("df", None)
    _set_state_default("df_original", None)
    _set_state_default("audio_source", "local_path")  # local_path | upload_folder
    _set_state_default("uploaded_audio_files", [])
    _set_state_default("uploaded_audio_index", {})

    with st.sidebar:
        st.header("Inputs")
        tk_ok = _tk_dialogs_supported()
        if not tk_ok:
            st.caption("Native file/folder picker not available (tkinter missing). You can still paste paths manually or use upload mode.")

        def _browse_excel_cb() -> None:
            if not _tk_dialogs_supported():
                return
            initialdir = ""
            try:
                p0 = Path((st.session_state.get("excel_path") or "").strip()).expanduser()
                if p0.is_file():
                    initialdir = str(p0.parent)
                elif p0.is_dir():
                    initialdir = str(p0)
            except Exception:
                initialdir = ""
            chosen = _tk_dialog(
                "open_file",
                title="Select segmented_dataset.xlsx",
                initialdir=initialdir or str(Path.cwd()),
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            )
            if chosen:
                st.session_state["excel_path"] = chosen

        def _browse_audio_root_cb() -> None:
            if not _tk_dialogs_supported():
                return
            initialdir = ""
            try:
                p0 = Path((st.session_state.get("audio_root") or "").strip()).expanduser()
                if p0.is_dir():
                    initialdir = str(p0)
                elif p0.is_file():
                    initialdir = str(p0.parent)
            except Exception:
                initialdir = ""
            chosen = _tk_dialog(
                "open_dir",
                title="Select segmented_audio folder",
                initialdir=initialdir or str(Path.cwd()),
                mustexist=True,
            )
            if chosen:
                st.session_state["audio_root"] = chosen

        def _browse_output_cb() -> None:
            if not _tk_dialogs_supported():
                return
            initialdir = ""
            initialfile = "annotated_dataset.xlsx"
            try:
                p0 = Path((st.session_state.get("output_path") or "").strip()).expanduser()
                if p0.is_dir():
                    initialdir = str(p0)
                elif str(p0):
                    initialdir = str(p0.parent)
                    if p0.name:
                        initialfile = p0.name
            except Exception:
                initialdir = ""
            chosen = _tk_dialog(
                "save_file",
                title="Choose output Excel file",
                initialdir=initialdir or str(Path.cwd()),
                initialfile=initialfile,
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            )
            if chosen:
                st.session_state["output_path"] = chosen

        dataset_mode = st.radio(
            "Load dataset from",
            options=["path", "upload"],
            index=0 if st.session_state["dataset_mode"] == "path" else 1,
            horizontal=True,
        )
        st.session_state["dataset_mode"] = dataset_mode

        loaded: Optional[LoadedDataset] = None
        excel_source_label = ""

        if dataset_mode == "upload":
            upload = st.file_uploader("Upload segmented_dataset.xlsx", type=["xlsx"])
            if upload is not None:
                try:
                    loaded = _read_excel_from_upload(upload)
                    excel_source_label = loaded.source_label
                except Exception as e:
                    st.error(f"Failed reading Excel upload: {e}")
        else:
            excel_path_str = st.text_input(
                "Path to segmented_dataset.xlsx",
                key="excel_path",
                placeholder=r"C:\...\Collected\segmented_dataset.xlsx",
            )
            st.button(
                "Browse…",
                use_container_width=True,
                disabled=not tk_ok,
                key="browse_excel",
                on_click=_browse_excel_cb,
            )
            if excel_path_str.strip():
                p = Path(excel_path_str).expanduser()
                if not p.is_absolute():
                    p = (Path.cwd() / p).resolve()
                if p.exists() and p.is_file():
                    try:
                        loaded = _read_excel_from_path(p)
                        excel_source_label = loaded.source_label
                    except Exception as e:
                        st.error(f"Failed reading Excel: {e}")
                else:
                    st.warning("Excel path not found yet.")

        st.divider()
        st.header("Audio")
        audio_source = st.radio(
            "Audio source",
            options=["local_path", "upload_folder"],
            format_func=lambda x: "Local folder path (recommended)" if x == "local_path" else "Select folder (uploads segmented_audio)",
            index=0 if st.session_state["audio_source"] == "local_path" else 1,
        )
        st.session_state["audio_source"] = audio_source

        uploaded_audio_files: List[Any] = []
        uploaded_audio_index: Dict[str, Any] = {}

        if audio_source == "upload_folder":
            st.caption("This uses your browser’s folder picker and uploads the selected files into the app session.")
            try:
                # Newer Streamlit versions may support directory upload via accept_multiple_files='directory'.
                uploaded_audio_files = st.file_uploader(
                    "Select segmented_audio folder (A/B/C/D/E)",
                    type=["wav", "flac", "mp3", "m4a", "ogg"],
                    accept_multiple_files="directory",  # type: ignore[arg-type]
                    key="uploaded_audio_dir",
                ) or []
            except TypeError:
                uploaded_audio_files = st.file_uploader(
                    "Select segmented_audio files (select all in A/B/C/D/E)",
                    type=["wav", "flac", "mp3", "m4a", "ogg"],
                    accept_multiple_files=True,
                    key="uploaded_audio_files",
                ) or []

            if uploaded_audio_files:
                uploaded_audio_index = _build_uploaded_audio_index(uploaded_audio_files)
                st.session_state["uploaded_audio_files"] = uploaded_audio_files
                st.session_state["uploaded_audio_index"] = uploaded_audio_index
                st.success(f"Loaded {len(uploaded_audio_files)} audio file(s) into the session.")
            else:
                st.session_state["uploaded_audio_files"] = []
                st.session_state["uploaded_audio_index"] = {}

        audio_root_str = st.text_input(
            "Path to segmented_audio folder",
            key="audio_root",
            placeholder=r"C:\...\Collected\segmented_audio",
            disabled=(audio_source == "upload_folder"),
        )
        st.button(
            "Browse…",
            use_container_width=True,
            disabled=(not tk_ok) or (audio_source == "upload_folder"),
            key="browse_audio_root",
            on_click=_browse_audio_root_cb,
        )
        st.caption(
            "Tip (Windows): in File Explorer, click the address bar, copy the folder path, and paste it here."
        )
        audio_root = Path(audio_root_str).expanduser() if audio_root_str.strip() else None
        if audio_root is not None and not audio_root.is_absolute():
            audio_root = (Path.cwd() / audio_root).resolve()

        if audio_source == "upload_folder":
            bucket_options = _buckets_from_uploads(st.session_state.get("uploaded_audio_files") or [])
        else:
            bucket_options = _list_bucket_options(audio_root) if audio_root is not None else ["A", "B", "C", "D", "E"]
        bucket = st.selectbox(
            "Choose audio bucket",
            options=bucket_options,
            index=bucket_options.index(st.session_state["bucket"]) if st.session_state["bucket"] in bucket_options else 0,
        )
        st.session_state["bucket"] = bucket

        st.divider()
        st.header("Export")
        autosave = st.checkbox("Auto-save on Next/Prev", value=bool(st.session_state["autosave"]))
        st.session_state["autosave"] = autosave

        # If output path is empty, set a default next to the Excel (if possible).
        # Must happen before the widget with key="output_path" is instantiated.
        if not (st.session_state.get("output_path") or "").strip():
            label = excel_source_label or str(st.session_state.get("excel_path") or "")
            st.session_state["output_path"] = str(_default_output_path(label))

        output_path_str = st.text_input(
            "Output Excel path",
            key="output_path",
            placeholder=r"C:\...\Collected\annotated_dataset.xlsx",
        )
        st.button(
            "Browse…",
            use_container_width=True,
            disabled=not tk_ok,
            key="browse_output",
            on_click=_browse_output_cb,
        )

    if loaded is None:
        st.info("Load `segmented_dataset.xlsx` from the sidebar to start.")
        return
    st.session_state["excel_source_label"] = excel_source_label

    # Persist edits across Streamlit reruns by keeping the dataframe in session_state.
    # If the dataset source changes, reset state to the newly loaded data.
    if st.session_state.get("dataset_id") != excel_source_label or st.session_state.get("df") is None:
        st.session_state["dataset_id"] = excel_source_label
        st.session_state["df"] = loaded.df.copy()
        st.session_state["df_original"] = loaded.df.copy()
        st.session_state["row_idx"] = 0

    df: pd.DataFrame = st.session_state["df"]
    df_original: Optional[pd.DataFrame] = st.session_state.get("df_original")
    if df.empty:
        st.warning("Excel loaded, but it has no rows.")
        return

    cols = list(df.columns.astype(str))
    if not cols:
        st.error("Excel loaded, but it has no columns.")
        return

    # Column selectors
    preferred_transcription = ["segment_transcription", "transcription", "text"]
    preferred_audio = ["segment_audio_file", "audio_file", "audio", "path"]

    st.session_state["transcription_col"] = _valid_selected_column(
        st.session_state.get("transcription_col", ""),
        cols,
        preferred_transcription,
    )
    st.session_state["audio_col"] = _valid_selected_column(
        st.session_state.get("audio_col", ""),
        cols,
        preferred_audio,
    )

    top_left, top_right = st.columns([2, 1])
    with top_left:
        st.subheader("Dataset")
        st.caption(f"Loaded: `{excel_source_label}` | rows={len(df)}")
    with top_right:
        with st.container():
            transcription_col = st.selectbox(
                "Transcription column",
                options=cols,
                index=cols.index(st.session_state["transcription_col"]),
            )
            st.session_state["transcription_col"] = transcription_col
            audio_col = st.selectbox(
                "Audio column",
                options=cols,
                index=cols.index(st.session_state["audio_col"]),
            )
            st.session_state["audio_col"] = audio_col

    n = len(df)
    idx = int(st.session_state["row_idx"])
    idx = max(0, min(n - 1, idx))
    st.session_state["row_idx"] = idx

    row = df.iloc[idx]
    transcription_col = st.session_state["transcription_col"]
    audio_col = st.session_state["audio_col"]

    current_text = _safe_str(row.get(transcription_col))
    original_text = ""
    if isinstance(df_original, pd.DataFrame) and transcription_col in df_original.columns:
        try:
            original_text = _safe_str(df_original.iloc[idx].get(transcription_col))
        except Exception:
            original_text = ""

    audio_value = _safe_str(row.get(st.session_state["audio_col"]))

    st.divider()
    header_cols = st.columns([1, 2, 2])
    with header_cols[0]:
        st.metric("Row", f"{idx + 1} / {n}")
    with header_cols[1]:
        jump = st.number_input("Jump to row #", min_value=1, max_value=n, value=idx + 1, step=1)
        if int(jump) != idx + 1:
            st.session_state["row_idx"] = int(jump) - 1
            st.rerun()
    with header_cols[2]:
        st.write("")

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Text")
        if original_text and original_text != current_text:
            st.text_area("Original (when loaded)", value=original_text, height=100, disabled=True)

        # Key includes row index so the widget refreshes when you navigate.
        new_text = st.text_area(
            f"Edit `{transcription_col}`",
            value=current_text,
            key=f"edit_{idx}_{transcription_col}",
            height=220,
            help="This edits the existing transcription column in-memory; export writes it to annotated_dataset.xlsx.",
        )

        action_cols = st.columns([1, 1, 1, 1, 1])
        with action_cols[0]:
            if st.button("Normalize numbers (Arabic)", use_container_width=True):
                normalized, err = _normalize_numbers_arabic(new_text)
                if err:
                    st.warning(err)
                df.at[df.index[idx], transcription_col] = normalized
                st.rerun()
        with action_cols[1]:
            if st.button("Prev", use_container_width=True, disabled=(idx <= 0)):
                df.at[df.index[idx], transcription_col] = new_text
                st.session_state["row_idx"] = max(0, idx - 1)
                if st.session_state["autosave"]:
                    _maybe_autosave(df)
                st.rerun()
        with action_cols[2]:
            if st.button("Next", use_container_width=True, disabled=(idx >= n - 1)):
                df.at[df.index[idx], transcription_col] = new_text
                st.session_state["row_idx"] = min(n - 1, idx + 1)
                if st.session_state["autosave"]:
                    _maybe_autosave(df)
                st.rerun()
        with action_cols[3]:
            if st.button("Save/export now", use_container_width=True):
                df.at[df.index[idx], transcription_col] = new_text
                written = _export_now(df, excel_source_label)
                if written is not None:
                    st.success(f"Wrote: {written}")
        with action_cols[4]:
            st.write("")

    with right:
        st.subheader("Audio")
        if not audio_value:
            st.info("Audio cell is empty for this row.")
        else:
            st.code(audio_value, language=None)

        audio_source = st.session_state.get("audio_source") or "local_path"
        if audio_source == "upload_folder":
            uploaded_idx: Dict[str, Any] = st.session_state.get("uploaded_audio_index") or {}
            if not uploaded_idx:
                st.info("Use the sidebar to select the `segmented_audio` folder for upload.")
            else:
                uf = _resolve_uploaded_audio(uploaded_idx, bucket, audio_value)
                if uf is None:
                    st.warning("Audio not found in uploaded folder for this row.")
                else:
                    try:
                        b = uf.getvalue()
                        st.audio(b, format=_mime_for_audio_name(_uploaded_relpath(uf) or getattr(uf, "name", "")))
                    except Exception as e:
                        st.error(f"Failed reading uploaded audio: {e}")
        else:
            if audio_root is None or not str(audio_root):
                st.info("Set `segmented_audio` path in the sidebar to enable playback.")
            else:
                resolved, cands = _resolve_audio_path(audio_root, bucket, audio_value)
                if resolved is None:
                    st.warning("Audio file not found. Tried:")
                    for c in cands[:8]:
                        st.write(f"- `{c}`")
                else:
                    st.caption(f"Resolved: `{resolved}`")
                    audio_bytes, err = _read_audio_bytes(resolved)
                    if err:
                        st.error(f"Failed reading audio: {err}")
                    elif audio_bytes is None:
                        st.error("Failed reading audio (unknown error).")
                    else:
                        st.audio(audio_bytes, format=_mime_for_audio_name(str(resolved)))


def _default_output_path(excel_source_label: str) -> Path:
    # If user provided a real path, write next to it; else write to CWD.
    if excel_source_label and not excel_source_label.startswith("upload:"):
        p = Path(excel_source_label)
        if p.is_file():
            return p.with_name("annotated_dataset.xlsx")
        if p.suffix.lower() == ".xlsx":
            return p.with_name("annotated_dataset.xlsx")
    return (Path.cwd() / "annotated_dataset.xlsx").resolve()


def _export_now(df: pd.DataFrame, excel_source_label: str) -> Optional[Path]:
    out_str = (st.session_state.get("output_path") or "").strip()
    if out_str:
        out = Path(out_str).expanduser()
        if not out.is_absolute():
            out = (Path.cwd() / out).resolve()
    else:
        out = _default_output_path(excel_source_label)

    try:
        written = _write_excel_safely(df, out)
        return written
    except Exception as e:
        st.error(f"Export failed: {e}")
        return None


def _maybe_autosave(df: pd.DataFrame) -> None:
    excel_source_label = str(st.session_state.get("excel_source_label") or "")
    _export_now(df, excel_source_label)


if __name__ == "__main__":
    main()

