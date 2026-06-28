import sys
import io
import os
import json
import traceback
import threading


# --------------------------------------------------
# UTF-8固定
# --------------------------------------------------
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")
else:
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", newline="\n")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", newline="\n")
else:
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", newline="\n")


# --------------------------------------------------
# 基本設定
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "dakanji", "model.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "dakanji", "labels.txt")

# デバッグ画像を書き出すなら True
DEBUG_SAVE_IMAGES = False

# 描画設定
CANVAS_SIZE = 256
MARGIN = 20
LINE_WIDTH = 16

# モデル入力サイズ
MODEL_H = 64
MODEL_W = 64

# 古め・低電力CPUでの起動負荷を抑える
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")


# --------------------------------------------------
# ログ / JSON出力
# --------------------------------------------------
def log(*args):
    print(*args, file=sys.stderr, flush=True)


def write_json(obj):
    try:
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    except Exception:
        # stdout 自体が死んでいる場合は何もできない
        pass


def write_event(event_type, **kwargs):
    obj = {"type": event_type}
    obj.update(kwargs)
    write_json(obj)


def emit_progress(percent, phase, message=None):
    obj = {"percent": int(percent), "phase": str(phase)}
    if message is not None:
        obj["message"] = str(message)
    write_event("progress", **obj)


def emit_ready(message=None):
    obj = {"percent": 100, "phase": "ready"}
    if message is not None:
        obj["message"] = str(message)
    write_event("ready", **obj)


def emit_fatal(phase, error, tb=None):
    write_event(
        "fatal",
        percent=0,
        phase=str(phase),
        message=str(error),
        error=str(error),
        traceback=str(tb or ""),
    )


# --------------------------------------------------
# 依存ライブラリ / モデル / ラベル読み込み
# --------------------------------------------------
np = None
Image = None
ImageDraw = None
interpreter = None
input_details = None
output_details = None
infer_lock = threading.Lock()
LABELS = []


def _load_interpreter_class():
    """
    TensorFlow本体には戻さず、LiteRT専用ランタイムを優先して使う。

    以前の TensorFlow fallback は、環境によって
    _pywrap_tensorflow_lite_metrics_wrapper のDLL依存で起動失敗するため廃止。
    ai-edge-litert が無い古い検証環境だけ、TensorFlowを含まない tflite_runtime を
    最後の保険として試す。
    """
    import_errors = []

    try:
        emit_progress(18, "litert_import_start", "LiteRT読込中")
        from ai_edge_litert.interpreter import Interpreter
        emit_progress(32, "litert_imported", "LiteRT読込")
        return Interpreter, "ai_edge_litert"
    except Exception as e:
        import_errors.append(("ai_edge_litert", repr(e)))
        log("ai_edge_litert unavailable:", repr(e))

    try:
        emit_progress(18, "tflite_runtime_import_start", "軽量推論器読込中")
        from tflite_runtime.interpreter import Interpreter
        emit_progress(32, "tflite_runtime_imported", "軽量推論器読込")
        return Interpreter, "tflite_runtime"
    except Exception as e:
        import_errors.append(("tflite_runtime", repr(e)))
        log("tflite_runtime unavailable:", repr(e))

    details = "; ".join(f"{name}: {err}" for name, err in import_errors)
    raise RuntimeError(
        "LiteRT推論器を読み込めませんでした。"
        "TensorFlow本体は使用しない構成に変更済みです。"
        "ビルド環境に ai-edge-litert をインストールしてください。"
        f" details=({details})"
    )


def _create_interpreter(InterpreterClass, model_bytes):
    """
    LiteRT / tflite_runtime の Interpreter を model_content で生成する。

    model_path を直接ネイティブ側へ渡すと、日本語など非ASCIIを含むパスで
    LiteRT側が model.tflite を開けない環境があるため、Python側で先に
    bytes として読み込み、model_content で渡す。

    num_threads 非対応のビルドでも落ちないようにフォールバックする。
    さらに、古いランタイムなどで model_content 自体が非対応だった場合のみ
    最後の保険として model_path に戻す。
    """
    try:
        return InterpreterClass(model_content=model_bytes, num_threads=1)
    except TypeError as e_content_threads:
        try:
            return InterpreterClass(model_content=model_bytes)
        except TypeError as e_content:
            log("model_content unsupported; fallback to model_path:", repr(e_content_threads), repr(e_content))
            try:
                return InterpreterClass(model_path=MODEL_PATH, num_threads=1)
            except TypeError:
                return InterpreterClass(model_path=MODEL_PATH)


def bootstrap():
    global np, Image, ImageDraw, interpreter, input_details, output_details, LABELS

    try:
        emit_progress(3, "python_start", "Python起動")

        emit_progress(6, "numpy_import_start", "NumPy読込中")
        import numpy as _np
        np = _np
        emit_progress(10, "numpy_imported", "NumPy読込")

        emit_progress(12, "pillow_import_start", "画像処理ライブラリ読込中")
        from PIL import Image as _Image, ImageDraw as _ImageDraw
        Image = _Image
        ImageDraw = _ImageDraw
        emit_progress(16, "pillow_imported", "画像処理ライブラリ読込")

        InterpreterClass, interpreter_backend = _load_interpreter_class()
        log("interpreter_backend =", interpreter_backend)

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"model.tflite が見つかりません: {MODEL_PATH}")
        if not os.path.exists(LABELS_PATH):
            raise FileNotFoundError(f"labels.txt が見つかりません: {LABELS_PATH}")

        emit_progress(45, "model_file_checked", "モデル確認")

        with open(MODEL_PATH, "rb") as f:
            model_bytes = f.read()
        emit_progress(52, "model_bytes_loaded", "モデル読込")

        interpreter = _create_interpreter(InterpreterClass, model_bytes)
        emit_progress(62, "interpreter_created", "推論器初期化")

        interpreter.resize_tensor_input(0, [1, MODEL_H, MODEL_W, 1])
        emit_progress(70, "tensor_resized", "入力テンソル設定")

        interpreter.allocate_tensors()
        emit_progress(84, "tensors_allocated", "テンソル確保")

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            text = f.read()

        # 改行だけ除去
        text = text.replace("\r", "").replace("\n", "")
        LABELS = list(text)
        emit_progress(96, "labels_loaded", "ラベル読込")

        log("recognizer started")
        log("MODEL_PATH =", MODEL_PATH)
        log("LABELS_PATH =", LABELS_PATH)
        log("input_details =", input_details)
        log("output_details =", output_details)
        log("labels_count =", len(LABELS))
        log("first_20_labels =", LABELS[:20])
        emit_ready("準備完了")

    except Exception as e:
        tb = traceback.format_exc()
        log("STARTUP_ERROR:", e)
        log(tb)
        emit_fatal("startup_error", e, tb)
        sys.exit(1)


# --------------------------------------------------
# strokes utility
# --------------------------------------------------
def parse_strokes(strokes_raw):
    try:
        return json.loads(strokes_raw) if isinstance(strokes_raw, str) else strokes_raw
    except Exception:
        return []


def calc_bounds(strokes):
    xs, ys = [], []

    for stroke in strokes:
        if not isinstance(stroke, list):
            continue
        for pt in stroke:
            if isinstance(pt, list) and len(pt) >= 2:
                try:
                    x = float(pt[0])
                    y = float(pt[1])
                except Exception:
                    continue
                xs.append(x)
                ys.append(y)

    if not xs or not ys:
        return None

    return min(xs), min(ys), max(xs), max(ys)


def classify_request_kind(strokes):
    xs, ys = [], []

    for stroke in strokes:
        if not isinstance(stroke, list):
            continue
        for pt in stroke:
            if isinstance(pt, list) and len(pt) >= 2:
                try:
                    x = float(pt[0])
                    y = float(pt[1])
                except Exception:
                    continue
                xs.append(x)
                ys.append(y)

    if not xs or not ys:
        return "unknown"

    max_side = max(max(xs), max(ys))
    return "normalized" if max_side <= 256 else "raw"


# --------------------------------------------------
# 描画
# app.py の render_strokes_to_rgba_alpha / preprocess_for_model を流用
# --------------------------------------------------
def render_strokes_to_rgba_alpha(
    strokes,
    canvas_size=CANVAS_SIZE,
    margin=MARGIN,
    line_width=LINE_WIDTH,
):
    """
    透明背景RGBAに白線を描く。
    最終的に alpha チャンネルをグレースケール画像として使う。
    """
    img = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    bounds = calc_bounds(strokes)

    if not bounds:
        return img

    min_x, min_y, max_x, max_y = bounds
    w = max(max_x - min_x, 1.0)
    h = max(max_y - min_y, 1.0)

    usable = canvas_size - margin * 2
    scale = min(usable / w, usable / h)

    scaled_w = w * scale
    scaled_h = h * scale

    offset_x = (canvas_size - scaled_w) / 2.0
    offset_y = (canvas_size - scaled_h) / 2.0

    draw = ImageDraw.Draw(img)

    for stroke in strokes:
        if not isinstance(stroke, list):
            continue

        pts = []
        for pt in stroke:
            if isinstance(pt, list) and len(pt) >= 2:
                try:
                    x = float(pt[0])
                    y = float(pt[1])
                except Exception:
                    continue

                x = (x - min_x) * scale + offset_x
                y = (y - min_y) * scale + offset_y
                pts.append((x, y))

        if len(pts) == 1:
            x, y = pts[0]
            r = line_width / 2.0
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, 255))
        elif len(pts) >= 2:
            draw.line(pts, fill=(255, 255, 255, 255), width=line_width)

    return img


def preprocess_for_model(strokes):
    rgba = render_strokes_to_rgba_alpha(
        strokes=strokes,
        canvas_size=CANVAS_SIZE,
        margin=MARGIN,
        line_width=LINE_WIDTH,
    )

    if DEBUG_SAVE_IMAGES:
        rgba.save(os.path.join(BASE_DIR, "debug_input_rgba.png"))

    # alpha をグレースケール相当として使う
    alpha = rgba.getchannel("A")
    resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR)
    alpha = alpha.resize((MODEL_W, MODEL_H), resampling)

    if DEBUG_SAVE_IMAGES:
        alpha.save(os.path.join(BASE_DIR, "debug_input_alpha_64.png"))

    arr = np.array(alpha).astype(np.float32)

    # app.py に合わせて 0..255 の float32 をそのまま渡す
    arr = arr.reshape(1, MODEL_H, MODEL_W, 1)

    if DEBUG_SAVE_IMAGES:
        vis = np.clip(arr[0, :, :, 0], 0, 255).astype(np.uint8)
        Image.fromarray(vis).save(os.path.join(BASE_DIR, "debug_input_to_model.png"))

    return arr


# --------------------------------------------------
# 推論
# app.py の predict_top_n を stdin/stdout 向けに移植
# --------------------------------------------------
def predict_top_n(strokes, n_best=20):
    x = preprocess_for_model(strokes)

    with infer_lock:
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(output_details[0]["index"]).copy()[0]

    top_idx = np.argsort(y)[::-1][:n_best]
    results = []

    for i in top_idx:
        ch = LABELS[i] if i < len(LABELS) else f"#{i}"
        results.append([ch, float(y[i])])

    log("top_10 =", results[:10])
    return results


# --------------------------------------------------
# コマンド処理
# --------------------------------------------------
def handle_ping(req):
    return {
        "id": req.get("id"),
        "ok": True,
        "pong": True
    }


def handle_recognize(req):
    strokes = parse_strokes(req.get("strokes", []))
    n_best = int(req.get("nBest", 10))
    source = req.get("source", "unknown")

    request_kind = classify_request_kind(strokes)

    log("---- recognize ----")
    log("source =", source)
    log("request_kind =", request_kind)
    log("stroke_count =", len(strokes))
    log("n_best =", n_best)

    scores = predict_top_n(strokes, n_best=n_best)

    return {
        "id": req.get("id"),
        "ok": True,
        "scores": scores
    }


# --------------------------------------------------
# メインループ
# --------------------------------------------------
def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        req = None
        try:
            req = json.loads(line)
            cmd = req.get("cmd")

            if cmd == "ping":
                res = handle_ping(req)
            elif cmd == "recognize":
                res = handle_recognize(req)
            else:
                res = {
                    "id": req.get("id"),
                    "ok": False,
                    "error": f"unknown command: {cmd}"
                }

        except Exception as e:
            log("ERROR:", e)
            log(traceback.format_exc())
            res = {
                "id": req.get("id") if isinstance(req, dict) else None,
                "ok": False,
                "error": str(e)
            }

        write_json(res)


if __name__ == "__main__":
    bootstrap()
    main()
