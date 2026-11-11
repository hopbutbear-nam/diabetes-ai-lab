#!/usr/bin/env bash
set -euo pipefail

if [ -f ".venv/Scripts/activate" ]; then
  source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

CLEAN=false
REPLAY_TEST=false

for arg in "$@"; do
  case "$arg" in
    --clean)
      CLEAN=true
      ;;
    --replay-test)
      REPLAY_TEST=true
      ;;
    *)
      echo "Tham số không hợp lệ: $arg"
      echo "Sử dụng: bash scripts/run_all.sh [--clean] [--replay-test]"
      exit 1
      ;;
  esac
done

if [ "$CLEAN" = true ]; then
  echo "=== [CLEAN] Dọn dẹp artifacts sạch ==="
  for d in models/*/; do
    if [ -d "$d" ]; then
      rm -rf "$d"
    fi
  done
  rm -f data/splits/*/indices.npz
  for f in reports/metrics/*; do
    if [ -f "$f" ] && [ "$(basename "$f")" != "test_lock.json" ]; then
      rm -f "$f"
    elif [ -d "$f" ]; then
      rm -rf "$f"
    fi
  done
  echo "Đã dọn dẹp xong."
fi

echo "=== (1) Kiểm tra môi trường (check_env.py) ==="
python scripts/check_env.py

echo "=== (2) Chuẩn bị dữ liệu (prepare_data.py) ==="
python scripts/prepare_data.py

echo "=== (3) Huấn luyện các mô hình ==="
for v in "" "--dedup"; do
  for t in multiclass binary; do
    if [ "$v" = "--dedup" ] && [ "$t" = "binary" ]; then
      continue
    fi
    for m in majority logreg xgb mlp; do
      echo "Huấn luyện: model=$m, task=$t, variant=${v:-full}"
      python scripts/train.py --config "configs/train_${m}.yaml" --task "$t" $v
    done
  done
done

echo "=== (4) Hiệu chuẩn và ngưỡng ==="
python scripts/calibrate_and_threshold.py --task multiclass
python scripts/calibrate_and_threshold.py --task binary
python scripts/calibrate_and_threshold.py --task multiclass --dedup

echo "=== (5) Giải thích mô hình toàn cục (explain_global.py) ==="
python scripts/explain_global.py

echo "=== (6) Đánh giá mô hình (evaluate.py) ==="
python scripts/evaluate.py --split val
LOCK_FILE="reports/metrics/test_lock.json"
if [ ! -f "$LOCK_FILE" ]; then
  python scripts/evaluate.py
else
  if [ "$REPLAY_TEST" = true ]; then
    echo "Phát lại đánh giá test (--force-reeval --reason 'run_all replay')..."
    python scripts/evaluate.py --force-reeval --reason "run_all replay"
  else
    echo "Đã có khóa đánh giá tại $LOCK_FILE. Bỏ qua evaluate.py (dùng --replay-test nếu muốn ép buộc)."
  fi
fi

echo "=== (7) Xuất mô hình tốt nhất (export_model.py) ==="
python scripts/export_model.py

echo "=== (8) Sinh các bảng báo cáo (make_reports.py) ==="
python scripts/make_reports.py
python scripts/make_reports.py --check

echo "=== (9) Chạy bộ kiểm thử (pytest) ==="
python -m pytest -q

echo "=== Hoàn tất run_all.sh thành công! ==="
