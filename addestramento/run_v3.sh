#!/bin/bash
# Run 3 del "rialzati": quattro fasi incatenate, la mano che tira su si toglie a tappe.
# Ogni fase riparte dai pesi della precedente; alla fine di ogni fase due video
# SENZA la mano (da terra e da accovacciato) e le altezze in diario.txt.
# Se il runtime si interrompe, rilanciando riprende dalla prima fase non finita.
#   bash run_v3.sh /content/drive/MyDrive/linuxday/g1_getup_v3
OUT=${1:-/content/drive/MyDrive/linuxday/g1_getup_v3}
MANO=(0.6 0.3 0.1 0.0)
PASSI=(150000000 100000000 100000000 100000000)
PPO="discounting=0.99 entropy_cost=0.01"
mkdir -p "$OUT"
PREV=""
for i in 0 1 2 3; do
  n=$((i+1)); D="$OUT/f$n"
  if [ -f "$D/stages/step_final.pkl" ]; then
    echo "fase $n gia' fatta, salto"; PREV="$D/stages/step_final.pkl"; continue
  fi
  R=""; [ -n "$PREV" ] && R="--restore $PREV"
  echo "$(date '+%d/%m %H:%M') fase $n: mano ${MANO[$i]}, ${PASSI[$i]} passi" | tee -a "$OUT/diario.txt"
  python train_loco.py --env G1Getup --register g1_getup --ppo_env G1JoystickFlatTerrain \
      --timesteps ${PASSI[$i]} --evals 8 --out "$D" --set assist=${MANO[$i]} \
      --ppo_set $PPO $R 2>&1 | grep -E '^\[|Fatto|Traceback|rror:'
  [ -f "$D/stages/step_final.pkl" ] || { echo "fase $n FALLITA" | tee -a "$OUT/diario.txt"; exit 1; }
  PREV="$D/stages/step_final.pkl"
  for P in terra accovacciato; do
    python render_getup.py --out "$D" --stages ${PASSI[$i]} --partenza $P --prove 4 --sheet \
        2>&1 | grep -E '^video|Traceback|rror:'
  done
  { echo "$(date '+%d/%m %H:%M') fase $n finita"; cat "$D/videos_getup/altezze.txt"; } >> "$OUT/diario.txt"
done
echo "$(date '+%d/%m %H:%M') run 3 finita" | tee -a "$OUT/diario.txt"
