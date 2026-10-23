"""Visualizzatore dal vivo del G1 con la rete addestrata (MuJoCo CPU, tempo reale).

  .venv/bin/mjpython vis_g1.py            # oppure doppio clic su avvia.command

Tasti (scelti tra quelli che il viewer MuJoCo non usa in modalità passiva):
  Invio        reset: robot in piedi, fermo (riparte con Spazio o Invio)
  Spazio       rete accesa / spenta
  frecce su/giù     avanti / indietro      (passi di 0,25 m/s, max 1)
  frecce sx/dx      gira a sinistra / destra (passi di 0,5 rad/s, max 1)
  8 / 9        di lato a sinistra / destra (passi di 0,25 m/s, max 0,5)
  .  (punto)   stop (comando zero: segna il passo sul posto)
  6            spinta laterale 0,7 m/s (si riprende)    7  spinta forte 1,5 m/s
  E            evoluzione: fase successiva della rete (pesi a caso -> ... -> finale)
Mouse: doppio clic su un corpo, poi Ctrl + trascina col tasto destro = spinta a mano.
"""
import os, time
import numpy as np
import mujoco
import mujoco.viewer

from politica_g1 import G1, Rete, CTRL_DT, QUI

FASI = [("pesi a caso (0 passi)", "step_000000000000.npz"),
        ("10 milioni di passi", "step_000010158080.npz"),
        ("30 milioni di passi", "step_000030474240.npz"),
        ("35 milioni di passi", "step_000035553280.npz"),
        ("40 milioni di passi", "step_000040632320.npz"),
        ("rete finale", "step_final.npz")]

K_ENTER, K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT = 257, 32, 265, 264, 263, 262
K_PUNTO, K_6, K_7, K_8, K_9, K_E = ord('.'), ord('6'), ord('7'), ord('8'), ord('9'), ord('E')

TASTI = ("Invio\nSpazio\nfrecce su/giu\nfrecce sx/dx\n8 / 9\n. (punto)\n6 / 7\nE\nCtrl+tasto dx",
         "reset in piedi\nrete accesa/spenta\navanti / indietro\ngira\ndi lato\nstop\n"
         "spinta / spinta forte\nfase della rete\nspingi col mouse")


def main():
  g = G1(cadute=True)
  reti = [Rete(os.path.join(QUI, "pesi", f)) for _, f in FASI]
  stato = dict(fase=len(FASI) - 1, accesa=False, fermo=True, spinta=1, msg="")
  tasti = []                      # i tasti arrivano dal thread del viewer: li gestisce il ciclo

  def reset(fermo=True):
    g.reset()
    stato["fermo"] = fermo

  def gestisci(k):
    c = g.cmd
    if k == K_ENTER:
      if stato["fermo"]:
        stato["fermo"] = False
      else:
        reset()
    elif k == K_SPACE:
      stato["accesa"] = not stato["accesa"]
      stato["fermo"] = False
    elif k == K_UP:    c[0] = min(c[0] + 0.25, 1.0)
    elif k == K_DOWN:  c[0] = max(c[0] - 0.25, -1.0)
    elif k == K_LEFT:  c[2] = min(c[2] + 0.5, 1.0)
    elif k == K_RIGHT: c[2] = max(c[2] - 0.5, -1.0)
    elif k == K_8:     c[1] = min(c[1] + 0.25, 0.5)
    elif k == K_9:     c[1] = max(c[1] - 0.25, -0.5)
    elif k == K_PUNTO: c[:] = 0
    elif k in (K_6, K_7):
      v = 0.7 if k == K_6 else 1.5
      q = g.d.qpos[3:7]
      yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
      lato = np.array([-np.sin(yaw), np.cos(yaw)])       # asse sinistro del robot
      g.d.qvel[0:2] += stato["spinta"] * v * lato
      stato["msg"] = f"SPINTA {'forte ' if k == K_7 else ''}{'da destra' if stato['spinta'] > 0 else 'da sinistra'}"
      stato["msg_t"] = time.perf_counter()
      stato["spinta"] *= -1
      stato["fermo"] = False
    elif k == K_E:
      cmd = g.cmd.copy()
      stato["fase"] = (stato["fase"] + 1) % len(FASI)
      reset(fermo=False)
      g.cmd[:] = cmd
      stato["accesa"] = True

  with mujoco.viewer.launch_passive(g.m, g.d, key_callback=tasti.append,
                                    show_left_ui=False, show_right_ui=False) as v:
    # camera che segue il bacino (la "track" dell'XML è troppo vicina: taglia la testa).
    # Rotella = zoom, trascina col tasto sinistro = ruota; Esc = camera libera.
    def camera():
      v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
      v.cam.trackbodyid = g.m.body("pelvis").id
      v.cam.distance, v.cam.elevation, v.cam.azimuth = 3.2, -12.0, 135.0
    prossimo = time.perf_counter()
    giri = 0
    ultimi_testi = None
    while v.is_running():
      with v.lock():
        if giri < 25:                 # il viewer all'avvio reimposta la camera: insisti per 0,5 s
          camera()
          giri += 1
        while tasti:
          gestisci(tasti.pop(0))
        if not stato["fermo"]:
          g.step(reti[stato["fase"]] if stato["accesa"] else None)
          if not (np.isfinite(g.d.qpos).all() and np.isfinite(g.d.qvel).all()):
            reset()                                    # fisica esplosa: si riparte in piedi
        c = g.cmd
        if stato["msg"] and time.perf_counter() - stato.get("msg_t", 0) > 1.5:
          stato["msg"] = ""
        rete = f"ACCESA - {FASI[stato['fase']][0]}" if stato["accesa"] else "SPENTA"
        testi = [
            (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, TASTI[0], TASTI[1]),
            (mujoco.mjtFontScale.mjFONTSCALE_200, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
             "Rete\nComando\n" + ("" if not (stato["fermo"] or stato["msg"]) else " "),
             f"{rete}\navanti {c[0]:+.2f}  lato {c[1]:+.2f}  gira {c[2]:+.2f}\n"
             + ("FERMO: Invio per partire" if stato["fermo"] else stato["msg"])),
        ]
      # set_texts va chiamato FUORI dal lock (aspetta il thread grafico: dentro il lock = stallo)
      # e solo quando il testo cambia.
      if testi != ultimi_testi:
        v.set_texts(testi)
        ultimi_testi = testi
      v.sync()
      prossimo += CTRL_DT
      attesa = prossimo - time.perf_counter()
      if attesa > 0:
        time.sleep(attesa)
      elif attesa < -0.2:                              # troppo indietro: non rincorrere
        prossimo = time.perf_counter()


if __name__ == "__main__":
  main()
