"""Video del G1 che prova a rialzarsi: ogni tentativo parte da terra.

A differenza di render_falls.py (il robot parte in piedi e aspettiamo che cada),
qui ogni episodio comincia con il robot buttato a terra in una posa a caso, e si
guarda cosa combina. Piu' tentativi di fila nello stesso video.

  python render_getup.py --out /content/drive/MyDrive/linuxday/g1_getup \
      --stages 20000000 60000000 100000000 --prove 4 --sheet

Stampa, per ogni video, a che altezza arriva il bacino in ogni tentativo
(0,755 m = in piedi): serve a capire a colpo d'occhio se si e' alzato davvero.
"""
import argparse, glob, os, pickle, re, time

import jax, jax.numpy as jp, mediapy as media, numpy as np

import g1_getup
from render_falls import contact_sheet
from render_loco import inference_factory, FRAME_SKIP

from mujoco_playground import registry


def make_runner(env, make_inf, n_steps):
  """Compila UNA volta: i pesi sono un argomento, cosi' tutte le fasi
  dell'addestramento usano la stessa funzione compilata."""

  def uno(inf, carry, _):
    state, rng = carry
    rng, k = jax.random.split(rng)
    azione, _ = inf(state.obs, k)
    ns = env.step(state, azione)
    return (ns, rng), ns

  @jax.jit
  def run(params, key):
    inf = make_inf(params, deterministic=True)
    rng, k_reset = jax.random.split(key)
    s0 = env.reset(k_reset)
    _, states = jax.lax.scan(lambda c, x: uno(inf, c, x), (s0, rng),
                             jp.arange(n_steps))
    return s0, states

  return run


def una_prova(env, run, params, seed, w=640, h=480):
  s0, states = jax.device_get(run(params, jax.random.PRNGKey(seed)))
  n = states.reward.shape[0]
  traj = [s0] + [jax.tree.map(lambda x: x[i], states)
                 for i in range(FRAME_SKIP - 1, n, FRAME_SKIP)]
  altezze = np.asarray([s.data.qpos[2] for s in traj])
  try:
    frames = env.render(traj, height=h, width=w, camera="track")
  except Exception:
    frames = env.render(traj, height=h, width=w)
  return frames, altezze


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--out", required=True)
  p.add_argument("--impl", default="warp", choices=["warp", "jax"])
  p.add_argument("--stages", type=int, nargs="+", required=True)
  p.add_argument("--prove", type=int, default=4, help="tentativi per video")
  p.add_argument("--seconds", type=float, default=8.0, help="durata di un tentativo")
  p.add_argument("--seed0", type=int, default=0)
  p.add_argument("--tag", default="")
  p.add_argument("--sheet", action="store_true", help="salva anche un provino PNG")
  p.add_argument("--partenza", default="terra", choices=["terra", "accovacciato"],
                 help="da dove parte ogni tentativo")
  p.add_argument("--set", nargs="*", default=[], metavar="CHIAVE=VALORE",
                 help="modifica la configurazione dell'ambiente (la mano resta a 0 se non la si chiede)")
  a = p.parse_args()

  g1_getup.register()
  cfg = registry.get_default_config(g1_getup.NOME)
  cfg.impl = a.impl
  cfg.naconmax = 256          # un robot solo: contatti abbondanti, costa nulla
  cfg.njmax = 29 * 2 + 64 * 4
  cfg.noise_config.level = 0.0
  if a.partenza == "terra":
    cfg.drop_from_height_prob = 1.0   # ogni tentativo parte da terra
  else:
    cfg.drop_from_height_prob = 0.0
  if "crouch_prob" in cfg:            # solo la versione 2 dell'ambiente
    cfg.crouch_prob = 0.0 if a.partenza == "terra" else 1.0
  import ast
  for kv in a.set:
    k, v = kv.split("=", 1)
    cfg[k] = ast.literal_eval(v)
  env = registry.load(g1_getup.NOME, config=cfg)
  _, make_inf = inference_factory(a.out)

  salvate = {int(re.findall(r"step_(\d+)", f)[0]): f
             for f in glob.glob(f"{a.out}/stages/step_*.pkl")
             if re.findall(r"step_(\d+)", f)}
  if not salvate:
    raise SystemExit(f"nessuna fase salvata in {a.out}/stages")
  os.makedirs(f"{a.out}/videos_getup", exist_ok=True)

  n_steps = int(a.seconds / env.dt)
  run = make_runner(env, make_inf, n_steps)
  fps = 1.0 / (env.dt * FRAME_SKIP)

  for target in a.stages:
    step = min(salvate, key=lambda s: abs(s - target))
    params = pickle.load(open(salvate[step], "rb"))
    t0 = time.time()
    frames, picchi = [], []
    for i in range(a.prove):
      f, altezze = una_prova(env, run, params, a.seed0 + i)
      frames += list(f)
      picchi.append((round(float(altezze[0]), 2), round(float(altezze.max()), 2),
                     round(float(altezze[-1]), 2)))
    suff = "" if a.partenza == "terra" else "_acc"
    nome = f"getup_{step/1e6:05.1f}M{suff}{a.tag}"
    media.write_video(f"{a.out}/videos_getup/{nome}.mp4", frames, fps=fps)
    if a.sheet:
      contact_sheet(frames, fps, f"{a.out}/videos_getup/{nome}.png", every=0.5)
    riga = (f"video: {nome}.mp4  bacino (partenza/massimo/fine) per tentativo: "
            f"{picchi}  [in piedi = 0.76]  ({time.time()-t0:.0f}s)")
    print(riga, flush=True)
    # anche su file: l'output di Colab viene troncato e le righe si perdono
    with open(f"{a.out}/videos_getup/altezze.txt", "a") as f:
      f.write(riga + "\n")


if __name__ == "__main__":
  main()
