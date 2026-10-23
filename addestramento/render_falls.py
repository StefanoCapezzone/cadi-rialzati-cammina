"""Video "cadute" del G1: il robot parte fermo, prova a camminare in avanti e,
quando cade, RESTA A TERRA per qualche secondo (motori fermi nella posa di riposo),
poi riparte da capo. Usa il modello con il corpo che tocca il pavimento (g1_falls).

  python render_falls.py --out /content/drive/MyDrive/linuxday/g1_walk --stages 0 10000000 20000000
  (--stages: passi approssimativi; per ciascuno si sceglie la fase salvata più vicina)
Stampa per ogni video quante cadute ci sono e quando.
"""
import argparse, glob, os, pickle, re, time
import jax, jax.numpy as jp, mediapy as media, numpy as np

import g1_falls
from mujoco_playground import registry
from render_loco import inference_factory

FRAME_SKIP = 2


def make_runner(env, make_inf, seconds, lie_seconds, stand_seconds, speed):
  """Compila UNA volta la simulazione; i pesi (params) sono un argomento, così
  le varie fasi dell'addestramento non richiedono una nuova compilazione."""
  dt = env.dt
  n, lie, stand = int(seconds / dt), int(lie_seconds / dt), int(stand_seconds / dt)
  cmds = jp.array([[0.0, 0.0, 0.0] if i < stand else [speed, 0.0, 0.0] for i in range(n)])

  def one(inf, carry, cmd):
    state, rng, ft, tsince = carry
    # all'inizio di ogni tentativo il robot sta fermo "stand" passi, poi cammina
    c = jp.where(tsince < stand, jp.zeros(3), cmd)
    state = state.replace(info={**state.info, "command": c})
    rng, k, kr = jax.random.split(rng, 3)
    a, _ = inf(state.obs, k)
    a = jp.where(ft > 0, jp.zeros_like(a), a)          # a terra: motori nella posa di riposo
    ns = env.step(state, a)
    fell_now = (ns.done > 0) & (ft == 0)
    ft2 = jp.where(ft > 0, ft + 1, jp.where(fell_now, 1, 0))
    again = ft2 > lie                                    # dopo "lie" passi a terra: nuovo tentativo
    rs = env.reset(kr)
    ns = jax.tree.map(lambda x, y: jp.where(again, x, y), rs, ns)
    ft2 = jp.where(again, 0, ft2)
    tsince2 = jp.where(again, 0, tsince + 1)
    return (ns, rng, ft2, tsince2), (ns, fell_now)

  @jax.jit
  def run(params, key, key2):
    inf = make_inf(params, deterministic=True)
    s0 = env.reset(key)
    _, (states, fell) = jax.lax.scan(lambda c, x: one(inf, c, x), (s0, key2, 0, 0), cmds)
    return s0, states, fell

  return run


def render(env, run, params, seed, w=640, h=480):
  dt = env.dt
  n = None
  s0, states, fell = jax.device_get(run(params, jax.random.PRNGKey(seed), jax.random.PRNGKey(seed + 7)))
  n = np.asarray(fell).shape[0]
  falls = [round(float(i * dt), 1) for i in np.nonzero(np.asarray(fell))[0]]
  traj = [s0] + [jax.tree.map(lambda x: x[i], states) for i in range(FRAME_SKIP - 1, n, FRAME_SKIP)]
  try:
    frames = env.render(traj, height=h, width=w, camera="track")
  except Exception:
    frames = env.render(traj, height=h, width=w)
  return frames, falls


def contact_sheet(frames, fps, path, every=0.5, cols=8, tw=200):
  from PIL import Image, ImageDraw
  th = []
  for i in range(0, len(frames), max(1, int(round(fps * every)))):
    im = Image.fromarray(np.asarray(frames[i]))
    im = im.resize((tw, int(tw * im.height / im.width)))
    ImageDraw.Draw(im).text((4, 4), f"{i / fps:.1f}s", fill=(255, 0, 0))
    th.append(np.asarray(im))
  while len(th) % cols:
    th.append(np.zeros_like(th[0]))
  rows = [np.concatenate(th[r:r + cols], 1) for r in range(0, len(th), cols)]
  Image.fromarray(np.concatenate(rows, 0)).save(path)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--out", required=True)
  p.add_argument("--impl", default="warp", choices=["warp", "jax"])
  p.add_argument("--stages", type=int, nargs="+", default=[0, 10_000_000, 20_000_000, 30_000_000, 40_000_000, 60_000_000])
  p.add_argument("--seconds", type=float, default=20.0)
  p.add_argument("--lie", type=float, default=2.5, help="secondi a terra dopo la caduta")
  p.add_argument("--stand", type=float, default=1.0, help="secondi fermo prima di partire")
  p.add_argument("--speed", type=float, default=0.7)
  p.add_argument("--seeds", type=int, nargs="+", default=[0], help="posizioni di partenza diverse")
  p.add_argument("--tag", default="", help="suffisso nel nome dei file")
  p.add_argument("--attrito", type=float, default=None, help="attrito del pavimento (normale ~0.8; 0.2 = ghiaccio)")
  p.add_argument("--sheet", action="store_true", help="salva anche un provino PNG (un fotogramma ogni 0.5 s)")
  a = p.parse_args()
  g1_falls.install(a.attrito)
  env_name, make_inf = inference_factory(a.out)
  cfg = registry.get_default_config(env_name)
  cfg.impl = a.impl
  cfg.naconmax = 256
  cfg.push_config.enable = False
  cfg.noise_config.level = 0.0
  env = registry.load(env_name, config=cfg)
  saved = {int(re.findall(r"step_(\d+)", f)[0]): f for f in glob.glob(f"{a.out}/stages/step_*.pkl")
           if re.findall(r"step_(\d+)", f)}
  os.makedirs(f"{a.out}/videos_cadute", exist_ok=True)
  run = make_runner(env, make_inf, a.seconds, a.lie, a.stand, a.speed)
  fps = 1.0 / (env.dt * FRAME_SKIP)
  for target in a.stages:
    step = min(saved, key=lambda s: abs(s - target))
    params = pickle.load(open(saved[step], "rb"))
    for seed in a.seeds:
      t0 = time.time()
      frames, falls = render(env, run, params, seed)
      name = f"cadute_{step/1e6:05.1f}M_s{seed}{a.tag}"
      path = f"{a.out}/videos_cadute/{name}.mp4"
      media.write_video(path, frames, fps=fps)
      if a.sheet:
        contact_sheet(frames, fps, f"{a.out}/videos_cadute/{name}.png")
      print(f"video: {name}.mp4  cadute: {len(falls)}  istanti (s): {falls}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
  main()
