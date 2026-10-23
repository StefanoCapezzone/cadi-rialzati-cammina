"""Tre camminate diverse della rete finale, una per file, per la clip "Cammina".

Non basta ripetere la stessa corsa tre volte: ogni percorso chiede al robot
qualcosa di diverso (avanti e curva / di lato e indietro / veloce), cosi' si
vede che obbedisce ai comandi e non recita una sequenza imparata a memoria.

  python render_walks.py --out /content/drive/MyDrive/linuxday/g1_walk
  python render_walks.py --out ... --routes avanti veloce --seed 3

Scrive {out}/videos/cammina_<nome>.mp4 (salta quelli gia' fatti).
"""
import argparse, glob, os, pickle, re

import jax, jax.numpy as jp, mediapy as media

from mujoco_playground import registry
from render_loco import inference_factory, FRAME_SKIP

# (secondi, [avanti m/s, laterale m/s, rotazione rad/s])
PERCORSI = {
    "avanti": [(1.0, [0.0, 0.0, 0.0]), (4.5, [0.8, 0.0, 0.0]),
               (3.5, [0.6, 0.0, 0.9]), (1.0, [0.0, 0.0, 0.0])],
    "lato":   [(0.8, [0.0, 0.0, 0.0]), (3.5, [0.0, 0.5, 0.0]),
               (3.0, [0.5, 0.4, 0.0]), (2.5, [-0.5, 0.0, 0.0]), (0.8, [0.0, 0.0, 0.0])],
    "veloce": [(0.8, [0.0, 0.0, 0.0]), (6.5, [1.3, 0.0, 0.0]), (1.2, [0.0, 0.0, 0.0])],
}


def render(env, inf, percorso, seed, w=640, h=480):
  dt = env.dt
  cmds = jp.concatenate([jp.tile(jp.array(c), (int(s / dt), 1)) for s, c in percorso])

  def one(carry, cmd):
    state, rng = carry
    state = state.replace(info={**state.info, "command": cmd})
    rng, k = jax.random.split(rng)
    a, _ = inf(state.obs, k)
    ns = env.step(state, a)
    rng, kr = jax.random.split(rng)
    rs = env.reset(kr)
    ns = jax.tree.map(lambda x, y: jp.where(ns.done > 0, x, y), rs, ns)
    return (ns, rng), ns

  @jax.jit
  def run(key):
    s0 = env.reset(key)
    _, states = jax.lax.scan(one, (s0, jax.random.PRNGKey(seed + 7)), cmds)
    return s0, states

  s0, states = jax.device_get(run(jax.random.PRNGKey(seed)))
  n = cmds.shape[0]
  traj = [s0] + [jax.tree.map(lambda x: x[i], states) for i in range(FRAME_SKIP - 1, n, FRAME_SKIP)]
  try:
    return env.render(traj, height=h, width=w, camera="track")
  except Exception:
    return env.render(traj, height=h, width=w)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--out", required=True)
  p.add_argument("--impl", default="warp", choices=["warp", "jax"])
  p.add_argument("--routes", nargs="+", default=list(PERCORSI), choices=list(PERCORSI))
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--stage", type=int, default=None,
                 help="passi della fase da usare (default: l'ultima salvata)")
  a = p.parse_args()

  env_name, make_inf = inference_factory(a.out)
  cfg = registry.get_default_config(env_name)
  cfg.impl = a.impl
  cfg.naconmax = 64
  cfg.push_config.enable = False
  cfg.noise_config.level = 0.0
  env = registry.load(env_name, config=cfg)

  saved = {int(re.findall(r"step_(\d+)", f)[0]): f
           for f in glob.glob(f"{a.out}/stages/step_*.pkl") if re.findall(r"step_(\d+)", f)}
  step = max(saved) if a.stage is None else min(saved, key=lambda s: abs(s - a.stage))
  inf = make_inf(pickle.load(open(saved[step], "rb")), deterministic=True)
  os.makedirs(f"{a.out}/videos", exist_ok=True)
  print(f"# fase {step/1e6:.1f}M passi, seed {a.seed}", flush=True)
  for nome in a.routes:
    dest = f"{a.out}/videos/cammina_{nome}.mp4"
    if os.path.exists(dest):
      print("gia' fatto:", os.path.basename(dest), flush=True)
      continue
    frames = render(env, inf, PERCORSI[nome], a.seed)
    media.write_video(dest, frames, fps=1.0 / (env.dt * FRAME_SKIP))
    durata = sum(s for s, _ in PERCORSI[nome])
    print(f"video: cammina_{nome}.mp4  {durata:.1f}s", flush=True)


if __name__ == "__main__":
  main()
