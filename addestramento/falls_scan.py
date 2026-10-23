"""Cerca le cadute buone SENZA rendere i video.

La simulazione e' veloce (secondi), il rendering e' lento (~4 minuti a video):
questo script prova molti seed / velocita' / attriti e stampa, per ciascuno,
quante cadute ci sono, quando avvengono e quanto e' durato ogni tentativo.
Si scelgono le combinazioni migliori e solo quelle si rendono con render_falls.py.

  python falls_scan.py --out /content/drive/MyDrive/linuxday/g1_walk \
      --stages 30000000 --seeds 0 1 2 3 4 5 6 7 --speed 0.7

Colonne stampate: fase, seed, numero di cadute, istanti (s), durata di ogni
tentativo (s) = da quando parte a quando cade.
"""
import argparse, glob, pickle, re, time

import jax, jax.numpy as jp, numpy as np

import g1_falls
from mujoco_playground import registry
from render_loco import inference_factory


def make_scanner(env, make_inf, seconds, lie_seconds, stand_seconds, speed):
  """Come make_runner di render_falls, ma restituisce SOLO le cadute:
  niente traiettoria da riportare a CPU, quindi e' molto piu' leggero."""
  dt = env.dt
  n, lie, stand = int(seconds / dt), int(lie_seconds / dt), int(stand_seconds / dt)
  cmds = jp.array([[0.0, 0.0, 0.0] if i < stand else [speed, 0.0, 0.0] for i in range(n)])

  def one(inf, carry, cmd):
    state, rng, ft, tsince = carry
    c = jp.where(tsince < stand, jp.zeros(3), cmd)
    state = state.replace(info={**state.info, "command": c})
    rng, k, kr = jax.random.split(rng, 3)
    a, _ = inf(state.obs, k)
    a = jp.where(ft > 0, jp.zeros_like(a), a)
    ns = env.step(state, a)
    fell_now = (ns.done > 0) & (ft == 0)
    ft2 = jp.where(ft > 0, ft + 1, jp.where(fell_now, 1, 0))
    again = ft2 > lie
    rs = env.reset(kr)
    ns = jax.tree.map(lambda x, y: jp.where(again, x, y), rs, ns)
    ft2 = jp.where(again, 0, ft2)
    tsince2 = jp.where(again, 0, tsince + 1)
    # avanzamento: quanto e' andato avanti il bacino rispetto all'inizio del tentativo
    return (ns, rng, ft2, tsince2), (fell_now, again)

  @jax.jit
  def run(params, key, key2):
    inf = make_inf(params, deterministic=True)
    s0 = env.reset(key)
    _, out = jax.lax.scan(lambda c, x: one(inf, c, x), (s0, key2, 0, 0), cmds)
    return out

  return run, dt


def episodi(fell, again, dt):
  """Da (cadute, ripartenze) ricava istanti di caduta e durata di ogni tentativo."""
  cadute = [float(i * dt) for i in np.nonzero(np.asarray(fell))[0]]
  riparte = [float(i * dt) for i in np.nonzero(np.asarray(again))[0]]
  durate, inizio = [], 0.0
  for c in cadute:
    durate.append(round(c - inizio, 1))
    successive = [r for r in riparte if r > c]
    inizio = successive[0] if successive else c
  return [round(c, 1) for c in cadute], durate


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--out", required=True)
  p.add_argument("--impl", default="warp", choices=["warp", "jax"])
  p.add_argument("--stages", type=int, nargs="+", required=True)
  p.add_argument("--seconds", type=float, default=20.0)
  p.add_argument("--lie", type=float, default=2.5)
  p.add_argument("--stand", type=float, default=1.0)
  p.add_argument("--speed", type=float, default=0.7)
  p.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
  p.add_argument("--attrito", type=float, default=None)
  a = p.parse_args()

  g1_falls.install(a.attrito)
  env_name, make_inf = inference_factory(a.out)
  cfg = registry.get_default_config(env_name)
  cfg.impl = a.impl
  cfg.naconmax = 256
  cfg.push_config.enable = False
  cfg.noise_config.level = 0.0
  env = registry.load(env_name, config=cfg)

  saved = {int(re.findall(r"step_(\d+)", f)[0]): f
           for f in glob.glob(f"{a.out}/stages/step_*.pkl") if re.findall(r"step_(\d+)", f)}
  run, dt = make_scanner(env, make_inf, a.seconds, a.lie, a.stand, a.speed)
  attrito = "normale" if a.attrito is None else f"attrito {a.attrito}"
  print(f"# velocita' {a.speed} m/s, {attrito}, {a.seconds:.0f}s per prova", flush=True)
  for target in a.stages:
    step = min(saved, key=lambda s: abs(s - target))
    params = pickle.load(open(saved[step], "rb"))
    for seed in a.seeds:
      t0 = time.time()
      fell, again = jax.device_get(run(params, jax.random.PRNGKey(seed), jax.random.PRNGKey(seed + 7)))
      cadute, durate = episodi(fell, again, dt)
      print(f"{step/1e6:6.1f}M  seed {seed}  cadute {len(cadute)}  "
            f"istanti {cadute}  durate {durate}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
  main()
