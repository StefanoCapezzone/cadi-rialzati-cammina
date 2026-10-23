"""Registra un video MP4 per ogni fase salvata (stesse pose di partenza).

Si lancia in un processo separato DOPO l'addestramento, così la GPU è libera:
  python render_stages.py --out /content/drive/MyDrive/linuxday/prova
Opzioni: --every N (una fase ogni N), --seeds 0 1 2, --stochastic
"""
import argparse, glob, os, pickle

import jax, mediapy as media
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import registry

import humanoid_getup

N_STEPS = 400    # 10 s di simulazione
FRAME_SKIP = 2   # disegna un passo su 2 (20 fotogrammi/s): il rendering è la parte lenta


def load_inference_factory(out):
  cfg = pickle.load(open(f"{out}/network_config.pkl", "rb"))
  pre = (running_statistics.normalize if cfg["normalize_observations"]
         else lambda x, y: x)
  nets = ppo_networks.make_ppo_networks(
      cfg["obs_size"], cfg["action_size"], preprocess_observations_fn=pre,
      **cfg["network_factory_kwargs"])
  return ppo_networks.make_inference_fn(nets)


def rollout_frames(env, inference_fn, seed=0, camera="side", w=640, h=480):
  """Simula N_STEPS passi in un unico blocco compilato (lax.scan): molto più
  veloce di un ciclo Python passo per passo, poi disegna i fotogrammi."""
  def one_step(carry, _):
    state, rng = carry
    rng, k = jax.random.split(rng)
    a, _ = inference_fn(state.obs, k)
    nstate = env.step(state, a)
    return (nstate, rng), nstate

  @jax.jit
  def run(key):
    s0 = env.reset(key)
    _, states = jax.lax.scan(one_step, (s0, jax.random.PRNGKey(seed + 100)),
                             None, length=N_STEPS)
    return s0, states

  s0, states = run(jax.random.PRNGKey(seed))
  states = jax.device_get(states)
  traj = [s0] + [jax.tree.map(lambda x: x[i], states)
                for i in range(FRAME_SKIP - 1, N_STEPS, FRAME_SKIP)]
  return env.render(traj, height=h, width=w, camera=camera)


def render_all(out, impl="warp", seeds=(0, 1, 2), every=1, deterministic=True):
  humanoid_getup.register()
  cfg = registry.get_default_config("HumanoidGetup")
  cfg.impl = impl
  cfg.naconmax = 4000   # un solo robot: bastano pochi contatti (risparmia memoria GPU)
  env = registry.load("HumanoidGetup", config=cfg)
  make_inference_fn = load_inference_factory(out)
  os.makedirs(f"{out}/videos", exist_ok=True)
  fps = 1.0 / (env.dt * FRAME_SKIP)
  stages = sorted(glob.glob(f"{out}/stages/step_*.pkl"))
  for i, pkl in enumerate(stages):
    if i % every and i != len(stages) - 1:
      continue
    name = os.path.basename(pkl).replace(".pkl", ".mp4")
    if os.path.exists(f"{out}/videos/{name}"):
      continue
    inf = make_inference_fn(pickle.load(open(pkl, "rb")), deterministic=deterministic)
    frames = []
    for s in seeds:
      frames += rollout_frames(env, inf, seed=s)
    media.write_video(f"{out}/videos/{name}", frames, fps=fps)
    print("video:", name, flush=True)


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--out", required=True)
  p.add_argument("--impl", default="warp", choices=["warp", "jax"])
  p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
  p.add_argument("--every", type=int, default=1)
  p.add_argument("--stochastic", action="store_true")
  a = p.parse_args()
  render_all(a.out, a.impl, tuple(a.seeds), a.every, not a.stochastic)
