"""Video delle fasi di un robot "joystick" (es. G1) con un percorso di comandi fisso:
cammina avanti, gira, cammina di lato... Uguale per ogni fase, così si confrontano.

  python render_loco.py --out /content/drive/MyDrive/linuxday/g1_walk --every 4
"""
import argparse, glob, os, pickle
import jax, jax.numpy as jp, mediapy as media
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import registry

FRAME_SKIP = 2
# (secondi, [vel_avanti m/s, vel_laterale m/s, rotazione rad/s])
SCRIPT = [(1.5, [0.0, 0.0, 0.0]), (5.0, [0.8, 0.0, 0.0]), (3.0, [0.6, 0.0, 0.8]),
          (3.0, [0.0, 0.4, 0.0]), (2.5, [0.0, 0.0, 0.0])]


def inference_factory(out):
  c = pickle.load(open(f"{out}/network_config.pkl", "rb"))
  pre = running_statistics.normalize if c["normalize_observations"] else (lambda x, y: x)
  nets = ppo_networks.make_ppo_networks(c["obs_size"], c["action_size"],
                                        preprocess_observations_fn=pre,
                                        **c["network_factory_kwargs"])
  return c["env"], ppo_networks.make_inference_fn(nets)


def render_stage(env, inf, seed, w=640, h=480, camera="track"):
  dt = env.dt
  cmds = jp.concatenate([jp.tile(jp.array(c), (int(s / dt), 1)) for s, c in SCRIPT])

  def one(carry, cmd):
    state, rng = carry
    state = state.replace(info={**state.info, "command": cmd})
    rng, k = jax.random.split(rng)
    a, _ = inf(state.obs, k)
    ns = env.step(state, a)
    # se cade (episodio finito) riparte da capo, come in addestramento:
    # nel video si vedono i tentativi uno dopo l'altro
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
    return env.render(traj, height=h, width=w, camera=camera)
  except Exception:
    return env.render(traj, height=h, width=w)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--out", required=True)
  p.add_argument("--impl", default="warp", choices=["warp", "jax"])
  p.add_argument("--every", type=int, default=1)
  p.add_argument("--seeds", type=int, nargs="+", default=[0])
  a = p.parse_args()
  env_name, make_inf = inference_factory(a.out)
  cfg = registry.get_default_config(env_name)
  cfg.impl = a.impl
  cfg.naconmax = 64
  cfg.push_config.enable = False     # niente spinte nei video
  cfg.noise_config.level = 0.0
  env = registry.load(env_name, config=cfg)
  os.makedirs(f"{a.out}/videos", exist_ok=True)
  stages = sorted(glob.glob(f"{a.out}/stages/step_*.pkl"))
  for i, pkl in enumerate(stages):
    if i % a.every and i != len(stages) - 1:
      continue
    name = os.path.basename(pkl).replace(".pkl", ".mp4")
    if os.path.exists(f"{a.out}/videos/{name}"):
      continue
    inf = make_inf(pickle.load(open(pkl, "rb")), deterministic=True)
    frames = []
    for s in a.seeds:
      frames += render_stage(env, inf, s)
    media.write_video(f"{a.out}/videos/{name}", frames, fps=1.0 / (env.dt * FRAME_SKIP))
    print("video:", name, flush=True)


if __name__ == "__main__":
  main()
