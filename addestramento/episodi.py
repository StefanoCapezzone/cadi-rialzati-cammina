"""Trova i confini degli episodi nei video delle cadute gia' renderizzati.

Ogni video e' una sequenza di tentativi: il robot parte, cammina, cade, resta a
terra qualche secondo coi motori spenti, poi riparte da capo. Alla ripartenza il
robot si teletrasporta all'origine e la telecamera salta: e' uno stacco netto,
che ffmpeg riconosce come cambio di scena. Da quegli stacchi si ricavano:
  ripartenza_i  = istante dello stacco
  caduta_i      = ripartenza_i - tempo a terra (2,5 s di default)
  tentativo_i   = da ripartenza_(i-1) (o 0) a caduta_i

  python episodi.py video1.mp4 video2.mp4 ...
  python episodi.py --dir /content/drive/MyDrive/linuxday/g1_walk/videos_cadute --filtro ghiaccio

Stampa una riga per video, pronta da usare per tagliare le clip.
"""
import argparse, glob, os, re, subprocess


def durata(path):
  r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "csv=p=0", path], capture_output=True, text=True)
  try:
    return float(r.stdout.strip())
  except ValueError:
    return 0.0


def stacchi(path, soglia):
  """Istanti in cui il fotogramma cambia di colpo: sono le ripartenze."""
  r = subprocess.run(["ffmpeg", "-hide_banner", "-i", path, "-an",
                      "-vf", f"select='gt(scene,{soglia})',showinfo",
                      "-f", "null", "-"], capture_output=True, text=True)
  t = [float(x) for x in re.findall(r"pts_time:([0-9.]+)", r.stderr)]
  fusi = []
  for x in sorted(t):
    if not fusi or x - fusi[-1] > 1.0:   # stacchi a meno di 1 s sono lo stesso salto
      fusi.append(round(x, 2))
  return [x for x in fusi if x > 0.5]


def main():
  p = argparse.ArgumentParser()
  p.add_argument("video", nargs="*")
  p.add_argument("--dir", default=None)
  p.add_argument("--filtro", default="", help="tiene solo i nomi che contengono questo testo")
  p.add_argument("--soglia", type=float, default=0.25)
  p.add_argument("--lie", type=float, default=2.5, help="secondi a terra dopo la caduta")
  a = p.parse_args()

  files = list(a.video)
  if a.dir:
    files += sorted(glob.glob(f"{a.dir}/*.mp4"))
  files = [f for f in files if a.filtro in os.path.basename(f)]
  if not files:
    raise SystemExit("nessun video")

  for f in files:
    d = durata(f)
    rip = stacchi(f, a.soglia)
    cadute = [round(r - a.lie, 1) for r in rip]
    inizi = [0.0] + rip
    tent = [f"{i:.1f}->{c:.1f}" for i, c in zip(inizi, cadute)]
    coda = f"  ultimo tentativo da {rip[-1]:.1f} (fine {d:.1f}, non cade)" if rip else ""
    print(f"{os.path.basename(f):38s} {d:5.1f}s  cadute {len(cadute)}: "
          f"{cadute}  tentativi: {tent}{coda}", flush=True)


if __name__ == "__main__":
  main()
