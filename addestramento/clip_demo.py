"""Taglia le clip della demo (Linux Day 2026) dai video su Drive.

Ogni clip e' una sequenza di EPISODI: un tentativo del robot (parte, cammina,
cade) tagliato stretto, senza i tempi morti a terra. Gli episodi si susseguono
senza pause, con didascalia in basso e contatore dei passi in alto a destra.

  python clip_demo.py --out /content/drive/MyDrive/linuxday/g1_walk
  python clip_demo.py --out ... --solo clip3_troppo_veloce

Gli istanti dei tagli vengono dal catalogo in cadute_istanti.md, ricavato con
episodi.py. Produce {out}/clip_demo/clip1_....mp4 ... clip6_....mp4
"""
import argparse, os, shlex, shutil, subprocess, tempfile

W, H = 640, 480
BANDA = 54          # altezza della fascia della didascalia
F_DID, F_CNT = 23, 19

GPU = "203 milioni di passi - 2 ore di GPU, 1,60 euro"
FINALE = 203_161_600


def font():
  import matplotlib
  for n in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
    p = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", n)
    if os.path.exists(p):
      return p
  raise SystemExit("font non trovato")


FONT = None


def passi(n):
  return f"{n:,}".replace(",", ".")


# --- descrizione delle clip ------------------------------------------------
# ep(video, da, a) = un episodio, dal secondo "da" al secondo "a" del video.
# "a" sta sempre circa 0,9 s dopo la caduta: si vede il tonfo, non l'attesa.
# rallenta=0.4 -> quel pezzo va al 40% della velocita'.
# did / cont sovrascrivono didascalia e contatore del singolo episodio.

def ep(video, da, a, rallenta=None, did=None, cont=None, dove="videos_cadute"):
  return dict(video=video, dove=dove, start=da, dur=round(a - da, 2),
              rallenta=rallenta, did=did, cont=cont)


def walk(video, da, a, **kw):
  return ep(video, da, a, dove="videos", **kw)


CLIP = {
    "clip1_giorno_zero": dict(
        did="Rete neurale appena inizializzata - 0 passi",
        cont=0,
        episodi=[
            ep("cadute_000.0M", 0.0, 1.8),
            ep("cadute_000.0M", 3.4, 5.5),
            ep("cadute_000.0M", 7.1, 8.8),
            ep("cadute_000.0M", 10.4, 12.3),
            ep("cadute_000.0M", 13.9, 15.8),
        ]),

    "clip2_ci_prova": dict(
        did="30 milioni di passi - circa 20 minuti di GPU",
        cont=30_500_000,
        episodi=[
            ep("cadute_030.5M", 0.6, 4.6),      # cade a 3.7
            ep("cadute_030.5M_s2", 0.6, 5.3),   # cade a 4.4
            ep("cadute_030.5M_s1", 2.0, 8.2),   # cade a 7.3
            ep("cadute_030.5M_s0", 4.6, 11.0),  # cade a 10.1, il tentativo piu' lungo
        ]),

    "clip3_troppo_veloce": dict(
        did="Stesso robot, gli chiediamo 1,8 m/s",
        cont=30_500_000,
        episodi=[
            ep("cadute_030.5M_s2_v18", 0.4, 5.0),            # cade a 4.1
            ep("cadute_030.5M_s0_v18", 0.4, 4.6),            # cade a 3.7
            ep("cadute_030.5M_s1_v18", 0.4, 2.2),            # cade a 2.9:
            ep("cadute_030.5M_s1_v18", 2.2, 3.8, rallenta=0.4),  # ...il volo al rallentatore
        ]),

    "clip4_cammina": dict(
        did=GPU,
        cont=FINALE,
        episodi=[
            walk("cammina_avanti", 0.5, 9.5),
            walk("cammina_lato", 0.5, 10.0),
            walk("cammina_veloce", 0.3, 8.2),
        ]),

    "clip5_ghiaccio": dict(
        did="Attrito 0,2 invece di 0,8: pavimento ghiacciato",
        cont=30_500_000,
        episodi=[
            ep("cadute_030.5M_s0_ghiaccio", 0.3, 2.8),                      # la spaccata
            ep("cadute_035.6M_s1_ghiaccio", 0.3, 3.3, cont=35_600_000),
            ep("cadute_030.5M_s1_ghiaccio", 1.5, 6.4),
            ep("cadute_035.6M_s1_ghiaccio", 4.8, 8.1, cont=35_600_000),
            ep("cadute_040.6M_s0_ghiaccio", 0.5, 8.5, cont=40_600_000,
               did="Stesso ghiaccio, robot addestrato"),
        ]),

    "clip6_cadute_assortite": dict(
        did="Ogni caduta e' un dato",
        cont=0,
        episodi=[
            ep("cadute_000.0M", 0.0, 1.8, cont=0),
            ep("cadute_005.1M", 3.2, 5.2, cont=5_100_000),
            ep("cadute_010.2M", 6.7, 8.4, cont=10_200_000),
            ep("cadute_015.2M", 10.5, 12.7, cont=15_200_000),
            ep("cadute_020.3M", 14.4, 16.9, cont=20_300_000),
            ep("cadute_025.4M_s0", 3.5, 6.6, cont=25_400_000),
            ep("cadute_030.5M_s1", 3.5, 8.2, cont=30_500_000),
            ep("cadute_030.5M_s2_v18", 8.0, 12.7, cont=30_500_000),
            ep("cadute_035.6M_s2", 10.5, 14.8, cont=35_600_000),
            walk("cammina_avanti", 1.0, 5.5, cont=FINALE, did=GPU),
        ]),
}


# --- montaggio -------------------------------------------------------------

def segmento(src, start, dur, didascalia, contatore, dest, rallenta=None):
  """Un pezzo di video con fascia + testi; rallenta=0.4 -> al 40% della velocita'."""
  d = tempfile.mkdtemp()
  t1, t2 = os.path.join(d, "did.txt"), os.path.join(d, "cnt.txt")
  open(t1, "w").write(didascalia)
  open(t2, "w").write("passi di addestramento: " + passi(contatore))
  vf = [
      f"scale={W}:{H}",
      f"drawbox=x=0:y=ih-{BANDA}:w=iw:h={BANDA}:color=black@0.60:t=fill",
      f"drawtext=fontfile={FONT}:textfile={t1}:fontcolor=white:fontsize={F_DID}"
      f":x=(w-text_w)/2:y=h-{BANDA}+({BANDA}-{F_DID})/2-2",
      f"drawtext=fontfile={FONT}:textfile={t2}:fontcolor=white:fontsize={F_CNT}"
      f":box=1:boxcolor=black@0.55:boxborderw=7:x=w-text_w-14:y=14",
  ]
  if rallenta:
    vf.insert(1, f"setpts={1/rallenta:.3f}*PTS")
  cmd = (f"ffmpeg -y -loglevel error -ss {start} -t {dur} -i {shlex.quote(src)} "
         f"-vf {shlex.quote(','.join(vf))} -r 25 -an -c:v libx264 -preset medium -crf 20 "
         f"-pix_fmt yuv420p {shlex.quote(dest)}")
  r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
  if r.returncode:
    print("ERRORE ffmpeg:", r.stderr[-600:], flush=True)
    raise SystemExit(1)
  return dest


def unisci(pezzi, dest):
  if len(pezzi) == 1:
    shutil.move(pezzi[0], dest)
    return dest
  lista = dest + ".txt"
  open(lista, "w").write("".join(f"file {shlex.quote(p)}\n" for p in pezzi))
  cmd = (f"ffmpeg -y -loglevel error -f concat -safe 0 -i {shlex.quote(lista)} "
         f"-c copy {shlex.quote(dest)}")
  r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
  os.remove(lista)
  if r.returncode:
    print("ERRORE concat:", r.stderr[-600:], flush=True)
    raise SystemExit(1)
  return dest


def main():
  global FONT
  p = argparse.ArgumentParser()
  p.add_argument("--out", required=True)
  p.add_argument("--solo", nargs="+", default=None, help="solo queste clip")
  a = p.parse_args()
  FONT = font()
  dest_dir = a.out + "/clip_demo"
  os.makedirs(dest_dir, exist_ok=True)
  tmp = tempfile.mkdtemp()
  totale = 0.0

  for nome, c in CLIP.items():
    if a.solo and nome not in a.solo:
      continue
    pezzi = []
    for i, e in enumerate(c["episodi"]):
      src = f"{a.out}/{e['dove']}/{e['video']}.mp4"
      if not os.path.exists(src):
        raise SystemExit(f"manca il video {src}")
      pezzi.append(segmento(src, e["start"], e["dur"],
                            e["did"] or c["did"],
                            c["cont"] if e["cont"] is None else e["cont"],
                            os.path.join(tmp, f"{nome}_{i}.mp4"),
                            rallenta=e["rallenta"]))
    d = unisci(pezzi, f"{dest_dir}/{nome}.mp4")
    dur = sum(e["dur"] / (e["rallenta"] or 1) for e in c["episodi"])
    totale += dur
    mb = os.path.getsize(d) / 1e6
    print(f"clip: {os.path.basename(d):28s} {len(c['episodi'])} episodi  "
          f"{dur:5.1f}s  {mb:.1f} MB", flush=True)
  print(f"totale: {totale:.0f} s ({totale/60:.1f} minuti)", flush=True)


if __name__ == "__main__":
  main()
