# Simulatore dal vivo: il G1 con la rete addestrata

Il visualizzatore nativo di MuJoCo con l'Unitree G1 e **la rete che abbiamo addestrato**, in tempo reale, **senza GPU e senza internet**. È quello usato sul palco del Linux Day (su un MacBook Pro Intel del 2020).

La rete gira in **numpy puro** (`politica_g1.py`): sono quattro moltiplicazioni di matrici. Non servono JAX, Brax né Playground.

## Installazione
```bash
cd simulatore
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt      # mujoco + numpy
.venv/bin/python politica_g1.py                # prova senza grafica
```
La prova deve stampare un bacino stabile intorno a 0,75 m e un avanzamento di circa 4,5 m in 10 s. Altre prove:
```bash
.venv/bin/python politica_g1.py --spenta           # rete spenta: crolla in ~1,5 s
.venv/bin/python politica_g1.py --cadute           # modello a collisioni complete (quello del visualizzatore)
.venv/bin/python politica_g1.py --cmd 0 0.4 0      # di lato
.venv/bin/python politica_g1.py --pesi pesi/step_000030474240.npz   # la rete a 30 milioni di passi
```
**Mac con processore Intel:** l'ultima versione di MuJoCo con ruote per macOS x86_64 è la 3.10.0 (`pip install mujoco==3.10.0`).

## Avvio
- **macOS:** doppio clic su `avvia.command`, oppure `.venv/bin/mjpython vis_g1.py` (su macOS il viewer passivo richiede `mjpython`, installato con MuJoCo).
- **Linux:** `.venv/bin/python vis_g1.py`.

Schermo intero dal pulsante verde della finestra (macOS). Tab mostra/nasconde i pannelli di MuJoCo.

## Tasti
| tasto | azione |
|---|---|
| Invio | reset: in piedi e fermo; un secondo Invio fa partire la fisica |
| Spazio | rete accesa / spenta |
| freccia su / giù | avanti / indietro (±0,25 m/s, max 1) |
| freccia sinistra / destra | gira (±0,5 rad/s, max 1) |
| 8 / 9 | di lato a sinistra / destra (±0,25 m/s, max 0,5) |
| . (punto) | stop: comando zero, segna il passo sul posto |
| 6 / 7 | spinta laterale 0,7 m/s / spinta forte 1,5 m/s |
| E | fase successiva della rete: pesi a caso → 10M → 30M → 35M → 40M passi → finale |
| mouse | doppio clic su un corpo, poi Ctrl + trascina col tasto destro: spinta a mano |

Non premere altre lettere, Backspace o i tasti 0–5: sono scorciatoie di visualizzazione di MuJoCo. Esc rende la telecamera libera.

## Una piccola dimostrazione (come al talk)
1. Avvia: il G1 è in piedi, fermo, rete spenta.
2. **Invio**: senza cervello, i motori tengono la posa… e il robot crolla.
3. **Invio**, poi **Spazio**: rete accesa, sta in piedi e segna il passo.
4. **Freccia su** due volte: cammina. Frecce laterali: gira. **8 / 9**: di lato. **Punto**: si ferma.
5. **6**: una spinta, barcolla e recupera. **7**: spinta forte, di solito cade.
6. **E** più volte: la stessa scena con la rete di diversi momenti dell'addestramento.

## Cosa c'è dentro
| file | |
|---|---|
| `vis_g1.py` | il visualizzatore (viewer passivo di MuJoCo, tasti, telecamera che segue il robot) |
| `politica_g1.py` | la rete in numpy e l'ambiente in MuJoCo "normale": osservazioni (103 numeri), passo di controllo (20 ms, 10 sotto-passi da 2 ms), modello a collisioni complete; si può usare anche da solo, senza grafica |
| `modello/` | il G1 **esattamente come in addestramento** (XML di MuJoCo Playground, mesh di MuJoCo Menagerie). È il modello "feetonly": collide col pavimento solo con i piedi; `politica_g1.py --cadute` (e il visualizzatore) ne genera una variante con collisioni su bacino, busto, testa e braccia, così il robot cade sul pavimento invece di attraversarlo |
| `pesi/` | i pesi della rete in `.npz`: la rete appena nata (0 passi), 10M, 30M, 35M, 40M e quella finale (203M). `chiavi_npz.txt` elenca le chiavi e le forme |

## Come si esegue la rete (se vuoi rifarlo altrove)
- **Ingressi**, in quest'ordine: velocità lineare del bacino (sensore `local_linvel_pelvis`, 3), giroscopio (`gyro_pelvis`, 3), gravità nel riferimento dell'IMU del bacino (3), comando [avanti, lato, rotazione] (3), angoli dei giunti meno la posa `knees_bent` (29), velocità dei giunti (29), ultima azione (29), fase del passo `[cos φ₀, cos φ₁, sin φ₀, sin φ₁]` (4).
- **Rete**: `(x − media) / dev.std`, poi MLP 512 → 256 → 128 con attivazione swish, uscita lineare di 58 numeri; **azione = tanh(prime 29 uscite)**.
- **Controllo**: `ctrl = posa knees_bent + 0,5 × azione`, tenuto per 10 passi di fisica da 2 ms. La fase avanza di `2π · 0,02 · 1,375` a ogni passo di controllo, partendo da `[0, π]`.

## Limiti
La rete è stata addestrata in MuJoCo Warp su GPU, qui gira su MuJoCo CPU: piccole differenze di fisica sono normali (cammina comunque come in addestramento). Non è mai stata provata su un G1 vero.
