# Le clip del talk

Sei clip (125 secondi in tutto), con didascalia in basso e contatore dei passi di addestramento in alto a destra. Ogni clip mette in fila più tentativi del robot, tagliati stretti: parte, prova, cade, e si passa al tentativo dopo. Licenza CC BY 4.0.

| clip | durata | cosa si vede |
|---|---|---|
| `clip1_giorno_zero.mp4` | 9,5 s | **0 passi.** La rete con i pesi a caso: cinque volte di fila il robot finisce a terra in meno di un secondo e mezzo |
| `clip2_ci_prova.mp4` | 21,3 s | **30 milioni di passi** (~20 minuti di GPU). Quattro tentativi sempre più lunghi: tiene l'equilibrio per qualche passo, poi cade |
| `clip3_troppo_veloce.mp4` | 14,6 s | Stesso robot, gli chiediamo **1,8 m/s**: falcate lunghissime e tre capitomboli, l'ultimo al rallentatore |
| `clip4_cammina.mp4` | 26,4 s | **203 milioni di passi** (2 ore di GPU, 1,60 €). Tre percorsi diversi con la stessa rete: avanti e curva; di lato, in diagonale e indietro; veloce a 1,3 m/s. Non cade mai |
| `clip5_ghiaccio.mp4` | 21,8 s | **Pavimento ghiacciato** (attrito 0,2 invece di 0,8): il robot a metà addestramento scivola quattro volte, quello addestrato lo attraversa senza cadere |
| `clip6_cadute_assortite.mp4` | 31,6 s | Nove cadute in ordine di addestramento crescente (da 0 a 35 milioni di passi), sempre più "dignitose", e in chiusura la camminata |

Le clip sono state prodotte con gli script in [`../addestramento/`](../addestramento/): `render_falls.py` e `render_walks.py` per i video sorgente, `episodi.py` per trovare gli istanti delle cadute (catalogo in `cadute_istanti.md`), `clip_demo.py` per il montaggio.
