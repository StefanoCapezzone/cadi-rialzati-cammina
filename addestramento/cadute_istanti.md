# Catalogo degli episodi nei video delle cadute (Drive: linuxday/g1_walk/videos_cadute)

Ricavato con `episodi.py` il 20/09/2026: rileva le ripartenze come stacchi di
scena e risale alle cadute. Verificato contro i tempi stampati da render_falls.py
su cadute_030.5M_s2 (4,4 / 7,9 / 11,3 / 14,7): coincidono.

**Come leggerlo**: dopo ogni caduta il robot resta a terra 2,5 s, poi riparte.
Quindi il tentativo n+1 comincia a (caduta_n + 2,5). Una caduta negli ultimi
2,5 s del video non viene rilevata (manca la ripartenza che la segnala).

Suffissi: _v12 / _v15 / _v18 = comando 1,2 / 1,5 / 1,8 m/s; _ghiaccio = attrito 0,2;
_s0 / _s1 / _s2 = posizione di partenza diversa. Senza suffisso: 0,7 m/s, attrito normale.

| video | durata | cadute (s) |
|---|---|---|
| cadute_000.0M | 20 | 0.9, 4.6, 7.9, 11.4, 14.9 |
| cadute_005.1M | 20 | 0.7, 4.3, 7.7, 11.0, 14.9 |
| cadute_010.2M | 20 | 0.8, 4.2, 7.5, 11.4, 15.2 |
| cadute_015.2M | 20 | 0.9, 4.5, 8.0, 11.8, 15.1 |
| cadute_020.3M | 20 | 0.9, 4.5, 8.0, 11.9, 16.0 |
| cadute_025.4M_s0 | 20 | 1.0, 5.7, 9.6, 14.6 |
| cadute_025.4M_s0_v15 | 15 | 1.1, 5.3, 9.7 |
| cadute_025.4M_s1 | 20 | 1.8, 5.5, 9.1, 13.4, 16.8 |
| cadute_025.4M_s1_v15 | 15 | 1.8, 6.0, 10.3 |
| cadute_025.4M_s2 | 20 | 1.5, 5.5, 8.9, 12.8 |
| cadute_025.4M_s2_v15 | 15 | 1.6, 6.1, 9.9 |
| cadute_030.5M | 20 | 3.7, 9.0, 12.9, 16.5 |
| cadute_030.5M_s0 | 20 | 10.1 |
| cadute_030.5M_s0_ghiaccio | 15 | 1.7 |
| cadute_030.5M_s0_v12 | 15 | 4.7, 11.3 |
| cadute_030.5M_s0_v15 | 15 | nessuna |
| cadute_030.5M_s0_v18 | 15 | 3.7, 11.8 |
| cadute_030.5M_s1 | 20 | 7.3, 14.5 |
| cadute_030.5M_s1_ghiaccio | 15 | 5.4 |
| cadute_030.5M_s1_v12 | 15 | 7.2 |
| cadute_030.5M_s1_v15 | 15 | 4.2, 8.2 |
| cadute_030.5M_s1_v18 | 15 | 2.9, 7.4 |
| cadute_030.5M_s2 | 20 | 4.4, 7.9, 11.3, 14.7 |
| cadute_030.5M_s2_v15 | 15 | 4.6, 10.8 |
| cadute_030.5M_s2_v18 | 15 | 4.1, 11.8 |
| cadute_035.6M_s0 | 20 | 3.8 |
| cadute_035.6M_s0_ghiaccio | 15 | nessuna |
| cadute_035.6M_s0_v12 / _v18 | 15 | nessuna |
| cadute_035.6M_s1 | 20 | 14.1 |
| cadute_035.6M_s1_ghiaccio | 15 | 2.3, 7.1 |
| cadute_035.6M_s1_v12 / _v18 | 15 | nessuna |
| cadute_035.6M_s2 | 20 | 5.7, 13.9 |
| cadute_040.6M e tutti i 040.6M_* | 15-20 | nessuna |

Da 35,6M in poi il robot quasi non cade piu', nemmeno a 1,8 m/s o sul ghiaccio:
e' la prova che regge condizioni mai viste in addestramento.
