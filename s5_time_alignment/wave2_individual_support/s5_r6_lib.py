# -*- coding: utf-8 -*-
"""AW-245 S5-R6, Bibliothek (Version 1.0): individuelle Unsicherheit des geschaetzten Uhrversatzes (S5). Nachtraeglich (post hoc), explorativ.

Verfahren und Kriterien der Identifizierbarkeit stammen unveraendert aus aw242_kern.py (SHA-256 wird im Lauf geprueft). Neu sind:
 (1) Leave-one-ECG-out: Offset ohne je ein vollstaendiges EKG (gleiche Gleichstandsregel), Abweichung zum Gesamt-Offset.
 (2) Permutations-p-Wert je Person: Statistik T = Minimum der Personenkurve (Median der absoluten Differenzen), Nullverteilung durch
     Permutation der EKG-Frequenzen innerhalb der Person ueber alle verwertbaren EKG (Konvention von AW-242 R2), danach Auswahl der
     vollstaendigen EKG. p = (1 + #{T_perm <= T_beob}) / (R + 1); bei hoechstens 6 verwertbaren EKG exakte Aufzaehlung aller Permutationen
     (p = #{T_perm <= T_beob} / m!).
 (3) Kalibrierung: Pseudo-Datensaetze, in denen die EKG-Frequenzen der Personen vorab permutiert sind (kein Zeitsignal), mit denselben
     Kriterien und p-Werten; liefert die erwartete Zahl "identifizierbar und p < 0,05" ohne Signal.
Keine Kennungen in Ausgaben; Personenzeilen nur unter <local path>.
"""
import itertools
import math
import sys

import numpy as np

sys.dont_write_bytecode = True

from aw242_kern import D, K, estimate_from_loss, loss_curve  # noqa: E402

EPS = 1e-12


def _min_loss_batch(V, H):
    """V (n,K), H (R,n) -> Minimum der Kurve je Permutation (R,)."""
    L = np.median(np.abs(V[None, :, :] - H[:, :, None]), axis=1)
    return L.min(axis=1)


def t_perm_mc(V, all_h, local_idx, R, rng, chunk=200):
    """Nullverteilung der Statistik T durch R zufaellige Permutationen von all_h (Laenge m), Auswahl local_idx."""
    m = len(all_h)
    out = np.empty(R)
    done = 0
    while done < R:
        r = min(chunk, R - done)
        perm = rng.random((r, m)).argsort(axis=1)
        H = all_h[perm][:, local_idx]
        out[done:done + r] = _min_loss_batch(V, H)
        done += r
    return out


def t_perm_exact(V, all_h, local_idx):
    """Alle m! Permutationen (m <= 6)."""
    m = len(all_h)
    perms = np.array(list(itertools.permutations(range(m))), dtype=int)
    H = all_h[perms][:, local_idx]
    return _min_loss_batch(V, H)


def p_value(t_obs, V, all_h, local_idx, R, rng, exact_max=6):
    """Rueckgabe (p, Zahl der Permutationen, exakt?)."""
    m = len(all_h)
    if m <= exact_max:
        t = t_perm_exact(V, all_h, local_idx)
        return float(np.mean(t <= t_obs + EPS)), int(math.factorial(m)), True
    t = t_perm_mc(V, all_h, local_idx, R, rng)
    return float((1 + np.sum(t <= t_obs + EPS)) / (R + 1)), int(R), False


def loo(V, h, full_offset):
    """Leave-one-ECG-out: Liste der Offsets und der Teilkriterien ohne Anzahl (ohne Mindestzahl 3) je Auslassung."""
    n = len(h)
    offs, oks = [], []
    for i in range(n):
        keep = np.arange(n) != i
        e = estimate_from_loss(loss_curve(V[keep], h[keep]), n - 1)
        offs.append(e['offset'])
        oks.append(bool(e['c_w'] and e['c_m'] and e['c_edge'] and e['c_conn']))
    offs = np.array(offs)
    return offs, np.array(oks), int(np.max(np.abs(offs - full_offset)))


def curve_estimate(V, h):
    """Kurve und Schaetzung (Dict aus estimate_from_loss) fuer Frequenzen h."""
    loss = loss_curve(V, h)
    return loss, estimate_from_loss(loss, len(h))


def p_all_eligible(p, t_obs, rng, R):
    """p-Wert mit Permutation ueber alle verwertbaren EKG der Person (Nullverteilung gleich fuer beobachtete und Pseudo-Frequenzen)."""
    return p_value(t_obs, p['V'], p['all_h'], p['local_idx'], R, rng)


def p_complete_only(p, t_obs, rng, R, h):
    """Sensitivitaet: Permutation nur ueber die vollstaendigen EKG der Person (h: deren Frequenzen)."""
    return p_value(t_obs, p['V'], np.asarray(h, float), np.arange(len(h)), R, rng)
