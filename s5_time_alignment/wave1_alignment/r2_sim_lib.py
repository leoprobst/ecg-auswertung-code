"""AW-242 R2: Bibliothek fuer die Simulation bekannter wahrer Versaetze (S5), getrennt von der Rechenstrecke testbar.

Modell: Die Referenzseite (Chartwerte, EKG-Zeitpunkte, Aufenthaltsgrenzen, Abdeckungsregel) ist real; die
EKG-Herzfrequenz wird kuenstlich erzeugt als latente Herzfrequenz zum Zeitpunkt t + tau (Versatz) plus Rauschen.
Latente Herzfrequenz (Modell A): lineare Interpolation der um die Dokumentationsverzoegerung d vorverlegten Chartwerte.
Latente Herzfrequenz (Modell B): Ornstein-Uhlenbeck-Prozess (Sigma, Theta aus den echten Chartserien), gegeben die
zwei benachbarten Chartwerte (Markov-Bruecke), d.h. realistische Autokorrelation zwischen den Chartzeitpunkten.
Erwartung des Schaetzers: tau + d (Verzoegerung verschiebt den Schaetzer um +d).
"""
import numpy as np
from aw242_kern import D, K, estimate_from_loss, loss_curve

LAGBINS = np.array([0, 5, 15, 30, 45, 75, 105, 165, 255, 405, 645, 1005, 1605, 2400]) * 60.0


def variogram(stays, robust=True):
    """stays: Liste von (times[s], values). Rueckgabe: Lag-Mitte (s), Halbvarianz der Differenzen, Anzahl je Klasse.

    robust=True: Halbvarianz aus dem Median der Betragsdifferenzen (Gauss: Median|d| = 0,6745 SD), weil die Kurvenkriterien
    medianbasiert sind und Ausreisser (Artefakte) den quadratischen Mittelwert dominieren.
    """
    nb = len(LAGBINS) - 1
    ssum = np.zeros(nb); cnt = np.zeros(nb); lag_sum = np.zeros(nb)
    hist = np.zeros((nb, 256))
    maxlag = LAGBINS[-1]
    for t, x in stays:
        n = len(t)
        for j in range(1, n):
            lag = t[j:] - t[:-j]
            if lag.min() > maxlag:
                break
            m = (lag > 0) & (lag <= maxlag)
            if not m.any():
                continue
            dd = (x[j:] - x[:-j])[m]
            b = np.searchsorted(LAGBINS, lag[m], side='left') - 1
            b = np.clip(b, 0, nb - 1)
            ssum += np.bincount(b, weights=dd ** 2, minlength=nb)
            cnt += np.bincount(b, minlength=nb)
            lag_sum += np.bincount(b, weights=lag[m], minlength=nb)
            ai = np.minimum(np.rint(np.abs(dd)).astype(int), 255)
            hist += np.bincount(b * 256 + ai, minlength=nb * 256).reshape(nb, 256)
    with np.errstate(invalid='ignore', divide='ignore'):
        lag_mid = lag_sum / cnt
        if robust:
            cs = np.cumsum(hist, axis=1) / np.maximum(hist.sum(axis=1, keepdims=True), 1)
            med = np.array([np.searchsorted(cs[i], 0.5) for i in range(nb)], float)
            gamma = 0.5 * (med / 0.6745) ** 2
        else:
            gamma = 0.5 * ssum / cnt
    return lag_mid, gamma, cnt


def fit_ou(lag_mid, gamma, cnt):
    """gamma(h) = nugget + sigma2 (1 - exp(-h/theta)); gewichtete kleinste Quadrate (Gewicht = Anzahl)."""
    from scipy.optimize import least_squares
    m = np.isfinite(gamma) & (cnt > 0)
    h = lag_mid[m]; g = gamma[m]; w = np.sqrt(cnt[m] / cnt[m].max())
    def res(p):
        nug, s2, th = p
        return w * (nug + s2 * (1 - np.exp(-h / th)) - g)
    best = None
    for th0 in (1800.0, 7200.0, 36000.0):
        sol = least_squares(res, [1.0, max(g.max(), 1.0), th0], bounds=([0, 0, 60.0], [200, 2000, 3e6]))
        if best is None or sol.cost < best.cost:
            best = sol
    nug, s2, th = best.x
    return dict(nugget=float(nug), sigma2=float(s2), theta_s=float(th), cost=float(best.cost))


def ou_moments(q, pt, vals, mu, sigma2, theta):
    """Bedingter Mittelwert und bedingte Varianz des OU-Prozesses zu den Zeiten q gegeben die Nachbarpunkte."""
    n = len(pt)
    idx = np.searchsorted(pt, q)
    left = idx - 1; right = idx
    hasl = left >= 0; hasr = right < n
    li = np.clip(left, 0, n - 1); ri = np.clip(right, 0, n - 1)
    xa = vals[li] - mu; xb = vals[ri] - mu
    h1 = np.where(hasl, q - pt[li], np.inf); h2 = np.where(hasr, pt[ri] - q, np.inf)
    ra = np.exp(-h1 / theta); rb = np.exp(-h2 / theta)
    H = np.where(hasl & hasr, pt[ri] - pt[li], np.inf)
    rab = np.exp(-H / theta)
    both = hasl & hasr
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        den = 1 - rab ** 2
        mean_b = mu + ((ra - rab * rb) * xa + (rb - rab * ra) * xb) / den
        var_b = sigma2 * (1 - (ra ** 2 + rb ** 2 - 2 * ra * rb * rab) / den)
    mean = np.where(both, mean_b, np.where(hasl, mu + ra * xa, np.where(hasr, mu + rb * xb, mu)))
    var = np.where(both, var_b, np.where(hasl, sigma2 * (1 - ra ** 2), np.where(hasr, sigma2 * (1 - rb ** 2), sigma2)))
    exact_l = hasl & (h1 < 1.0); exact_r = hasr & (h2 < 1.0)
    degenerate = both & (H < 60.0)
    mean = np.where(exact_l, vals[li], np.where(exact_r, vals[ri], mean))
    var = np.where(exact_l | exact_r, 0.0, var)
    if degenerate.any():
        with np.errstate(invalid='ignore', divide='ignore'):
            w = np.where(H > 0, h1 / np.where(H > 0, H, 1), 0.5)
        lin = vals[li] * (1 - w) + vals[ri] * w
        sel = degenerate & ~(exact_l | exact_r)
        mean = np.where(sel, lin, mean)
        var = np.where(sel, 0.0, var)
    var = np.clip(np.nan_to_num(var, nan=sigma2), 0.0, None)
    return mean, var


def latent_B(q, pt, vals, mu, sigma2, theta, rng):
    """OU-Bruecke: Stichprobe der latenten Frequenz zu den Zeiten q gegeben die Chartpunkte (pt, vals)."""
    mean, var = ou_moments(q, pt, vals, mu, sigma2, theta)
    return mean + np.sqrt(var) * rng.standard_normal(len(q))


def latent_A(q, pt, vals):
    return np.interp(q, pt, vals)


def person_sim(pdata, tau_s, delay, model, ou, sig_eps, rng, noise_scale=None):
    """Erzeugt pseudo-EKG-Frequenzen fuer die vollstaendigen EKG einer Person.

    pdata: dict mit t (n,), stay_ids (n,), stays: dict stay -> (times, values, mu); h_real (n,)
    delay: ('const', Sekunden) oder ('emp', Stichprobe in Sekunden)
    """
    t = pdata['t']; n = len(t)
    x = np.empty(n)
    for s in np.unique(pdata['stay_ids']):
        m = pdata['stay_ids'] == s
        times, vals, mu = pdata['stays'][s]
        if delay[0] == 'const':
            pt = times - delay[1]; v = vals
        else:
            d = rng.choice(delay[1], size=len(times))
            pt = times - d
            o = np.argsort(pt, kind='stable'); pt = pt[o]; v = vals[o]
        q = t[m] + tau_s
        if model == 'A':
            x[m] = latent_A(q, pt, v)
        else:
            x[m] = latent_B(q, pt, v, mu, ou['sigma2'], ou['theta_s'], rng)
    se = sig_eps if noise_scale is None else sig_eps * noise_scale
    h = x + se * rng.standard_normal(n)
    return np.clip(h, 30.0, 220.0)


def run_scenario(persons, model, tau_min, delay, ou, sig_eps, seed_seq, noise_scale_fn=None):
    """Ein Durchlauf ueber alle Personen; Rueckgabe Liste von Dicts je Person (ok, offset, width, near zone)."""
    rng = np.random.default_rng(seed_seq)
    res = []
    for p in persons:
        ns = noise_scale_fn(p) if noise_scale_fn else None
        h = person_sim(p, tau_min * 60.0, delay, model, ou, sig_eps, rng, ns)
        loss = loss_curve(p['V'], h)
        e = estimate_from_loss(loss, len(h))
        e['fold'] = p['fold']
        res.append(e)
    return res
