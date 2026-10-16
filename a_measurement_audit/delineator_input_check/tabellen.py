"""Markdown tables from the aggregate CSV files (inputs for the report)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys
from pathlib import Path
import pandas as pd
sys.dont_write_bytecode = True

RES = Path(str(data_path("a.aw196_ergebnis")))
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else RES / 'tabellen.md'
NAMES = {'qrs_duration_ms': 'QRS duration', 'qt_interval_ms': 'QT', 'qtc_fridericia_ms': 'QTcF', 'pr_interval_ms': 'PR', 'p_duration_ms': 'P duration',
         'p_onset': 'P onset', 'p_offset': 'P offset', 'qrs_onset': 'QRS onset', 'qrs_offset': 'QRS offset', 't_offset': 'T offset'}
KL = {'praktisch abhaengig': 'dependent', 'praktisch unabhaengig': 'independent', 'begrenzt abhaengig': 'limited', 'nicht beurteilbar': 'n.a.'}


def f(x, d=1):
    if pd.isna(x):
        return 'n/a'
    return f'{x:.{d}f}'.replace('.', ',')


def table(df, cols, heads):
    lines = ['| ' + ' | '.join(heads) + ' |', '|' + '|'.join(['---'] * len(heads)) + '|']
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join(c(r) for c in cols) + ' |')
    return '\n'.join(lines)


def contrast_table(df):
    cols = [lambda r: NAMES[r.groesse], lambda r: str(int(r.n_paare)), lambda r: f(r.diskordant_pct), lambda r: f(r.anteil_identisch_pct),
            lambda r: f(r.bias_ms, 2), lambda r: f'{f(r.loa_unten_ms)} to {f(r.loa_oben_ms)}', lambda r: f(r.H_ms) + (f' ({f(r.H_ci_unten)} to {f(r.H_ci_oben)})' if 'H_ci_unten' in r and not pd.isna(r.H_ci_unten) else ''),
            lambda r: f(r.mad_ms, 2), lambda r: f(r.S_ms, 2), lambda r: KL[r.klasse]]
    heads = ['Quantity', 'Pairs', 'discordant %', 'identical %', 'Bias ms', 'LoA ms', 'H ms (95% CI)', 'mean abs. diff.', 'S ms', 'Class']
    return table(df, cols, heads)


def main():
    parts = []
    m = pd.read_csv(RES / 'mimic_kontraste_konsensschlag.csv')
    for k in m.kontrast.unique():
        parts.append(f'### MIMIC, {k}\n\n' + contrast_table(m[m.kontrast == k]) + '\n')
    b = pd.read_csv(RES / 'mimic_kontraste_je_schlag_zusatz.csv')
    for k in b.kontrast.unique():
        parts.append(f'### MIMIC per beat (addition), {k}\n\n' + contrast_table(b[b.kontrast == k].assign(n_paare=lambda d: d.n_paare)) + '\n')
    if (RES / 'ludb_fehler_gegen_experte.csv').exists():
        e = pd.read_csv(RES / 'ludb_fehler_gegen_experte.csv')
        cols = [lambda r: r.variante, lambda r: NAMES[r.groesse], lambda r: str(int(r.n_gepaart)), lambda r: f(r.ext_mean_ms, 2), lambda r: f(r.ext_sd_ms, 2), lambda r: f(r.ext_median_ms),
                lambda r: f(r.ext_mae_ms, 2), lambda r: f(r.med_mean_ms, 2), lambda r: f(r.med_mae_ms, 2)]
        parts.append('### LUDB, error against the expert (variant minus expert)\n\n' + table(e, cols, ['Variant', 'Fiducial point', 'Pairs', 'Bias ms (extreme ref.)', 'SD ms', 'Median ms', 'MAE ms', 'Bias ms (median ref.)', 'MAE ms (median ref.)']) + '\n')
        s = pd.read_csv(RES / 'ludb_kontraste_messpunkte.csv')
        for k in s.kontrast.unique():
            sub = s[s.kontrast == k]
            parts.append(f'### LUDB fiducial points per reference beat, {k}\n\n' + contrast_table(sub) + '\n')
            cols = [lambda r: NAMES[r.groesse], lambda r: f(r.delta_bias_gegen_experte_ms, 2), lambda r: f(r.delta_mae_gegen_experte_ms, 2), lambda r: 'yes' if r.fehlerrelevant else 'no']
            parts.append(f'Effect on the error against the expert, {k}\n\n' + table(sub, cols, ['Fiducial point', 'Delta bias ms', 'Delta MAE ms', 'error-relevant']) + '\n')
        mk = pd.read_csv(RES / 'ludb_kontraste_marker.csv')
        for k in mk.kontrast.unique():
            parts.append(f'### LUDB markers per recording, {k}\n\n' + contrast_table(mk[mk.kontrast == k]) + '\n')
        me = pd.read_csv(RES / 'ludb_marker_gegen_experte.csv')
        cols = [lambda r: r.variante, lambda r: NAMES[r.groesse], lambda r: str(int(r.fehler_n)), lambda r: f(r.fehler_mean_ms, 2), lambda r: f(r.fehler_sd_ms, 2), lambda r: f(r.fehler_mae_ms, 2)]
        parts.append('### LUDB, markers against expert intervals (variant minus expert)\n\n' + table(me, cols, ['Variant', 'Marker', 'Recordings', 'Bias ms', 'SD ms', 'MAE ms']) + '\n')
    if (RES / 'ptb_kontraste_konsensschlag.csv').exists():
        p = pd.read_csv(RES / 'ptb_kontraste_konsensschlag.csv')
        for k in p.kontrast.unique():
            parts.append(f'### PTB, {k}\n\n' + contrast_table(p[p.kontrast == k]) + '\n')
        pb = pd.read_csv(RES / 'ptb_kontraste_je_schlag_zusatz.csv')
        for k in pb.kontrast.unique():
            parts.append(f'### PTB per beat (addition), {k}\n\n' + contrast_table(pb[pb.kontrast == k]) + '\n')
    OUT.write_text('\n'.join(parts), encoding='utf8')
    print(OUT)


if __name__ == '__main__':
    main()
