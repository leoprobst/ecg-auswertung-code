"""Pinned, unmodified scientific model data. No downloads during GUI startup."""
from pathlib import Path
import hashlib
import json
import urllib.request

COMMIT = "e8ef6dbad6c1bf4446428f1f73dd0a4d164af40d"
MODELS = {
    "ohara-2011": "0ba58e14b76714741b3125d638eb03806c1be48eadb56c578cc821c141344c27",
    "tentusscher-2006": "82009bdcfc069f71cf29763d075382532be31cddd8d9f883fa0fc00ec79e290e",
    "courtemanche-1998": "6cadfaa1e13937ae4ae3df16bc907ad0670fbb8fe8f38969e36bdaecd9e98afa",
    "stewart-2009": "158326f1349fb69fd535bb0faf65cafa43f92c307103e2ce8e3bc0b192d2d019",
    "fabbri-2017": "077a2965e48bc10f6d7f7a99d29ea748f339495dada6881dce442ed7858a3806",
}
ASSETS = Path(__file__).resolve().parents[2] / "assets" / "ionic_models"


def model_path(name):
    if name not in MODELS:
        raise ValueError("Unregistered ionic model")
    path = ASSETS / (name + ".mmt")
    if not path.is_file():
        raise FileNotFoundError("Run python -m vcg_biomarker.simulation.model_assets once")
    if hashlib.sha256(path.read_bytes()).hexdigest() != MODELS[name]:
        raise ValueError("Ionic model hash mismatch; a modified model needs its own version")
    return path


def install():
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, digest in MODELS.items():
        path = ASSETS / (name + ".mmt")
        if path.exists():
            model_path(name)
            continue
        url = f"https://raw.githubusercontent.com/myokit/models/{COMMIT}/c/{name}.mmt"
        data = urllib.request.urlopen(url, timeout=30).read(2_000_000)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Downloaded scientific model failed pinned hash validation")
        with path.open("xb") as handle:
            handle.write(data)
    manifest = {"source": "https://github.com/myokit/models", "commit": COMMIT, "sha256": MODELS,
                "attribution": "Unmodified curated models; original authors and Myokit curators credited in each model header.",
                "usage": "Reuse invited by source repository; retain all model metadata and original paper references."}
    (ASSETS / "provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    install()
