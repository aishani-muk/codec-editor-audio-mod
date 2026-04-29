"""
Tonic identification for Indian art music.

Primary: look up a Saraga ``.ctonic.txt`` / ``.tonic`` file if provided.
Fallback: Essentia's ``TonicIndianArtMusic`` algorithm
  (Gulati, Bellur, Salamon, Ishwar, Murthy, Serra.
   "Automatic tonic identification in Indian art music: approaches and
   evaluation." Journal of New Music Research 43(1):53–71, 2014.)
  Reported accuracy: ~93–94% on the IAM-tonic dataset (both Carnatic
  and Hindustani subsets).

Final fallback: a caller-supplied default Hz (e.g. C4 = 261.63).
"""

from __future__ import annotations

import warnings
from pathlib import Path


def load_tonic_file(tonic_path: str) -> float:
    """Read a Saraga-style plain-text tonic file (one float Hz on line 1)."""
    with open(tonic_path) as f:
        return float(f.read().strip())


def estimate_tonic_from_file(
    wav_path: str, sr: int = 44100,
) -> float:
    """Estimate the tonic of a recording using Essentia's Gulati estimator.

    Essentia's ``TonicIndianArtMusic`` prefers 44.1 kHz input; we resample
    via ``MonoLoader``.
    """
    import essentia.standard as es

    audio = es.MonoLoader(filename=str(wav_path), sampleRate=sr)()
    return float(es.TonicIndianArtMusic()(audio))


def _find_saraga_tonic_file(stem: str, tonic_dir: str) -> str | None:
    """Find a Saraga-style tonic annotation matching ``stem`` under ``tonic_dir``.

    Searches for ``{stem}.ctonic.txt``, ``{stem}.tonic``, or any sibling
    file containing ``stem`` in its name.
    """
    root = Path(tonic_dir)
    for pat in (
        f"**/{stem}.ctonic.txt",
        f"**/{stem}.tonic",
        f"**/*{stem}*.ctonic.txt",
    ):
        hits = list(root.glob(pat))
        if hits:
            return str(hits[0])
    return None


def resolve_tonic(
    wav_path: str,
    stem: str | None = None,
    tonic_dir: str | None = None,
    default_hz: float = 261.63,
) -> tuple[float, str]:
    """Resolve tonic frequency following the fallback chain.

    Returns ``(tonic_hz, source)`` where ``source`` is one of
    ``"saraga_file"``, ``"essentia_gulati"``, or ``"default"``.
    """
    if stem is None:
        stem = Path(wav_path).stem

    if tonic_dir:
        t_path = _find_saraga_tonic_file(stem, tonic_dir)
        if t_path:
            try:
                return load_tonic_file(t_path), "saraga_file"
            except Exception as e:
                warnings.warn(f"Failed to parse tonic file {t_path}: {e}")

    # Essentia fallback.
    try:
        return estimate_tonic_from_file(wav_path), "essentia_gulati"
    except ImportError:
        warnings.warn("Essentia not installed; using default tonic")
    except Exception as e:
        warnings.warn(f"Essentia tonic estimation failed ({type(e).__name__}: {e}); "
                      f"using default tonic")

    return default_hz, "default"
