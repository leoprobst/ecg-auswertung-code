"""Elektromechanische 3D-Illustration des Herzens für das ECG-Labor.

Alle Zeiten stammen aus dem ECG (Schlagzeitleiste, Delineation, Rhythmus- und
Leitungsmuster), alle Amplituden aus physiologischen Vorlagen. Das Paket liest
nur aus der Laboranalyse; es schreibt nichts in Befund, Export oder Pipeline und
importiert nichts aus ``vcg_biomarker.mimic``.
"""
from __future__ import annotations

__all__ = ["HeartModel", "default_model"]

from .model import HeartModel, default_model  # noqa: E402
