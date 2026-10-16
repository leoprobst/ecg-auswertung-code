from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Optional

import pandas as pd

from ..core.biomarkers import BiomarkerRegistry
from .baseline_registry import BaselineRegistry
from .label_registry import definitions_by_id
from .models import BaselineDefinition, ClinicalDisplayTerm, StageDescriptor
from .pathology_groups import pathology_group_for_endpoint


STAGE_GROUP_NAMES_DE = {
    "prepare": "Daten vorbereiten",
    "measure": "Marker berechnen",
    "analyze": "Auswerten",
    "review": "Prüfen und berichten",
}


STAGE_DESCRIPTORS: tuple[StageDescriptor, ...] = (
    StageDescriptor(
        "score-discovery", "analyze", "Forschungsscores entwickeln",
        "Begrenzte verschachtelte Score-Suche auf Trainingspatienten; keine klinische Freigabe.",
        "Formeln, Kalibration, Vergleichsmodelle und fehlende Therapie-/Mappingreferenzen.",
        ("validate", "audit"),
    ),
    StageDescriptor(
        "inventory",
        "prepare",
        "Datenquellen prüfen",
        "Prüft, ob alle benötigten MIMIC-Dateien, Versionen und Spalten vorhanden sind.",
        "Eine verständliche Liste vorhandener Daten und konkreter Blocker.",
    ),
    StageDescriptor(
        "genetic-feasibility",
        "prepare",
        "Genetische Daten einordnen",
        "Prüft die QRS-T-Winkel-GWAS-Dateien und trennt mögliche Kontextnutzung von nicht zulässigen Kausal- oder PRS-Analysen.",
        "Quellinventar mit explizitem MR-/PRS-Go-No-Go; es werden keine genetischen Kausalclaims berechnet.",
        ("inventory",),
    ),
    StageDescriptor(
        "benchmark",
        "prepare",
        "Rechnerleistung bestimmen",
        "Ermittelt automatisch eine schnelle, speichersichere Workerzahl.",
        "Empfohlene CPU- und Speichereinstellungen für diesen Rechner.",
    ),
    StageDescriptor(
        "ingest",
        "prepare",
        "Kerndaten vorbereiten",
        "Überführt unveränderte MIMIC-Rohdaten in einheitliche, schnell lesbare Tabellen.",
        "Standardisierte Patienten-, Aufenthalts-, ECG- und Echo-Tabellen.",
        ("inventory",),
    ),
    StageDescriptor(
        "ingest-events",
        "prepare",
        "Klinische Ereignisse vorbereiten",
        "Standardisiert Laborwerte, Diagnosen, Prozeduren und Medikamente mit Zeitangaben.",
        "Zeitlich vergleichbare klinische Ereignistabellen.",
        ("ingest",),
    ),
    StageDescriptor(
        "build-index",
        "prepare",
        "ECGs mit Patienten verknüpfen",
        "Erstellt für jedes ECG die eindeutige Verbindung zu Patient, Aufenthalt und Split.",
        "ECG-Masterindex mit genau einer Zeile pro ECG.",
        ("ingest",),
    ),
    StageDescriptor(
        "parse-reports",
        "prepare",
        "Maschinenbefunde einordnen",
        "Liest maschinelle ECG-Texte mit Negation und Unsicherheit als Vergleichsarm.",
        "Strukturierte maschinelle Befundmuster, nicht der klinische Goldstandard.",
        ("build-index",),
    ),
    StageDescriptor(
        "parse-notes",
        "prepare",
        "Arztbriefe und Befunde einordnen",
        "Liest Entlassungsbriefe und Röntgen-Thorax-Berichte mit Negation und Unsicherheit auf ein kuratiertes Muster-Set (Risikofaktoren, radiologische Herz-/Lungenbefunde, Diagnose-Erwähnungen).",
        "Aufnahmebezogene Freitext-Signale, kein vollständiges NLP und kein klinischer Goldstandard.",
        ("ingest",),
    ),
    StageDescriptor(
        "build-evidence",
        "prepare",
        "Klinische Referenzen zuordnen",
        "Ordnet Labor, Echo, Diagnosen, Prozeduren und Medikamente zeitlich jedem ECG zu.",
        "Nachvollziehbare Evidenzzeilen mit Quelle und Zeitabstand.",
        ("ingest-events", "build-index"),
    ),
    StageDescriptor(
        "attest-event-evidence",
        "review",
        "Ereigniszuordnung prüfen",
        "Kontrolliert, dass optimierte und direkte Ereigniszuordnung dasselbe Ergebnis liefern.",
        "Technischer Gleichheitsnachweis für klinische Ereignisse.",
        ("build-evidence",),
    ),
    StageDescriptor(
        "build-labels",
        "prepare",
        "Klinische Zielgruppen bilden",
        "Bildet positive, negative, unklare und nicht verfügbare Referenzgruppen.",
        "Transparente klinische Endpunkte mit Evidenz und Ausschlüssen.",
        ("build-evidence",),
    ),
    StageDescriptor(
        "attest-labels",
        "review",
        "Zielgruppen wissenschaftlich prüfen",
        "Prüft Fall-/Kontrollregeln, Zeitfenster und Leakage-Schutz.",
        "Freigabe oder konkrete Blocker für die Markerberechnung.",
        ("build-labels",),
    ),
    StageDescriptor(
        "audit-biomarkers",
        "review",
        "Biomarker-Formeln prüfen",
        "Prüft Implementierung, Einheiten, Wertebereiche, Redundanz und synthetisches Verhalten.",
        "Auditbericht mit gesperrten oder korrigierbaren Markern.",
    ),
    StageDescriptor(
        "pilot-go-no-go",
        "review",
        "Pilot-Freigabe zusammenstellen",
        "Prüft Software-, Freeze-, 1.000-/10.000-Pilot- und GUI-Gates, ohne den Voll-Lauf zu starten.",
        "Maschinenlesbarer Go/No-Go-Bericht mit allen offenen Abweichungen.",
        ("validate-pilot-10000",),
    ),
    StageDescriptor(
        "validate-delineation",
        "review",
        "Wellenerkennung extern prüfen",
        "Prüft Wellengrenzen und Robustheit getrennt an sechs unabhängigen PhysioNet-Datensätzen.",
        "Datensatzspezifische Fehler-, Erkennungs- und Abstentionswerte; fehlende Referenzdaten werden offen ausgewiesen.",
    ),
    StageDescriptor(
        "extract-features",
        "measure",
        "ECG- und VCG-Marker berechnen",
        "Erkennt Wellen, erzeugt VCGs und berechnet die ausgewählten Biomarker für alle geeigneten ECGs.",
        "Pausierbare Feature-Partitionen mit Signal- und Messqualität.",
        ("attest-labels", "audit-biomarkers"),
    ),
    StageDescriptor(
        "marker-redundancy",
        "measure",
        "Markergruppen bestimmen",
        "Vergleicht jeden Marker mit jedem anderen auf einer gezogenen "
        "Stichprobe und fasst zusammen, was sich statistisch nicht "
        "unterscheiden lässt. Ohne diese Gruppierung erscheint ein Befund "
        "in einer Trefferliste so oft, wie es Schreibweisen dafür gibt. "
        "Kein Marker wird entfernt; die Gruppe weist aus, welche Zeilen "
        "denselben Befund darstellen.",
        "Gruppentabelle mit Vertreter je Gruppe und der Übereinstimmung "
        "mit klassischen Kriterien.",
        ("extract-features",),
    ),
    StageDescriptor(
        "control-panel",
        "review",
        "Kontrolltafel rechnen",
        "Prüft die fertigen Ergebnisse gegen Zusammenhänge mit vorab "
        "bekannter Richtung (Kalium hebt die T-Welle, Kalzium verkürzt die "
        "QT-Zeit) und gegen Negativkontrollen ohne Mechanismus. Deren "
        "Trefferquote ist die gemessene Falsch-Positiv-Rate der Kohorte und "
        "enthält Konfundierung, die eine FDR-Korrektur nicht erfasst. Eine "
        "nicht wiedergefundene Positivkontrolle wird protokolliert und "
        "bricht den Lauf nicht ab.",
        "Kontrolltafel mit Richtung, Signifikanz und gemessenem Rauschboden, "
        "dazu ein lesbarer Bericht.",
        ("validate",),
    ),
    StageDescriptor(
        "segment-availability",
        "review",
        "Messbarkeit nach Endpunktstatus prüfen",
        "Ermittelt je Endpunkt, wie oft jedes Wellensegment messbar war, "
        "getrennt für Fälle und Kontrollen. Gleiche Ausfallraten kosten nur "
        "Aussagekraft; unterschiedliche bedeuten, dass der Marker den "
        "Endpunkt teilweise über seine eigene Abwesenheit abbildet. Die "
        "Richtung der so entstehenden Verzerrung ist nicht bestimmbar.",
        "Verfügbarkeitstabelle je Endpunkt und Segment mit dem Abstand "
        "zwischen Fällen und Kontrollen.",
        ("extract-features", "build-labels"),
    ),
    StageDescriptor(
        "reference-ranges",
        "review",
        "Referenzbereiche je Marker, Alter, Geschlecht und Frequenz",
        "Der Baustein, ohne den aus einem Marker kein Befund wird. Eine "
        "Leitlinie braucht sechs Dinge; das Projekt hatte zwei davon, und "
        "der fehlende, an dem alles andere hängt, ist der Referenzbereich. "
        "Diese Stufe legt eine Vergleichskohorte fest -- Patienten ohne "
        "kardiale Diagnose, ohne herzwirksames Medikament, ohne Device, ohne "
        "Intensivepisode, mit brauchbarer Signalqualität -- und berechnet "
        "je Marker und Schicht die 2,5./50./97,5. Perzentile mit "
        "Bootstrap-Intervall und Fallzahl. Ausdrücklich **keine Normwerte**: "
        "das sind Krankenhauspatienten ohne dokumentierte Herzerkrankung, "
        "nicht Gesunde. Jede Zeile trägt die Population und die "
        "VCG-Transformation mit, weil ein Schwellenwert ohne beides "
        "unbrauchbar ist.",
        "Referenztabelle je Marker x Geschlecht x Altersdekade x "
        "Frequenzband, mit Fallzahl, Bootstrap-Intervall und der "
        "Begründung, wo eine Schicht zu klein zum Schätzen ist.",
        ("extract-features", "build-evidence"),
    ),
    StageDescriptor(
        "transform-agreement",
        "review",
        "Umrechnung zwischen den beiden VCG-Transformationen",
        "Ein Zwölf-Kanal-ECG ist kein Vektorkardiogramm -- X, Y und Z "
        "werden rechnerisch erzeugt, und die zwei gebräuchlichen Verfahren "
        "sind sich nicht einig. Der räumliche QRS-T-Winkel Gesunder liegt "
        "bei 44-65 Grad nach Kors und 66-81 Grad nach inversem Dower: rund "
        "zwanzig Grad Unterschied, die aus der Rechnung stammen und nicht "
        "aus dem Herzen. Diese Stufe misst je Marker Bias, "
        "Uebereinstimmungsgrenzen und die Umrechnungsgerade an den echten "
        "Aufzeichnungen. Ohne sie ist jeder Schwellenwert dieses Projekts "
        "ausserhalb dieses Projekts unbenutzbar.",
        "Umrechnungstabelle je Marker mit Bias, Limits of Agreement, "
        "Rangkorrelation und Einstufung, ob die beiden Verfahren nach "
        "Umrechnung austauschbar sind.",
        ("extract-features",),
    ),
    StageDescriptor(
        "reporting-guideline",
        "review",
        "Mindest-Berichtssatz und Befundungsalgorithmus",
        "Der Zusammenbau des eigentlichen Projektziels. Ein Marker kommt "
        "nicht durch Auswahl in den Berichtssatz, sondern durch vier "
        "gemessene Bedingungen: er hat einen Referenzbereich, seine "
        "Wiederholbarkeit ist bekannt, seine Transformationsabhängigkeit "
        "ist gemessen, und er hat entweder einen Literaturanker oder eine "
        "replizierte Assoziation. Wer eine Bedingung verfehlt, steht mit "
        "der verfehlten Bedingung in der Tabelle. Dazu die vier "
        "Pflichtangaben neben jedem Wert und je klinischer Frage die "
        "Anspruchsobergrenze -- ein ECG-definierter Endpunkt bleibt "
        "analytische Konkordanz, gleich wie hoch die Fläche unter der "
        "Kurve ist.",
        "Mindest-Berichtssatz mit Ausschlussgründen, Befundungsalgorithmus "
        "mit Operating Points und Anspruchsobergrenze, Pflichtangaben und "
        "die Formel für die kleinste nachweisbare Änderung.",
        ("reference-ranges", "repeatability", "transform-agreement"),
    ),
    StageDescriptor(
        "ext-icd-crosscheck",
        "review",
        "Labels gegen eine zweite Ableitung prüfen",
        "MIMIC-IV-ECG-Ext-ICD verknüpft dieselben Aufzeichnungen nach "
        "eigenen Regeln mit ICD-10-Codes aus denselben Quelltabellen. Wo "
        "die beiden Ableitungen auseinandergehen, ist eine von beiden "
        "fehlerhaft; von innen ist das nicht feststellbar. Da Kohorte und "
        "Aufzeichnungen identisch sind, belegt Übereinstimmung keine "
        "Übertragbarkeit.",
        "Vier-Felder-Vergleich je Endpunkt, getrennt nach Aufenthalt und "
        "Vorgeschichte, dazu die Lage der publizierten 20 Folds zur eigenen "
        "Train-/Validation-Trennung.",
        ("build-labels",),
    ),
    StageDescriptor(
        "two-stage-selection",
        "analyze",
        "Bestätigungskandidaten auswählen",
        "Wählt je Endpunkt die stärksten Marker nach vorab festgelegter "
        "Regel und ausschließlich aus dem Trainingssplit, damit Treffer des "
        "breiten Screens einer bestätigenden Prüfung zugeführt werden "
        "können. Der Validierungssplit geht in die Auswahl nicht ein.",
        "Auswahlliste je Endpunkt und ein Bestätigungsplan, in dem "
        "vorab festgelegte und ausgewählte Paare getrennt ausgewiesen sind.",
        ("validate", "marker-redundancy"),
    ),
    StageDescriptor(
        "screen-broad-axes",
        "analyze",
        "Breite Achsen screenen",
        "Prüft jeden Marker gegen jede hinreichend häufige Größe der "
        "Datenbank: Laborwerte, Prozeduren, Erreger, Intensivmessungen und "
        "Fallgruppen. Die Intensivmessungs-Achse wird zuerst gerechnet; sie "
        "ist die einzige mit invasiv gemessener Referenz und damit die "
        "einzige, bei der ein Treffer eine Surrogatfrage stellt.",
        "Trefferliste je Achse, getrennt nach Trainings- und "
        "Validierungssplit, mit Effektuntergrenzen und Mehrfachtestkorrektur.",
        ("extract-features", "marker-redundancy"),
    ),
    StageDescriptor(
        "deep-combination-search",
        "analyze",
        "Markerkombinationen suchen",
        "Ermittelt je Endpunkt, welche Marker zusammen mehr leisten als "
        "einzeln, und misst den Zugewinn jeder weiteren Komponente an "
        "Patienten, die nicht in die Auswahl eingegangen sind. Die Laufzeit "
        "ist nach oben offen; da nach jeder Tiefe geschrieben wird, bleibt "
        "bei einem Abbruch jede abgeschlossene Tiefe vollständig erhalten.",
        "Tiefenkurve je Endpunkt mit dem Zugewinn jeder zusätzlichen "
        "Komponente.",
        ("extract-features",),
    ),
    StageDescriptor(
        "repeatability",
        "measure",
        "Messwiederholbarkeit bestimmen",
        "Sucht Paare von ECGs desselben Patienten ohne Ereignis dazwischen "
        "und misst, wie stark ein Marker allein durch die Wiederholung "
        "schwankt. Ohne diese Zahl lässt sich eine gemessene Änderung "
        "nicht von Messrauschen unterscheiden.",
        "Wiederholbarkeitstabelle je Marker mit Streuung und "
        "Verlaufsdaten je Patient.",
        ("extract-features",),
    ),
    StageDescriptor(
        "external-validation",
        "review",
        "An einer fremden Kohorte nachprüfen",
        "Wendet die eingefrorenen Richtungen und Schwellen auf PTB-XL an. "
        "Die dortigen Labels stammen vom selben ECG, die hiesigen aus Echo "
        "und Labor; ein Teil des Leistungsunterschieds geht auf diese "
        "verschiedenen Referenzarten zurück und nicht auf die Population.",
        "Übertragungsmatrix mit den AUC beider Kohorten je Markerpaar.",
        ("validate", "freeze"),
    ),
    StageDescriptor(
        "population-cohorts",
        "analyze",
        "CODE-15 und SaMi-Trop auswerten",
        "Berechnet den klinisch vorab festgelegten Core-/Sentinel-Markerstand in der grossen Primärversorgungs- und der prospektiven Chagas-Kohorte, patientengetrennt und mit kohortenspezifischen Claims.",
        "ECG-Referenzverteilungen, externe Befundkonkordanz, Mortalitätsmodelle, Coverage und ein explizit standortkonfundierter Chagas-Kontrast.",
        ("audit-biomarkers",),
    ),
    StageDescriptor(
        "refresh-phenome-promotions",
        "analyze",
        "Beförderungskandidaten aktualisieren",
        "Sammelt aus dem breiten Phänom-Screen die Treffer, die eine "
        "bestätigende Prüfung verdienen, und schreibt sie mit ihrem "
        "Stand fort.",
        "Kandidatenliste mit Status und Begründung je Marker-Endpunkt-Paar.",
        ("discover-phenome",),
    ),
    StageDescriptor(
        "group-vcg",
        "analyze",
        "Durchschnitts-VCG zweier Gruppen vergleichen",
        "Mittelt die Vektorschleifen aller Fälle und aller Kontrollen "
        "eines Endpunkts und stellt sie einander gegenüber. Zeigt die Art "
        "des morphologischen Unterschieds, die aus einem AUC-Wert nicht "
        "hervorgeht.",
        "Gemittelte Schleifen beider Gruppen mit Vertrauensbereich aus "
        "Bootstrap.",
        ("extract-features", "build-labels"),
    ),
    StageDescriptor(
        "within-subject-change",
        "analyze",
        "Intraindividuelle Änderungen rechnen",
        "Vergleicht zwei ECGs derselben Person um ein Ereignis herum: vor "
        "und nach der Dialyse gegen die Kaliumänderung, sowie über die "
        "Medikamentendosis innerhalb der behandelten Gruppe. Anatomie, "
        "Geschlecht und chronische Erkrankungen kürzen sich heraus, statt "
        "modelliert zu werden. Der Dosis-Arm ist unterpowert; seine "
        "Nullbefunde belegen keine Abwesenheit eines Effekts.",
        "Änderungstabelle je Design mit Rangkorrelation, Fallzahl und "
        "erwarteter Richtung.",
        ("extract-features", "ingest-events"),
    ),
    StageDescriptor(
        "voltage-mass-discordance",
        "analyze",
        "Spannungs-Massen-Diskordanz rechnen",
        "Rekonstruiert die linksventrikuläre Masse nach Devereux aus den "
        "Echo-Rohmaßen, die MIMIC nicht direkt ausweist, und bildet das "
        "Verhältnis aus QRS-Spannung und Masse. Geringe Spannung bei hoher "
        "Masse ist das klassische Zeichen einer infiltrativen "
        "Kardiomyopathie. Bewertet wird nur bei verdickter Wand, da die "
        "Screening-Frage sonst nicht gestellt wird.",
        "Diskordanztabelle je Endpunkt mit den AUC des Verhältnisses und "
        "seiner beiden Bestandteile, damit das Verhältnis sich gegen beide "
        "behaupten muss.",
        ("extract-features", "build-evidence"),
    ),
    StageDescriptor(
        "time-to-event",
        "analyze",
        "Zeit bis zum Ereignis rechnen",
        "Verwendet die Aufzeichnung als Zeitnullpunkt und schätzt, welche "
        "Marker die folgenden zwölf Monate vorhersagen; ein Diagnosecode "
        "wird dafür nicht benötigt. Jeder Marker wird zweimal geschätzt: "
        "auf dem ersten ECG je Patient und auf allen ECGs mit "
        "clusterrobuster Varianz. Übereinstimmende Schätzungen zeigen, dass "
        "das Ergebnis nicht von einem einzelnen Design getragen wird.",
        "Hazard Ratios je Standardabweichung mit Konfidenzintervall, "
        "Konkordanz und der größten Abweichung zwischen den beiden "
        "Designs.",
        ("extract-features", "build-index"),
    ),
    StageDescriptor(
        "occlusion-yield",
        "analyze",
        "Ausbeute der Infarktkriterien prüfen",
        "Vergleicht die klassischen Infarktkriterien mit einem Endpunkt, "
        "der das ECG nicht heranzieht: Troponinanstieg, Katheterintervention "
        "binnen eines Tages und koronare Beschwerden. Anschließend wird "
        "geprüft, ob die VCG-Marker ihre Trennschärfe behalten, wenn alle "
        "Aufzeichnungen mit erfülltem ST-Hebungskriterium entfernt werden. "
        "Der Endpunkt ist ein Revaskularisations-Surrogat ohne "
        "angiographischen Verschlussnachweis.",
        "Ausbeutetabelle je Kriterium und Trennschärfe der Marker unter den "
        "Kriterium-Negativen, jeweils mit dem Hinweis, dass der Endpunkt ein "
        "Revaskularisations-Surrogat ohne Angiographie ist.",
        ("extract-features", "build-labels"),
    ),
    StageDescriptor(
        "thesis-tests",
        "analyze",
        "Thesen prüfen",
        "Rechnet die 28 elektrophysiologischen und klinischen VCG-Thesen "
        "genau in der Fassung, die vor dem Lauf festgelegt wurde: je These "
        "Marker, Zeitfenster, Paarbildung und Widerlegungskriterium. "
        "Nur Train und Validation; der gesperrte Test bleibt unberührt. "
        "Widerlegungen werden wie Bestätigungen berichtet.",
        "Je These und Teilaussage Schätzer mit Intervall, Werte getrennt für "
        "Train und Validation und das Urteil bestätigt, widerlegt, "
        "unentschieden oder nicht auswertbar mit Grund.",
        ("extract-features", "build-labels", "reference-ranges"),
    ),
    StageDescriptor(
        "physiological-gap-markers",
        "measure",
        "Alter-/BMI-Gap-Marker trainieren",
        "Sagt Alter und BMI aus dem Core-Markerpanel vorher (Train-only, "
        "patientengruppierte Kreuzvalidierung); die Abweichung vom "
        "tatsächlichen Wert wird als neuer Marker gespeichert.",
        "Trainierte Gap-Modelle und Gap-Werte je ECG für Train/Validation.",
        ("extract-features",),
    ),
    StageDescriptor(
        "extract-features-pilot-1000",
        "measure",
        "Technischen 1.000-ECG-Test rechnen",
        "Prüft Wellenerkennung, Markerformeln, Schreiben und Resume an 1.000 Entwicklungs-ECGs.",
        "Nicht einfrierbare Vorschau; mindestens 80 Prozent müssen technisch erfolgreich sein.",
        ("attest-labels", "audit-biomarkers"),
    ),
    StageDescriptor(
        "validate-pilot-1000",
        "analyze",
        "1.000-ECG-Test auswerten",
        "Prüft, ob Ergebnis- und Grenzwerttabellen aus dem kleinen Pilot korrekt entstehen.",
        "Technisches Gate ohne klinische Leistungsbehauptung.",
        ("extract-features-pilot-1000",),
    ),
    StageDescriptor(
        "extract-features-pilot-10000",
        "measure",
        "10.000-ECG-Pilot rechnen",
        "Prüft Stabilität, Ressourcen und Partitionierung mit dem klinischen Core- und Sentinel-Panel vor dem Monatslauf.",
        "Nicht einfrierbarer Entwicklungspilot; Discovery wurde im 1.000er Methodenpilot geprüft und wird im Monatslauf wieder einbezogen.",
        ("validate-pilot-1000",),
    ),
    StageDescriptor(
        "validate-pilot-10000",
        "analyze",
        "10.000-ECG-Pilot auswerten",
        "Prüft alle Core-/Sentinel-Einzelmarker und Endpunkte als Skalierungs-, Ressourcen- und Resume-Test; Bootstrap-Konfidenzen wurden bereits im 1.000er Methodenpilot geprüft.",
        "Technisches Freigabe-Gate ohne Triple-Suche oder klinische Validation-Bestätigung.",
        ("extract-features-pilot-10000",),
    ),
    StageDescriptor(
        "validate",
        "analyze",
        "Marker und Grenzwerte prüfen",
        "Vergleicht Einzelmarker und kleine Scores in Train und Validation mit klinischen Referenzen.",
        "Leistungswerte, Grenzwerte, Stabilität und Vergleich mit klassischen Regeln.",
        ("extract-features",),
    ),
    StageDescriptor(
        "therapy-response",
        "analyze",
        "Medikamenten-Ansprechen prüfen",
        "Paart je AV-Knoten-wirksamer Medikamentengabe (z. B. Adenosin) das "
        "nächste ECG davor mit dem nächsten ECG danach (±2h) und vergleicht "
        "Rhythmus- und Frequenzmuster -- Momentaufnahmen, keine Telemetrie.",
        "Vorher/Nachher-ECG-Paare mit Rhythmus-Delta und gebootstrapptem "
        "Normalisierungs-Effekt je Medikamentenklasse.",
        ("extract-features",),
    ),
    StageDescriptor(
        "discover-phenome",
        "analyze",
        "Weitere Zusammenhänge suchen",
        "Sucht breit nach bisher nicht priorisierten Pathologie- und Kontextassoziationen.",
        "Nach Evidenz sortierte Hypothesen zur späteren Detailanalyse.",
        ("extract-features",),
    ),
    StageDescriptor(
        "report",
        "review",
        "Ergebnisse aufbereiten",
        "Erzeugt Tabellen, Visualisierungen, Review-Fälle und druckbare Zusammenfassungen.",
        "Nachvollziehbare Berichte mit Snapshot- und Versionsangaben.",
        ("validate",),
    ),
    StageDescriptor(
        "audit",
        "review",
        "Gesamtlauf abnehmen",
        "Kontrolliert Vollständigkeit, Hashes, Splits, Testschutz und wissenschaftliche Mindestanforderungen.",
        "Klare Freigabe oder eine Liste noch offener Punkte.",
        ("report",),
    ),
)


_STAGES = {item.stage_id: item for item in STAGE_DESCRIPTORS}


GLOSSARY_DE = {
    "Kohorte": "Eine nach festen Regeln ausgewählte Gruppe von Patienten oder ECGs.",
    "Klinisches Ziel": "Der Zustand, gegen den ein ECG- oder VCG-Marker geprüft wird.",
    "Beurteilbar": "Die benötigte klinische Referenz wurde tatsächlich erhoben; fehlende Daten gelten nicht als gesund.",
    "Prävalenz": "Anteil positiver Fälle unter allen beurteilbaren Fällen und Kontrollen.",
    "Train": "Patientengruppe zum Bestimmen von Richtung, Skalierung und Grenzwerten.",
    "Validation": "Getrennte Patientengruppe zur Prüfung, ob ein im Training gefundener Zusammenhang stabil bleibt.",
    "Locked Test": "Versiegelte Patientengruppe, die erst nach dem wissenschaftlichen Freeze einmalig ausgewertet wird.",
    "Zeitfenster": "Erlaubter zeitlicher Abstand zwischen ECG und klinischer Referenzmessung.",
    "Sensitivität": "Anteil der klinisch positiven Fälle, die das Kriterium erkennt.",
    "Spezifität": "Anteil der klinisch negativen Kontrollen, die das Kriterium korrekt nicht markiert.",
    "FDR q": "Für die große Zahl getesteter Marker korrigierte statistische Irrtumswahrscheinlichkeit.",
    "Explorativ": "Hypothesengenerierend; noch kein klinisch bestätigtes Diagnosekriterium.",
}


# Display names deliberately avoid internal database terminology. The internal
# IDs remain available in provenance panels and exports.
_ENDPOINT_NAMES_DE = {
    "troponin_positive_any": "Erhöhtes Troponin nahe dem ECG",
    "acute_troponin_change_proxy": "Akute Troponinveränderung",
    "acute_mi_high_specificity": "Akuter Herzinfarkt, strenger Rückblick-Proxy",
    "acute_mi_probable": "Wahrscheinlicher akuter Herzinfarkt, Rückblick-Proxy",
    "omi_revascularization_proxy": "Akutes Koronarereignis mit Revaskularisation",
    "myocardial_injury_demand_context": "Myokardschädigung im Belastungs-/Demand-Kontext",
    "echo_lvh_strict": "Linksherzmuskelverdickung im Echo, streng",
    "echo_lvh_sensitive": "Linksherzmuskelverdickung im Echo, sensitiv",
    "lv_wall_thickening": "Verdickte linke Herzwand",
    "lv_dilatation_proxy": "Erweiterter linker Ventrikel",
    "concentric_remodeling_proxy": "Konzentrisches Remodeling",
    "reduced_lvef_strict": "Deutlich reduzierte Auswurffraktion unter 40 %",
    "reduced_lvef": "Reduzierte Auswurffraktion unter 50 %",
    "preserved_lvef": "Erhaltene Auswurffraktion ab 50 %",
    "echo_lae": "Vergrößerter linker Vorhof im Echo",
    "echo_rae": "Vergrößerter rechter Vorhof im Echo",
    "echo_rv_dilation": "Erweiterter rechter Ventrikel im Echo",
    "echo_rv_systolic_dysfunction": "Eingeschränkte Rechtsherzfunktion im Echo",
    "elevated_rvsp_or_pasp": "Erhöhter pulmonaler Druck im Echo",
    "echo_pulmonary_hypertension": "Pulmonale Hypertonie im Echo",
    "regional_wall_motion_abnormality": "Regionale Wandbewegungsstörung",
    "aortic_stenosis": "Aortenstenose, mindestens mittelgradig",
    "aortic_regurgitation": "Aorteninsuffizienz, mindestens mittelgradig",
    "mitral_stenosis": "Mitralstenose, mindestens mittelgradig",
    "mitral_regurgitation": "Mitralinsuffizienz, mindestens mittelgradig",
    "tricuspid_regurgitation": "Trikuspidalinsuffizienz, mindestens mittelgradig",
    "significant_valvular_disease": "Bedeutsame Klappenerkrankung",
    "aortic_stenosis_severe": "Schwere Aortenstenose",
    "aortic_regurgitation_severe": "Schwere Aorteninsuffizienz",
    "mitral_stenosis_severe": "Schwere Mitralstenose",
    "mitral_regurgitation_severe": "Schwere Mitralinsuffizienz",
    "tricuspid_regurgitation_severe": "Schwere Trikuspidalinsuffizienz",
    "significant_valvular_disease_severe": "Mindestens eine schwere Klappenerkrankung",
    "qtc_gt_480": "Verlängerte herzfrequenzkorrigierte QT-Zeit über 480 ms",
    "qtc_gt_500": "Stark verlängerte herzfrequenzkorrigierte QT-Zeit über 500 ms",
    "hypokalemia_near_ecg": "Kaliummangel nahe dem ECG",
    "severe_hypokalemia_near_ecg": "Schwerer Kaliummangel nahe dem ECG",
    "hyperkalemia_near_ecg": "Kaliumüberschuss nahe dem ECG",
    "severe_hyperkalemia_near_ecg": "Schwerer Kaliumüberschuss nahe dem ECG",
    "hypomagnesemia_near_ecg": "Magnesiummangel nahe dem ECG",
    "hypocalcemia_near_ecg": "Kalziummangel nahe dem ECG",
    "new_onset_af_30d": "Neu auftretendes Vorhofflimmern innerhalb 30 Tagen",
    "new_onset_af_90d": "Neu auftretendes Vorhofflimmern innerhalb 90 Tagen",
    "new_onset_af_1y": "Neu auftretendes Vorhofflimmern innerhalb eines Jahres",
    "old_mi_icd": "Früherer Herzinfarkt",
    "mi_or_revascularization_history": "Früherer Herzinfarkt oder Revaskularisation",
    "heart_failure_icd": "Herzinsuffizienz im Behandlungskontext",
    "elevated_ntprobnp": "Erhöhtes NT-proBNP nahe dem ECG",
    "acute_decompensated_heart_failure_proxy": "Akut dekompensierte Herzinsuffizienz, Proxy",
    "pulmonary_embolism_icd": "Lungenembolie im Behandlungskontext",
    "pulmonary_hypertension_icd": "Pulmonale Hypertonie im Behandlungskontext",
    "right_heart_strain_proxy": "Rechtsherzbelastung, kombinierter Proxy",
    "cardiomyopathy_icd": "Kardiomyopathie im Behandlungskontext",
    "pericarditis_icd": "Perikarditis im Behandlungskontext",
    "myocarditis_icd": "Myokarditis mit passendem Kontext",
    "endocarditis_high_confidence_proxy": "Endokarditis, strenger Rückblick-Proxy",
    "vt_vf_icd_or_event": "Ventrikuläre Tachykardie oder Kammerflimmern",
    "cardiac_arrest": "Herz-Kreislauf-Stillstand",
    "in_hospital_mortality": "Tod während des Krankenhausaufenthalts",
    "mortality_30d": "Sterblichkeit innerhalb 30 Tagen",
    "qt_prolonging_drug_exposure": "QT-verlängerndes Medikament nahe dem ECG",
    "antiarrhythmic_exposure_near_ecg": "Klasse-I/III-Antiarrhythmikum nahe dem ECG",
    "digoxin_exposure_near_ecg": "Digoxin nahe dem ECG",
}


_ENDPOINT_DEFINITIONS_DE = {
    "troponin_positive_any": "Troponin liegt im festgelegten Zeitfenster über der testspezifischen Referenzgrenze; dies bedeutet Myokardschädigung, nicht automatisch Herzinfarkt.",
    "acute_troponin_change_proxy": "Serielle Troponinwerte zeigen eine vorab definierte relevante Änderung mit mindestens einem erhöhten Wert.",
    "acute_mi_high_specificity": "Kombination aus dynamischer Troponinschädigung, akutem MI-Code und zeitnah dokumentierter PCI; kein Angiografie-Goldstandard.",
    "acute_mi_probable": "Breiterer retrospektiver MI-Kontext aus Troponin, Kodierung und Koronarintervention oder ACS-Behandlung.",
    "omi_revascularization_proxy": "Akute Schädigung plus ACS-/Brustschmerzkontext und zeitnahe PCI; keine bewiesene Koronarokklusion.",
    "myocardial_injury_demand_context": "Troponinerhöhung zusammen mit Sepsis, Schock, Hypoxie, Tachyarrhythmie, Anämie oder Nierenversagen ohne klare Koronarintervention.",
    "qtc_gt_480": "Maschinell ermittelte, nach Fridericia korrigierte QT-Zeit liegt über 480 Millisekunden; ECG-interner technischer Vergleich.",
    "qtc_gt_500": "Maschinell ermittelte, nach Fridericia korrigierte QT-Zeit liegt über 500 Millisekunden; ECG-interner technischer Vergleich.",
}


def stage_descriptor(stage_id: object) -> StageDescriptor:
    normalized = str(stage_id or "")
    return _STAGES.get(
        normalized,
        StageDescriptor(
            normalized or "unknown",
            "review",
            "Technische Aufgabe",
            "Interner, nicht näher beschriebener Verarbeitungsschritt.",
            "Technisches Artefakt; Details im Ereignisprotokoll.",
        ),
    )


@lru_cache(maxsize=1)
def clinical_display_terms() -> tuple[ClinicalDisplayTerm, ...]:
    definitions = definitions_by_id()
    terms: list[ClinicalDisplayTerm] = []
    for endpoint_id, definition in definitions.items():
        group = pathology_group_for_endpoint(endpoint_id)
        name = _ENDPOINT_NAMES_DE.get(endpoint_id, definition.display_name)
        short_definition = _ENDPOINT_DEFINITIONS_DE.get(
            endpoint_id,
            _generic_endpoint_definition(endpoint_id, name),
        )
        window = _window_summary(
            definition.window_before_hours,
            definition.window_after_hours,
        )
        source = ", ".join(definition.source_tables)
        terms.append(
            ClinicalDisplayTerm(
                term_id=endpoint_id,
                display_name_de=name,
                category_de=group.display_name_de,
                short_definition_de=short_definition,
                source_summary_de=(
                    f"Quelle: {source or 'nicht angegeben'}; Zeitbezug: {window}."
                ),
                synonyms=(
                    definition.display_name,
                    endpoint_id.replace("_", " "),
                ),
            )
        )
    return tuple(
        sorted(
            terms,
            key=lambda item: (
                item.category_de.casefold(),
                item.display_name_de.casefold(),
            ),
        )
    )


def clinical_display_term(endpoint_id: object) -> ClinicalDisplayTerm:
    normalized = str(endpoint_id or "")
    for term in clinical_display_terms():
        if term.term_id == normalized:
            return term
    return ClinicalDisplayTerm(
        term_id=normalized,
        display_name_de=normalized.replace("_", " ").strip().capitalize()
        or "Unbekanntes klinisches Ziel",
        category_de="Weitere klinische Ziele",
        short_definition_de="Für dieses interne Ziel fehlt eine Anzeigeübersetzung.",
    )


def search_clinical_terms(text: str = "") -> tuple[ClinicalDisplayTerm, ...]:
    needle = str(text or "").strip().casefold()
    if not needle:
        return clinical_display_terms()
    return tuple(
        term for term in clinical_display_terms() if needle in term.search_text
    )


@lru_cache(maxsize=1)
def _baseline_registry() -> BaselineRegistry:
    return BaselineRegistry()


def _classical_criterion_guidance(
    feature_id: str,
) -> Optional[BaselineDefinition]:
    """Look up the literature-sourced guideline definition for a
    `classical_*` feature id, so its Sollwert/formula can be shown as a
    plausibility check even though classical criteria are not
    BiomarkerRegistry entries (they are computed by ClassicalECGCriteria
    directly, not via the FeatureContext catalog). Every
    BaselineDefinition in baseline_registry.py uses baseline_id ==
    criterion_id, so this is a direct lookup, not a search."""
    prefix = "classical_"
    if not feature_id.startswith(prefix):
        return None
    criterion_id = feature_id[len(prefix):]
    try:
        return _baseline_registry().get(criterion_id)
    except KeyError:
        return None


def biomarker_display_name(feature_id: object) -> str:
    normalized = str(feature_id or "")
    try:
        definition = BiomarkerRegistry.definition(normalized)
    except ValueError:
        baseline = _classical_criterion_guidance(normalized)
        if baseline is not None:
            return baseline.display_name
        return _humanize_identifier(normalized)
    description = str(definition.description or "").strip().rstrip(".")
    if description and len(description) <= 88:
        return description
    return _humanize_identifier(normalized)


def biomarker_tooltip(feature_id: object) -> str:
    normalized = str(feature_id or "")
    try:
        definition = BiomarkerRegistry.definition(normalized)
    except ValueError:
        baseline = _classical_criterion_guidance(normalized)
        if baseline is not None:
            parts = [
                f"Leitlinienkriterium: {baseline.display_name}",
                f"Sollwert/Formel: {baseline.formula}",
            ]
            if baseline.notes:
                parts.append(baseline.notes)
            if baseline.source_title:
                parts.append(f"Quelle: {baseline.source_title}")
            return "\n".join(parts)
        return f"Interner Biomarker: {normalized}"
    parts = [definition.description]
    if definition.formula:
        parts.append(f"Formel: {definition.formula}")
    if definition.unit:
        parts.append(f"Einheit: {definition.unit}")
    if definition.family:
        parts.append(f"Familie: {definition.family}")
    from vcg_biomarker.core.projection_metadata import projection_metadata
    projection = projection_metadata(normalized)
    if projection is not None:
        axes = " / ".join(projection["axes"])
        parts.extend([
            f"2D-Projektion: {projection['plane']} | Achsenfolge: {axes}",
            "Positive Achsen: X links, Y inferior, Z posterior. Achsenumkehr verändert den dargestellten Drehsinn.",
            "Transformationsabhängiger Forschungsmarker; keine validierte klinische Schwelle.",
            "Katalogbeschreibung, kein Nachweis einer Neuberechnung historischer Ergebnisse.",
        ])
    return "\n".join(part for part in parts if part)


def annotate_result_display(records: Iterable[dict]) -> list[dict]:
    output: list[dict] = []
    for record in records:
        item = dict(record)
        endpoint_id = item.get("endpoint_id", item.get("target_id", ""))
        term = clinical_display_term(endpoint_id)
        item["clinical_target"] = term.display_name_de
        item["clinical_category"] = term.category_de
        item["clinical_target_definition"] = term.short_definition_de
        feature_id = item.get("feature_id", item.get("vcg_feature_id", ""))
        if feature_id:
            item["biomarker_name"] = biomarker_display_name(feature_id)
        output.append(item)
    return output


def annotate_result_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add display labels while retaining every reproducible internal ID."""

    result = frame.copy()
    endpoint_column = next(
        (
            column
            for column in ("endpoint_id", "label_id")
            if column in result
        ),
        None,
    )
    if endpoint_column is not None:
        endpoint_ids = result[endpoint_column].fillna("").astype(str)
        terms = {
            endpoint_id: clinical_display_term(endpoint_id)
            for endpoint_id in endpoint_ids.unique()
        }
        if "clinical_target_de" not in result:
            result.insert(
                0,
                "clinical_target_de",
                endpoint_ids.map(lambda value: terms[value].display_name_de),
            )
        if "clinical_category_de" not in result:
            result.insert(
                min(1, len(result.columns)),
                "clinical_category_de",
                endpoint_ids.map(lambda value: terms[value].category_de),
            )
        if "clinical_target_definition_de" not in result:
            result["clinical_target_definition_de"] = endpoint_ids.map(
                lambda value: terms[value].short_definition_de
            )
    feature_column = next(
        (
            column
            for column in ("feature_id", "vcg_feature_id")
            if column in result
        ),
        None,
    )
    if feature_column is not None and "biomarker_name_de" not in result:
        feature_ids = result[feature_column].fillna("").astype(str)
        names = {
            feature_id: biomarker_display_name(feature_id)
            for feature_id in feature_ids.unique()
        }
        result.insert(
            min(2, len(result.columns)),
            "biomarker_name_de",
            feature_ids.map(names),
        )
    return result


def _generic_endpoint_definition(endpoint_id: str, name: str) -> str:
    if endpoint_id.startswith("echo_") or endpoint_id in {
        "lv_wall_thickening",
        "lv_dilatation_proxy",
        "concentric_remodeling_proxy",
        "reduced_lvef",
        "reduced_lvef_strict",
        "preserved_lvef",
        "regional_wall_motion_abnormality",
    }:
        return f"{name} wurde aus einer zeitnahen strukturierten Echo-Messung abgeleitet."
    if "kalemia" in endpoint_id or "magnesemia" in endpoint_id or "calcemia" in endpoint_id:
        return f"{name} wurde aus einem einheitengeprüften Laborwert nahe dem ECG abgeleitet."
    if endpoint_id.startswith("new_onset_af"):
        return f"{name}; vorher bekannte oder im Index-ECG vorhandene AF/AFlutter-Fälle sind ausgeschlossen."
    if "exposure" in endpoint_id:
        return f"{name} wurde aus dokumentierter Gabe oder einer zum ECG aktiven Verordnung abgeleitet."
    if endpoint_id.endswith("_icd") or "_icd_" in endpoint_id:
        return f"{name} beruht auf dokumentierter klinischer Kodierung und wird als Behandlungskontext, nicht als perfekter Goldstandard, interpretiert."
    if "mortality" in endpoint_id or endpoint_id == "cardiac_arrest":
        return f"{name} ist ein klinischer Ausgangs- beziehungsweise Prognoseendpunkt."
    return f"{name} wurde nach einer versionierten klinischen Referenzregel gebildet."


def _window_summary(before: Optional[float], after: Optional[float]) -> str:
    if before is None and after is None:
        return "kein festes Zeitfenster"
    if before is None:
        return f"bis {abs(float(after or 0)):g} Stunden nach dem ECG"
    if after is None:
        return f"ab {abs(float(before)):g} Stunden vor dem ECG"
    return f"{abs(float(before)):g} Stunden davor bis {abs(float(after)):g} Stunden danach"


def _humanize_identifier(value: str) -> str:
    replacements = {
        "qrs": "QRS",
        "qrst": "QRST",
        "vcg": "VCG",
        "ecg": "ECG",
        "svg": "SVG",
        "dtw": "DTW",
        "qtc": "QTc",
        "pca": "PCA",
        "st": "ST",
    }
    words = [item for item in value.replace("/", " ").split("_") if item]
    rendered = [replacements.get(word.casefold(), word) for word in words]
    if not rendered:
        return "Unbenannter Biomarker"
    text = " ".join(rendered)
    return text[0].upper() + text[1:]


__all__ = [
    "GLOSSARY_DE",
    "STAGE_DESCRIPTORS",
    "STAGE_GROUP_NAMES_DE",
    "annotate_result_display",
    "biomarker_display_name",
    "biomarker_tooltip",
    "clinical_display_term",
    "clinical_display_terms",
    "search_clinical_terms",
    "stage_descriptor",
]
