"""
statistics_service.py
---------------------
Zählt und speichert Besucherstatistiken für den Museumsbetrieb.

Speicherformat (visitor_stats.yaml):
    2026-03-24:
      durchgaenge: 12   # Anzahl Aufnahme-Batches
      personen: 18      # Summe erkannter echte Personen

Zuständigkeiten:
- record_session(face_count): Zählt einen Durchgang + Personen für heute
- get_today() / get_period() / get_month() / get_total(): Aggregierte Abfragen
- reset(): Löscht alle gespeicherten Statistikdaten
- Atomares Speichern via temp file + os.replace() (museumssicher)
- Automatisches Bereinigen von Einträgen älter als retention_days

Datenschutzhinweis:
- Es werden KEINE Personenbilder, Uhrzeiten oder Einzelprofile gespeichert.
- Nur tägliche Aggregate (Anzahl Durchgänge + Anzahl Personen).

AUTOR: Florian Hoeft
"""

import os
import yaml
from datetime import date, timedelta


class StatisticsService:
    """
    Verwaltet Besucherstatistiken im Museumsbetrieb.

    Daten werden täglich in visitor_stats.yaml gespeichert.
    Schreibvorgänge sind atomar, um Datenverlust bei Stromausfall zu vermeiden.
    Einträge älter als retention_days werden beim nächsten Schreiben automatisch gelöscht.
    """

    def __init__(self, stats_file="visitor_stats.yaml", enabled=True, retention_days=365):
        """
        :param stats_file: Pfad zur Statistik-YAML-Datei.
        :param enabled: Wenn False, werden keine Daten aufgezeichnet.
        :param retention_days: Tageseinträge älter als dieser Wert werden automatisch gelöscht.
        """
        self.stats_file = stats_file
        self.enabled = enabled
        self.retention_days = retention_days

    def record_session(self, face_count, developer_mode=False):
        """
        Zählt einen abgeschlossenen Durchgang und die Anzahl erkannter Personen.
        Wird bei status == "BATCH" in handle_pipeline_result() aufgerufen.

        Im Developer-Mode wird nichts gezählt, damit Testläufe die Besucherzahlen
        nicht verfälschen.

        :param face_count: Anzahl echter erkannter Personen (nicht Pool-Personen).
        :param developer_mode: Wenn True, wird kein Eintrag geschrieben.
        """
        if not self.enabled or developer_mode:
            return
        data = self._load()
        today = str(date.today())
        day = data.setdefault(today, {"durchgaenge": 0, "personen": 0})
        day["durchgaenge"] = day.get("durchgaenge", 0) + 1
        day["personen"] = day.get("personen", 0) + max(0, face_count)
        self._prune_old_entries(data)
        self._save(data)

    def get_today(self):
        """
        Gibt die Statistik für den heutigen Tag zurück.
        :return: Dict mit 'durchgaenge' und 'personen', beide >= 0.
        """
        data = self._load()
        today = str(date.today())
        return dict(data.get(today, {"durchgaenge": 0, "personen": 0}))

    def get_period(self, mode="weekly"):
        """
        Aggregiert Statistiken über einen Zeitraum.

        :param mode: "daily" = letzte 7 Tage, "weekly" = letzte 28 Tage.
        :return: Dict mit 'durchgaenge', 'personen', 'tage' (Anzahl Tage mit Daten).
        """
        days = 7 if mode == "daily" else 28
        return self._aggregate_days(days)

    def get_month(self):
        """
        Aggregiert Statistiken für den aktuellen Kalendermonat.
        :return: Dict mit 'durchgaenge', 'personen', 'tage'.
        """
        today = date.today()
        # Alle Tage vom 1. des Monats bis heute
        days_since_month_start = today.day
        return self._aggregate_days(days_since_month_start)

    def get_total(self):
        """
        Gibt die Gesamtstatistik über alle gespeicherten Tage zurück.
        :return: Dict mit 'durchgaenge', 'personen', 'tage'.
        """
        data = self._load()
        total_durchgaenge = 0
        total_personen = 0
        tage_mit_daten = 0
        for entry in data.values():
            if not isinstance(entry, dict):
                continue
            total_durchgaenge += entry.get("durchgaenge", 0)
            total_personen += entry.get("personen", 0)
            tage_mit_daten += 1
        return {
            "durchgaenge": total_durchgaenge,
            "personen": total_personen,
            "tage": tage_mit_daten,
        }

    def reset(self):
        """
        Löscht alle gespeicherten Statistikdaten.
        Wird vom Admin-Menü-Reset aufgerufen.
        """
        try:
            if os.path.exists(self.stats_file):
                os.remove(self.stats_file)
        except OSError:
            pass

    def _aggregate_days(self, num_days):
        """
        Hilfsmethode: summiert Daten für die letzten num_days Tage.
        :param num_days: Anzahl Tage rückwirkend ab heute.
        :return: Dict mit 'durchgaenge', 'personen', 'tage'.
        """
        data = self._load()
        today = date.today()
        total_durchgaenge = 0
        total_personen = 0
        tage_mit_daten = 0
        for i in range(num_days):
            day_str = str(today - timedelta(days=i))
            entry = data.get(day_str)
            if not isinstance(entry, dict):
                continue
            total_durchgaenge += entry.get("durchgaenge", 0)
            total_personen += entry.get("personen", 0)
            tage_mit_daten += 1
        return {
            "durchgaenge": total_durchgaenge,
            "personen": total_personen,
            "tage": tage_mit_daten,
        }

    def _prune_old_entries(self, data):
        """
        Entfernt Einträge, die älter als retention_days sind.
        Wird vor jedem Speichern aufgerufen (in-place).
        :param data: Zu bereinigendes Daten-Dict.
        """
        cutoff = str(date.today() - timedelta(days=self.retention_days))
        old_keys = [k for k in data if isinstance(k, str) and k < cutoff]
        for k in old_keys:
            del data[k]

    def _load(self):
        """
        Lädt die Statistikdatei mit Validierung der einzelnen Einträge.
        Defekte Einträge werden übersprungen statt die ganze Datei zu verlieren.
        :return: Dict mit validierten Tageseinträgen.
        """
        if not os.path.exists(self.stats_file):
            return {}
        try:
            with open(self.stats_file, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}
        # Nur valide Einträge übernehmen: Schlüssel muss YYYY-MM-DD sein, Wert ein Dict
        validated = {}
        for key, value in raw.items():
            if not isinstance(key, str) or len(key) != 10:
                continue
            if not isinstance(value, dict):
                continue
            validated[key] = {
                "durchgaenge": max(0, int(value.get("durchgaenge", 0))),
                "personen": max(0, int(value.get("personen", 0))),
            }
        return validated

    def _save(self, data):
        """
        Speichert die Statistikdatei atomar via temp file + os.replace().
        Verhindert Datenverlust bei Stromausfall im Museumsbetrieb.
        :param data: Zu speicherndes Statistik-Dictionary.
        """
        tmp_path = self.stats_file + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=True, allow_unicode=True)
            os.replace(tmp_path, self.stats_file)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
