//+------------------------------------------------------------------+
//| Export_MacroCalendar_MT5_CSV.mq5                                 |
//| Expert Advisor - exportiert den eingebauten MT5-Wirtschafts-     |
//| kalender (MetaQuotes) inkl. ACTUAL, Forecast, Previous, Revised  |
//| und MetaQuotes' Impact-Einstufung (positiv/negativ fuer die      |
//| Waehrung) als CSV fuer das TACO-Lab-Makro-Dashboard.             |
//|                                                                   |
//| AUSFUEHRUNG:                                                     |
//|   1) Datei nach <MT5-Datenordner>/MQL5/Experts/ kopieren und in  |
//|      MetaEditor kompilieren (F7).                                |
//|   2) EA auf irgendeinen Chart ziehen (Symbol/TF egal) und        |
//|      laufen lassen. Handelt nicht, schreibt nur Dateien.         |
//|   3) Exportiert beim Start und danach alle InpTimerMinutes.      |
//|                                                                   |
//| AUSGABE (gemeinsamer Ordner, FILE_COMMON):                        |
//|   <Common>/Files/TACO_macro_recent.csv   (letzte InpRecentDays   |
//|        + naechste InpForwardDays, bei jedem Timer-Lauf)          |
//|   <Common>/Files/TACO_macro_history.csv  (ab InpHistoryFrom,     |
//|        beim Start + einmal pro Tag)                              |
//|                                                                   |
//| Zeitstempel werden nach UTC umgerechnet (aktueller Server-Offset, |
//| fuer historische Werte ggf. +-1h Sommerzeit-Versatz - fuer die   |
//| Tagesauswertung irrelevant).                                     |
//+------------------------------------------------------------------+
#property copyright "TACO Lab"
#property version   "1.00"

input datetime InpHistoryFrom  = D'2015.01.01'; // Historie ab
input int      InpRecentDays   = 60;            // Recent-Datei: Tage zurueck
input int      InpForwardDays  = 14;            // Recent-Datei: Tage voraus
input int      InpTimerMinutes = 10;            // Export-Intervall (Minuten)
input bool     InpIncludeLow   = true;          // auch Low-Impact exportieren

string   g_currencies[] = {"USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"};
datetime g_last_history_day = 0;

//+------------------------------------------------------------------+
int OnInit()
  {
   EventSetTimer(MathMax(1, InpTimerMinutes) * 60);
   RunExport(true);
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason) { EventKillTimer(); }

void OnTimer()
  {
   datetime today = StringToTime(TimeToString(TimeGMT(), TIME_DATE));
   RunExport(today != g_last_history_day);
  }

void OnTick() {}

//+------------------------------------------------------------------+
void RunExport(const bool with_history)
  {
   datetime now_gmt = TimeGMT();
   int recent = ExportRange("TACO_macro_recent.csv",
                            now_gmt - (datetime)InpRecentDays * 86400,
                            now_gmt + (datetime)InpForwardDays * 86400);
   Print("TACO Makro-Kalender: recent exportiert, Zeilen: ", recent);
   if(with_history)
     {
      int hist = ExportRange("TACO_macro_history.csv", InpHistoryFrom, now_gmt);
      g_last_history_day = StringToTime(TimeToString(now_gmt, TIME_DATE));
      Print("TACO Makro-Kalender: Historie exportiert, Zeilen: ", hist);
     }
  }

//+------------------------------------------------------------------+
string ImportanceLabel(const ENUM_CALENDAR_EVENT_IMPORTANCE imp)
  {
   switch(imp)
     {
      case CALENDAR_IMPORTANCE_HIGH:     return "High";
      case CALENDAR_IMPORTANCE_MODERATE: return "Medium";
      case CALENDAR_IMPORTANCE_LOW:      return "Low";
      default:                           return "None";
     }
  }

string ImpactLabel(const ENUM_CALENDAR_EVENT_IMPACT imp)
  {
   if(imp == CALENDAR_IMPACT_POSITIVE) return "positive";
   if(imp == CALENDAR_IMPACT_NEGATIVE) return "negative";
   return "";
  }

string Clean(string s)
  {
   StringReplace(s, ",", " ");
   StringReplace(s, ";", " ");
   StringReplace(s, "\"", "");
   StringReplace(s, "\r", " ");
   StringReplace(s, "\n", " ");
   return s;
  }

string Num(const bool has, const double v)
  {
   return has ? DoubleToString(v, 6) : "";
  }

//+------------------------------------------------------------------+
int ExportRange(const string filename, const datetime from_gmt, const datetime to_gmt)
  {
   // Kalenderzeiten sind Trade-Server-Zeit -> Offset zu GMT
   long server_offset = (long)(TimeTradeServer() - TimeGMT());
   server_offset = (long)MathRound(server_offset / 1800.0) * 1800;
   datetime from_srv = (datetime)(from_gmt + server_offset);
   datetime to_srv   = (datetime)(to_gmt + server_offset);

   string tmp = filename + ".tmp";
   int h = FileOpen(tmp, FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON, ',');
   if(h == INVALID_HANDLE)
     {
      Print("FEHLER: Datei nicht oeffenbar: ", tmp, " Code ", GetLastError());
      return 0;
     }
   FileWrite(h, "value_id", "event_id", "event_code", "currency", "country", "event",
             "importance", "sector", "frequency", "unit", "multiplier", "time_utc", "period",
             "actual", "forecast", "previous", "revised_previous", "impact");

   int rows = 0;
   for(int c = 0; c < ArraySize(g_currencies); c++)
     {
      MqlCalendarEvent events[];
      int n_events = CalendarEventByCurrency(g_currencies[c], events);
      for(int e = 0; e < n_events; e++)
        {
         if(events[e].importance == CALENDAR_IMPORTANCE_NONE)
            continue;
         if(!InpIncludeLow && events[e].importance == CALENDAR_IMPORTANCE_LOW)
            continue;
         MqlCalendarCountry country;
         string country_code = "";
         if(CalendarCountryById(events[e].country_id, country))
            country_code = country.code;

         MqlCalendarValue values[];
         int n_values = CalendarValueHistoryByEvent(events[e].id, values, from_srv, to_srv);
         for(int v = 0; v < n_values; v++)
           {
            datetime t_utc = (datetime)(values[v].time - server_offset);
            FileWrite(h,
                      (string)values[v].id,
                      (string)events[e].id,
                      Clean(events[e].event_code),
                      g_currencies[c],
                      country_code,
                      Clean(events[e].name),
                      ImportanceLabel(events[e].importance),
                      EnumToString(events[e].sector),
                      EnumToString(events[e].frequency),
                      EnumToString(events[e].unit),
                      EnumToString(events[e].multiplier),
                      TimeToString(t_utc, TIME_DATE | TIME_MINUTES),
                      TimeToString(values[v].period, TIME_DATE),
                      Num(values[v].HasActualValue(), values[v].GetActualValue()),
                      Num(values[v].HasForecastValue(), values[v].GetForecastValue()),
                      Num(values[v].HasPreviousValue(), values[v].GetPreviousValue()),
                      Num(values[v].HasRevisedValue(), values[v].GetRevisedValue()),
                      ImpactLabel(values[v].impact_type));
            rows++;
           }
        }
     }
   FileClose(h);
   if(!FileMove(tmp, FILE_COMMON, filename, FILE_COMMON | FILE_REWRITE))
      Print("FEHLER: Umbenennen ", tmp, " -> ", filename, " Code ", GetLastError());
   return rows;
  }
//+------------------------------------------------------------------+
