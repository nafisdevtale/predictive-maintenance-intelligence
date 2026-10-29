"use client";

import React from "react";
import { SimulatorStepResult } from "@/lib/api";
import { Download, FileSpreadsheet, AlertCircle, Database, Award, ExternalLink } from "lucide-react";

interface ReportsViewProps {
  sessionReadings: SimulatorStepResult[];
}

export function ReportsView({ sessionReadings }: ReportsViewProps) {
  const handleExportCSV = () => {
    if (sessionReadings.length === 0) return;

    const headers = [
      "Tick",
      "Scenario",
      "Machine_Type",
      "Air_temperature_K",
      "Process_temperature_K",
      "Rotational_speed_rpm",
      "Torque_Nm",
      "Tool_wear_min",
      "Tool_Cycle",
      "Tool_Replaced",
      "Estimated_Anomaly_Score",
      "Predicted_Label",
      "Status",
      "Risk_Level",
      "Is_Simulated",
    ];

    const rows = sessionReadings.map((item) => [
      item.metadata.tick,
      item.metadata.scenario,
      item.reading.Type,
      item.reading["Air temperature [K]"],
      item.reading["Process temperature [K]"],
      item.reading["Rotational speed [rpm]"],
      item.reading["Torque [Nm]"],
      item.reading["Tool wear [min]"],
      item.metadata.tool_cycle,
      item.metadata.tool_replaced ? "TRUE" : "FALSE",
      item.prediction.estimated_anomaly_score,
      item.prediction.predicted_label,
      item.prediction.status,
      item.prediction.risk_level,
      "TRUE",
    ]);

    const csvContent =
      "data:text/csv;charset=utf-8," +
      [headers.join(","), ...rows.map((e) => e.join(","))].join("\n");

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `predictive_maintenance_session_telemetry_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6">
      {/* Export Header */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-wrap items-center justify-between gap-4">
        <div>
          <h3 className="text-base font-bold text-slate-900">Session Telemetry Audit & CSV Export</h3>
          <p className="text-xs text-slate-500 mt-1">
            Recorded {sessionReadings.length} sequential readings in this active browser session.
          </p>
        </div>

        <button
          onClick={handleExportCSV}
          disabled={sessionReadings.length === 0}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold shadow-sm transition ${
            sessionReadings.length > 0
              ? "bg-indigo-600 hover:bg-indigo-700 text-white cursor-pointer"
              : "bg-slate-100 text-slate-400 cursor-not-allowed"
          }`}
        >
          <Download className="w-4 h-4" /> Export Session CSV ({sessionReadings.length})
        </button>
      </div>

      {/* Session Telemetry Table */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
        <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-3 flex items-center gap-2">
          <FileSpreadsheet className="w-4 h-4 text-slate-500" /> Active Session Readings
        </h4>

        {sessionReadings.length === 0 ? (
          <div className="p-8 text-center text-slate-500 bg-slate-50 rounded-lg border border-dashed border-slate-300">
            <AlertCircle className="w-8 h-8 text-slate-400 mx-auto mb-2" />
            <p className="text-sm font-medium text-slate-700">No session telemetry recorded yet.</p>
            <p className="text-xs text-slate-500 mt-1">
              Navigate to the <strong>Live Simulator</strong> tab and start streaming to generate and log readings here.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto max-h-96">
            <table className="w-full text-left text-xs border border-slate-200 rounded-lg">
              <thead className="bg-slate-50 text-slate-700 uppercase font-semibold sticky top-0 border-b border-slate-200">
                <tr>
                  <th className="px-3 py-2.5">Tick</th>
                  <th className="px-3 py-2.5">Scenario</th>
                  <th className="px-3 py-2.5">Type</th>
                  <th className="px-3 py-2.5 text-right">Air Temp</th>
                  <th className="px-3 py-2.5 text-right">Proc Temp</th>
                  <th className="px-3 py-2.5 text-right">RPM</th>
                  <th className="px-3 py-2.5 text-right">Torque</th>
                  <th className="px-3 py-2.5 text-right">Wear</th>
                  <th className="px-3 py-2.5 text-right">Score</th>
                  <th className="px-3 py-2.5 text-center">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 font-mono text-[11px]">
                {sessionReadings.map((item, idx) => (
                  <tr key={idx} className="hover:bg-slate-50">
                    <td className="px-3 py-2 font-bold text-slate-900">#{item.metadata.tick}</td>
                    <td className="px-3 py-2 font-sans capitalize text-slate-600">{item.metadata.scenario}</td>
                    <td className="px-3 py-2 font-sans font-semibold text-slate-700">{item.reading.Type}</td>
                    <td className="px-3 py-2 text-right">{item.reading["Air temperature [K]"].toFixed(1)} K</td>
                    <td className="px-3 py-2 text-right">{item.reading["Process temperature [K]"].toFixed(1)} K</td>
                    <td className="px-3 py-2 text-right">{item.reading["Rotational speed [rpm]"]}</td>
                    <td className="px-3 py-2 text-right">{item.reading["Torque [Nm]"].toFixed(1)}</td>
                    <td className="px-3 py-2 text-right">{item.reading["Tool wear [min]"]}m</td>
                    <td className="px-3 py-2 text-right font-bold text-indigo-700">
                      {item.prediction.estimated_anomaly_score.toFixed(4)}
                    </td>
                    <td className="px-3 py-2 text-center font-sans">
                      <span
                        className={`text-[9px] font-bold px-2 py-0.5 rounded-full ${
                          item.prediction.status === "MAINTENANCE_REQUIRED"
                            ? "bg-red-100 text-red-800"
                            : item.prediction.status === "ATTENTION_ADVISED"
                            ? "bg-amber-100 text-amber-800"
                            : "bg-emerald-100 text-emerald-800"
                        }`}
                      >
                        {item.prediction.status.replace("_", " ")}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Attribution & Benchmark Citations */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
        <h4 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
          <Database className="w-4 h-4 text-indigo-600" /> Benchmark Dataset Attribution & Governance
        </h4>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-600">
          <div className="p-4 bg-slate-50 rounded-lg border border-slate-100">
            <span className="font-bold text-slate-800 block mb-1">Primary Citation:</span>
            <p className="italic leading-relaxed">
              Matzka, S. (2020). Explainable Artificial Intelligence for Predictive Maintenance Applications.
              Third International Conference on Artificial Intelligence for Industries (AI4I 2020), pp. 69-74. IEEE.
            </p>
            <a
              href="https://doi.org/10.1109/AI4I49448.2020.00023"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-indigo-600 font-medium hover:underline mt-2"
            >
              DOI: 10.1109/AI4I49448.2020.00023 <ExternalLink className="w-3 h-3" />
            </a>
          </div>

          <div className="p-4 bg-slate-50 rounded-lg border border-slate-100">
            <span className="font-bold text-slate-800 block mb-1">License & Repository:</span>
            <p className="leading-relaxed">
              Published under <strong>Creative Commons Attribution 4.0 International (CC BY 4.0)</strong> license by Stephan Matzka, HTW Berlin, hosted at the UCI Machine Learning Repository.
            </p>
            <a
              href="https://doi.org/10.24432/C5HS5C"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-indigo-600 font-medium hover:underline mt-2"
            >
              UCI DOI: 10.24432/C5HS5C <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
