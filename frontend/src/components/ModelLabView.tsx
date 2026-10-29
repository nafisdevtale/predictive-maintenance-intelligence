"use client";

import React, { useState } from "react";
import { MetricsResponse } from "@/lib/api";
import { Cpu, ShieldAlert, BarChart3, CheckCircle, Crosshair, ArrowUpDown } from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  Cell,
} from "recharts";

interface ModelLabViewProps {
  metrics: MetricsResponse | null;
  loading: boolean;
}

export function ModelLabView({ metrics, loading }: ModelLabViewProps) {
  const [selectedPartition, setSelectedPartition] = useState<"oof" | "test">("test");
  const [selectedCutoff, setSelectedCutoff] = useState<"optimal" | "default">("optimal");

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] text-slate-500">
        <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-slate-700 mb-4"></div>
        <p className="text-sm">Loading comprehensive model evaluation matrices...</p>
      </div>
    );
  }

  if (!metrics) {
    return (
      <div className="p-6 bg-slate-50 rounded-lg text-center text-slate-600">
        Model metrics currently unavailable.
      </div>
    );
  }

  // Extract selected metrics
  const oofRF =
    selectedCutoff === "optimal"
      ? metrics.out_of_fold_validation_results.random_forest.f2_optimal_threshold.optimal_threshold_metrics
      : metrics.out_of_fold_validation_results.random_forest.default_threshold_0_5;

  const oofLR = metrics.out_of_fold_validation_results.logistic_regression.default_threshold_0_5;

  const testRF =
    selectedCutoff === "optimal"
      ? metrics.held_out_test_results.random_forest.training_selected_threshold
      : metrics.held_out_test_results.random_forest.default_threshold_0_5;

  const testLR = metrics.held_out_test_results.logistic_regression.default_threshold_0_5;

  const currentRF = selectedPartition === "test" ? testRF : oofRF;
  const currentLR = selectedPartition === "test" ? testLR : oofLR;

  const cm = currentRF.confusion_matrix;

  // Comparison Bar Chart Data
  const comparisonData = [
    {
      metric: "PR-AUC (AP)",
      "Random Forest": currentRF.average_precision_pr_auc,
      "Logistic Regression": currentLR.average_precision_pr_auc,
    },
    {
      metric: "ROC-AUC",
      "Random Forest": currentRF.roc_auc,
      "Logistic Regression": currentLR.roc_auc,
    },
    {
      metric: "Recall (Sens.)",
      "Random Forest": currentRF.recall,
      "Logistic Regression": currentLR.recall,
    },
    {
      metric: "Precision",
      "Random Forest": currentRF.precision,
      "Logistic Regression": currentLR.precision,
    },
    {
      metric: "F1 Score",
      "Random Forest": currentRF.f1_score,
      "Logistic Regression": currentLR.f1_score,
    },
    {
      metric: "F2 Score",
      "Random Forest": currentRF.f2_score,
      "Logistic Regression": currentLR.f2_score,
    },
    {
      metric: "Bal. Accuracy",
      "Random Forest": currentRF.balanced_accuracy,
      "Logistic Regression": currentLR.balanced_accuracy,
    },
  ];

  // Feature Importance Data
  const importanceData = Object.entries(metrics.feature_importance_rf).map(([name, val]) => ({
    name,
    importance: Number((val * 100).toFixed(1)),
  }));

  return (
    <div className="space-y-6">
      {/* Controls Bar */}
      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-wrap items-center justify-between gap-4">
        <div>
          <span className="text-xs font-bold text-slate-500 uppercase tracking-wider block">Evaluation Mode</span>
          <div className="inline-flex rounded-lg border border-slate-200 p-1 bg-slate-50 mt-1">
            <button
              onClick={() => setSelectedPartition("test")}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition ${
                selectedPartition === "test" ? "bg-white text-slate-900 shadow-sm" : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Held-Out Test Set (Unseen 2,000)
            </button>
            <button
              onClick={() => setSelectedPartition("oof")}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition ${
                selectedPartition === "oof" ? "bg-white text-slate-900 shadow-sm" : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Train 5-Fold Cross-Validation (OOF 8,000)
            </button>
          </div>
        </div>

        <div>
          <span className="text-xs font-bold text-slate-500 uppercase tracking-wider block">Decision Cutoff Strategy</span>
          <div className="inline-flex rounded-lg border border-slate-200 p-1 bg-slate-50 mt-1">
            <button
              onClick={() => setSelectedCutoff("optimal")}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition ${
                selectedCutoff === "optimal"
                  ? "bg-amber-600 text-white shadow-sm"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              F2-Optimal ({metrics.metadata.selected_threshold.toFixed(2)})
            </button>
            <button
              onClick={() => setSelectedCutoff("default")}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition ${
                selectedCutoff === "default"
                  ? "bg-slate-800 text-white shadow-sm"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Default (0.50)
            </button>
          </div>
        </div>
      </div>

      {/* Side-by-Side Model Scorecards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Random Forest Card */}
        <div className="bg-white p-6 rounded-xl border-2 border-indigo-500 shadow-sm relative">
          <span className="absolute -top-3 right-4 bg-indigo-600 text-white text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full">
            Selected Production Champion
          </span>
          <div className="flex items-center gap-2">
            <Cpu className="w-5 h-5 text-indigo-600" />
            <h3 className="text-base font-bold text-slate-900">Random Forest Classifier</h3>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Ensemble of 150 regularized trees with balanced subsampling and physics feature inputs.
          </p>

          <div className="grid grid-cols-3 gap-2 mt-4 text-center">
            <div className="p-2 bg-indigo-50 rounded-lg">
              <span className="text-[10px] text-indigo-700 font-semibold uppercase">PR-AUC (AP)</span>
              <div className="text-xl font-bold text-indigo-900">{currentRF.average_precision_pr_auc.toFixed(3)}</div>
            </div>
            <div className="p-2 bg-indigo-50 rounded-lg">
              <span className="text-[10px] text-indigo-700 font-semibold uppercase">F2 Score</span>
              <div className="text-xl font-bold text-indigo-900">{currentRF.f2_score.toFixed(3)}</div>
            </div>
            <div className="p-2 bg-indigo-50 rounded-lg">
              <span className="text-[10px] text-indigo-700 font-semibold uppercase">Recall</span>
              <div className="text-xl font-bold text-indigo-900">{(currentRF.recall * 100).toFixed(1)}%</div>
            </div>
          </div>
        </div>

        {/* Logistic Regression Card */}
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-slate-500" />
            <h3 className="text-base font-bold text-slate-900">Logistic Regression Baseline</h3>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Linear baseline with balanced class weights and L-BFGS solver.
          </p>

          <div className="grid grid-cols-3 gap-2 mt-4 text-center">
            <div className="p-2 bg-slate-50 rounded-lg">
              <span className="text-[10px] text-slate-600 font-semibold uppercase">PR-AUC (AP)</span>
              <div className="text-xl font-bold text-slate-700">{currentLR.average_precision_pr_auc.toFixed(3)}</div>
            </div>
            <div className="p-2 bg-slate-50 rounded-lg">
              <span className="text-[10px] text-slate-600 font-semibold uppercase">F2 Score</span>
              <div className="text-xl font-bold text-slate-700">{currentLR.f2_score.toFixed(3)}</div>
            </div>
            <div className="p-2 bg-slate-50 rounded-lg">
              <span className="text-[10px] text-slate-600 font-semibold uppercase">Recall</span>
              <div className="text-xl font-bold text-slate-700">{(currentLR.recall * 100).toFixed(1)}%</div>
            </div>
          </div>
        </div>
      </div>

      {/* Comparison Chart & Confusion Matrix */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Metric Comparison Chart */}
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
          <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
            Head-to-Head Metric Comparison ({selectedPartition === "test" ? "Test Set" : "Train OOF"})
          </h4>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparisonData} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
                <XAxis dataKey="metric" stroke="#94a3b8" fontSize={10} angle={-20} textAnchor="end" />
                <YAxis domain={[0, 1]} stroke="#94a3b8" fontSize={10} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Bar dataKey="Random Forest" fill="#4f46e5" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Logistic Regression" fill="#94a3b8" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Confusion Matrix Interactive Visualizer */}
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-col justify-between">
          <div>
            <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
              Confusion Matrix: Random Forest (Cutoff = {currentRF.threshold})
            </h4>
            <p className="text-xs text-slate-500 mb-4">
              Real counts from {selectedPartition === "test" ? "the 2,000 held-out test split" : "the 8,000 training OOF split"}.
            </p>

            <div className="grid grid-cols-2 gap-3 max-w-sm mx-auto">
              <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-lg text-center">
                <span className="text-[10px] font-bold text-emerald-800 uppercase block">True Negatives</span>
                <span className="text-2xl font-bold text-emerald-700">{cm.true_negatives.toLocaleString()}</span>
                <span className="text-[10px] text-emerald-600 block mt-0.5">Correct Normal</span>
              </div>

              <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg text-center">
                <span className="text-[10px] font-bold text-amber-800 uppercase block">False Positives</span>
                <span className="text-2xl font-bold text-amber-700">{cm.false_positives.toLocaleString()}</span>
                <span className="text-[10px] text-amber-600 block mt-0.5">False Alarm Inspection</span>
              </div>

              <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-center">
                <span className="text-[10px] font-bold text-red-800 uppercase block">False Negatives</span>
                <span className="text-2xl font-bold text-red-700">{cm.false_negatives.toLocaleString()}</span>
                <span className="text-[10px] text-red-600 block mt-0.5">Missed Breakdown</span>
              </div>

              <div className="p-3 bg-indigo-50 border border-indigo-200 rounded-lg text-center">
                <span className="text-[10px] font-bold text-indigo-800 uppercase block">True Positives</span>
                <span className="text-2xl font-bold text-indigo-700">{cm.true_positives.toLocaleString()}</span>
                <span className="text-[10px] text-indigo-600 block mt-0.5">Caught Failure</span>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] text-slate-600">
            {currentRF.operational_tradeoff_note}
          </div>
        </div>
      </div>

      {/* Feature Importance Bar Chart */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
        <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-1">
          Random Forest Feature Importance (MDI)
        </h4>
        <p className="text-xs text-slate-500 mb-4">
          Physics-derived features (<code className="text-indigo-600 font-mono">Power [W]</code> and <code className="text-indigo-600 font-mono">Temp difference [K]</code>) contribute over 29% of decision split gain.
        </p>

        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={importanceData} layout="vertical" margin={{ top: 5, right: 30, left: 100, bottom: 5 }}>
              <XAxis type="number" domain={[0, 25]} unit="%" fontSize={10} />
              <YAxis dataKey="name" type="category" fontSize={10} width={130} />
              <Tooltip />
              <Bar dataKey="importance" fill="#4f46e5" radius={[0, 4, 4, 0]}>
                {importanceData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.name.includes("[W]") || entry.name.includes("difference") ? "#10b981" : "#4f46e5"}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
