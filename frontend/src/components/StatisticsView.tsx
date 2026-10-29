"use client";

import React from "react";
import { StatisticsResponse } from "@/lib/api";
import { Scale, CheckCircle2, AlertCircle, FileText, TrendingUp, Info } from "lucide-react";

interface StatisticsViewProps {
  stats: StatisticsResponse | null;
  loading: boolean;
}

export function StatisticsView({ stats, loading }: StatisticsViewProps) {
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] text-slate-500">
        <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-slate-700 mb-4"></div>
        <p className="text-sm">Loading SciPy hypothesis testing results and FDR corrections...</p>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="p-6 bg-slate-50 rounded-lg text-center text-slate-600">
        Statistical analysis data currently unavailable.
      </div>
    );
  }

  const cat = stats.categorical_test;

  return (
    <div className="space-y-6">
      {/* Overview Banner */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex items-start gap-4">
        <Scale className="w-8 h-8 text-indigo-600 flex-shrink-0 mt-1" />
        <div>
          <h3 className="text-base font-bold text-slate-900">SciPy Statistical Hypothesis Testing Engine</h3>
          <p className="text-xs text-slate-600 mt-1 leading-relaxed">
            Non-parametric two-sample <strong>Mann-Whitney U tests</strong> were conducted across all numeric and physics-engineered predictors comparing failure (n = 339) against normal operating machines (n = 9,661). Multiple comparison family-wise error rates are controlled via <strong>Benjamini-Hochberg (BH) False Discovery Rate (FDR)</strong> adjustments at alpha = 0.05.
          </p>
        </div>
      </div>

      {/* Numeric Hypothesis Testing Table */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <h4 className="text-sm font-bold text-slate-900 uppercase tracking-wider mb-1">
          Numeric Sensor & Feature Comparisons (Mann-Whitney U with FDR Correction)
        </h4>
        <p className="text-xs text-slate-500 mb-4">
          All 7 tested features maintained statistical significance (adjusted p &lt; 0.05) after rigorous Benjamini-Hochberg FDR correction.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-600 border border-slate-200 rounded-lg">
            <thead className="bg-slate-50 text-slate-700 uppercase font-semibold border-b border-slate-200">
              <tr>
                <th className="px-3 py-3">Feature</th>
                <th className="px-3 py-3 text-right">Non-Fail Median</th>
                <th className="px-3 py-3 text-right">Fail Median</th>
                <th className="px-3 py-3 text-right">Δ Median</th>
                <th className="px-3 py-3 text-right">Rank-Biserial (r)</th>
                <th className="px-3 py-3 text-right">Raw p-value</th>
                <th className="px-3 py-3 text-right">FDR Adjusted p</th>
                <th className="px-3 py-3 text-center">FDR Significant?</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 font-mono text-[11px]">
              {stats.numeric_tests.map((test, idx) => (
                <tr key={idx} className={test.is_significant_fdr ? "bg-white hover:bg-slate-50" : "bg-slate-50"}>
                  <td className="px-3 py-2.5 font-sans font-medium text-slate-900">{test.feature}</td>
                  <td className="px-3 py-2.5 text-right">{test.median_non_failure.toFixed(2)}</td>
                  <td className="px-3 py-2.5 text-right">{test.median_failure.toFixed(2)}</td>
                  <td className={`px-3 py-2.5 text-right font-bold ${test.median_difference > 0 ? "text-indigo-600" : "text-amber-600"}`}>
                    {test.median_difference > 0 ? `+${test.median_difference.toFixed(2)}` : test.median_difference.toFixed(2)}
                  </td>
                  <td className="px-3 py-2.5 text-right font-bold">
                    {test.rank_biserial_r > 0 ? `+${test.rank_biserial_r.toFixed(4)}` : test.rank_biserial_r.toFixed(4)}
                  </td>
                  <td className="px-3 py-2.5 text-right text-slate-500">{test.raw_p_value.toExponential(2)}</td>
                  <td className="px-3 py-2.5 text-right text-indigo-700 font-semibold">{test.fdr_adjusted_p_value.toExponential(2)}</td>
                  <td className="px-3 py-2.5 text-center font-sans">
                    {test.is_significant_fdr ? (
                      <span className="inline-flex items-center gap-1 text-[10px] bg-emerald-100 text-emerald-800 font-bold px-2 py-0.5 rounded-full">
                        <CheckCircle2 className="w-3 h-3" /> YES
                      </span>
                    ) : (
                      <span className="text-[10px] bg-slate-200 text-slate-700 px-2 py-0.5 rounded-full">NO</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Categorical Chi-Square Contingency Test */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
          <h4 className="text-sm font-bold text-slate-900 uppercase tracking-wider mb-2">
            Machine Variant Contingency (Type vs Failure)
          </h4>
          <p className="text-xs text-slate-500 mb-4">
            Contingency cross-tabulation and empirical failure percentage per machine quality type.
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border border-slate-200 rounded-lg">
              <thead className="bg-slate-50 text-slate-700 uppercase font-semibold">
                <tr>
                  <th className="px-3 py-2">Variant</th>
                  <th className="px-3 py-2 text-right">Normal (0)</th>
                  <th className="px-3 py-2 text-right">Failure (1)</th>
                  <th className="px-3 py-2 text-right">Failure Rate (%)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {Object.entries(cat.breakdown_by_type).map(([tName, item]) => (
                  <tr key={tName}>
                    <td className="px-3 py-2.5 font-bold text-slate-900">Type {tName}</td>
                    <td className="px-3 py-2.5 text-right font-mono">{(item.total_count - item.failure_count).toLocaleString()}</td>
                    <td className="px-3 py-2.5 text-right font-mono font-bold text-red-600">{item.failure_count.toLocaleString()}</td>
                    <td className="px-3 py-2.5 text-right font-mono font-bold text-slate-800">{item.failure_rate_pct.toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-col justify-between">
          <div>
            <h4 className="text-sm font-bold text-slate-900 uppercase tracking-wider mb-2">
              Chi-Square Test of Independence Metrics
            </h4>
            <div className="space-y-3 mt-4 text-xs">
              <div className="flex justify-between pb-2 border-b border-slate-100">
                <span className="text-slate-500">Chi-Square Statistic (χ²)</span>
                <span className="font-mono font-bold text-slate-900">{cat.chi2_statistic.toFixed(4)}</span>
              </div>
              <div className="flex justify-between pb-2 border-b border-slate-100">
                <span className="text-slate-500">Degrees of Freedom (df)</span>
                <span className="font-mono font-bold text-slate-900">{cat.degrees_of_freedom}</span>
              </div>
              <div className="flex justify-between pb-2 border-b border-slate-100">
                <span className="text-slate-500">Asymptotic p-value</span>
                <span className="font-mono font-bold text-indigo-700">{cat.p_value.toExponential(3)}</span>
              </div>
              <div className="flex justify-between pb-2 border-b border-slate-100">
                <span className="text-slate-500">Cramer&apos;s V Effect Size</span>
                <span className="font-mono font-bold text-amber-700">{cat.cramers_v.toFixed(4)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Cochran&apos;s Expected Counts Valid</span>
                <span className="font-bold text-emerald-600">
                  {cat.assumptions.assumptions_met ? "Yes (100% cells ≥ 5)" : "No"}
                </span>
              </div>
            </div>
          </div>

          <p className="text-[11px] text-slate-500 mt-4 pt-3 border-t border-slate-100">
            {stats.interpretations.machine_type_association.interpretation}
          </p>
        </div>
      </div>

      {/* Cautious Analytical Interpretations */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
        <h4 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
          <Info className="w-4 h-4 text-indigo-600" /> Cautious Scientific Interpretations & Limitations
        </h4>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-600">
          <div className="p-4 bg-slate-50 rounded-lg border border-slate-100 space-y-1">
            <span className="font-bold text-slate-800 block">1. Correlation Does Not Prove Causation</span>
            <p className="leading-relaxed">{stats.interpretations.causality_caveat}</p>
          </div>

          <div className="p-4 bg-slate-50 rounded-lg border border-slate-100 space-y-1">
            <span className="font-bold text-slate-800 block">2. Synthetic Generator Benchmark</span>
            <p className="leading-relaxed">{stats.interpretations.dataset_nature}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
