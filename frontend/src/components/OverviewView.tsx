"use client";

import React from "react";
import { OverviewResponse } from "@/lib/api";
import { ShieldCheck, Cpu, Database, Activity, Layers, Wrench, BarChart2 } from "lucide-react";
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip, BarChart, Bar, XAxis, YAxis } from "recharts";

interface OverviewViewProps {
  data: OverviewResponse | null;
  loading: boolean;
}

export function OverviewView({ data, loading }: OverviewViewProps) {
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] text-slate-500">
        <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-slate-700 mb-4"></div>
        <p className="text-sm">Loading dataset overview and system telemetry...</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="p-6 bg-slate-50 rounded-lg text-center text-slate-600">
        Overview data currently unavailable. Ensure the backend server is running.
      </div>
    );
  }

  const classData = [
    { name: "Normal Operation (0)", value: data.class_distribution.normal_count, color: "#10b981" },
    { name: "Machine Failure (1)", value: data.class_distribution.failure_count, color: "#ef4444" },
  ];

  const variantData = Object.entries(data.quality_variants).map(([name, item]) => ({
    name,
    count: item.count,
    pct: item.pct,
  }));

  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Benchmark Records</span>
            <Database className="w-5 h-5 text-indigo-500" />
          </div>
          <div className="mt-3">
            <span className="text-3xl font-bold text-slate-900">{data.total_observations.toLocaleString()}</span>
            <p className="text-xs text-slate-500 mt-1">AI4I 2020 Milling Telemetry</p>
          </div>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Failure Rate</span>
            <Activity className="w-5 h-5 text-red-500" />
          </div>
          <div className="mt-3">
            <span className="text-3xl font-bold text-red-600">{data.class_distribution.failure_rate_pct}%</span>
            <p className="text-xs text-slate-500 mt-1">{data.class_distribution.failure_count} total failure events</p>
          </div>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Deployed Model</span>
            <Cpu className="w-5 h-5 text-emerald-500" />
          </div>
          <div className="mt-3">
            <span className="text-2xl font-bold text-slate-900">{data.selected_model.model_name}</span>
            <p className="text-xs text-emerald-600 mt-1 font-medium">Selected via Train-OOF PR-AUC</p>
          </div>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Operating Cutoff</span>
            <ShieldCheck className="w-5 h-5 text-amber-500" />
          </div>
          <div className="mt-3">
            <span className="text-3xl font-bold text-amber-600">{data.selected_model.decision_threshold.toFixed(2)}</span>
            <p className="text-xs text-slate-500 mt-1">F2-Optimal Recall Prioritization</p>
          </div>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
            <BarChart2 className="w-4 h-4 text-slate-500" /> Class Balance Imbalance (96.6% vs 3.4%)
          </h3>
          <p className="text-xs text-slate-500 mt-1">
            Severe real-world class imbalance (~28:1 ratio) requiring cost-sensitive and balanced learning.
          </p>
          <div className="h-60 mt-4">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={classData}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={80}
                  paddingAngle={4}
                  dataKey="value"
                  label={({ name, percent }) => `${name.split(" ")[0]} ${(percent * 100).toFixed(1)}%`}
                >
                  {classData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
            <Layers className="w-4 h-4 text-slate-500" /> Machine Variant Distribution (Type)
          </h3>
          <p className="text-xs text-slate-500 mt-1">
            Quality variants: Low (50%), Medium (30%), High (20%) with differing baseline durability.
          </p>
          <div className="h-60 mt-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={variantData} layout="vertical" margin={{ top: 10, right: 30, left: 20, bottom: 5 }}>
                <XAxis type="number" domain={[0, 7000]} />
                <YAxis dataKey="name" type="category" width={80} />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Engineering & Methodology Architecture */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
        <h3 className="text-base font-bold text-slate-900">Strict Data Science Pipeline Architecture</h3>
        <p className="text-xs text-slate-500 mt-1">
          Guaranteed leakage prevention and statistical rigor implemented across all phases.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
          <div className="p-4 bg-slate-50 rounded-lg border border-slate-100">
            <div className="flex items-center gap-2 text-indigo-700 font-semibold text-sm">
              <ShieldCheck className="w-4 h-4" /> 1. Leakage Prevention
            </div>
            <p className="text-xs text-slate-600 mt-2 leading-relaxed">
              Target indicators (<code className="text-indigo-600 font-mono">TWF, HDF, PWF, OSF, RNF</code>) and serial tags (<code className="text-indigo-600 font-mono">UDI, Product ID</code>) are 100% excluded from model predictors.
            </p>
          </div>

          <div className="p-4 bg-slate-50 rounded-lg border border-slate-100">
            <div className="flex items-center gap-2 text-emerald-700 font-semibold text-sm">
              <Wrench className="w-4 h-4" /> 2. Physics Feature Engineering
            </div>
            <p className="text-xs text-slate-600 mt-2 leading-relaxed">
              Derived thermal dissipation proxy (<code className="text-emerald-600 font-mono">Process Temp - Air Temp</code>) and mechanical power (<code className="text-emerald-600 font-mono">Torque × Speed × 2π/60</code>) accounting for &gt;29% of feature importance.
            </p>
          </div>

          <div className="p-4 bg-slate-50 rounded-lg border border-slate-100">
            <div className="flex items-center gap-2 text-amber-700 font-semibold text-sm">
              <Activity className="w-4 h-4" /> 3. Quarantined Test Split
            </div>
            <p className="text-xs text-slate-600 mt-2 leading-relaxed">
              Held-out 20% test set was never accessed during preprocessor fitting, cross-validation, or threshold tuning. Decision cutoff (0.31) tuned solely on training out-of-fold F2-score.
            </p>
          </div>
        </div>
      </div>

      {/* Sensor Specifications Table */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">Permitted Sensor Telemetry Inputs</h3>
        <p className="text-xs text-slate-500 mt-1">
          Operational domain ranges derived from the empirical UCI AI4I 2020 distribution.
        </p>

        <div className="overflow-x-auto mt-4">
          <table className="w-full text-left text-xs text-slate-600 border border-slate-200 rounded-lg">
            <thead className="bg-slate-50 text-slate-700 uppercase font-semibold border-b border-slate-200">
              <tr>
                <th className="px-4 py-3">Telemetry Feature</th>
                <th className="px-4 py-3">Type</th>
                <th className="px-4 py-3">Valid Range</th>
                <th className="px-4 py-3">Nominal Center</th>
                <th className="px-4 py-3">Physical Domain Meaning</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              <tr>
                <td className="px-4 py-2.5 font-medium text-slate-900">Machine Type</td>
                <td className="px-4 py-2.5">Categorical</td>
                <td className="px-4 py-2.5">L, M, H</td>
                <td className="px-4 py-2.5">L</td>
                <td className="px-4 py-2.5">Product quality variant with differing metallurgy tolerances</td>
              </tr>
              <tr>
                <td className="px-4 py-2.5 font-medium text-slate-900">Air temperature [K]</td>
                <td className="px-4 py-2.5">Continuous</td>
                <td className="px-4 py-2.5">295.3 - 304.5 K</td>
                <td className="px-4 py-2.5">300.0 K</td>
                <td className="px-4 py-2.5">Ambient environment temperature in shop floor</td>
              </tr>
              <tr>
                <td className="px-4 py-2.5 font-medium text-slate-900">Process temperature [K]</td>
                <td className="px-4 py-2.5">Continuous</td>
                <td className="px-4 py-2.5">305.7 - 313.8 K</td>
                <td className="px-4 py-2.5">310.0 K</td>
                <td className="px-4 py-2.5">Spindle / cutting zone friction temperature</td>
              </tr>
              <tr>
                <td className="px-4 py-2.5 font-medium text-slate-900">Rotational speed [rpm]</td>
                <td className="px-4 py-2.5">Integer</td>
                <td className="px-4 py-2.5">1,168 - 2,886 rpm</td>
                <td className="px-4 py-2.5">1,503 rpm</td>
                <td className="px-4 py-2.5">Milling spindle rotation velocity</td>
              </tr>
              <tr>
                <td className="px-4 py-2.5 font-medium text-slate-900">Torque [Nm]</td>
                <td className="px-4 py-2.5">Continuous</td>
                <td className="px-4 py-2.5">3.8 - 76.6 Nm</td>
                <td className="px-4 py-2.5">40.0 Nm</td>
                <td className="px-4 py-2.5">Motor resistance torque under cutting engagement</td>
              </tr>
              <tr>
                <td className="px-4 py-2.5 font-medium text-slate-900">Tool wear [min]</td>
                <td className="px-4 py-2.5">Integer</td>
                <td className="px-4 py-2.5">0 - 253 min</td>
                <td className="px-4 py-2.5">108 min</td>
                <td className="px-4 py-2.5">Cumulative cutting insert operational duration</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
