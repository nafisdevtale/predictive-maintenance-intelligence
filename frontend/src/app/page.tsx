"use client";

import React, { useState, useEffect } from "react";
import {
  checkBackendHealth,
  fetchOverviewData,
  fetchModelMetrics,
  fetchStatistics,
  OverviewResponse,
  MetricsResponse,
  StatisticsResponse,
  SimulatorStepResult,
} from "@/lib/api";
import {
  LayoutDashboard,
  Activity,
  Cpu,
  BarChart2,
  FileSpreadsheet,
  AlertTriangle,
  RefreshCw,
  Server,
} from "lucide-react";
import { EducationalDisclaimerBanner, ApiDisconnectedBanner } from "@/components/Disclaimers";
import { OverviewView } from "@/components/OverviewView";
import { SimulatorView } from "@/components/SimulatorView";
import { ModelLabView } from "@/components/ModelLabView";
import { StatisticsView } from "@/components/StatisticsView";
import { ReportsView } from "@/components/ReportsView";

type TabType = "overview" | "simulator" | "model-lab" | "statistics" | "reports";

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabType>("overview");
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);

  // Cached API data
  const [overview, setOverview] = useState<OverviewResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [statistics, setStatistics] = useState<StatisticsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  // Client-side session simulator history
  const [sessionReadings, setSessionReadings] = useState<SimulatorStepResult[]>([]);

  const loadAllData = async () => {
    setLoading(true);
    try {
      await checkBackendHealth();
      setApiConnected(true);

      const [ov, met, stat] = await Promise.all([
        fetchOverviewData(),
        fetchModelMetrics(),
        fetchStatistics(),
      ]);

      setOverview(ov);
      setMetrics(met);
      setStatistics(stat);
    } catch (err) {
      setApiConnected(false);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAllData();
  }, []);

  const handleSessionReadingAdded = (newStep: SimulatorStepResult) => {
    setSessionReadings((prev) => [newStep, ...prev]);
  };

  const decisionThreshold = overview?.selected_model.decision_threshold ?? 0.31;

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col font-sans">
      {/* Top Application Header */}
      <header className="bg-slate-900 border-b border-slate-800 text-white sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center font-bold text-white shadow-md">
              PM
            </div>
            <div>
              <h1 className="text-base font-bold tracking-tight text-white flex items-center gap-2">
                Predictive Maintenance Intelligence System
                <span className="text-[10px] font-semibold bg-indigo-900 text-indigo-300 px-2 py-0.5 rounded border border-indigo-700">
                  UCI AI4I 2020
                </span>
              </h1>
              <p className="text-[11px] text-slate-400">Condition-Based Anomaly Estimation & Diagnostic Intelligence</p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 text-xs">
              <span
                className={`w-2 h-2 rounded-full ${
                  apiConnected === true ? "bg-emerald-400 animate-pulse" : apiConnected === false ? "bg-red-400" : "bg-amber-400"
                }`}
              />
              <span className="text-slate-300 hidden sm:inline">
                {apiConnected === true
                  ? `Backend: Online (${overview?.selected_model.model_name ?? "RF"})`
                  : apiConnected === false
                  ? "Backend: Disconnected"
                  : "Checking Backend..."}
              </span>
            </div>

            <button
              onClick={loadAllData}
              className="p-1.5 text-slate-400 hover:text-white rounded-md hover:bg-slate-800 transition"
              title="Refresh Data"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Navigation Tabs Bar */}
        <div className="bg-slate-950 border-t border-slate-800/80 px-4 sm:px-6 lg:px-8">
          <div className="max-w-7xl mx-auto flex space-x-1 sm:space-x-2 overflow-x-auto py-2 scrollbar-none">
            <button
              onClick={() => setActiveTab("overview")}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition ${
                activeTab === "overview"
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              }`}
            >
              <LayoutDashboard className="w-4 h-4" /> Overview
            </button>

            <button
              onClick={() => setActiveTab("simulator")}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition ${
                activeTab === "simulator"
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              }`}
            >
              <Activity className="w-4 h-4" /> Live Simulator
            </button>

            <button
              onClick={() => setActiveTab("model-lab")}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition ${
                activeTab === "model-lab"
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              }`}
            >
              <Cpu className="w-4 h-4" /> Model Lab
            </button>

            <button
              onClick={() => setActiveTab("statistics")}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition ${
                activeTab === "statistics"
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              }`}
            >
              <BarChart2 className="w-4 h-4" /> Statistical Analysis
            </button>

            <button
              onClick={() => setActiveTab("reports")}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition ${
                activeTab === "reports"
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              }`}
            >
              <FileSpreadsheet className="w-4 h-4" /> Reports & Data
              {sessionReadings.length > 0 && (
                <span className="ml-1 bg-indigo-500/30 text-indigo-300 text-[10px] px-1.5 py-0.2 rounded-full font-mono">
                  {sessionReadings.length}
                </span>
              )}
            </button>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <EducationalDisclaimerBanner />

        {apiConnected === false && <ApiDisconnectedBanner onRetry={loadAllData} />}

        {activeTab === "overview" && <OverviewView data={overview} loading={loading} />}

        {activeTab === "simulator" && (
          <SimulatorView
            onSessionReadingAdded={handleSessionReadingAdded}
            decisionThreshold={decisionThreshold}
          />
        )}

        {activeTab === "model-lab" && <ModelLabView metrics={metrics} loading={loading} />}

        {activeTab === "statistics" && <StatisticsView stats={statistics} loading={loading} />}

        {activeTab === "reports" && <ReportsView sessionReadings={sessionReadings} />}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 py-6 mt-12 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 space-y-1">
          <p className="font-medium text-slate-700">Predictive Maintenance Intelligence System — College Data Science Capstone</p>
          <p>
            Trained on the synthetic UCI AI4I 2020 Predictive Maintenance Dataset (DOI: 10.24432/C5HS5C, CC BY 4.0).
          </p>
          <p className="text-[11px] text-slate-400">
            Designed for dual Vercel project deployment: FastAPI backend at root, Next.js App Router at frontend/.
          </p>
        </div>
      </footer>
    </div>
  );
}
