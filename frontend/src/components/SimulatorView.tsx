"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  initializeSimulator,
  stepSimulator,
  SimulatorStepResult,
  SensorReading,
} from "@/lib/api";
import {
  Play,
  Pause,
  RotateCcw,
  SkipForward,
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Thermometer,
  Gauge,
  Zap,
} from "lucide-react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ReferenceLine,
} from "recharts";

interface SimulatorViewProps {
  onSessionReadingAdded: (reading: SimulatorStepResult) => void;
  decisionThreshold: number;
}

export function SimulatorView({ onSessionReadingAdded, decisionThreshold }: SimulatorViewProps) {
  const [scenario, setScenario] = useState<string>("normal");
  const [machineType, setMachineType] = useState<string>("L");
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [stepIntervalMs, setStepIntervalMs] = useState<number>(1500);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Simulator state stored in client memory (Vercel serverless friendly)
  const [simState, setSimState] = useState<any>(null);
  const [latestStep, setLatestStep] = useState<SimulatorStepResult | null>(null);
  const [telemetryHistory, setTelemetryHistory] = useState<any[]>([]);

  const isRunningRef = useRef(isRunning);
  isRunningRef.current = isRunning;
  const simStateRef = useRef(simState);
  simStateRef.current = simState;

  // Initialize simulator
  const handleInitialize = async (selectedScenario = scenario, selectedType = machineType) => {
    setLoading(true);
    setError(null);
    try {
      const res = await initializeSimulator(selectedScenario, selectedType, Date.now() % 10000);
      setSimState(res.state);
      setLatestStep(null);
      setTelemetryHistory([]);
    } catch (err: any) {
      setError(err.message || "Failed to initialize simulator");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    handleInitialize();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Step function
  const handleStep = async () => {
    if (!simStateRef.current) return;
    try {
      const result = await stepSimulator(simStateRef.current);
      setSimState(result.next_state);
      setLatestStep(result);
      onSessionReadingAdded(result);

      // Add to Recharts history buffer (max 30 ticks)
      const dataPoint = {
        tick: result.metadata.tick,
        score: result.prediction.estimated_anomaly_score,
        air_temp: result.reading["Air temperature [K]"],
        proc_temp: result.reading["Process temperature [K]"],
        speed: result.reading["Rotational speed [rpm]"],
        torque: result.reading["Torque [Nm]"],
        tool_wear: result.reading["Tool wear [min]"],
      };

      setTelemetryHistory((prev) => {
        const next = [...prev, dataPoint];
        return next.length > 30 ? next.slice(next.length - 30) : next;
      });
    } catch (err: any) {
      setError(err.message || "Error advancing simulation step");
      setIsRunning(false);
    }
  };

  // Interval loop
  useEffect(() => {
    if (!isRunning) return;

    const timer = setInterval(() => {
      handleStep();
    }, stepIntervalMs);

    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isRunning, stepIntervalMs]);

  const handleScenarioChange = (newScenario: string) => {
    setScenario(newScenario);
    setIsRunning(false);
    handleInitialize(newScenario, machineType);
  };

  const handleTypeChange = (newType: string) => {
    setMachineType(newType);
    setIsRunning(false);
    handleInitialize(scenario, newType);
  };

  const currentScore = latestStep?.prediction.estimated_anomaly_score ?? 0.0;
  const currentStatus = latestStep?.prediction.status ?? "NORMAL_OPERATION";
  const currentReading = latestStep?.reading;

  return (
    <div className="space-y-6">
      {/* Simulation Notice */}
      <div className="bg-indigo-50 border border-indigo-200 px-4 py-3 rounded-xl flex items-center justify-between text-xs text-indigo-900">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-indigo-600 animate-pulse" />
          <span>
            <strong>Simulated Telemetry Stream:</strong> Generating physics-grounded random-walk inputs within empirical AI4I 2020 domain ranges. Client-driven stateless execution.
          </span>
        </div>
        <span className="bg-indigo-200 text-indigo-800 font-bold px-2 py-0.5 rounded text-[10px] uppercase">
          SYNTHETIC TELEMETRY
        </span>
      </div>

      {error && (
        <div className="p-4 bg-red-50 border-l-4 border-red-500 text-red-700 text-xs rounded-r-lg">
          {error}
        </div>
      )}

      {/* Control Panel */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
          <div>
            <label className="block text-xs font-semibold text-slate-700 uppercase mb-1">
              Operational Scenario
            </label>
            <select
              value={scenario}
              onChange={(e) => handleScenarioChange(e.target.value)}
              className="w-full text-sm border border-slate-300 rounded-lg px-3 py-2 bg-slate-50 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value="normal">Normal Operation (Baseline)</option>
              <option value="elevated_load">Elevated Load (High Torque / Low Speed)</option>
              <option value="thermal_stress">Thermal Stress (Poor Heat Dissipation)</option>
              <option value="high_tool_wear">High Tool Wear (End of Life Cycle)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 uppercase mb-1">
              Machine Quality Variant
            </label>
            <select
              value={machineType}
              onChange={(e) => handleTypeChange(e.target.value)}
              className="w-full text-sm border border-slate-300 rounded-lg px-3 py-2 bg-slate-50 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value="L">Type L (Low Quality - 60% of fleet)</option>
              <option value="M">Type M (Medium Quality - 30% of fleet)</option>
              <option value="H">Type H (High Quality - 10% of fleet)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 uppercase mb-1">
              Stream Pace
            </label>
            <select
              value={stepIntervalMs}
              onChange={(e) => setStepIntervalMs(Number(e.target.value))}
              className="w-full text-sm border border-slate-300 rounded-lg px-3 py-2 bg-slate-50 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value={1000}>Fast (1.0 sec / tick)</option>
              <option value={1500}>Normal (1.5 sec / tick)</option>
              <option value={2500}>Slow (2.5 sec / tick)</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            {!isRunning ? (
              <button
                onClick={() => setIsRunning(true)}
                disabled={loading || !simState}
                className="flex-1 flex items-center justify-center gap-2 bg-emerald-600 hover:bg-emerald-700 text-white font-medium py-2 px-4 rounded-lg text-sm shadow transition"
              >
                <Play className="w-4 h-4" /> Start
              </button>
            ) : (
              <button
                onClick={() => setIsRunning(false)}
                className="flex-1 flex items-center justify-center gap-2 bg-amber-600 hover:bg-amber-700 text-white font-medium py-2 px-4 rounded-lg text-sm shadow transition"
              >
                <Pause className="w-4 h-4" /> Pause
              </button>
            )}

            <button
              onClick={handleStep}
              disabled={isRunning || loading || !simState}
              className="p-2 border border-slate-300 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg transition"
              title="Step Single Tick"
            >
              <SkipForward className="w-5 h-5" />
            </button>

            <button
              onClick={() => {
                setIsRunning(false);
                handleInitialize();
              }}
              className="p-2 border border-slate-300 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg transition"
              title="Reset Simulator"
            >
              <RotateCcw className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>

      {/* Live Status & Gauge Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Anomaly Gauge Card */}
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-col justify-between">
          <div>
            <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">
              Random Forest Model Assessment
            </span>
            <div className="mt-4 flex items-baseline justify-between">
              <div>
                <span className="text-4xl font-extrabold text-slate-900">
                  {currentScore.toFixed(4)}
                </span>
                <span className="text-xs text-slate-500 ml-2">Score [0.0 - 1.0]</span>
              </div>
              <div
                className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${
                  currentStatus === "MAINTENANCE_REQUIRED"
                    ? "bg-red-100 text-red-800 border border-red-300"
                    : currentStatus === "ATTENTION_ADVISED"
                    ? "bg-amber-100 text-amber-800 border border-amber-300"
                    : "bg-emerald-100 text-emerald-800 border border-emerald-300"
                }`}
              >
                {currentStatus.replace("_", " ")}
              </div>
            </div>

            {/* Score Bar with Threshold */}
            <div className="mt-4 relative pt-1">
              <div className="overflow-hidden h-3 text-xs flex rounded-full bg-slate-100 border border-slate-200">
                <div
                  style={{ width: `${Math.min(currentScore * 100, 100)}%` }}
                  className={`shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center transition-all duration-300 ${
                    currentScore >= decisionThreshold ? "bg-red-500" : "bg-emerald-500"
                  }`}
                />
              </div>
              {/* Threshold pointer */}
              <div
                className="absolute top-0 w-0.5 h-5 bg-slate-900"
                style={{ left: `${decisionThreshold * 100}%` }}
                title={`Decision Cutoff: ${decisionThreshold}`}
              />
              <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                <span>0.00 (Safe)</span>
                <span className="font-semibold text-slate-700">Cutoff: {decisionThreshold.toFixed(2)}</span>
                <span>1.00 (Failure)</span>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-4 border-t border-slate-100 text-[11px] text-slate-500">
            {latestStep?.prediction.score_interpretation ||
              "Run the simulator to stream live synthetic telemetry into the model."}
          </div>
        </div>

        {/* Current Live Sensor Readings */}
        <div className="lg:col-span-2 bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">
              Instantaneous Sensor Readings (Tick #{latestStep?.metadata.tick ?? 0})
            </span>
            {latestStep?.metadata.tool_replaced && (
              <span className="text-xs bg-indigo-100 text-indigo-700 font-semibold px-2 py-0.5 rounded">
                Tool Insert Replaced (Cycle #{latestStep.metadata.tool_cycle})
              </span>
            )}
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <div className="flex items-center gap-1.5 text-slate-500 text-xs font-medium">
                <Thermometer className="w-3.5 h-3.5" /> Air Temperature
              </div>
              <div className="text-lg font-bold text-slate-900 mt-1">
                {currentReading ? `${currentReading["Air temperature [K]"]} K` : "--"}
              </div>
              <span className="text-[10px] text-slate-400">Range: 295.3 - 304.5 K</span>
            </div>

            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <div className="flex items-center gap-1.5 text-slate-500 text-xs font-medium">
                <Thermometer className="w-3.5 h-3.5 text-red-500" /> Process Temp
              </div>
              <div className="text-lg font-bold text-slate-900 mt-1">
                {currentReading ? `${currentReading["Process temperature [K]"]} K` : "--"}
              </div>
              <span className="text-[10px] text-slate-400">Range: 305.7 - 313.8 K</span>
            </div>

            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <div className="flex items-center gap-1.5 text-slate-500 text-xs font-medium">
                <Gauge className="w-3.5 h-3.5 text-blue-500" /> Rotational Speed
              </div>
              <div className="text-lg font-bold text-slate-900 mt-1">
                {currentReading ? `${currentReading["Rotational speed [rpm]"]} rpm` : "--"}
              </div>
              <span className="text-[10px] text-slate-400">Range: 1,168 - 2,886 rpm</span>
            </div>

            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <div className="flex items-center gap-1.5 text-slate-500 text-xs font-medium">
                <Zap className="w-3.5 h-3.5 text-amber-500" /> Torque
              </div>
              <div className="text-lg font-bold text-slate-900 mt-1">
                {currentReading ? `${currentReading["Torque [Nm]"]} Nm` : "--"}
              </div>
              <span className="text-[10px] text-slate-400">Range: 3.8 - 76.6 Nm</span>
            </div>

            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <div className="flex items-center gap-1.5 text-slate-500 text-xs font-medium">
                <Clock className="w-3.5 h-3.5 text-indigo-500" /> Tool Wear
              </div>
              <div className="text-lg font-bold text-slate-900 mt-1">
                {currentReading ? `${currentReading["Tool wear [min]"]} min` : "--"}
              </div>
              <span className="text-[10px] text-slate-400">Cycle: #{latestStep?.metadata.tool_cycle ?? 1}</span>
            </div>

            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <div className="flex items-center gap-1.5 text-slate-500 text-xs font-medium">
                <Zap className="w-3.5 h-3.5 text-emerald-600" /> Derived Power
              </div>
              <div className="text-lg font-bold text-slate-900 mt-1">
                {currentReading
                  ? `${Math.round(
                      (currentReading["Torque [Nm]"] *
                        currentReading["Rotational speed [rpm]"] *
                        2 *
                        Math.PI) /
                        60
                    )} W`
                  : "--"}
              </div>
              <span className="text-[10px] text-slate-400">Base load: ~2,860 W</span>
            </div>
          </div>
        </div>
      </div>

      {/* Real-time Recharts Graphs */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Chart 1: Anomaly Score Over Time */}
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
          <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
            Model Estimated Anomaly Score Trend
          </h4>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={telemetryHistory}>
                <XAxis dataKey="tick" stroke="#94a3b8" fontSize={10} />
                <YAxis domain={[0, 1]} stroke="#94a3b8" fontSize={10} />
                <Tooltip />
                <ReferenceLine
                  y={decisionThreshold}
                  stroke="#ef4444"
                  strokeDasharray="3 3"
                  label={{ value: `Cutoff (${decisionThreshold})`, fill: "#ef4444", fontSize: 10 }}
                />
                <Line
                  type="monotone"
                  dataKey="score"
                  stroke="#6366f1"
                  strokeWidth={2}
                  dot={{ r: 2 }}
                  isAnimationActive={false}
                  name="Anomaly Score"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 2: Mechanical Coupling (Torque & Speed) */}
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
          <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
            Mechanical Telemetry (Torque vs Rotational Speed)
          </h4>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={telemetryHistory}>
                <XAxis dataKey="tick" stroke="#94a3b8" fontSize={10} />
                <YAxis yAxisId="left" stroke="#3b82f6" fontSize={10} domain={[1000, 2500]} />
                <YAxis yAxisId="right" orientation="right" stroke="#f59e0b" fontSize={10} domain={[10, 80]} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="speed"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                  name="Speed [rpm]"
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="torque"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                  name="Torque [Nm]"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
