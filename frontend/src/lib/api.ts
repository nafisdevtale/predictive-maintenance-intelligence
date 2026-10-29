/**
 * Typed API Client for the Predictive Maintenance Intelligence FastAPI Backend.
 */

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/+$/, "") || "http://localhost:8000";

export interface HealthResponse {
  status: string;
  model_ready: boolean;
  model_name: string;
  version: string;
}

export interface OverviewResponse {
  dataset_name: string;
  doi: string;
  license: string;
  total_observations: number;
  class_distribution: {
    normal_count: number;
    failure_count: number;
    failure_rate_pct: number;
  };
  quality_variants: {
    [key: string]: { count: number; pct: number };
  };
  sensor_specifications: {
    [key: string]: { min: number; max: number; default: number; unit?: string; options?: string[] };
  };
  selected_model: {
    model_name: string;
    decision_threshold: number;
    selection_criterion: string;
    feature_count: number;
    features: string[];
  };
  synthetic_benchmark_notice: string;
}

export interface MetricsResponse {
  dataset_split: {
    n_train: number;
    n_test: number;
    train_failures: number;
    test_failures: number;
    train_failure_rate_pct: number;
    test_failure_rate_pct: number;
  };
  metadata: {
    selected_model: string;
    selected_threshold: number;
    threshold_selection_method: string;
    training_sample_count: number;
    test_sample_count: number;
    features: string[];
  };
  out_of_fold_validation_results: {
    logistic_regression: {
      default_threshold_0_5: any;
      f2_optimal_threshold: any;
    };
    random_forest: {
      default_threshold_0_5: any;
      f2_optimal_threshold: any;
    };
  };
  held_out_test_results: {
    logistic_regression: {
      default_threshold_0_5: any;
      training_selected_threshold: any;
    };
    random_forest: {
      default_threshold_0_5: any;
      training_selected_threshold: any;
    };
  };
  feature_importance_rf: Record<string, number>;
  feature_coefficients_lr: Record<string, number>;
}

export interface StatisticsResponse {
  sample_size: number;
  alpha_threshold: number;
  multiple_testing_correction: string;
  numeric_tests: Array<{
    feature: string;
    n_non_failure: number;
    n_failure: number;
    median_non_failure: number;
    iqr_non_failure: number;
    median_failure: number;
    iqr_failure: number;
    median_difference: number;
    u_statistic: number;
    raw_p_value: number;
    fdr_adjusted_p_value: number;
    rank_biserial_r: number;
    is_significant_fdr: boolean;
  }>;
  categorical_test: {
    status: string;
    contingency_table: Record<string, Record<string, number>>;
    breakdown_by_type: Record<string, { total_count: number; failure_count: number; failure_rate_pct: number }>;
    chi2_statistic: number;
    degrees_of_freedom: number;
    p_value: number;
    is_significant: boolean;
    cramers_v: number;
    assumptions: {
      min_expected_frequency: number;
      pct_expected_under_5: number;
      assumptions_met: boolean;
    };
  };
  interpretations: {
    dataset_nature: string;
    causality_caveat: string;
    fdr_correction_summary: string;
    strongest_numeric_association: {
      feature: string;
      rank_biserial_r: number;
      median_failure: number;
      median_non_failure: number;
    };
    machine_type_association: {
      chi2_stat: number;
      p_value: number;
      cramers_v: number;
      is_significant: boolean;
      interpretation: string;
    };
  };
}

export interface PredictionResult {
  model_name: string;
  decision_threshold: number;
  estimated_anomaly_score: number;
  predicted_label: number;
  status: "NORMAL_OPERATION" | "ATTENTION_ADVISED" | "MAINTENANCE_REQUIRED";
  risk_level: "LOW" | "ELEVATED" | "HIGH";
  score_interpretation: string;
  safety_disclaimer: string;
}

export interface SensorReading {
  Type: "L" | "M" | "H";
  "Air temperature [K]": number;
  "Process temperature [K]": number;
  "Rotational speed [rpm]": number;
  "Torque [Nm]": number;
  "Tool wear [min]": number;
}

export interface SimulatorStepResult {
  next_state: any;
  reading: SensorReading;
  metadata: {
    tick: number;
    scenario: string;
    tool_cycle: number;
    tool_replaced: boolean;
    is_simulated: boolean;
    disclaimer: string;
  };
  prediction: PredictionResult;
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorDetail = `Request failed with status ${res.status}`;
    try {
      const errJson = await res.json();
      if (errJson.detail) errorDetail = errJson.detail;
    } catch {
      // Ignore json parse error
    }
    throw new Error(errorDetail);
  }
  return res.json() as Promise<T>;
}

export async function checkBackendHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/health`, { cache: "no-store" });
  return handleResponse<HealthResponse>(res);
}

export async function fetchOverviewData(): Promise<OverviewResponse> {
  const res = await fetch(`${API_BASE_URL}/api/overview`, { cache: "no-store" });
  return handleResponse<OverviewResponse>(res);
}

export async function fetchModelMetrics(): Promise<MetricsResponse> {
  const res = await fetch(`${API_BASE_URL}/api/model/metrics`, { cache: "no-store" });
  return handleResponse<MetricsResponse>(res);
}

export async function fetchStatistics(): Promise<StatisticsResponse> {
  const res = await fetch(`${API_BASE_URL}/api/statistics`, { cache: "no-store" });
  return handleResponse<StatisticsResponse>(res);
}

export async function predictTelemetry(input: SensorReading): Promise<PredictionResult> {
  const res = await fetch(`${API_BASE_URL}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  return handleResponse<PredictionResult>(res);
}

export async function initializeSimulator(
  scenario: string = "normal",
  machineType: string = "L",
  seed: number = 42
): Promise<{ state: any; scenario: string; machine_type: string; is_simulated: boolean }> {
  const res = await fetch(`${API_BASE_URL}/api/simulator/initialize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scenario, machine_type: machineType, seed }),
  });
  return handleResponse(res);
}

export async function stepSimulator(state: any): Promise<SimulatorStepResult> {
  const res = await fetch(`${API_BASE_URL}/api/simulator/step`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ state }),
  });
  return handleResponse<SimulatorStepResult>(res);
}
