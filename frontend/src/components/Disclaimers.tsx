import React from "react";
import { AlertTriangle, Info } from "lucide-react";

export function EducationalDisclaimerBanner() {
  return (
    <div className="bg-amber-50 border-l-4 border-amber-500 p-4 rounded-r-lg shadow-sm mb-6">
      <div className="flex items-start space-x-3">
        <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
        <div className="text-xs md:text-sm text-amber-800 leading-relaxed">
          <span className="font-semibold uppercase tracking-wide">Academic & Educational Disclaimer:</span>{" "}
          This system is an academic machine learning decision-support demonstration trained on the synthetic{" "}
          <strong>UCI AI4I 2020 Predictive Maintenance benchmark</strong>. All telemetries, simulator readings, and
          anomaly scores are educational statistical estimates and must <strong>never</strong> be used as
          safety-critical diagnostic controls or physical shutdown mechanisms in real-world machinery.
        </div>
      </div>
    </div>
  );
}

export function ApiDisconnectedBanner({ onRetry }: { onRetry: () => void }) {
  return (
    <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded-r-lg shadow-sm mb-6">
      <div className="flex items-start justify-between">
        <div className="flex items-start space-x-3">
          <AlertTriangle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-red-800">
            <span className="font-semibold">FastAPI Backend Disconnected:</span> Unable to connect to the prediction
            server at <code className="bg-red-100 px-1 py-0.5 rounded text-xs">NEXT_PUBLIC_API_BASE_URL</code>.
            Ensure the FastAPI server is running locally:
            <pre className="mt-2 bg-slate-900 text-slate-100 p-2 rounded text-xs font-mono">
              .venv/bin/uvicorn app:app --port 8000 --reload
            </pre>
          </div>
        </div>
        <button
          onClick={onRetry}
          className="ml-4 px-3 py-1.5 text-xs font-medium bg-red-600 text-white rounded hover:bg-red-700 transition"
        >
          Retry Connection
        </button>
      </div>
    </div>
  );
}
