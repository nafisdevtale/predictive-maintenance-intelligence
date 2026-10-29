import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Predictive Maintenance Intelligence System",
  description:
    "Industrial Condition-Based Monitoring and Anomaly Estimation Dashboard (UCI AI4I 2020 Benchmark)",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased min-h-screen bg-slate-50 text-slate-900">{children}</body>
    </html>
  );
}
