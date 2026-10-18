/**
 * PowerSense Pro -- Energy Dashboard
 *
 * Single-file React dashboard for the PowerSense Pro industrial IoT pipeline.
 * Reads raw readings + latest derived values directly from Supabase (anon
 * key, read-only), and reads ML predictions + the combined summary from the
 * Render-hosted FastAPI pipeline. Falls back to realistic mock data whenever
 * Supabase is not configured or a fetch fails, so the dashboard is always
 * demoable even with nothing wired up yet.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { createClient } from "@supabase/supabase-js";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";
import {
  Zap,
  Activity,
  Gauge,
  TrendingUp,
  AlertTriangle,
  Sun,
  Moon,
  DollarSign,
  Leaf,
  Clock,
  Cpu,
  CheckCircle2,
  XCircle,
  PauseCircle,
  Flame,
  ShieldAlert,
  WifiOff,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const SUPABASE_URL = "https://cgvjqmqdbsdneukkhbfc.supabase.co";
const SUPABASE_ANON_KEY = "sb_publishable_lmNUWoyMAImqbsKshbKbBw_XcLVbQqe";
const RENDER_URL = "https://powersense-6iuu.onrender.com";
const DEVICE_ID = "table_fan"; // matches the device_id actually posting to energy_readings_public
const REFRESH_INTERVAL = 30000; // 30 seconds

// Display-only tariff used for the power-trend tooltip's instantaneous cost
// estimate. The authoritative cost figures (Cost Summary card) always come
// from derived_values.cost_per_hour / daily_cost_projected / monthly_cost_projected.
const DISPLAY_TARIFF_RATE = 9; // Rs / kWh

const READINGS_LIMIT = 100;

// Public, read-only view/table exposed for the dashboard's publishable key --
// distinct from the base `energy_readings` table the backend pipeline reads
// with its privileged service key.
const READINGS_TABLE = "energy_readings_public";

const isSupabaseConfigured =
  typeof SUPABASE_URL === "string" &&
  typeof SUPABASE_ANON_KEY === "string" &&
  SUPABASE_URL.length > 0 &&
  SUPABASE_ANON_KEY.length > 0 &&
  !SUPABASE_URL.includes("YOUR_") &&
  !SUPABASE_ANON_KEY.includes("YOUR_");

const supabase = isSupabaseConfigured
  ? createClient(SUPABASE_URL, SUPABASE_ANON_KEY)
  : null;

// ---------------------------------------------------------------------------
// Mock data (used whenever Supabase / the API is unreachable or unconfigured)
// ---------------------------------------------------------------------------
const mockReading = {
  voltage: 228.5,
  current: 8.2,
  active_power: 1640,
  active_energy: 12.4,
  frequency: 49.8,
  power_factor: 0.87,
  device_id: "hackathon_machine",
  created_at: new Date().toISOString(),
};

const mockDerived = {
  apparent_power: 1874,
  reactive_power: 875,
  pf_status: "good",
  cost_per_hour: 14.76,
  daily_cost_projected: 118.1,
  monthly_cost_projected: 3070,
  motor_health_score: 78,
  motor_health_label: "good",
  machine_state: "normal",
  load_factor: 82,
  thermal_stress_index: 67.2,
  thd_estimate: 24.6,
  carbon_footprint_kg: 1.345,
  voltage_deviation_pct: -0.65,
  idle_cost_per_hour: 1.44,
};

const mockPredictions = {
  anomaly: { label: "normal", confidence: 0.94, value: 0.023 },
  forecast: { value: 13.2, details: { estimated_cost: 118.8 } },
  fault: {
    label: "NORMAL",
    details: { severity: "none", action: "Continue normal operation", evidence: [] },
  },
  health: { value: 78, label: "good", details: { rul_days: 142, trend_direction: "stable" } },
};

/** 100 synthetic readings at the real device cadence (5s), newest-first --
 * matches the shape of the live Supabase query so downstream code never has
 * to care which source produced the data. */
function generateMockReadings(count = READINGS_LIMIT) {
  const now = Date.now();
  const out = [];
  for (let i = 0; i < count; i++) {
    const t = new Date(now - i * 5000);
    if (i === 0) {
      // Keep the newest point exactly on-spec with the provided mock reading.
      out.push({ ...mockReading, created_at: t.toISOString() });
      continue;
    }
    const jitter = () => Math.random() - 0.5;
    const cyclic = Math.sin(i / 14) * 110;
    out.push({
      id: `mock-${i}`,
      device_id: DEVICE_ID,
      created_at: t.toISOString(),
      voltage: +(228.5 + jitter() * 3).toFixed(1),
      current: +(8.2 + jitter() * 0.6).toFixed(2),
      active_power: Math.max(0, +(1640 + cyclic + jitter() * 40).toFixed(0)),
      active_energy: +(12.4 - i * 0.0009).toFixed(3),
      frequency: +(49.8 + jitter() * 0.15).toFixed(2),
      power_factor: +(0.87 + jitter() * 0.015).toFixed(3),
    });
  }
  return out; // newest-first, matching Supabase's order('created_at', desc)
}

// ---------------------------------------------------------------------------
// Small formatting / math helpers
// ---------------------------------------------------------------------------
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const num = (v, fallback = 0) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
};
const fmt = (v, digits = 1) => num(v).toFixed(digits);
const rupee = (v, digits = 2) =>
  `₹${num(v).toLocaleString("en-IN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}`;

function healthColor(score) {
  const s = num(score);
  if (s > 85) return "var(--green)";
  if (s > 70) return "var(--accent)";
  if (s > 50) return "var(--yellow)";
  if (s > 30) return "var(--orange)";
  return "var(--red)";
}

function pfColor(pf) {
  const v = num(pf);
  if (v >= 0.95) return "var(--green)";
  if (v >= 0.85) return "var(--accent)";
  if (v >= 0.7) return "var(--yellow)";
  return "var(--red)";
}

function severityColor(severity) {
  switch (String(severity || "").toLowerCase()) {
    case "critical":
      return "var(--red)";
    case "high":
      return "var(--red)";
    case "medium":
      return "var(--yellow)";
    case "low":
      return "var(--accent)";
    default:
      return "var(--green)";
  }
}

/** Loosely bucket whatever label the anomaly model returns into one of the
 * four conceptual states the spec calls for, without hiding the real label. */
function anomalyBucketColor(label) {
  const l = String(label || "").toLowerCase();
  if (l.includes("severe") || l.includes("critical")) return "var(--red)";
  if (l.includes("strong") || l === "anomaly") return "var(--orange)";
  if (l.includes("border") || l.includes("warn")) return "var(--yellow)";
  if (l.includes("normal")) return "var(--green)";
  return "var(--text-dim)";
}

function formatTime(iso) {
  try {
    return new Date(iso).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return "--:--:--";
  }
}

/** Shorter HH:MM label for the chart's x-axis ticks -- the full HH:MM:SS
 * from formatTime() is kept separately for the tooltip, since cramming
 * seconds into every axis tick is what caused label overlap on narrow
 * (mobile) widths. */
function formatTimeShort(iso) {
  try {
    return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return "--:--";
  }
}

/** Simple least-squares slope of y vs. x, plus R^2, used for the client-side
 * PF degradation estimate (the backend does not expose a dedicated
 * prediction for this, so it is computed here from the recent readings
 * window). R^2 matters as much as the slope itself: sensor noise with no
 * real trend still produces a nonzero slope by chance, and dividing a
 * near-threshold gap by a near-zero noise slope explodes into a wildly
 * misleading "crosses the threshold in 0.3 days" -- R^2 close to 0 is the
 * signal that the slope is noise, not a trend worth extrapolating. */
function linearSlope(points) {
  const n = points.length;
  if (n < 2) return { slope: 0, intercept: points[0]?.y ?? 0, r2: 0 };
  let sumX = 0,
    sumY = 0,
    sumXY = 0,
    sumXX = 0;
  for (const { x, y } of points) {
    sumX += x;
    sumY += y;
    sumXY += x * y;
    sumXX += x * x;
  }
  const denom = n * sumXX - sumX * sumX;
  if (denom === 0) return { slope: 0, intercept: sumY / n, r2: 0 };
  const slope = (n * sumXY - sumX * sumY) / denom;
  const intercept = (sumY - slope * sumX) / n;

  const meanY = sumY / n;
  let ssRes = 0;
  let ssTot = 0;
  for (const { x, y } of points) {
    const pred = slope * x + intercept;
    ssRes += (y - pred) ** 2;
    ssTot += (y - meanY) ** 2;
  }
  const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;
  return { slope, intercept, r2 };
}

// ---------------------------------------------------------------------------
// SVG semicircle gauge math
// ---------------------------------------------------------------------------
function polarToCartesian(cx, cy, r, angleDeg) {
  const rad = (angleDeg * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function describeArc(cx, cy, r, startAngle, endAngle) {
  const start = polarToCartesian(cx, cy, r, startAngle);
  const end = polarToCartesian(cx, cy, r, endAngle);
  const largeArcFlag = endAngle - startAngle <= 180 ? "0" : "1";
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} 1 ${end.x} ${end.y}`;
}

// ---------------------------------------------------------------------------
// Data fetching
// ---------------------------------------------------------------------------
async function fetchReadings() {
  if (!supabase) return { data: generateMockReadings(), live: false };
  try {
    const { data, error } = await supabase
      .from(READINGS_TABLE)
      .select("*")
      .eq("device_id", DEVICE_ID)
      .order("created_at", { ascending: false })
      .limit(READINGS_LIMIT);
    if (error) throw error;
    if (!data || data.length === 0) {
      return { data: generateMockReadings(), live: false };
    }
    return { data, live: true };
  } catch (err) {
    console.error("[PowerSense] fetchReadings failed, using mock data:", err);
    return { data: generateMockReadings(), live: false };
  }
}

async function fetchDerived() {
  if (!supabase) return mockDerived;
  try {
    const { data, error } = await supabase
      .from("derived_values")
      .select("*")
      .eq("device_id", DEVICE_ID)
      .order("timestamp", { ascending: false })
      .limit(1);
    if (error) throw error;
    if (!data || data.length === 0) return mockDerived;
    return data[0];
  } catch (err) {
    console.error("[PowerSense] fetchDerived failed, using mock data:", err);
    return mockDerived;
  }
}

async function fetchPredictions() {
  try {
    const res = await fetch(`${RENDER_URL}/predictions/${DEVICE_ID}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    const preds = json?.predictions ?? {};
    return Object.keys(preds).length > 0 ? preds : mockPredictions;
  } catch (err) {
    console.error("[PowerSense] fetchPredictions failed, using mock data:", err);
    return mockPredictions;
  }
}

async function fetchSummary() {
  try {
    const res = await fetch(`${RENDER_URL}/summary/${DEVICE_ID}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error("[PowerSense] fetchSummary failed:", err);
    return null;
  }
}

// ---------------------------------------------------------------------------
// Small reusable pieces
// ---------------------------------------------------------------------------
function ProgressBar({ label, value, min, max, unit, digits = 1, colorFn }) {
  const v = num(value);
  const pct = clamp(((v - min) / (max - min)) * 100, 0, 100);
  const color = colorFn ? colorFn(v) : "var(--accent)";
  return (
    <div className="ps-reading-row">
      <div className="ps-reading-head">
        <span className="ps-reading-label">{label}</span>
        <span className="ps-reading-value">
          {fmt(v, digits)} {unit}
        </span>
      </div>
      <div className="ps-track">
        <div
          className="ps-fill"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  );
}

function StatCard({ icon: Icon, label, value, unit, sub, color }) {
  return (
    <div className="ps-card ps-stat">
      <div className="ps-stat-icon" style={{ color: color || "var(--accent)" }}>
        <Icon size={18} />
      </div>
      <div className="ps-stat-body">
        <div className="ps-stat-label">{label}</div>
        <div className="ps-stat-value" style={{ color: color || "var(--text)" }}>
          {value}
          {unit ? <span className="ps-stat-unit"> {unit}</span> : null}
        </div>
        {sub ? <div className="ps-stat-sub">{sub}</div> : null}
      </div>
    </div>
  );
}

function PredictionCard({ icon: Icon, title, trained, children }) {
  return (
    <div className="ps-card ps-pred-card">
      <div className="ps-pred-head">
        <Icon size={16} />
        <span>{title}</span>
      </div>
      {trained ? (
        <div className="ps-pred-body">{children}</div>
      ) : (
        <div className="ps-pred-training">
          <PauseCircle size={14} />
          <span>Model training&hellip;</span>
        </div>
      )}
    </div>
  );
}

function CustomTooltip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const power = num(payload[0]?.value);
  const cost = (power / 1000) * DISPLAY_TARIFF_RATE;
  const fullTime = payload[0]?.payload?.fullTime;
  return (
    <div className="ps-tooltip">
      <div className="ps-tooltip-time">{fullTime}</div>
      <div className="ps-tooltip-row">
        <span>Power</span>
        <strong>{fmt(power, 0)} W</strong>
      </div>
      <div className="ps-tooltip-row">
        <span>Est. cost</span>
        <strong>{rupee(cost, 2)}/hr</strong>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function EnergyDashboard() {
  const [theme, setTheme] = useState("dark");
  const [readings, setReadings] = useState(() => generateMockReadings());
  const [derived, setDerived] = useState(mockDerived);
  const [predictions, setPredictions] = useState(mockPredictions);
  const [summary, setSummary] = useState(null);
  const [liveMode, setLiveMode] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [isFetching, setIsFetching] = useState(false);

  const loadAll = useCallback(async () => {
    setIsFetching(true);
    const [readingsResult, derivedResult, predictionsResult, summaryResult] =
      await Promise.allSettled([
        fetchReadings(),
        fetchDerived(),
        fetchPredictions(),
        fetchSummary(),
      ]);

    if (readingsResult.status === "fulfilled") {
      setReadings(readingsResult.value.data);
      setLiveMode(readingsResult.value.live);
    }
    if (derivedResult.status === "fulfilled") setDerived(derivedResult.value);
    if (predictionsResult.status === "fulfilled") setPredictions(predictionsResult.value);
    if (summaryResult.status === "fulfilled") setSummary(summaryResult.value);

    setLastUpdated(new Date());
    setIsFetching(false);
  }, []);

  useEffect(() => {
    loadAll();
    const id = setInterval(loadAll, REFRESH_INTERVAL);
    return () => clearInterval(id);
  }, [loadAll]);

  const latestReading = readings[0] || mockReading;

  const chartData = useMemo(() => {
    return [...readings].reverse().map((r) => ({
      time: formatTimeShort(r.created_at),
      fullTime: formatTime(r.created_at),
      power: num(r.active_power),
    }));
  }, [readings]);

  const peakPowerToday = useMemo(() => {
    if (!readings.length) return 0;
    return Math.max(...readings.map((r) => num(r.active_power)));
  }, [readings]);

  // "Today" is bounded by the last 100 fetched readings (per the data-source
  // spec), not a true calendar-day window -- the pipeline posts every 5s, so
  // 100 rows covers only the most recent few minutes in a live deployment.

  const machineState = String(derived?.machine_state || "off").toLowerCase();
  const isOverload = machineState === "overload";
  const isIdle = machineState === "idle";

  const dailyCarbon = useMemo(() => {
    const perHour = num(derived?.carbon_footprint_kg);
    const costPerHour = num(derived?.cost_per_hour);
    const dailyCost = num(derived?.daily_cost_projected);
    const shiftHours = costPerHour > 0 ? dailyCost / costPerHour : 8;
    return perHour * shiftHours;
  }, [derived]);

  // Client-side PF degradation estimate: the backend does not expose a
  // dedicated "pf_degradation" prediction, so this linearly projects the
  // recent power_factor trend to the point it would cross the 0.85 penalty
  // threshold used by the engine's pf_penalty_pct rule.
  const pfTrend = useMemo(() => {
    const series = [...readings]
      .reverse()
      .filter((r) => r.power_factor != null)
      .map((r, i) => ({ x: i, y: num(r.power_factor) }));
    if (series.length < 10) return null;

    const latestPf = series[series.length - 1].y;
    const { slope, r2 } = linearSlope(series);
    const perSampleSeconds = 5; // device posting cadence

    if (latestPf < 0.85) {
      return { status: "active", daysUntil: 0, latestPf };
    }
    // Require both a declining slope AND a fit good enough to trust the
    // extrapolation -- flat, noisy PF readings fit a near-zero slope with
    // r^2 close to 0, which is exactly the case that must NOT be projected
    // forward (see linearSlope's comment on why r^2 is the guard here).
    const R2_MIN = 0.3;
    if (slope >= -1e-6 || r2 < R2_MIN) {
      return { status: "stable", daysUntil: null, latestPf };
    }
    const samplesUntilThreshold = (0.85 - latestPf) / slope;
    const daysUntil = clamp((samplesUntilThreshold * perSampleSeconds) / 86400, 0.1, 3650);
    return { status: "degrading", daysUntil, latestPf };
  }, [readings]);

  const faultPrediction = predictions?.fault;
  const faultDetails = faultPrediction?.details || {};
  const faultSeverity = String(faultDetails.severity || "none").toLowerCase();
  const faultIsBad = faultSeverity === "high" || faultSeverity === "critical";

  const toggleTheme = () => setTheme((t) => (t === "dark" ? "light" : "dark"));

  return (
    <div className="ps-dashboard" data-theme={theme}>
      <style>{CSS}</style>

      <header className="ps-header">
        <div className="ps-header-left">
          <Zap size={22} className="ps-logo-icon" />
          <h1>PowerSense Pro</h1>
          <span className="ps-badge">
            <Cpu size={12} /> {DEVICE_ID}
          </span>
          {!liveMode && (
            <span className="ps-badge ps-badge-muted">
              <WifiOff size={12} /> Demo data
            </span>
          )}
        </div>
        <div className="ps-header-right">
          <span className="ps-updated">
            <Clock size={13} />
            {lastUpdated ? lastUpdated.toLocaleTimeString() : "--:--:--"}
            {isFetching && <span className="ps-pulse-dot" />}
          </span>
          <button className="ps-theme-toggle" onClick={toggleTheme} aria-label="Toggle theme">
            {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </div>
      </header>

      <main className="ps-main">
        {/* ---- Machine status banner ---- */}
        <section
          className={`ps-banner ps-banner-${machineState} ${isOverload ? "ps-banner-flash" : ""}`}
        >
          <div className="ps-banner-icon">
            {machineState === "off" && <PauseCircle size={22} />}
            {machineState === "idle" && <PauseCircle size={22} />}
            {machineState === "overload" && <AlertTriangle size={22} />}
            {!["off", "idle", "overload"].includes(machineState) && (
              <CheckCircle2 size={22} />
            )}
          </div>
          <div className="ps-banner-text">
            <div className="ps-banner-title">{machineState.toUpperCase().replace("_", " ")}</div>
            <div className="ps-banner-sub">
              {machineState === "off" && "Machine is powered off."}
              {machineState === "idle" &&
                `Idle -- burning ${rupee(derived?.idle_cost_per_hour, 2)}/hr with no useful output.`}
              {machineState === "overload" &&
                `Load factor ${fmt(derived?.load_factor, 0)}% exceeds safe limits -- act now.`}
              {machineState === "normal" && "Operating within normal parameters."}
              {machineState === "light_load" && "Running under light load."}
              {machineState === "heavy_load" && "Running under heavy load -- monitor closely."}
            </div>
          </div>
        </section>

        {/* ---- KPI grid ---- */}
        <section className="ps-kpi-grid">
          <StatCard
            icon={Zap}
            label="Total Energy Today"
            value={fmt(latestReading.active_energy, 2)}
            unit="kWh"
          />
          <StatCard
            icon={Activity}
            label="Current Power"
            value={fmt(derived?.active_power ?? latestReading.active_power, 0)}
            unit="W"
            color={isOverload ? "var(--red)" : undefined}
          />
          <StatCard
            icon={Gauge}
            label="Power Factor"
            value={fmt(derived?.power_factor ?? latestReading.power_factor, 3)}
            color={num(derived?.power_factor ?? latestReading.power_factor) < 0.85 ? "var(--red)" : "var(--green)"}
          />
          <StatCard
            icon={ShieldAlert}
            label="Motor Health Score"
            value={fmt(derived?.motor_health_score, 0)}
            unit="/100"
            color={healthColor(derived?.motor_health_score)}
            sub={derived?.motor_health_label}
          />
          <StatCard
            icon={TrendingUp}
            label="Peak Power Today"
            value={fmt(peakPowerToday, 0)}
            unit="W"
          />
          <StatCard
            icon={DollarSign}
            label="Idle Cost Today"
            value={rupee(derived?.idle_cost_per_hour, 2)}
            sub={`${rupee(derived?.idle_cost_per_hour, 2)}/hr if idle`}
          />
        </section>

        {/* ---- Power trend + Live readings ---- */}
        <section className="ps-grid-2col">
          <div className="ps-card ps-chart-card">
            <div className="ps-card-title">
              Power Trend <span className="ps-card-subtitle">(last {readings.length} readings)</span>
            </div>
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={chartData} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
                <defs>
                  <linearGradient id="psPowerGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.55} />
                    <stop offset="100%" stopColor="var(--accent)" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--grid-line)" vertical={false} />
                <XAxis
                  dataKey="time"
                  tick={{ fill: "var(--text-dim)", fontSize: 11 }}
                  interval="preserveStartEnd"
                  minTickGap={32}
                  axisLine={{ stroke: "var(--grid-line)" }}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: "var(--text-dim)", fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  width={48}
                  label={{ value: "W", position: "insideTopLeft", fill: "var(--text-dim)", fontSize: 11 }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Area
                  type="monotone"
                  dataKey="power"
                  stroke="var(--accent)"
                  strokeWidth={2}
                  fill="url(#psPowerGradient)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="ps-card ps-readings-card">
            <div className="ps-card-title">Live Readings</div>
            <ProgressBar
              label="Voltage"
              value={latestReading.voltage}
              min={0}
              max={300}
              unit="V"
              digits={1}
            />
            <ProgressBar
              label="Current"
              value={latestReading.current}
              min={0}
              max={10}
              unit="A"
              digits={2}
            />
            <ProgressBar
              label="Active Power"
              value={latestReading.active_power}
              min={0}
              max={2000}
              unit="W"
              digits={0}
            />
            <ProgressBar
              label="Power Factor"
              value={latestReading.power_factor}
              min={0}
              max={1}
              unit=""
              digits={3}
              colorFn={pfColor}
            />
            <ProgressBar
              label="Frequency"
              value={latestReading.frequency}
              min={45}
              max={65}
              unit="Hz"
              digits={2}
            />
            <ProgressBar
              label="THD Estimate"
              value={derived?.thd_estimate}
              min={0}
              max={100}
              unit="%"
              digits={1}
            />
          </div>
        </section>

        {/* ---- Health gauge + DL predictions ---- */}
        <section className="ps-grid-2col-rev">
          <div className="ps-card ps-gauge-card">
            <div className="ps-card-title">Motor Health</div>
            <HealthGauge score={num(derived?.motor_health_score)} />
            <div className="ps-gauge-score" style={{ color: healthColor(derived?.motor_health_score) }}>
              {fmt(derived?.motor_health_score, 0)}
              <span className="ps-gauge-max">/100</span>
            </div>
            <div className="ps-gauge-label">{derived?.motor_health_label || "unknown"}</div>
          </div>

          <div className="ps-pred-grid">
            <PredictionCard icon={Activity} title="Anomaly Status" trained={!!predictions?.anomaly}>
              <div
                className="ps-pred-main"
                style={{ color: anomalyBucketColor(predictions?.anomaly?.label) }}
              >
                {(predictions?.anomaly?.label || "unknown").toString().replace(/_/g, " ")}
              </div>
              <div className="ps-pred-detail">
                Confidence {fmt(num(predictions?.anomaly?.confidence) * 100, 0)}%
              </div>
              <div className="ps-pred-detail ps-pred-detail-dim">
                score {fmt(predictions?.anomaly?.value, 3)}
              </div>
            </PredictionCard>

            <PredictionCard icon={TrendingUp} title="24h Energy Forecast" trained={!!predictions?.forecast}>
              <div className="ps-pred-main">{fmt(predictions?.forecast?.value, 1)} kWh</div>
              <div className="ps-pred-detail">
                Est. cost {rupee(predictions?.forecast?.details?.estimated_cost, 0)}
              </div>
            </PredictionCard>

            <PredictionCard icon={Flame} title="PF Degradation" trained={!!pfTrend}>
              {pfTrend?.status === "active" && (
                <>
                  <div className="ps-pred-main" style={{ color: "var(--red)" }}>
                    Penalty active now
                  </div>
                  <div className="ps-pred-detail">PF {fmt(pfTrend.latestPf, 3)} is below 0.85</div>
                </>
              )}
              {pfTrend?.status === "stable" && (
                <>
                  <div className="ps-pred-main" style={{ color: "var(--green)" }}>
                    Stable
                  </div>
                  <div className="ps-pred-detail">No degrading trend detected</div>
                </>
              )}
              {pfTrend?.status === "degrading" && (
                <>
                  <div className="ps-pred-main" style={{ color: "var(--yellow)" }}>
                    {fmt(pfTrend.daysUntil, 1)} days
                  </div>
                  <div className="ps-pred-detail">until PF crosses the 0.85 penalty threshold</div>
                </>
              )}
              <div className="ps-pred-detail ps-pred-detail-dim">client-side trend estimate</div>
            </PredictionCard>

            <PredictionCard icon={AlertTriangle} title="Fault Detection" trained={!!faultPrediction}>
              <div className="ps-pred-main" style={{ color: severityColor(faultSeverity) }}>
                {faultPrediction?.label || "NORMAL"}
              </div>
              <div className="ps-pred-detail">{faultDetails.action || "Continue normal operation"}</div>
            </PredictionCard>
          </div>
        </section>

        {/* ---- Active faults ---- */}
        <section className={`ps-card ps-fault-card ${faultIsBad ? "ps-fault-card-bad" : ""}`}>
          <div className="ps-card-title">
            <AlertTriangle size={16} style={{ marginRight: 6 }} />
            Active Faults
          </div>
          <div className="ps-fault-row">
            <span
              className="ps-fault-badge"
              style={{
                background: severityColor(faultSeverity),
                color: "#0d1420",
              }}
            >
              {faultSeverity.toUpperCase()}
            </span>
            <span className="ps-fault-type">{faultPrediction?.label || "NORMAL"}</span>
          </div>
          {Array.isArray(faultDetails.evidence) && faultDetails.evidence.length > 0 ? (
            <ul className="ps-evidence-list">
              {faultDetails.evidence.map((e, i) => (
                <li key={i}>{e}</li>
              ))}
            </ul>
          ) : (
            <div className="ps-fault-clear">
              <CheckCircle2 size={14} /> No active fault conditions.
            </div>
          )}
          <div className="ps-fault-action">
            <strong>Action:</strong> {faultDetails.action || "Continue normal operation"}
          </div>
          {summary?.alert_count > 0 && (
            <div className="ps-fault-footnote">
              {summary.alert_count} active alert{summary.alert_count === 1 ? "" : "s"} logged
              {summary.critical_alert_count > 0
                ? ` (${summary.critical_alert_count} critical)`
                : ""}
              .
            </div>
          )}
        </section>

        {/* ---- Cost + carbon ---- */}
        <section className="ps-grid-2col">
          <div className="ps-card">
            <div className="ps-card-title">
              <DollarSign size={16} style={{ marginRight: 6 }} />
              Cost Summary
            </div>
            <div className="ps-cost-grid">
              <div className="ps-cost-item">
                <div className="ps-cost-label">Per hour</div>
                <div className="ps-cost-value">{rupee(derived?.cost_per_hour, 2)}</div>
              </div>
              <div className="ps-cost-item">
                <div className="ps-cost-label">Daily projected</div>
                <div className="ps-cost-value">{rupee(derived?.daily_cost_projected, 2)}</div>
              </div>
              <div className="ps-cost-item">
                <div className="ps-cost-label">Monthly projected</div>
                <div className="ps-cost-value">{rupee(derived?.monthly_cost_projected, 0)}</div>
              </div>
            </div>
          </div>

          <div className="ps-card">
            <div className="ps-card-title">
              <Leaf size={16} style={{ marginRight: 6 }} />
              Carbon Footprint
            </div>
            <div className="ps-cost-grid">
              <div className="ps-cost-item">
                <div className="ps-cost-label">Current</div>
                <div className="ps-cost-value">{fmt(derived?.carbon_footprint_kg, 3)} kg/hr</div>
              </div>
              <div className="ps-cost-item">
                <div className="ps-cost-label">Daily projected</div>
                <div className="ps-cost-value">{fmt(dailyCarbon, 2)} kg CO&#8322;</div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Motor health gauge (SVG semicircle)
// ---------------------------------------------------------------------------
function HealthGauge({ score }) {
  const clamped = clamp(score, 0, 100);
  const cx = 120;
  const cy = 120;
  const r = 95;
  const strokeWidth = 18;

  const trackPath = describeArc(cx, cy, r, 180, 360);
  const valueAngle = 180 + (clamped / 100) * 180;
  const valuePath = describeArc(cx, cy, r, 180, valueAngle);
  const color = healthColor(clamped);

  return (
    <svg viewBox="0 0 240 136" className="ps-gauge-svg" role="img" aria-label="Motor health gauge">
      <path d={trackPath} fill="none" stroke="var(--track)" strokeWidth={strokeWidth} strokeLinecap="round" />
      <path
        d={valuePath}
        fill="none"
        stroke={color}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        style={{ transition: "stroke 0.3s ease" }}
      />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------
const CSS = `
.ps-dashboard, .ps-dashboard * { box-sizing: border-box; }

.ps-dashboard[data-theme="dark"] {
  --bg: #0d1420;
  --card: #1a2332;
  --card-border: rgba(255,255,255,0.07);
  --text: #e6edf5;
  --text-dim: #8b96a8;
  --accent: #00d4ff;
  --green: #10b981;
  --yellow: #f59e0b;
  --orange: #f97316;
  --red: #ef4444;
  --track: #2e3c54;
  --grid-line: rgba(255,255,255,0.08);
  --tooltip-bg: #0f1623;
}

.ps-dashboard[data-theme="light"] {
  --bg: #f3f4f6;
  --card: #ffffff;
  --card-border: rgba(0,0,0,0.08);
  --text: #111827;
  --text-dim: #6b7280;
  --accent: #0891b2;
  --green: #10b981;
  --yellow: #f59e0b;
  --orange: #ea580c;
  --red: #ef4444;
  --track: #e5e7eb;
  --grid-line: rgba(0,0,0,0.08);
  --tooltip-bg: #ffffff;
}

.ps-dashboard {
  min-height: 100vh;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  padding-bottom: 32px;
  transition: background 0.2s ease, color 0.2s ease;
}

.ps-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 16px clamp(16px, 4vw, 32px);
  border-bottom: 1px solid var(--card-border);
  flex-wrap: wrap;
}

.ps-header-left {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.ps-logo-icon { color: var(--accent); }

.ps-header-left h1 {
  font-size: clamp(16px, 2.4vw, 20px);
  margin: 0;
  letter-spacing: -0.01em;
}

.ps-badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  font-weight: 600;
  padding: 4px 9px;
  border-radius: 999px;
  background: rgba(0, 212, 255, 0.12);
  color: var(--accent);
  border: 1px solid rgba(0, 212, 255, 0.25);
}

.ps-badge-muted {
  background: rgba(139, 150, 168, 0.12);
  color: var(--text-dim);
  border-color: var(--card-border);
}

.ps-header-right {
  display: flex;
  align-items: center;
  gap: 14px;
}

.ps-updated {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--text-dim);
}

.ps-pulse-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--accent);
  animation: ps-pulse 1s ease-in-out infinite;
}

@keyframes ps-pulse {
  0%, 100% { opacity: 0.3; transform: scale(0.8); }
  50% { opacity: 1; transform: scale(1.2); }
}

.ps-theme-toggle {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border-radius: 8px;
  border: 1px solid var(--card-border);
  background: var(--card);
  color: var(--text);
  cursor: pointer;
}

.ps-theme-toggle:hover { border-color: var(--accent); color: var(--accent); }

.ps-main {
  display: flex;
  flex-direction: column;
  gap: 18px;
  padding: 20px clamp(16px, 4vw, 32px) 0;
  max-width: 1400px;
  margin: 0 auto;
}

.ps-card {
  background: var(--card);
  border: 1px solid var(--card-border);
  border-radius: 14px;
  padding: 18px;
}

.ps-card-title {
  font-size: 13px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text-dim);
  margin-bottom: 14px;
  display: flex;
  align-items: center;
}

.ps-card-subtitle {
  text-transform: none;
  font-weight: 400;
  letter-spacing: normal;
  margin-left: 6px;
  opacity: 0.7;
}

/* ---- status banner ---- */
.ps-banner {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 16px 20px;
  border-radius: 14px;
  border: 2px solid transparent;
}

.ps-banner-off { background: rgba(139,150,168,0.12); color: var(--text-dim); }
.ps-banner-idle { background: rgba(245,158,11,0.12); border-color: rgba(245,158,11,0.3); }
.ps-banner-normal, .ps-banner-light_load { background: rgba(16,185,129,0.12); border-color: rgba(16,185,129,0.3); }
.ps-banner-heavy_load { background: rgba(249,115,22,0.14); border-color: rgba(249,115,22,0.35); }
.ps-banner-overload { background: rgba(239,68,68,0.14); border-color: var(--red); }

.ps-banner-flash { animation: ps-flash 1.1s ease-in-out infinite; }

@keyframes ps-flash {
  0%, 100% { box-shadow: 0 0 0 2px var(--red), 0 0 20px rgba(239,68,68,0.2); }
  50% { box-shadow: 0 0 0 2px var(--red), 0 0 36px rgba(239,68,68,0.55); }
}

.ps-banner-title { font-size: 16px; font-weight: 800; letter-spacing: 0.02em; }
.ps-banner-sub { font-size: 13px; color: var(--text-dim); margin-top: 2px; }

/* ---- KPI grid ---- */
.ps-kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 14px;
}

.ps-stat {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}

.ps-stat-icon {
  width: 36px;
  height: 36px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0,212,255,0.1);
  flex-shrink: 0;
}

.ps-stat-label { font-size: 12px; color: var(--text-dim); margin-bottom: 4px; }
.ps-stat-value { font-size: 22px; font-weight: 800; line-height: 1.1; }
.ps-stat-unit { font-size: 13px; font-weight: 500; color: var(--text-dim); }
.ps-stat-sub { font-size: 11px; color: var(--text-dim); margin-top: 3px; text-transform: capitalize; }

/* ---- two-column rows ---- */
.ps-grid-2col {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 14px;
}

.ps-grid-2col-rev {
  display: grid;
  grid-template-columns: 300px 1fr;
  gap: 14px;
}

@media (max-width: 900px) {
  .ps-grid-2col, .ps-grid-2col-rev { grid-template-columns: 1fr; }
}

/* ---- live readings ---- */
.ps-reading-row { margin-bottom: 14px; }
.ps-reading-row:last-child { margin-bottom: 0; }

.ps-reading-head {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  margin-bottom: 6px;
}

.ps-reading-label { color: var(--text-dim); }
.ps-reading-value { font-weight: 700; }

.ps-track {
  height: 7px;
  border-radius: 999px;
  background: var(--track);
  overflow: hidden;
}

.ps-fill {
  height: 100%;
  border-radius: 999px;
  transition: width 0.4s ease;
}

/* ---- chart tooltip ---- */
.ps-tooltip {
  background: var(--tooltip-bg);
  border: 1px solid var(--card-border);
  border-radius: 8px;
  padding: 8px 10px;
  font-size: 12px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.25);
}

.ps-tooltip-time { color: var(--text-dim); margin-bottom: 4px; font-size: 11px; }
.ps-tooltip-row { display: flex; justify-content: space-between; gap: 16px; color: var(--text); }

/* ---- gauge ---- */
.ps-gauge-card { display: flex; flex-direction: column; align-items: center; text-align: center; }
.ps-gauge-svg { width: 100%; max-width: 240px; }
.ps-gauge-score { font-size: 32px; font-weight: 800; margin-top: -18px; }
.ps-gauge-max { font-size: 14px; color: var(--text-dim); font-weight: 500; }
.ps-gauge-label { font-size: 13px; color: var(--text-dim); text-transform: capitalize; margin-top: 2px; }

/* ---- predictions ---- */
.ps-pred-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 14px;
}

.ps-pred-card { display: flex; flex-direction: column; }

.ps-pred-head {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 12px;
  font-weight: 700;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.03em;
  margin-bottom: 10px;
}

.ps-pred-main { font-size: 20px; font-weight: 800; text-transform: capitalize; }
.ps-pred-detail { font-size: 12px; color: var(--text-dim); margin-top: 3px; }
.ps-pred-detail-dim { opacity: 0.7; font-style: italic; }

.ps-pred-training {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  color: var(--text-dim);
  font-style: italic;
  margin-top: auto;
  padding-top: 8px;
}

/* ---- faults card ---- */
.ps-fault-card-bad { border-color: var(--red); box-shadow: 0 0 0 1px var(--red); }

.ps-fault-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }

.ps-fault-badge {
  font-size: 11px;
  font-weight: 800;
  padding: 3px 9px;
  border-radius: 999px;
  letter-spacing: 0.03em;
}

.ps-fault-type { font-size: 16px; font-weight: 700; }

.ps-evidence-list {
  margin: 0 0 12px;
  padding-left: 18px;
  font-size: 13px;
  color: var(--text-dim);
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.ps-fault-clear {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  color: var(--green);
  margin-bottom: 12px;
}

.ps-fault-action { font-size: 13px; }
.ps-fault-footnote { font-size: 12px; color: var(--text-dim); margin-top: 8px; }

/* ---- cost / carbon ---- */
.ps-cost-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 14px;
}

.ps-cost-label { font-size: 12px; color: var(--text-dim); margin-bottom: 4px; }
.ps-cost-value { font-size: 20px; font-weight: 800; }

@media (max-width: 480px) {
  .ps-stat-value { font-size: 19px; }
  .ps-card { padding: 14px; }
}
`;
