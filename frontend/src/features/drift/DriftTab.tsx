import { useEffect, useRef, useState } from "react"
import {
  AlertCircle,
  ArrowDownLeft,
  ArrowUpRight,
  LoaderCircle,
  RefreshCw,
  Wind,
} from "lucide-react"
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty"
import { Separator } from "@/components/ui/separator"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ApiError, getDriftAnalysis } from "@/lib/api"
import type { DriftEnvironment, DriftMode, DriftOrigin, DriftResponse } from "@/types/api"
import { EnvironmentalConditionsChart } from "./EnvironmentalConditionsChart"

function formatVector(vector: DriftEnvironment["current"], decimals: number) {
  return `${vector.u.toFixed(decimals)} m/s E | ${vector.v.toFixed(decimals)} m/s N`
}

function formatCoordinate(value: number, positive: string, negative: string) {
  return `${Math.abs(value).toFixed(4)}° ${value >= 0 ? positive : negative}`
}

function trajectoryTime(point: DriftResponse["trajectory"][number], mode: DriftMode) {
  return mode === "forecast" ? `+${point.hour ?? 0} h` : `${point.hoursBeforeDetection ?? 0} h`
}

function TrajectoryChart({ response }: { response: DriftResponse }) {
  const lastPoint = response.trajectory[response.trajectory.length - 1]
  if (!lastPoint) return null

  const chartData = response.trajectory.map((point) => ({
    ...point,
    timeLabel: trajectoryTime(point, response.mode),
  }))
  const endLabel = response.mode === "forecast"
    ? `+${response.forecastHours}h · ${lastPoint.totalDriftKm.toFixed(1)} km`
    : `-${response.forecastHours}h · ${lastPoint.totalDriftKm.toFixed(1)} km`

  return (
    <div className="h-[340px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 24, right: 22, left: 4, bottom: 16 }}>
          <CartesianGrid stroke="#E7E8EA" strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="eastKm"
            tickLine={false}
            axisLine={{ stroke: "#D9DDE1" }}
            tick={{ fill: "#6B7075", fontSize: 11 }}
            tickFormatter={(value) => `${Number(value).toFixed(0)}`}
            label={{ value: "East displacement (km)", position: "insideBottom", offset: -8, fill: "#6B7075", fontSize: 11 }}
          />
          <YAxis
            type="number"
            dataKey="northKm"
            tickLine={false}
            axisLine={{ stroke: "#D9DDE1" }}
            tick={{ fill: "#6B7075", fontSize: 11 }}
            tickFormatter={(value) => `${Number(value).toFixed(0)}`}
            label={{ value: "North displacement (km)", angle: -90, position: "insideLeft", fill: "#6B7075", fontSize: 11 }}
            width={52}
          />
          <Tooltip
            cursor={{ stroke: "#C8FF3D", strokeWidth: 1 }}
            formatter={(value, name) => [
              `${Number(value).toFixed(2)} km`,
              name === "northKm" ? "North displacement" : "East displacement",
            ]}
            labelFormatter={(_, payload) => payload?.[0]?.payload?.timeLabel ?? "Trajectory"}
          />
          <Line
            type="monotone"
            dataKey="northKm"
            stroke="#202124"
            strokeWidth={2.5}
            dot={{ r: 2.5, fill: "#202124", stroke: "#202124" }}
            activeDot={{ r: 5, fill: "#C8FF3D", stroke: "#202124", strokeWidth: 2 }}
          />
          <ReferenceDot
            x={0}
            y={0}
            r={5}
            fill="#202124"
            stroke="#C8FF3D"
            strokeWidth={2}
            label={{ value: response.mode === "forecast" ? "Spill Start" : "Detected Spill", position: "top", fill: "#202124", fontSize: 11 }}
          />
          <ReferenceDot
            x={lastPoint.eastKm}
            y={lastPoint.northKm}
            r={5}
            fill="#C8FF3D"
            stroke="#202124"
            strokeWidth={2}
            label={{ value: endLabel, position: "top", fill: "#202124", fontSize: 11 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

function EnvironmentMetrics({ environment }: { environment: DriftEnvironment }) {
  return (
    <div className="grid gap-3 md:grid-cols-3">
      <div className="rounded-md border border-border bg-background p-4">
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">Ocean Current</p>
        <p className="mt-2 font-mono text-sm font-semibold text-foreground">{formatVector(environment.current, 2)}</p>
      </div>
      <div className="rounded-md border border-border bg-background p-4">
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">Surface Wind</p>
        <p className="mt-2 font-mono text-sm font-semibold text-foreground">{formatVector(environment.wind, 1)}</p>
      </div>
      <div className="rounded-md border border-border bg-background p-4">
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">Combined Drift</p>
        <p className="mt-2 font-mono text-sm font-semibold text-foreground">{environment.combinedDrift.speed.toFixed(2)} m/s</p>
      </div>
    </div>
  )
}

function SummaryMetrics({ response }: { response: DriftResponse }) {
  const distance = response.mode === "forecast"
    ? response.summary.predictedMovementKm
    : response.summary.estimatedSourceDisplacementKm
  const isForecast = response.mode === "forecast"

  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <div className="rounded-md border border-border bg-background p-4">
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">{isForecast ? "Predicted Movement" : "Estimated Source Displacement"}</p>
        <p className="mt-2 font-mono text-2xl font-semibold text-foreground">{distance?.toFixed(1) ?? "—"} km</p>
      </div>
      <div className="rounded-md border border-border bg-background p-4">
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">Direction</p>
        <p className="mt-2 font-mono text-2xl font-semibold text-foreground">{response.summary.direction}</p>
      </div>
      <div className="rounded-md border border-border bg-background p-4">
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">Average Drift Speed</p>
        <p className="mt-2 font-mono text-2xl font-semibold text-foreground">{response.summary.averageDriftSpeedMps.toFixed(2)} m/s</p>
      </div>
      <div className="rounded-md border border-border bg-background p-4">
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">{isForecast ? "Forecast Horizon" : "Hindcast Window"}</p>
        <p className="mt-2 font-mono text-2xl font-semibold text-foreground">{response.forecastHours} h</p>
      </div>
    </div>
  )
}

function CheckpointTable({ response }: { response: DriftResponse }) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Time</TableHead>
          <TableHead>East (km)</TableHead>
          <TableHead>North (km)</TableHead>
          <TableHead>Total Movement (km)</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {response.checkpoints.map((checkpoint) => (
          <TableRow key={`${response.mode}-${checkpoint.time}`}>
            <TableCell className="font-mono font-medium">{checkpoint.time > 0 ? `+${checkpoint.time}` : checkpoint.time} h</TableCell>
            <TableCell className="font-mono">{checkpoint.eastKm.toFixed(2)}</TableCell>
            <TableCell className="font-mono">{checkpoint.northKm.toFixed(2)}</TableCell>
            <TableCell className="font-mono">{checkpoint.totalMovementKm.toFixed(2)}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

function DetailedTrajectory({ response }: { response: DriftResponse }) {
  const isForecast = response.mode === "forecast"
  return (
    <details className="group rounded-md border border-border bg-white">
      <summary className="cursor-pointer list-none px-4 py-3 text-sm font-semibold text-foreground marker:hidden">
        <span className="flex items-center justify-between gap-3">
          Detailed {isForecast ? "Forecast" : "Hindcast"} Data
          <span className="text-xs font-normal text-muted-foreground group-open:hidden">Expand</span>
        </span>
      </summary>
      <div className="overflow-x-auto border-t border-border px-4 py-3">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>{isForecast ? "Hour" : "Hours Before Detection"}</TableHead>
              <TableHead>{isForecast ? "East Drift (km)" : "East Hindcast (km)"}</TableHead>
              <TableHead>{isForecast ? "North Drift (km)" : "North Hindcast (km)"}</TableHead>
              <TableHead>{isForecast ? "Total Drift (km)" : "Total Hindcast (km)"}</TableHead>
              <TableHead>Latitude</TableHead>
              <TableHead>Longitude</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {response.trajectory.map((point) => (
              <TableRow key={`${response.mode}-${point.hour ?? point.hoursBeforeDetection}`}>
                <TableCell className="font-mono">{isForecast ? `+${point.hour}` : point.hoursBeforeDetection} h</TableCell>
                <TableCell className="font-mono">{point.eastKm.toFixed(4)}</TableCell>
                <TableCell className="font-mono">{point.northKm.toFixed(4)}</TableCell>
                <TableCell className="font-mono">{point.totalDriftKm.toFixed(4)}</TableCell>
                <TableCell className="font-mono">{point.latitude?.toFixed(6) ?? "—"}</TableCell>
                <TableCell className="font-mono">{point.longitude?.toFixed(6) ?? "—"}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </details>
  )
}

export function DriftTab({ origin }: { origin?: DriftOrigin }) {
  const [mode, setMode] = useState<DriftMode>("forecast")
  const [hours, setHours] = useState(24)
  const [response, setResponse] = useState<DriftResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [retryToken, setRetryToken] = useState(0)
  const requestId = useRef(0)
  const originX = origin?.x
  const originY = origin?.y
  const originCrs = origin?.crs

  useEffect(() => {
    if (originX === undefined || originY === undefined || !originCrs) {
      return
    }

    const currentRequestId = ++requestId.current
    const timer = window.setTimeout(async () => {
      setLoading(true)
      setError(null)
      setResponse(null)
      try {
        const nextResponse = await getDriftAnalysis({
          origin: { x: originX, y: originY },
          crs: originCrs,
          hours,
          mode,
        })
        if (currentRequestId === requestId.current) setResponse(nextResponse)
      } catch (caught) {
        if (currentRequestId === requestId.current) {
          setError(caught instanceof ApiError ? caught.message : "Environmental forecast unavailable.")
        }
      } finally {
        if (currentRequestId === requestId.current) setLoading(false)
      }
    }, 300)

    return () => window.clearTimeout(timer)
  }, [hours, mode, originCrs, originX, originY, retryToken])

  const isForecast = mode === "forecast"

  return (
    <section className="space-y-6" aria-labelledby="drift-title">
      <div>
        <h2 id="drift-title" className="text-2xl font-semibold tracking-tight">Drift Simulation</h2>
        <p className="mt-1 text-sm text-muted-foreground">Environmental forcing, cumulative drift trajectory, and forecast movement.</p>
      </div>

      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <CardTitle className="text-base">Forecast Controls</CardTitle>
              <CardDescription>Choose the environmental analysis direction and time window.</CardDescription>
            </div>
            <div className="inline-flex rounded-md border border-border bg-slate-50 p-1" aria-label="Drift analysis mode">
              <Button size="sm" variant={isForecast ? "secondary" : "ghost"} className={isForecast ? "bg-[#c8ff3d] text-[#202124] hover:bg-[#c8ff3d]" : ""} onClick={() => setMode("forecast")}>
                <ArrowUpRight className="size-3.5" /> Forecast
              </Button>
              <Button size="sm" variant={!isForecast ? "secondary" : "ghost"} className={!isForecast ? "bg-[#c8ff3d] text-[#202124] hover:bg-[#c8ff3d]" : ""} onClick={() => setMode("hindcast")}>
                <ArrowDownLeft className="size-3.5" /> Hindcast
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between gap-3 text-sm">
            <label htmlFor="drift-hours" className="font-medium text-foreground">{isForecast ? "Forecast horizon" : "Hindcast window"}</label>
            <span className="font-mono text-muted-foreground">{hours} hours</span>
          </div>
          <input id="drift-hours" type="range" min={1} max={48} value={hours} onChange={(event) => setHours(Number(event.target.value))} className="h-2 w-full cursor-pointer accent-[#202124]" />
          <div className="flex justify-between text-xs text-muted-foreground"><span>1 h</span><span>24 h</span><span>48 h</span></div>
        </CardContent>
      </Card>

      {!origin ? (
        <Card>
          <CardContent className="p-0">
            <Empty className="min-h-56 border-0 bg-background">
              <EmptyHeader>
                <EmptyMedia variant="icon"><Wind className="size-5" /></EmptyMedia>
                <EmptyTitle>Spill origin required</EmptyTitle>
                <EmptyDescription>Upload and run a GeoTIFF in Tab 1 before requesting an environmental forecast.</EmptyDescription>
              </EmptyHeader>
            </Empty>
          </CardContent>
        </Card>
      ) : (
        <>
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Spill Origin</CardTitle>
              <CardDescription>Origin passed from the selected Tab 1 detection candidate.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4 sm:grid-cols-3">
              <div><p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">Longitude</p><p className="mt-1 font-mono text-sm font-semibold">{origin.longitude !== undefined ? formatCoordinate(origin.longitude, "E", "W") : "Derived by backend"}</p></div>
              <div><p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">Latitude</p><p className="mt-1 font-mono text-sm font-semibold">{origin.latitude !== undefined ? formatCoordinate(origin.latitude, "N", "S") : "Derived by backend"}</p></div>
              <div><p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">CRS</p><p className="mt-1 font-mono text-sm font-semibold">{origin.crs}</p></div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div><CardTitle className="flex items-center gap-2 text-base"><Wind className="size-4 text-sky-600" />Environmental Conditions</CardTitle><CardDescription>Real forcing values returned by the Python NetCDF analysis.</CardDescription></div>
                {response && <Badge variant="outline">{response.environment.forcingMode}</Badge>}
              </div>
            </CardHeader>
            <CardContent>
              {loading && <div className="flex min-h-40 items-center justify-center gap-2 text-sm text-muted-foreground"><LoaderCircle className="size-4 animate-spin" /> Loading environmental forecast...</div>}
              {!loading && error && <Alert variant="destructive"><AlertCircle className="size-4" /><AlertTitle>Environmental forecast unavailable</AlertTitle><AlertDescription className="flex flex-wrap items-center justify-between gap-3"><span>{error}</span><Button size="sm" variant="outline" onClick={() => setRetryToken((value) => value + 1)}><RefreshCw className="size-3.5" /> Retry</Button></AlertDescription></Alert>}
              {!loading && !error && response && <div className="space-y-5"><EnvironmentMetrics environment={response.environment} /><EnvironmentalConditionsChart data={response.environmentalForcing} /><p className="text-xs text-muted-foreground">Forcing mode: {response.environment.forcingMode}{response.environment.syntheticVariation ? " · prototype variation applied" : ""}.</p></div>}
            </CardContent>
            <Separator />
            <CardContent className="pt-4 text-xs text-muted-foreground">Current and wind values are returned in m/s; the combined drift uses the backend forcing calculation.{response?.environment.syntheticVariation ? " Temporal variation is shown using prototype forcing until time-series NetCDF data is available." : ""}</CardContent>
          </Card>

          {response && !loading && !error && <Card>
            <CardHeader>
              <div className="flex flex-wrap items-start justify-between gap-3"><div><CardTitle className="text-base">{response.forecastHours}-Hour {isForecast ? "Forecast" : "Hindcast"}</CardTitle><CardDescription>{isForecast ? "Predicted future spill movement from the detected origin." : "Estimated backtracked positions from the detected spill."}</CardDescription></div><Badge variant="secondary">{isForecast ? "Forecast" : "Potential Source Region"}</Badge></div>
            </CardHeader>
            <CardContent className="space-y-6">
              <SummaryMetrics response={response} />
              <div className="grid gap-6 lg:grid-cols-[minmax(0,1.6fr)_minmax(320px,1fr)]"><div className="min-w-0 rounded-md border border-border bg-white p-4"><div className="mb-3 flex items-center justify-between gap-3"><h3 className="text-sm font-semibold text-foreground">Predicted Spill Movement</h3><span className="text-xs text-muted-foreground">East / North displacement</span></div><TrajectoryChart response={response} /></div><div className="min-w-0 rounded-md border border-border bg-white p-4"><h3 className="mb-3 text-sm font-semibold text-foreground">{isForecast ? "Forecast" : "Hindcast"} Checkpoints</h3><div className="overflow-x-auto"><CheckpointTable response={response} /></div></div></div>
              <DetailedTrajectory response={response} />
              <Alert className="border-sky-100 bg-sky-50 text-slate-700"><AlertDescription>{response.note}</AlertDescription></Alert>
            </CardContent>
          </Card>}
        </>
      )}
    </section>
  )
}
