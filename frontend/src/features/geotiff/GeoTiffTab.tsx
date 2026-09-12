import { useRef, useState, type ChangeEvent } from "react"
import {
  ArrowRight,
  CheckCircle2,
  CircleAlert,
  Crosshair,
  FileImage,
  Info,
  Map,
  Play,
  Ruler,
  ServerOff,
  Upload,
} from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { ApiError, detectGeoTiff } from "@/lib/api"
import type {
  DetectionCandidate,
  DetectionResult,
  ImageryPanel,
} from "@/types/api"

type AnalysisState = "idle" | "ready" | "loading" | "success" | "error"

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function formatCoordinate(value: number, positive: string, negative: string) {
  return `${Math.abs(value).toFixed(4)}° ${value >= 0 ? positive : negative}`
}

function Metric({
  label,
  value,
  detail,
  className = "",
}: {
  label: string
  value: string
  detail?: string
  className?: string
}) {
  return (
    <div className={`min-w-0 space-y-1 ${className}`}>
      <p className="text-[0.68rem] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
        {label}
      </p>
      <p className="truncate font-mono text-sm font-semibold text-foreground sm:text-base">
        {value}
      </p>
      {detail && <p className="text-xs text-muted-foreground">{detail}</p>}
    </div>
  )
}

function ImagePanel({ panel, index }: { panel: ImageryPanel; index: string }) {
  return (
    <article className="min-w-0 space-y-2">
      <div className="flex items-center justify-between gap-3 text-xs">
        <h4 className="truncate font-semibold text-foreground">{index}. {panel.title}</h4>
        {panel.technicalLabel && (
          <span className="shrink-0 font-mono text-[0.68rem] text-muted-foreground">
            {panel.technicalLabel}
          </span>
        )}
      </div>
      <div
        className={`overflow-hidden rounded-md border bg-slate-950 ${
          panel.highlighted ? "border-sky-500 ring-1 ring-sky-100" : "border-border"
        }`}
      >
        <img
          src={panel.src}
          alt={panel.title}
          className="aspect-[4/3] h-auto w-full object-contain"
        />
      </div>
      {panel.caption && <p className="text-xs leading-relaxed text-muted-foreground">{panel.caption}</p>}
    </article>
  )
}

function ImageryInspection({ imagery }: { imagery: DetectionResult["imagery"] }) {
  const panels = [
    imagery?.backscatter ? { key: "backscatter", panel: imagery.backscatter, index: "01" } : null,
    imagery?.adaptiveMask ? { key: "adaptiveMask", panel: imagery.adaptiveMask, index: "02" } : null,
    imagery?.polygonOverlay ? { key: "polygonOverlay", panel: imagery.polygonOverlay, index: "03" } : null,
  ].filter(Boolean) as Array<{ key: string; panel: ImageryPanel; index: string }>

  if (panels.length === 0) return null

  return (
    <Card>
      <CardHeader className="border-b border-border py-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <CardTitle className="flex items-center gap-2 text-sm uppercase tracking-wide">
            <Map className="size-4 text-sky-600" />
            SAR GeoTIFF Imagery Inspection
          </CardTitle>
          <p className="font-mono text-xs text-muted-foreground">Processed imagery</p>
        </div>
      </CardHeader>
      <CardContent className="grid gap-6 pt-6 md:grid-cols-3">
        {panels.map(({ key, panel, index }) => (
          <ImagePanel key={key} panel={panel} index={index} />
        ))}
      </CardContent>
    </Card>
  )
}

function SpatialGeolocation({ result }: { result: DetectionResult }) {
  const spatial = result.spatial
  const raster = result.raster
  if (!spatial && !raster) return null

  const centroid =
    spatial?.latitude !== undefined && spatial.longitude !== undefined
      ? `${formatCoordinate(spatial.latitude, "N", "S")}, ${formatCoordinate(spatial.longitude, "E", "W")}`
      : null
  const pixelCentroid =
    spatial?.pixelX !== undefined && spatial.pixelY !== undefined
      ? `(X: ${spatial.pixelX}, Y: ${spatial.pixelY})`
      : null
  const resolution = raster?.groundSamplingDistance ??
    (raster?.resolution !== undefined ? `${raster.resolution} m/px` : null)
  const bounds = raster?.bounds ??
    (raster?.width !== undefined && raster.height !== undefined
      ? `${raster.width} × ${raster.height} px`
      : null)

  const metrics = [
    centroid && { label: "Spatial Centroid (WGS84)", value: centroid },
    pixelCentroid && { label: "Raster Pixel Centroid", value: pixelCentroid },
    spatial?.crs && { label: "Coordinate System (CRS)", value: spatial.crs },
    resolution && { label: "Ground Sampling Distance", value: resolution },
  ].filter(Boolean) as Array<{ label: string; value: string }>

  if (metrics.length === 0 && !bounds) return null

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Crosshair className="size-4 text-sky-600" />
          Spatial Geolocation &amp; Coordinate Reference
        </CardTitle>
        <CardDescription>Geospatial values returned by the calibrated raster pipeline.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {metrics.map((metric) => (
            <Metric key={metric.label} {...metric} />
          ))}
        </div>
        {bounds && (
          <div className="grid gap-2 rounded-md border border-border bg-background p-3 sm:grid-cols-[180px_1fr] sm:items-center">
            <p className="text-xs font-semibold text-muted-foreground">Raster Bounding Box</p>
            <p className="font-mono text-xs text-foreground">{bounds}</p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function MorphologicalCharacterization({ result }: { result: DetectionResult }) {
  const geometry = result.geometry
  if (!geometry) return null

  const primary = [
    geometry.area !== undefined && { label: "Surface Area Estimate", value: geometry.area.toLocaleString(), unit: geometry.areaUnit ?? "px²" },
    geometry.perimeter !== undefined && { label: "Contour Perimeter", value: geometry.perimeter.toLocaleString(), unit: geometry.distanceUnit ?? "px" },
    geometry.aspectRatio !== undefined && { label: "Aspect Ratio & Elongation", value: geometry.aspectRatio.toFixed(2), unit: "" },
  ].filter(Boolean) as Array<{ label: string; value: string; unit: string }>
  const secondary = [
    geometry.length !== undefined && { label: "Approx. Major Length", value: `${geometry.length.toLocaleString()} ${geometry.distanceUnit ?? "px"}` },
    geometry.width !== undefined && { label: "Approx. Minor Width", value: `${geometry.width.toLocaleString()} ${geometry.distanceUnit ?? "px"}` },
    geometry.orientation !== undefined && { label: "Orientation / Major Axis", value: `${geometry.orientation.toFixed(1)}°` },
  ].filter(Boolean) as Array<{ label: string; value: string }>

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <CardTitle className="text-base">Primary Slick Morphological Characterization</CardTitle>
            <CardDescription>Computed geometric properties for the selected detection candidate.</CardDescription>
          </div>
          {result.morphologyLabel && <Badge variant="outline">{result.morphologyLabel}</Badge>}
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="grid gap-4 md:grid-cols-3">
          {primary.map((metric) => (
            <div key={metric.label} className="rounded-md border border-border bg-background p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{metric.label}</p>
              <p className="mt-2 font-mono text-2xl font-semibold tracking-tight text-foreground">
                {metric.value} <span className="text-sm font-medium text-muted-foreground">{metric.unit}</span>
              </p>
            </div>
          ))}
        </div>
        {secondary.length > 0 && (
          <div className="grid gap-3 border-t border-border pt-5 sm:grid-cols-3">
            {secondary.map((metric) => (
              <Metric key={metric.label} {...metric} />
            ))}
          </div>
        )}
        {result.interpretation && (
          <Alert className="border-sky-100 bg-sky-50 text-slate-700">
            <Info className="size-4 text-sky-600" />
            <AlertTitle>Morphological Interpretation</AlertTitle>
            <AlertDescription>{result.interpretation}</AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  )
}

function CandidateTable({ candidates }: { candidates: DetectionCandidate[] }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <CardTitle className="text-base">Detection Candidates Ranking</CardTitle>
            <CardDescription>Candidate records returned by the classical ranking pipeline.</CardDescription>
          </div>
          <Badge variant="secondary">{candidates.length} candidates</Badge>
        </div>
      </CardHeader>
      <CardContent className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Candidate</TableHead>
              <TableHead>Rank</TableHead>
              <TableHead>Classical Score</TableHead>
              <TableHead>Area (px²)</TableHead>
              <TableHead>Aspect Ratio</TableHead>
              <TableHead>Mean Intensity</TableHead>
              <TableHead>Compactness</TableHead>
              <TableHead>Border Touching</TableHead>
              <TableHead>Simulation Target</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {candidates.map((candidate) => (
              <TableRow key={candidate.id} className={candidate.selected ? "bg-sky-50/70" : undefined}>
                <TableCell className="font-medium">{candidate.id}</TableCell>
                <TableCell>{candidate.rank}</TableCell>
                <TableCell className="font-mono">{candidate.score?.toFixed(3) ?? "—"}</TableCell>
                <TableCell className="font-mono">{candidate.area?.toLocaleString() ?? "—"}</TableCell>
                <TableCell className="font-mono">{candidate.aspectRatio?.toFixed(2) ?? "—"}</TableCell>
                <TableCell className="font-mono">{candidate.meanIntensity?.toFixed(2) ?? "—"}</TableCell>
                <TableCell className="font-mono">{candidate.compactness?.toFixed(3) ?? "—"}</TableCell>
                <TableCell>{candidate.borderTouching === undefined ? "—" : candidate.borderTouching ? "Yes" : "No"}</TableCell>
                <TableCell>{candidate.selected ? <Badge variant="outline">Active</Badge> : "Available"}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}

function DetectionResultView({ result, onContinue }: { result: DetectionResult; onContinue?: () => void }) {
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold tracking-tight">Analysis Results</h3>
          <p className="text-sm text-muted-foreground">Values below are supplied by the detection response.</p>
        </div>
        <Badge className="gap-1.5 bg-[#c8ff3d] text-[#202124] hover:bg-[#c8ff3d]">
          <CheckCircle2 className="size-3.5" />
          {result.status}
        </Badge>
      </div>
      <ImageryInspection imagery={result.imagery} />
      <SpatialGeolocation result={result} />
      <MorphologicalCharacterization result={result} />
      {result.candidates && result.candidates.length > 0 && <CandidateTable candidates={result.candidates} />}
      {result.previewUrl && !result.imagery && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Detection Preview</CardTitle>
          </CardHeader>
          <CardContent>
            <img src={result.previewUrl} alt="Detection result" className="max-h-[520px] w-full rounded-md border object-contain" />
          </CardContent>
        </Card>
      )}
      <Card className="border-sky-100 bg-sky-50/60">
        <CardContent className="flex flex-col gap-4 p-5 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-sm font-semibold text-foreground">Next pipeline action</p>
            <p className="text-sm text-muted-foreground">Continue to the ocean drift simulation with the selected result.</p>
          </div>
          <Button onClick={onContinue} disabled={!onContinue}>
            Open Drift Simulation
            <ArrowRight className="size-4" />
          </Button>
        </CardContent>
      </Card>
    </div>
  )
}

export function GeoTiffTab({
  onContinue,
  onDetectionResult,
}: {
  onContinue?: () => void
  onDetectionResult?: (result: DetectionResult | null) => void
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [state, setState] = useState<AnalysisState>("idle")
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<DetectionResult | null>(null)

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const nextFile = event.target.files?.[0] ?? null
    setError(null)
    setResult(null)
    onDetectionResult?.(null)

    if (!nextFile) {
      setFile(null)
      setState("idle")
      return
    }

    if (!/\.(tif|tiff)$/i.test(nextFile.name)) {
      setFile(null)
      setState("error")
      setError("Please select a .tif or .tiff GeoTIFF file.")
      return
    }

    setFile(nextFile)
    setState("ready")
  }

  async function handleDetection() {
    if (!file) return

    setState("loading")
    setError(null)
    try {
      const detection = await detectGeoTiff(file)
      setResult(detection)
      onDetectionResult?.(detection)
      setState("success")
    } catch (caught) {
      setState("error")
      onDetectionResult?.(null)
      setError(
        caught instanceof ApiError
          ? caught.message
          : "The detection request could not be completed."
      )
    }
  }

  const engine = result?.engine

  return (
    <section className="space-y-8" aria-labelledby="geotiff-title">
      <div>
        <h2 id="geotiff-title" className="text-3xl font-semibold tracking-tight text-foreground">
          Maritime Oil Spill Detection &amp; Vessel Tracking
        </h2>
        <p className="mt-2 text-sm text-muted-foreground sm:text-base">
          Production GIS Pipeline — SAR GeoTIFF Extraction, Oil-Slick Segmentation &amp; AIS Trajectory Correlation
        </p>
      </div>

      <Card className="bg-white">
        <CardContent className="flex flex-col gap-4 p-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
            <span className="font-semibold text-foreground">Classification Engine:</span>
            <span className="inline-flex items-center gap-2 text-muted-foreground">
              <span className="size-2 rounded-full bg-amber-500" />
              {engine?.name ?? "Backend status pending"}
            </span>
            <span className="hidden text-border sm:inline">•</span>
            <span className="text-xs text-muted-foreground">
              {engine?.note ?? "Runtime metadata will be shown when the Python API returns it."}
            </span>
          </div>
          <div className="flex flex-wrap gap-2 font-mono text-[0.68rem] text-muted-foreground">
            {engine?.hardware && <Badge variant="secondary">Hardware: {engine.hardware}</Badge>}
            {engine?.precision && <Badge variant="secondary">Precision: {engine.precision}</Badge>}
            {engine?.objective && <Badge variant="secondary">Objective: {engine.objective}</Badge>}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="border-b border-border py-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <CardTitle className="flex items-center gap-2 text-sm uppercase tracking-wide">
              <FileImage className="size-4 text-sky-600" />
              GeoTIFF Raster Ingestion Hub
            </CardTitle>
            <span className="font-mono text-xs text-muted-foreground">Format: GeoTIFF (.tif, .tiff)</span>
          </div>
        </CardHeader>
        <CardContent className="p-6">
          <input
            ref={inputRef}
            className="sr-only"
            type="file"
            accept=".tif,.tiff,image/tiff"
            onChange={handleFileChange}
            aria-label="Upload GeoTIFF"
          />

          {!file ? (
            <div className="flex min-h-64 flex-col items-center justify-center gap-5 rounded-lg border border-dashed border-slate-300 bg-slate-50/40 px-6 py-10 text-center">
              <div className="flex size-12 items-center justify-center rounded-full border border-border bg-white text-sky-600 shadow-sm">
                <Upload className="size-5" />
              </div>
              <div className="space-y-2">
                <h3 className="text-lg font-semibold text-foreground">Upload Satellite GeoTIFF Imagery</h3>
                <p className="max-w-xl text-sm text-muted-foreground">
                  Select a calibrated SAR GeoTIFF scene for oil-spill detection and spatial analysis.
                </p>
              </div>
              <div className="flex flex-wrap justify-center gap-2">
                <Button type="button" onClick={() => inputRef.current?.click()}>
                  <Upload className="size-4" />
                  Select GeoTIFF (.tif)
                </Button>
                <Badge variant="outline" className="h-9 px-3 font-mono font-normal">
                  .tif / .tiff · backend validated
                </Badge>
              </div>
            </div>
          ) : (
            <div className="flex flex-col gap-4 rounded-lg border border-border bg-slate-50/50 p-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex min-w-0 items-center gap-3">
                <div className="flex size-10 shrink-0 items-center justify-center rounded-md bg-white text-sky-600 shadow-sm">
                  <FileImage className="size-5" />
                </div>
                <div className="min-w-0">
                  <p className="truncate text-sm font-semibold text-foreground">{file.name}</p>
                  <p className="font-mono text-xs text-muted-foreground">
                    {formatBytes(file.size)} · {file.type || "GeoTIFF"}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Badge variant="secondary">GeoTIFF</Badge>
                <Badge variant={state === "error" ? "destructive" : "secondary"}>
                  {state === "loading" ? "Processing" : state === "success" ? "Complete" : "Ready"}
                </Badge>
              </div>
            </div>
          )}

          {state === "loading" && (
            <div className="mt-5 space-y-2">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Processing satellite scene...</span>
                <span>In progress</span>
              </div>
              <Progress value={null} />
            </div>
          )}

          {error && (
            <Alert variant="destructive" className="mt-5">
              <ServerOff className="size-4" />
              <AlertTitle>Unable to process GeoTIFF</AlertTitle>
              <AlertDescription>{error} Try another file or verify that the Python API is running.</AlertDescription>
            </Alert>
          )}
        </CardContent>
        <Separator />
        <CardFooter className="flex flex-col items-stretch gap-3 p-4 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-xs text-muted-foreground">Accepted formats remain limited to the existing GeoTIFF contract.</p>
          <Button disabled={!file || state === "loading"} onClick={handleDetection}>
            <Play className="size-4" />
            Run Detection
          </Button>
        </CardFooter>
      </Card>

      {!file && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <CircleAlert className="size-3.5" />
          Detection results, imagery, and geometry appear after a real backend response.
        </div>
      )}

      {file && state === "ready" && !result && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Ruler className="size-3.5" />
          File selected locally; no raster metadata is inferred before backend processing.
        </div>
      )}

      {result && <DetectionResultView result={result} onContinue={onContinue} />}
    </section>
  )
}
