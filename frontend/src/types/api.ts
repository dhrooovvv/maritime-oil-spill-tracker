export interface SpillGeometry {
  area?: number
  perimeter?: number
  length?: number
  width?: number
  areaUnit?: string
  distanceUnit?: string
  orientation?: number
  aspectRatio?: number
}

export interface DetectionCandidate {
  id: string
  rank: number
  score?: number
  area?: number
  aspectRatio?: number
  meanIntensity?: number
  compactness?: number
  borderTouching?: boolean
  selected?: boolean
}

export interface ImageryPanel {
  src: string
  title: string
  caption?: string
  technicalLabel?: string
  highlighted?: boolean
}

export interface RasterMetadata {
  width?: number
  height?: number
  resolution?: number
  bounds?: string
  groundSamplingDistance?: string
}

export interface SpatialInformation {
  longitude?: number
  latitude?: number
  pixelX?: number
  pixelY?: number
  x?: number
  y?: number
  crs?: string
  epsg?: number
}

export interface DetectionResult {
  status: string
  confidence?: number
  geometry?: SpillGeometry
  spatial?: SpatialInformation
  raster?: RasterMetadata
  imagery?: {
    backscatter?: ImageryPanel
    adaptiveMask?: ImageryPanel
    polygonOverlay?: ImageryPanel
  }
  candidates?: DetectionCandidate[]
  morphologyLabel?: string
  interpretation?: string
  engine?: {
    name?: string
    status?: string
    note?: string
    hardware?: string
    precision?: string
    objective?: string
  }
  metadata?: Record<string, string | number | null>
  previewUrl?: string
}

export interface EnvironmentalPoint {
  hour: number
  currentSpeed?: number
  windSpeed?: number
  netSpeed?: number
}

export type DriftMode = "forecast" | "hindcast"

export interface DriftOrigin {
  x: number
  y: number
  crs: string
  longitude?: number
  latitude?: number
}

export interface DriftTrajectoryPoint {
  hour?: number
  hoursBeforeDetection?: number
  eastKm: number
  northKm: number
  totalDriftKm: number
  latitude?: number
  longitude?: number
}

export interface DriftVector {
  u: number
  v: number
  speed: number
}

export interface DriftEnvironment {
  current: DriftVector
  wind: DriftVector
  combinedDrift: DriftVector
  forcingMode: string
  syntheticVariation?: boolean
}

export interface DriftSummary {
  predictedMovementKm?: number
  estimatedSourceDisplacementKm?: number
  direction: string
  averageDriftSpeedMps: number
}

export interface DriftCheckpoint {
  time: number
  eastKm: number
  northKm: number
  totalMovementKm: number
}

export interface DriftResponse {
  mode: DriftMode
  forecastHours: number
  origin: DriftOrigin & {
    longitude: number
    latitude: number
  }
  environment: DriftEnvironment
  environmentalForcing: EnvironmentalPoint[]
  summary: DriftSummary
  trajectory: DriftTrajectoryPoint[]
  checkpoints: DriftCheckpoint[]
  note: string
}

export interface DriftRequest {
  origin: Pick<DriftOrigin, "x" | "y">
  crs: string
  hours: number
  mode: DriftMode
}

/** Backward-compatible name for code that previously imported DriftForecast. */
export type DriftForecast = DriftResponse

export interface SuspectVessel {
  vesselName: string
  mmsi: string
  vesselType?: string
  distanceMeters?: number
  suspicionScore?: number
}
