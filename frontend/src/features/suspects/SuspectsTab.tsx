import { useRef, useState } from "react"
import type { LucideIcon } from "lucide-react"
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Clock3,
  Compass,
  Info,
  MapPin,
  Navigation,
  RefreshCw,
  Route,
  Ship,
  Target,
  TrendingUp,
} from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
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

type Priority = "HIGH" | "MEDIUM" | "LOW"

type TimelineEvent = {
  time: string
  location: string
  speedKnots: number
  heading: number
}

type EvidenceMetric = {
  label: string
  score: number
  value: string
  explanation: string
  icon: LucideIcon
}

type TrackPoint = {
  x: number
  y: number
}

type Candidate = {
  rank: number
  vesselName: string
  mmsi: string
  vesselType: string
  flag: string
  distanceKm: number
  trajectoryMatch: number
  timeMatch: number
  confidence: number
  priority: Priority
  closestApproachTime: string
  closestApproachHeading: number
  closestApproachSpeed: number
  evidence: EvidenceMetric[]
  timeline: TimelineEvent[]
  track: TrackPoint[]
}

// Prototype-only data remains isolated until a vessel-correlation API contract exists.
const prototypeVesselMatch: {
  mode: "prototype"
  incidentId: string
  scoringModel: Array<{ label: string; weight: number }>
  candidates: Candidate[]
} = {
  mode: "prototype",
  incidentId: "SPILL-2026-IN01",
  scoringModel: [
    { label: "Spatial proximity", weight: 30 },
    { label: "Trajectory alignment", weight: 30 },
    { label: "Temporal alignment", weight: 25 },
    { label: "Vessel characteristics", weight: 15 },
  ],
  candidates: [
    {
      rank: 1,
      vesselName: "PACIFIC_VOYAGER",
      mmsi: "413210000",
      vesselType: "Tanker — Crude Oil",
      flag: "India",
      distanceKm: 0.8,
      trajectoryMatch: 91,
      timeMatch: 97,
      confidence: 91.4,
      priority: "HIGH",
      closestApproachTime: "-1h 12m",
      closestApproachHeading: 71,
      closestApproachSpeed: 7.4,
      evidence: [
        { label: "Spatial proximity", score: 94, value: "0.8 km", explanation: "Closest recorded approach to the reconstructed spill origin.", icon: MapPin },
        { label: "Trajectory alignment", score: 91, value: "91%", explanation: "Historical track follows the hindcast corridor into the source region.", icon: Route },
        { label: "Temporal alignment", score: 97, value: "97%", explanation: "AIS timestamps overlap the estimated spill formation window.", icon: Clock3 },
        { label: "Vessel characteristics", score: 88, value: "88%", explanation: "Vessel class is compatible with the prototype source profile.", icon: Ship },
      ],
      timeline: [
        { time: "-31h", location: "31.2 km from source region", speedKnots: 12.4, heading: 64 },
        { time: "-24h", location: "22.1 km from source region", speedKnots: 11.8, heading: 72 },
        { time: "-18h", location: "13.8 km from source region", speedKnots: 11.4, heading: 78 },
        { time: "-12h", location: "6.8 km from source region", speedKnots: 11.2, heading: 82 },
        { time: "-6h", location: "4.6 km from source region", speedKnots: 9.8, heading: 76 },
        { time: "-1h", location: "0.8 km minimum approach", speedKnots: 7.4, heading: 71 },
      ],
      track: [
        { x: 72, y: 232 },
        { x: 148, y: 214 },
        { x: 230, y: 191 },
        { x: 315, y: 163 },
        { x: 398, y: 139 },
        { x: 476, y: 130 },
        { x: 548, y: 161 },
        { x: 608, y: 189 },
      ],
    },
    {
      rank: 2,
      vesselName: "OCEAN_HORIZON",
      mmsi: "413780000",
      vesselType: "Product Tanker",
      flag: "India",
      distanceKm: 3.4,
      trajectoryMatch: 84,
      timeMatch: 89,
      confidence: 82.7,
      priority: "MEDIUM",
      closestApproachTime: "-3h 18m",
      closestApproachHeading: 84,
      closestApproachSpeed: 9.1,
      evidence: [
        { label: "Spatial proximity", score: 82, value: "3.4 km", explanation: "Entered the investigation radius but remained outside the closest approach.", icon: MapPin },
        { label: "Trajectory alignment", score: 84, value: "84%", explanation: "Track partially overlaps the reconstructed drift corridor.", icon: Route },
        { label: "Temporal alignment", score: 89, value: "89%", explanation: "AIS history overlaps most of the estimated time window.", icon: Clock3 },
        { label: "Vessel characteristics", score: 80, value: "80%", explanation: "Product tanker profile is relevant but less specific to the scenario.", icon: Ship },
      ],
      timeline: [
        { time: "-31h", location: "38.4 km from source region", speedKnots: 13.1, heading: 71 },
        { time: "-24h", location: "28.7 km from source region", speedKnots: 12.6, heading: 76 },
        { time: "-18h", location: "18.9 km from source region", speedKnots: 11.7, heading: 80 },
        { time: "-12h", location: "10.5 km from source region", speedKnots: 10.2, heading: 83 },
        { time: "-6h", location: "5.2 km from source region", speedKnots: 9.4, heading: 84 },
        { time: "-3h", location: "3.4 km minimum approach", speedKnots: 9.1, heading: 84 },
      ],
      track: [{ x: 70, y: 235 }, { x: 148, y: 220 }, { x: 226, y: 202 }, { x: 306, y: 181 }, { x: 388, y: 165 }, { x: 468, y: 164 }, { x: 548, y: 183 }, { x: 612, y: 212 }],
    },
    {
      rank: 3,
      vesselName: "EASTERN_TRADER",
      mmsi: "419120000",
      vesselType: "Chemical Tanker",
      flag: "India",
      distanceKm: 7.8,
      trajectoryMatch: 72,
      timeMatch: 81,
      confidence: 74.3,
      priority: "MEDIUM",
      closestApproachTime: "-7h 41m",
      closestApproachHeading: 102,
      closestApproachSpeed: 10.6,
      evidence: [
        { label: "Spatial proximity", score: 68, value: "7.8 km", explanation: "Observed within the wider investigation radius.", icon: MapPin },
        { label: "Trajectory alignment", score: 72, value: "72%", explanation: "Track approaches the corridor but diverges before the source region.", icon: Route },
        { label: "Temporal alignment", score: 81, value: "81%", explanation: "Partial overlap with the estimated spill formation window.", icon: Clock3 },
        { label: "Vessel characteristics", score: 76, value: "76%", explanation: "Chemical tanker class is a relevant but non-specific indicator.", icon: Ship },
      ],
      timeline: [
        { time: "-31h", location: "46.2 km from source region", speedKnots: 12.8, heading: 92 },
        { time: "-24h", location: "35.5 km from source region", speedKnots: 12.2, heading: 95 },
        { time: "-18h", location: "24.1 km from source region", speedKnots: 11.9, heading: 99 },
        { time: "-12h", location: "14.8 km from source region", speedKnots: 11.0, heading: 101 },
        { time: "-7h", location: "7.8 km minimum approach", speedKnots: 10.6, heading: 102 },
        { time: "-1h", location: "9.6 km from source region", speedKnots: 10.1, heading: 106 },
      ],
      track: [{ x: 76, y: 222 }, { x: 156, y: 208 }, { x: 238, y: 193 }, { x: 318, y: 185 }, { x: 397, y: 191 }, { x: 478, y: 213 }, { x: 555, y: 231 }, { x: 618, y: 239 }],
    },
    {
      rank: 4,
      vesselName: "SEA_FALCON",
      mmsi: "477120300",
      vesselType: "Bulk Carrier",
      flag: "Hong Kong",
      distanceKm: 12.6,
      trajectoryMatch: 51,
      timeMatch: 63,
      confidence: 58.9,
      priority: "LOW",
      closestApproachTime: "-11h 08m",
      closestApproachHeading: 214,
      closestApproachSpeed: 13.2,
      evidence: [
        { label: "Spatial proximity", score: 49, value: "12.6 km", explanation: "Outside the primary screening radius.", icon: MapPin },
        { label: "Trajectory alignment", score: 51, value: "51%", explanation: "Track has limited overlap with the reconstructed corridor.", icon: Route },
        { label: "Temporal alignment", score: 63, value: "63%", explanation: "Only a partial timestamp overlap is available.", icon: Clock3 },
        { label: "Vessel characteristics", score: 57, value: "57%", explanation: "Bulk carrier profile provides weak scenario-specific evidence.", icon: Ship },
      ],
      timeline: [
        { time: "-31h", location: "59.0 km from source region", speedKnots: 14.2, heading: 195 },
        { time: "-24h", location: "46.7 km from source region", speedKnots: 13.8, heading: 202 },
        { time: "-18h", location: "32.8 km from source region", speedKnots: 13.5, heading: 208 },
        { time: "-12h", location: "18.7 km from source region", speedKnots: 13.2, heading: 213 },
        { time: "-11h", location: "12.6 km minimum approach", speedKnots: 13.2, heading: 214 },
        { time: "-1h", location: "19.3 km from source region", speedKnots: 13.0, heading: 219 },
      ],
      track: [{ x: 70, y: 214 }, { x: 150, y: 198 }, { x: 230, y: 180 }, { x: 310, y: 161 }, { x: 392, y: 145 }, { x: 474, y: 129 }, { x: 554, y: 113 }, { x: 618, y: 99 }],
    },
    {
      rank: 5,
      vesselName: "INDIAN_STAR",
      mmsi: "419650000",
      vesselType: "Container Ship",
      flag: "India",
      distanceKm: 18.2,
      trajectoryMatch: 39,
      timeMatch: 54,
      confidence: 47.1,
      priority: "LOW",
      closestApproachTime: "-19h 32m",
      closestApproachHeading: 286,
      closestApproachSpeed: 16.1,
      evidence: [
        { label: "Spatial proximity", score: 35, value: "18.2 km", explanation: "Outside the investigation radius used for primary matching.", icon: MapPin },
        { label: "Trajectory alignment", score: 39, value: "39%", explanation: "Track does not closely follow the hindcast corridor.", icon: Route },
        { label: "Temporal alignment", score: 54, value: "54%", explanation: "Limited timing overlap with the prototype spill window.", icon: Clock3 },
        { label: "Vessel characteristics", score: 48, value: "48%", explanation: "Container ship class is not a strong scenario match.", icon: Ship },
      ],
      timeline: [
        { time: "-31h", location: "74.1 km from source region", speedKnots: 17.2, heading: 271 },
        { time: "-24h", location: "58.2 km from source region", speedKnots: 16.9, heading: 276 },
        { time: "-18h", location: "39.7 km from source region", speedKnots: 16.4, heading: 282 },
        { time: "-19h", location: "18.2 km minimum approach", speedKnots: 16.1, heading: 286 },
        { time: "-12h", location: "25.8 km from source region", speedKnots: 15.8, heading: 290 },
        { time: "-1h", location: "42.4 km from source region", speedKnots: 15.4, heading: 295 },
      ],
      track: [{ x: 72, y: 112 }, { x: 150, y: 128 }, { x: 230, y: 144 }, { x: 312, y: 160 }, { x: 395, y: 174 }, { x: 478, y: 186 }, { x: 554, y: 197 }, { x: 618, y: 205 }],
    },
  ],
}

function priorityClass(priority: Priority) {
  if (priority === "HIGH") return "border-red-200 bg-red-50 text-red-700"
  if (priority === "MEDIUM") return "border-amber-200 bg-amber-50 text-amber-700"
  return "border-slate-200 bg-slate-50 text-slate-600"
}

function formatScore(score: number) {
  return score.toFixed(1)
}

function SummaryMetric({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="rounded-lg border border-border bg-slate-50/70 p-4">
      <p className="text-[0.68rem] font-semibold uppercase tracking-[0.12em] text-muted-foreground">{label}</p>
      <p className="mt-2 text-2xl font-semibold tracking-tight text-foreground">{value}</p>
      <p className="mt-1 text-xs text-muted-foreground">{hint}</p>
    </div>
  )
}

function EvidenceRow({ metric }: { metric: EvidenceMetric }) {
  const Icon = metric.icon
  return (
    <div className="space-y-2 rounded-lg border border-border/80 bg-white p-4">
      <div className="flex items-start gap-3">
        <span className="flex size-8 shrink-0 items-center justify-center rounded-md bg-sky-50 text-sky-700">
          <Icon className="size-4" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between gap-3">
            <p className="text-sm font-semibold text-foreground">{metric.label}</p>
            <span className="font-mono text-sm font-semibold text-sky-700">{metric.value}</span>
          </div>
          <Progress value={metric.score} className="mt-3 [&_[data-slot=progress-indicator]]:bg-sky-600" />
          <p className="mt-2 text-xs leading-5 text-muted-foreground">{metric.explanation}</p>
        </div>
      </div>
    </div>
  )
}

function TrackDiagram({ candidate }: { candidate: Candidate }) {
  const points = candidate.track.map((point) => `${point.x},${point.y}`).join(" ")
  const closest = candidate.track[Math.max(0, candidate.track.length - 3)]

  return (
    <div className="overflow-hidden rounded-lg border border-slate-200 bg-slate-950 p-3">
      <svg viewBox="0 0 700 300" className="h-auto w-full" role="img" aria-label={`Schematic AIS track for ${candidate.vesselName}`}>
        <defs>
          <pattern id="track-grid" width="35" height="35" patternUnits="userSpaceOnUse">
            <path d="M 35 0 L 0 0 0 35" fill="none" stroke="#20304b" strokeWidth="1" />
          </pattern>
          <marker id="track-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L0,6 L7,3 z" fill="#38bdf8" />
          </marker>
        </defs>
        <rect width="700" height="300" rx="8" fill="url(#track-grid)" />
        <path d="M40 260 C160 210 200 236 302 184 S470 90 660 45" fill="none" stroke="#334155" strokeWidth="1" strokeDasharray="4 6" />
        <circle cx="555" cy="142" r="38" fill="#0ea5e91a" stroke="#38bdf8" strokeDasharray="4 4" />
        <text x="555" y="98" textAnchor="middle" fill="#7dd3fc" fontSize="12">Potential source region</text>
        <circle cx="555" cy="142" r="7" fill="#fbbf24" stroke="#fff" strokeWidth="2" />
        <path d="M543 142h24M555 130v24" stroke="#fbbf24" strokeWidth="1.5" />
        <text x="575" y="146" fill="#fde68a" fontSize="12">Spill origin</text>
        <polyline points={points} fill="none" stroke="#38bdf8" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" markerEnd="url(#track-arrow)" />
        {candidate.track.map((point, index) => (
          <circle key={`${point.x}-${point.y}`} cx={point.x} cy={point.y} r={index === candidate.track.length - 3 ? 7 : 4} fill={index === candidate.track.length - 3 ? "#fb7185" : "#bae6fd"} stroke="#0f172a" strokeWidth="2" />
        ))}
        <circle cx={closest.x} cy={closest.y} r="13" fill="none" stroke="#fb7185" strokeWidth="2" strokeDasharray="3 3" />
        <text x={closest.x + 16} y={closest.y - 10} fill="#fecdd3" fontSize="12">Closest approach</text>
        <text x="24" y="280" fill="#94a3b8" fontSize="11">Historical AIS positions · schematic relative view · simulated</text>
      </svg>
      <div className="mt-2 flex flex-wrap gap-x-5 gap-y-2 px-1 text-xs text-slate-300">
        <span className="inline-flex items-center gap-2"><span className="size-2 rounded-full bg-sky-400" />Vessel track</span>
        <span className="inline-flex items-center gap-2"><span className="size-2 rounded-full bg-rose-400" />Closest approach</span>
        <span className="inline-flex items-center gap-2"><span className="size-2 rounded-full bg-amber-300" />Spill origin</span>
      </div>
    </div>
  )
}

export function SuspectsTab() {
  const [selectedRank, setSelectedRank] = useState(1)
  const [notice, setNotice] = useState<string | null>(null)
  const trackRef = useRef<HTMLDivElement>(null)
  const selectedCandidate = prototypeVesselMatch.candidates.find((candidate) => candidate.rank === selectedRank) ?? prototypeVesselMatch.candidates[0]

  function showNotice(message: string) {
    setNotice(message)
    window.setTimeout(() => setNotice(null), 4200)
  }

  return (
    <section className="space-y-6" aria-labelledby="suspects-title">
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h2 id="suspects-title" className="text-2xl font-semibold tracking-tight">Suspect Match</h2>
            <Badge variant="outline" className="border-sky-200 bg-sky-50 text-sky-700">Prototype AIS Correlation</Badge>
          </div>
          <p className="mt-1 text-sm text-muted-foreground">Review vessel proximity and investigative evidence returned by the Python backend.</p>
        </div>
        <div className="inline-flex w-fit items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-700">
          <span className="size-2 rounded-full bg-emerald-500" />Analysis Ready
        </div>
      </div>

      <Alert className="border-amber-200 bg-amber-50/70">
        <AlertTriangle className="text-amber-600" />
        <div className="flex w-full flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <AlertTitle className="text-xs font-semibold uppercase tracking-[0.14em] text-amber-800">Potential source vessel identified</AlertTitle>
            <AlertDescription className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-foreground">
              <span className="text-lg font-semibold">{selectedCandidate.vesselName}</span>
              <span className="text-sm text-muted-foreground">MMSI {selectedCandidate.mmsi} · {selectedCandidate.vesselType}</span>
            </AlertDescription>
          </div>
          <div className="flex items-center gap-5 rounded-lg border border-amber-200 bg-white/70 px-4 py-3">
            <div><p className="text-[0.65rem] font-semibold uppercase tracking-[0.12em] text-muted-foreground">Correlation confidence</p><p className="mt-1 text-3xl font-semibold tracking-tight text-foreground">{formatScore(selectedCandidate.confidence)}%</p></div>
            <div className="min-w-28"><Badge className={priorityClass(selectedCandidate.priority)}>Priority {selectedCandidate.priority}</Badge><Progress value={selectedCandidate.confidence} className="mt-3 w-28 [&_[data-slot=progress-indicator]]:bg-amber-500" /></div>
          </div>
        </div>
      </Alert>

      <p className="-mt-3 text-xs text-muted-foreground">Correlation confidence describes prototype evidence strength, not probability of causation. Proximity alone does not establish vessel responsibility.</p>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <SummaryMetric label="Distance to spill origin" value={`${selectedCandidate.distanceKm.toFixed(1)} km`} hint="Closest recorded approach" />
        <SummaryMetric label="Hindcast compatibility" value={`${selectedCandidate.evidence[1].score + 3}%`} hint="Prototype source corridor" />
        <SummaryMetric label="Trajectory alignment" value={`${selectedCandidate.trajectoryMatch}%`} hint="Historical track overlap" />
        <SummaryMetric label="Time window match" value={`${selectedCandidate.timeMatch}%`} hint="AIS and spill window overlap" />
      </div>

      <Card>
        <CardHeader className="gap-2 sm:flex-row sm:items-start sm:justify-between">
          <div><CardTitle className="text-base">Ranked vessel candidates</CardTitle><CardDescription>Sorted by prototype correlation score using simulated historical AIS records.</CardDescription></div>
          <Badge variant="outline" className="w-fit font-mono text-[0.68rem]">{prototypeVesselMatch.candidates.length} candidates · {prototypeVesselMatch.incidentId}</Badge>
        </CardHeader>
        <CardContent className="px-0 pb-0">
          <div className="overflow-x-auto">
            <Table>
              <TableHeader><TableRow><TableHead className="pl-6">Rank</TableHead><TableHead>Vessel</TableHead><TableHead>MMSI</TableHead><TableHead>Type</TableHead><TableHead>Distance</TableHead><TableHead>Trajectory</TableHead><TableHead>Time</TableHead><TableHead>Confidence</TableHead><TableHead className="pr-6">Priority</TableHead></TableRow></TableHeader>
              <TableBody>
                {prototypeVesselMatch.candidates.map((candidate) => (
                  <TableRow key={candidate.mmsi} className={`cursor-pointer transition-colors hover:bg-sky-50/60 ${candidate.rank === selectedCandidate.rank ? "bg-sky-50/80" : ""}`} onClick={() => setSelectedRank(candidate.rank)} aria-selected={candidate.rank === selectedCandidate.rank}>
                    <TableCell className="pl-6 font-mono text-xs font-semibold text-muted-foreground">#{candidate.rank}</TableCell>
                    <TableCell><div className="font-semibold text-foreground">{candidate.vesselName}</div><div className="text-xs text-muted-foreground">{candidate.flag}</div></TableCell>
                    <TableCell className="font-mono text-xs">{candidate.mmsi}</TableCell>
                    <TableCell className="whitespace-nowrap text-xs">{candidate.vesselType}</TableCell>
                    <TableCell className="font-mono text-xs font-semibold">{candidate.distanceKm.toFixed(1)} km</TableCell>
                    <TableCell className="font-mono text-xs">{candidate.trajectoryMatch}%</TableCell>
                    <TableCell className="font-mono text-xs">{candidate.timeMatch}%</TableCell>
                    <TableCell className="font-mono text-xs font-semibold">{formatScore(candidate.confidence)}%</TableCell>
                    <TableCell className="pr-6"><Badge variant="outline" className={priorityClass(candidate.priority)}>{candidate.priority}</Badge></TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
          <div className="border-t border-border bg-slate-50/60 px-6 py-3 text-xs text-muted-foreground">Prototype / Simulated AIS Data · Click a row to review that candidate. Ranking is an investigative screening aid, not attribution.</div>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
        <Card>
          <CardHeader><CardTitle className="text-base">Candidate Evidence</CardTitle><CardDescription>{selectedCandidate.vesselName} · MMSI {selectedCandidate.mmsi} · {selectedCandidate.vesselType}</CardDescription></CardHeader>
          <CardContent className="space-y-3">
            {selectedCandidate.evidence.map((metric) => <EvidenceRow key={metric.label} metric={metric} />)}
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle className="text-base">Correlation Score</CardTitle><CardDescription>Prototype scoring model · selected candidate</CardDescription></CardHeader>
          <CardContent>
            <div className="flex items-end justify-between gap-4"><div><p className="text-4xl font-semibold tracking-tight text-foreground">{formatScore(selectedCandidate.confidence)}%</p><p className="mt-1 text-sm text-muted-foreground">Overall correlation</p></div><Badge variant="outline" className={priorityClass(selectedCandidate.priority)}>Priority {selectedCandidate.priority}</Badge></div>
            <Progress value={selectedCandidate.confidence} className="mt-5 h-2 [&_[data-slot=progress-indicator]]:bg-sky-600" />
            <Separator className="my-5" />
            <div className="space-y-3">
              {prototypeVesselMatch.scoringModel.map((item) => <div key={item.label} className="flex items-center justify-between text-sm"><span className="text-muted-foreground">{item.label}</span><span className="font-mono font-semibold text-foreground">{item.weight}%</span></div>)}
            </div>
            <div className="mt-5 flex items-start gap-2 rounded-md bg-slate-50 p-3 text-xs leading-5 text-muted-foreground"><Info className="mt-0.5 size-4 shrink-0 text-sky-600" />Weights are transparent prototype parameters. They have not been scientifically validated or calibrated against confirmed incidents.</div>
          </CardContent>
        </Card>
      </div>

      <Card ref={trackRef}>
        <CardHeader className="gap-2 sm:flex-row sm:items-start sm:justify-between"><div><CardTitle className="text-base">Historical AIS Track</CardTitle><CardDescription>Relative schematic for {selectedCandidate.vesselName}; positions are simulated and not a geographic map.</CardDescription></div><Badge variant="outline" className="w-fit border-slate-200">Simulated track</Badge></CardHeader>
        <CardContent><TrackDiagram candidate={selectedCandidate} /></CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle className="text-base">AIS Timeline</CardTitle><CardDescription>Mock historical positions leading into the reconstructed spill source region.</CardDescription></CardHeader>
        <CardContent>
          <div className="grid gap-3 md:grid-cols-3 xl:grid-cols-6">
            {selectedCandidate.timeline.map((event) => (
              <div key={`${event.time}-${event.location}`} className="relative rounded-lg border border-border bg-white p-3 before:absolute before:-top-2 before:left-5 before:size-3 before:rotate-45 before:border-l before:border-t before:border-border before:bg-white md:before:-left-2 md:before:top-5 md:before:border-b md:before:border-l md:before:border-t-0">
                <div className="flex items-center justify-between gap-2"><span className="font-mono text-sm font-semibold text-sky-700">{event.time}</span><span className="size-2 rounded-full bg-sky-500" /></div>
                <p className="mt-3 min-h-10 text-xs leading-5 text-muted-foreground">{event.location}</p>
                <Separator className="my-3" />
                <p className="text-xs text-muted-foreground"><span className="font-mono font-semibold text-foreground">{event.speedKnots.toFixed(1)} kn</span> · heading <span className="font-mono font-semibold text-foreground">{event.heading.toString().padStart(3, "0")}°</span></p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle className="text-base">Closest AIS Approach</CardTitle><CardDescription>Selected vessel approach metrics from simulated records.</CardDescription></CardHeader>
        <CardContent className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <SummaryMetric label="Minimum distance" value={`${selectedCandidate.distanceKm.toFixed(1)} km`} hint="To reconstructed origin" />
          <SummaryMetric label="Approach time" value={selectedCandidate.closestApproachTime} hint="Relative to spill detection" />
          <SummaryMetric label="Heading" value={`${selectedCandidate.closestApproachHeading.toString().padStart(3, "0")}°`} hint="At closest approach" />
          <SummaryMetric label="Speed" value={`${selectedCandidate.closestApproachSpeed.toFixed(1)} kn`} hint="At closest approach" />
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle className="text-base">Investigation Notes</CardTitle><CardDescription>Interpretation guardrails for this prototype review.</CardDescription></CardHeader>
        <CardContent className="grid gap-3 md:grid-cols-2">
          <div className="space-y-3 text-sm text-muted-foreground">
            <p className="flex gap-3"><CheckCircle2 className="mt-0.5 size-4 shrink-0 text-emerald-600" />{selectedCandidate.vesselName} is the highest-ranked candidate in the current simulated AIS set.</p>
            <p className="flex gap-3"><TrendingUp className="mt-0.5 size-4 shrink-0 text-sky-600" />The track shows temporal and spatial overlap with the reconstructed source corridor.</p>
            <p className="flex gap-3"><Compass className="mt-0.5 size-4 shrink-0 text-sky-600" />The closest approach occurred {selectedCandidate.closestApproachTime} at {selectedCandidate.closestApproachSpeed.toFixed(1)} kn.</p>
          </div>
          <div className="rounded-lg border border-amber-200 bg-amber-50/60 p-4 text-sm leading-6 text-amber-900"><strong>Uncertainty:</strong> AIS coverage gaps, drift-model error, environmental look-alikes, and synthetic records can change the ranking. Proximity and correlation are screening signals only and do not establish responsibility.</div>
        </CardContent>
      </Card>

      <div className="flex flex-wrap items-center gap-2 border-t border-border pt-5">
        <Button onClick={() => { setSelectedRank(1); showNotice("Prototype correlation reset to the highest-ranked candidate.") }}><Target className="size-4" />Run Correlation</Button>
        <Button variant="outline" onClick={() => { setSelectedRank(1); showNotice("Prototype evidence recalculated from the isolated sample dataset.") }}><RefreshCw className="size-4" />Recalculate</Button>
        <Button variant="outline" onClick={() => trackRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })}><Navigation className="size-4" />View Full Track</Button>
        <Button variant="outline" onClick={() => showNotice("Export is available when the vessel-correlation API is connected.")}><ArrowRight className="size-4" />Export Investigation</Button>
        <Button variant="ghost" onClick={() => showNotice("Return to Tab 2 to review the hindcast source corridor.")}>Back to Hindcast</Button>
      </div>
      {notice && <div className="rounded-md border border-sky-200 bg-sky-50 px-4 py-3 text-sm text-sky-800" role="status">{notice}</div>}
      <p className="text-xs text-muted-foreground">⚠️ Prototype Mode — Synthetic AIS Data. No live vessel service is connected. The displayed candidate is a prime suspect candidate for investigation, not a confirmed source vessel.</p>
    </section>
  )
}
