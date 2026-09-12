import { Upload, Waves } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { TooltipProvider } from "@/components/ui/tooltip"
import { DriftTab } from "@/features/drift/DriftTab"
import { GeoTiffTab } from "@/features/geotiff/GeoTiffTab"
import { SuspectsTab } from "@/features/suspects/SuspectsTab"
import { useState } from "react"
import type { DetectionResult, DriftOrigin } from "@/types/api"

type PipelineTab = "detection" | "drift" | "suspects"

const pipelineSteps: Array<{ value: PipelineTab; number: string; title: string; description: string }> = [
  { value: "detection", number: "01", title: "Satellite GeoTIFF Analysis", description: "Segmentation & metrics" },
  { value: "drift", number: "02", title: "Ocean Drift Simulation", description: "Hydrodynamic particle forecast" },
  { value: "suspects", number: "03", title: "Vessel Suspect Match", description: "AIS historical trajectory correlation" },
]

function App() {
  const [activeTab, setActiveTab] = useState<PipelineTab>("detection")
  const [resetToken, setResetToken] = useState(0)
  const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null)

  const spatial = detectionResult?.spatial
  const spillOrigin: DriftOrigin | undefined = (
    spatial?.x !== undefined &&
    spatial.y !== undefined &&
    spatial.crs
  )
    ? {
        x: spatial.x,
        y: spatial.y,
        crs: spatial.crs,
        longitude: spatial.longitude,
        latitude: spatial.latitude,
      }
    : undefined

  function startNewScene() {
    setResetToken((value) => value + 1)
    setDetectionResult(null)
    setActiveTab("detection")
  }

  return (
    <TooltipProvider>
      <div className="min-h-screen bg-background text-foreground">
        <main className="mx-auto flex w-full max-w-[1440px] flex-col gap-8 px-5 pb-8 sm:px-8 lg:px-12">
          <header className="-mx-5 flex min-h-16 items-center justify-between gap-4 border-b border-border bg-white px-5 py-3 sm:-mx-8 sm:px-8 lg:-mx-12 lg:px-12">
            <div className="flex min-w-0 items-center gap-3">
              <div className="flex size-9 shrink-0 items-center justify-center rounded-lg border border-sky-200 bg-sky-50 text-sky-600">
                <Waves className="size-5" />
              </div>
              <div className="flex min-w-0 items-center gap-3">
                <span className="truncate text-base font-semibold tracking-tight text-foreground">OceanTrace</span>
                <Badge variant="outline" className="hidden border-sky-200 bg-sky-50 text-sky-700 sm:inline-flex">
                  SAR Analysis Pipeline
                </Badge>
                <span className="hidden items-center gap-1.5 text-xs font-medium text-emerald-600 md:inline-flex">
                  <span className="size-2 rounded-full bg-emerald-500" />
                  Operational
                </span>
              </div>
            </div>
            <div className="flex shrink-0 items-center gap-2">
              <Button variant="outline" size="sm" onClick={startNewScene}>
                <Upload className="size-3.5" />
                <span className="hidden sm:inline">Upload New Scene</span>
                <span className="sm:hidden">New Scene</span>
              </Button>
              <Badge variant="secondary" className="hidden font-mono text-[0.68rem] md:inline-flex">
                Session pending
              </Badge>
            </div>
          </header>

          <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as PipelineTab)} className="w-full">
            <TabsList variant="line" className="!h-auto grid w-full grid-cols-1 items-stretch rounded-lg border border-border bg-white p-0 sm:grid-cols-3">
              {pipelineSteps.map((step) => {
                const isActive = activeTab === step.value
                return (
                  <TabsTrigger
                    key={step.value}
                    value={step.value}
                    className="flex min-h-16 items-center justify-start gap-3 rounded-none border-b border-border px-4 py-3 text-left after:bg-sky-600 first:rounded-t-lg last:rounded-b-lg sm:min-h-20 sm:border-b-0 sm:border-r sm:last:border-r-0 sm:first:rounded-l-lg sm:first:rounded-tr-none sm:last:rounded-r-lg sm:last:rounded-bl-none"
                  >
                    <span className={`flex size-7 shrink-0 items-center justify-center rounded-full border font-mono text-xs ${isActive ? "border-sky-600 bg-sky-600 text-white" : "border-border bg-slate-50 text-muted-foreground"}`}>
                      {step.number}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-sm font-semibold text-foreground">{step.title}</span>
                      <span className="block truncate text-xs font-normal text-muted-foreground">{step.description}</span>
                    </span>
                    <Badge variant={isActive ? "secondary" : "outline"} className="hidden shrink-0 text-[0.65rem] font-normal lg:inline-flex">
                      {isActive ? "Active Step" : "Pending"}
                    </Badge>
                  </TabsTrigger>
                )
              })}
            </TabsList>

            <TabsContent value="detection" className="pt-8">
              <GeoTiffTab key={resetToken} onContinue={() => setActiveTab("drift")} onDetectionResult={setDetectionResult} />
            </TabsContent>
            <TabsContent value="drift" className="pt-8">
              <DriftTab origin={spillOrigin} />
            </TabsContent>
            <TabsContent value="suspects" className="pt-8">
              <SuspectsTab />
            </TabsContent>
          </Tabs>
        </main>
      </div>
    </TooltipProvider>
  )
}

export default App
