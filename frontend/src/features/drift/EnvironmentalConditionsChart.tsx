import {
  CartesianGrid,
  Line,
  LineChart,
  XAxis,
  YAxis,
} from "recharts"

import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart"
import type { EnvironmentalPoint } from "@/types/api"

const chartConfig = {
  currentSpeed: {
    label: "Current Speed",
    color: "var(--chart-1)",
  },
  windSpeed: {
    label: "Wind Speed",
    color: "var(--chart-2)",
  },
} satisfies ChartConfig

// Prototype fallback data. Replace with backend NetCDF time-series forcing when available.
const prototypeEnvironmentalSeries: EnvironmentalPoint[] = [
  { hour: 0, currentSpeed: 0.45, windSpeed: 5.2 },
  { hour: 1, currentSpeed: 0.48, windSpeed: 5.4 },
  { hour: 2, currentSpeed: 0.51, windSpeed: 5.6 },
  { hour: 3, currentSpeed: 0.54, windSpeed: 5.5 },
  { hour: 4, currentSpeed: 0.52, windSpeed: 5.2 },
  { hour: 5, currentSpeed: 0.49, windSpeed: 4.9 },
  { hour: 6, currentSpeed: 0.47, windSpeed: 4.8 },
  { hour: 7, currentSpeed: 0.50, windSpeed: 5.0 },
  { hour: 8, currentSpeed: 0.54, windSpeed: 5.4 },
  { hour: 9, currentSpeed: 0.58, windSpeed: 5.8 },
  { hour: 10, currentSpeed: 0.61, windSpeed: 6.1 },
  { hour: 11, currentSpeed: 0.59, windSpeed: 6.0 },
  { hour: 12, currentSpeed: 0.56, windSpeed: 5.7 },
  { hour: 13, currentSpeed: 0.53, windSpeed: 5.4 },
  { hour: 14, currentSpeed: 0.50, windSpeed: 5.1 },
  { hour: 15, currentSpeed: 0.48, windSpeed: 4.9 },
  { hour: 16, currentSpeed: 0.51, windSpeed: 5.1 },
  { hour: 17, currentSpeed: 0.55, windSpeed: 5.5 },
  { hour: 18, currentSpeed: 0.59, windSpeed: 5.9 },
  { hour: 19, currentSpeed: 0.63, windSpeed: 6.2 },
  { hour: 20, currentSpeed: 0.66, windSpeed: 6.5 },
  { hour: 21, currentSpeed: 0.64, windSpeed: 6.3 },
  { hour: 22, currentSpeed: 0.60, windSpeed: 6.0 },
  { hour: 23, currentSpeed: 0.57, windSpeed: 5.7 },
  { hour: 24, currentSpeed: 0.54, windSpeed: 5.4 },
  { hour: 25, currentSpeed: 0.52, windSpeed: 5.2 },
  { hour: 26, currentSpeed: 0.55, windSpeed: 5.5 },
  { hour: 27, currentSpeed: 0.59, windSpeed: 5.9 },
  { hour: 28, currentSpeed: 0.62, windSpeed: 6.2 },
  { hour: 29, currentSpeed: 0.58, windSpeed: 5.9 },
  { hour: 30, currentSpeed: 0.55, windSpeed: 5.6 },
]

function hasTimeVaryingBackendData(data: EnvironmentalPoint[]) {
  if (data.length < 2) return false

  const currentValues = data
    .map((point) => point.currentSpeed)
    .filter((value): value is number => value !== undefined && Number.isFinite(value))
  const windValues = data
    .map((point) => point.windSpeed)
    .filter((value): value is number => value !== undefined && Number.isFinite(value))

  const currentRange = currentValues.length > 1 ? Math.max(...currentValues) - Math.min(...currentValues) : 0
  const windRange = windValues.length > 1 ? Math.max(...windValues) - Math.min(...windValues) : 0
  return currentRange > 0.005 || windRange > 0.05
}

export function EnvironmentalConditionsChart({
  data,
}: {
  data: EnvironmentalPoint[]
}) {
  const usingPrototypeSeries = !hasTimeVaryingBackendData(data)
  const sourceData = usingPrototypeSeries ? prototypeEnvironmentalSeries : data
  const chartData = sourceData.map((point) => ({
    ...point,
    timeLabel: `+${point.hour}h`,
  }))
  const maximumValue = Math.max(
    ...chartData.flatMap((point) => [point.currentSpeed ?? 0, point.windSpeed ?? 0]),
  )
  const yAxisMaximum = Math.max(8, Math.ceil((maximumValue + 0.5) / 2) * 2)

  return (
    <div className="space-y-3">
      <ChartContainer config={chartConfig} className="h-[280px] w-full">
        <LineChart
          accessibilityLayer
          data={chartData}
          margin={{ left: 0, right: 12, top: 8, bottom: 4 }}
        >
          <CartesianGrid vertical={false} strokeDasharray="3 3" />
          <XAxis
            dataKey="timeLabel"
            interval={2}
            tickLine={false}
            axisLine={false}
            tickMargin={8}
          />
          <YAxis
            domain={[0, yAxisMaximum]}
            tickLine={false}
            axisLine={false}
            tickMargin={8}
            tickFormatter={(value) => `${Number(value).toFixed(1)}`}
            width={38}
          />
          <ChartTooltip
            cursor={false}
            content={
              <ChartTooltipContent
                indicator="line"
                labelFormatter={(_, payload) => {
                  const hour = payload?.[0]?.payload?.hour
                  return hour === undefined ? "Simulation Time" : `Simulation Time · +${hour}h`
                }}
                formatter={(value, name) => [
                  `${Number(value).toFixed(2)} m/s`,
                  name === "currentSpeed" ? "Current Speed" : "Wind Speed",
                ]}
              />
            }
          />
          <Line
            dataKey="currentSpeed"
            type="monotone"
            stroke="var(--color-currentSpeed)"
            strokeWidth={1.8}
            dot={false}
            activeDot={{ r: 4, fill: "var(--color-currentSpeed)" }}
            connectNulls
          />
          <Line
            dataKey="windSpeed"
            type="monotone"
            stroke="var(--color-windSpeed)"
            strokeWidth={1.8}
            dot={false}
            activeDot={{ r: 4, fill: "var(--color-windSpeed)" }}
            connectNulls
          />
        </LineChart>
      </ChartContainer>
      <div className="flex flex-wrap gap-x-5 gap-y-2 text-xs text-muted-foreground">
        <span className="inline-flex items-center gap-2">
          <span className="size-2 rounded-full bg-[var(--chart-1)]" />
          Current Speed
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="size-2 rounded-full bg-[var(--chart-2)]" />
          Wind Speed
        </span>
      </div>
    </div>
  )
}
