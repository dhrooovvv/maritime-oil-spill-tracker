import type { DetectionResult, DriftRequest, DriftResponse } from "@/types/api"

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000"
).replace(/\/$/, "")

export class ApiError extends Error {
  readonly status?: number

  constructor(message: string, status?: number) {
    super(message)
    this.name = "ApiError"
    this.status = status
  }
}

async function readError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string; message?: string }
    return body.detail ?? body.message ?? response.statusText
  } catch {
    return response.statusText || "The backend returned an unexpected error."
  }
}

export async function detectGeoTiff(file: File): Promise<DetectionResult> {
  const formData = new FormData()
  formData.append("file", file)

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/api/detect`, {
      method: "POST",
      body: formData,
    })
  } catch {
    throw new ApiError(
      "The Python API is not reachable. Start the backend before running detection."
    )
  }

  if (!response.ok) {
    throw new ApiError(await readError(response), response.status)
  }

  let payload: unknown
  try {
    payload = await response.json()
  } catch {
    throw new ApiError("The backend returned an invalid JSON response.", response.status)
  }

  if (
    !payload ||
    typeof payload !== "object" ||
    typeof (payload as { status?: unknown }).status !== "string"
  ) {
    throw new ApiError("The backend returned a malformed detection response.", response.status)
  }

  return payload as DetectionResult
}

export async function getDriftAnalysis(params: DriftRequest): Promise<DriftResponse> {

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/api/drift`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(params),
    })
  } catch {
    throw new ApiError(
      "Unable to connect to the OceanTrace Python analysis service."
    )
  }

  if (!response.ok) {
    throw new ApiError(await readError(response), response.status)
  }

  let payload: unknown
  try {
    payload = await response.json()
  } catch {
    throw new ApiError("The drift service returned invalid JSON.", response.status)
  }

  if (
    !payload ||
    typeof payload !== "object" ||
    !Array.isArray((payload as { trajectory?: unknown }).trajectory)
  ) {
    throw new ApiError("The drift service returned a malformed response.", response.status)
  }

  return payload as DriftResponse
}
