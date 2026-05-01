<script setup lang="ts">
type PredictResponse = {
  model: string
  threshold: number
  label: string
  defect_type?: string
  defect_confidence?: number | null
  score: number | null
  pred_label: number | null
  anomaly_map_base64: string | null
  anomaly_overlay_base64?: string | null
  pred_mask_overlay_base64?: string | null
  // Classical CV
  defect_count: number | null
  largest_defect_area: number | null
  classical_overlay_base64?: string | null
}

const config = useRuntimeConfig()
const apiBase = config.public.apiBase

const colorMode = useColorMode()
const isDark = computed(() => colorMode.value === 'dark')
function toggleTheme() {
  colorMode.preference = isDark.value ? 'light' : 'dark'
}

const threshold = ref(0.5)
const imageFile = ref<File | null>(null)
const imagePreview = ref('')
const loading = ref(false)
const errorMessage = ref('')
const result = ref<PredictResponse | null>(null)

const zoomImageSrc = ref('')
const zoomOpen = ref(false)

function openZoom(src: string) {
  if (!src) return
  zoomImageSrc.value = src
  zoomOpen.value = true
}

function closeZoom() {
  zoomOpen.value = false
  zoomImageSrc.value = ''
}

const heatmapSrc = computed(() => {
  if (!result.value?.anomaly_map_base64) return ''
  return `data:image/png;base64,${result.value.anomaly_map_base64}`
})

const anomalyOverlaySrc = computed(() => {
  if (!result.value?.anomaly_overlay_base64) return ''
  return `data:image/png;base64,${result.value.anomaly_overlay_base64}`
})

const predMaskOverlaySrc = computed(() => {
  if (!result.value?.pred_mask_overlay_base64) return ''
  return `data:image/png;base64,${result.value.pred_mask_overlay_base64}`
})

const classicalOverlaySrc = computed(() => {
  if (!result.value?.classical_overlay_base64) return ''
  return `data:image/png;base64,${result.value.classical_overlay_base64}`
})

const labelColor = computed(() => {
  if (result.value?.label === 'DEFECT') return 'error'
  if (result.value?.label === 'GOOD') return 'success'
  return 'neutral'
})

const defectTypeText = computed(() => {
  const defect = result.value?.defect_type
  if (!defect) return 'N/A'
  return defect.replaceAll('_', ' ').toUpperCase()
})

const defectTypeColor = computed(() => {
  const defect = result.value?.defect_type
  if (!defect) return 'neutral'
  if (defect === 'good') return 'success'
  if (defect === 'crack') return 'error'
  if (defect === 'faulty_imprint') return 'warning'
  if (defect === 'poke') return 'info'
  if (defect === 'scratch') return 'warning'
  if (defect === 'squeeze') return 'error'
  return 'neutral'
})

const scorePercent = computed(() => {
  const s = result.value?.score
  if (s == null) return 0
  return Math.min(Math.max(s, 0), 1) * 100
})

const gaugePath = computed(() => {
  const r = 80
  const cx = 100
  const cy = 100
  const startAngle = 135
  const endAngle = 45
  const rad = (deg: number) => (deg * Math.PI) / 180
  const x1 = cx + r * Math.cos(rad(startAngle))
  const y1 = cy + r * Math.sin(rad(startAngle))
  const x2 = cx + r * Math.cos(rad(endAngle))
  const y2 = cy + r * Math.sin(rad(endAngle))
  return `M ${x1} ${y1} A ${r} ${r} 0 1 1 ${x2} ${y2}`
})

const gaugeNeedle = computed(() => {
  const r = 70
  const cx = 100
  const cy = 100
  const angle = 135 + (scorePercent.value / 100) * 270
  const rad = (deg: number) => (deg * Math.PI) / 180
  return {
    x: cx + r * Math.cos(rad(angle)),
    y: cy + r * Math.sin(rad(angle)),
  }
})

const barWidth = computed(() => {
  const s = result.value?.score
  if (s == null) return 0
  return Math.min(Math.max(s, 0), 1) * 100
})

const thresholdBarWidth = computed(() => {
  const t = result.value?.threshold
  if (t == null) return 0
  return Math.min(Math.max(t, 0), 1) * 100
})

function onImageChanged(event: Event) {
  const input = event.target as HTMLInputElement
  const selected = input.files?.[0] ?? null
  imageFile.value = selected
  result.value = null

  if (imagePreview.value) {
    URL.revokeObjectURL(imagePreview.value)
    imagePreview.value = ''
  }

  if (selected) {
    imagePreview.value = URL.createObjectURL(selected)
  }
}

async function runPrediction() {
  errorMessage.value = ''
  result.value = null

  if (!imageFile.value) {
    errorMessage.value = 'Please upload an image.'
    return
  }

  loading.value = true

  try {
    const formData = new FormData()
    formData.append('model_name', 'capsule')
    formData.append('threshold', String(threshold.value))
    formData.append('image', imageFile.value)

    const data = await $fetch<PredictResponse>(`${apiBase}/predict`, {
      method: 'POST',
      body: formData
    })
    result.value = data
  } catch (error: any) {
    errorMessage.value = error?.data?.detail || error?.message || 'Prediction request failed.'
  } finally {
    loading.value = false
  }
}

</script>

<template>
  <UApp>
    <NuxtRouteAnnouncer />
    <UContainer class="py-8 max-w-6xl">
      <div class="space-y-6">
        <UCard class="rounded-2xl border border-default/70 shadow-sm">
          <template #header>
            <div class="flex items-center justify-between">
              <h1 class="text-2xl font-semibold">EfficientAD Detection</h1>
              <UButton
                variant="ghost"
                color="neutral"
                :icon="isDark ? 'i-lucide-sun' : 'i-lucide-moon'"
                @click="toggleTheme"
              >
                {{ isDark ? 'Light' : 'Dark' }}
              </UButton>
            </div>
          </template>

          <div class="grid gap-4 md:grid-cols-2">
            <div class="space-y-2">
              <label class="block text-sm font-medium">Input image</label>
              <input
                class="block w-full rounded-md border border-default px-3 py-2 bg-default text-sm"
                type="file"
                accept="image/png,image/jpg,image/jpeg"
                @change="onImageChanged"
              >
            </div>
            <div class="space-y-2">
              <label class="block text-sm font-medium">Threshold (0–1)</label>
              <input
                v-model.number="threshold"
                class="block w-full rounded-md border border-default px-3 py-2 bg-default text-sm"
                type="number"
                step="0.01"
                min="0"
                max="1"
              >
            </div>
          </div>

          <div class="mt-4 flex items-center gap-3">
            <UButton :loading="loading" icon="i-lucide-play" @click="runPrediction">
              Run prediction
            </UButton>
            <UBadge variant="soft" color="neutral">Model: capsule (fixed)</UBadge>
            <UBadge variant="soft" color="neutral">Threshold: {{ threshold.toFixed(2) }}</UBadge>
            <UBadge variant="soft" color="neutral">API: {{ apiBase }}</UBadge>
          </div>

          <UAlert
            v-if="errorMessage"
            class="mt-4"
            color="error"
            variant="soft"
            title="Request failed"
            :description="errorMessage"
          />
        </UCard>

        <div class="grid gap-6 md:grid-cols-2">
          <UCard class="rounded-2xl border border-default/70 shadow-sm">
            <template #header>
              <h2 class="font-semibold">Input Preview</h2>
            </template>
            <img v-if="imagePreview" :src="imagePreview" alt="Input preview" class="h-[320px] w-full rounded-md border border-default object-contain bg-elevated">
            <p v-else class="text-sm text-muted">No image selected yet.</p>
          </UCard>

          <UCard class="rounded-2xl border border-default/70 shadow-sm">
            <template #header>
              <h2 class="font-semibold">Prediction Summary</h2>
            </template>
            <div v-if="result" class="space-y-4 flex flex-col items-center justify-center h-full">
              <!-- Gauge Chart -->
              <div class="flex justify-center">
                <svg viewBox="0 0 200 120" class="w-48 h-28">
                  <defs>
                    <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stop-color="#22c55e" />
                      <stop offset="50%" stop-color="#eab308" />
                      <stop offset="100%" stop-color="#ef4444" />
                    </linearGradient>
                  </defs>
                  <path :d="gaugePath" fill="none" stroke="url(#gaugeGrad)" stroke-width="12" stroke-linecap="round" />
                  <circle cx="100" cy="100" r="4" fill="#374151" />
                  <line x1="100" y1="100" :x2="gaugeNeedle.x" :y2="gaugeNeedle.y" stroke="#374151" stroke-width="3" stroke-linecap="round" />
                  <text x="100" y="115" text-anchor="middle" font-size="14" font-weight="bold" fill="#374151">{{ result.score?.toFixed(3) ?? 'N/A' }}</text>
                </svg>
              </div>

              <!-- Bar Chart -->
              <div class="space-y-2">
                <div class="flex items-center gap-2 text-xs">
                  <span class="w-16 font-medium">Score</span>
                  <div class="flex-1 h-4 bg-gray-100 rounded-full overflow-hidden">
                    <div class="h-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 transition-all duration-500" :style="{ width: barWidth + '%' }"></div>
                  </div>
                  <span class="w-12 text-right">{{ result.score?.toFixed(3) }}/{{ threshold.toFixed(2) }}</span>
                </div>
              </div>

              <!-- Info -->
              <div class="space-y-1 text-sm pt-2 border-t border-default">
                <p class="flex items-center gap-2">
                  <span class="font-medium">Label:</span>
                  <UBadge :color="labelColor" variant="soft">{{ result.label }}</UBadge>
                </p>
                <p class="flex items-center gap-2">
                  <span class="font-medium">Defect type:</span>
                  <UBadge :color="defectTypeColor" variant="soft">{{ defectTypeText }}</UBadge>
                </p>
                <p><span class="font-medium">Defect count (CV):</span> {{ result.defect_count ?? 'N/A' }}</p>
                <p><span class="font-medium">Largest area (CV):</span> {{ result.largest_defect_area ? result.largest_defect_area.toFixed(0) + ' px²' : 'N/A' }}</p>
              </div>
            </div>
            <p v-else class="text-sm text-muted">Run a prediction to see results.</p>
          </UCard>
        </div>

        <div class="grid gap-6 md:grid-cols-2">
          <UCard class="rounded-2xl border border-default/70 shadow-sm">
            <template #header>
              <h2 class="font-semibold">Anomaly Map</h2>
            </template>
            <img v-if="heatmapSrc" :src="heatmapSrc" alt="Anomaly map" class="h-[240px] w-full rounded-md border border-default object-contain bg-elevated cursor-pointer hover:opacity-90 transition-opacity" @click="openZoom(heatmapSrc)">
            <p v-else class="text-sm text-muted">No anomaly map available.</p>
          </UCard>

          <UCard class="rounded-2xl border border-default/70 shadow-sm">
            <template #header>
              <h2 class="font-semibold">Image + Anomaly Map</h2>
            </template>
            <img v-if="anomalyOverlaySrc" :src="anomalyOverlaySrc" alt="Image and anomaly map overlay" class="h-[240px] w-full rounded-md border border-default object-contain bg-elevated cursor-pointer hover:opacity-90 transition-opacity" @click="openZoom(anomalyOverlaySrc)">
            <p v-else class="text-sm text-muted">No anomaly overlay available.</p>
          </UCard>
        </div>

        <div class="grid gap-6 md:grid-cols-2">
          <UCard class="rounded-2xl border border-default/70 shadow-sm">
            <template #header>
              <h2 class="font-semibold">Image + Pred Mask</h2>
            </template>
            <img v-if="predMaskOverlaySrc" :src="predMaskOverlaySrc" alt="Image and prediction mask overlay" class="h-[240px] w-full rounded-md border border-default object-contain bg-elevated cursor-pointer hover:opacity-90 transition-opacity" @click="openZoom(predMaskOverlaySrc)">
            <p v-else class="text-sm text-muted">No prediction mask overlay available.</p>
          </UCard>

          <UCard class="rounded-2xl border border-default/70 shadow-sm">
            <template #header>
              <h2 class="font-semibold">Image + Contours (CV)</h2>
            </template>
            <img v-if="classicalOverlaySrc" :src="classicalOverlaySrc" alt="Image and contour overlay from classical CV" class="h-[240px] w-full rounded-md border border-default object-contain bg-elevated cursor-pointer hover:opacity-90 transition-opacity" @click="openZoom(classicalOverlaySrc)">
            <p v-else class="text-sm text-muted">No classical overlay available.</p>
          </UCard>
        </div>

        <!-- Zoom Modal -->
        <div v-if="zoomOpen" class="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4" @click="closeZoom">
          <div class="relative max-h-[90vh] max-w-[90vw]">
            <img :src="zoomImageSrc" alt="Zoomed" class="max-h-[90vh] max-w-[90vw] rounded-lg border border-white/20 object-contain">
            <button class="absolute -top-3 -right-3 bg-white text-black rounded-full w-8 h-8 flex items-center justify-center font-bold shadow-lg hover:bg-gray-200" @click.stop="closeZoom">✕</button>
          </div>
        </div>
      </div>
    </UContainer>
  </UApp>
</template>
