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
}

const config = useRuntimeConfig()
const apiBase = config.public.apiBase

const threshold = ref(8.0)
const imageFile = ref<File | null>(null)
const imagePreview = ref('')
const loading = ref(false)
const errorMessage = ref('')
const result = ref<PredictResponse | null>(null)

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
    <UContainer class="py-8 max-w-5xl">
      <div class="space-y-6">
        <UCard class="rounded-2xl border border-default/70 shadow-sm">
          <template #header>
            <div class="space-y-1">
              <h1 class="text-2xl font-semibold">EfficientAD Detection</h1>
            </div>
          </template>

          <div class="grid gap-4 md:grid-cols-1">
            <div class="space-y-2">
              <label class="block text-sm font-medium">Input image</label>
              <input
                class="block w-full rounded-md border border-default px-3 py-2 bg-default text-sm"
                type="file"
                accept="image/png,image/jpg,image/jpeg"
                @change="onImageChanged"
              >
            </div>
          </div>

          <div class="mt-4 flex items-center gap-3">
            <UButton :loading="loading" icon="i-lucide-play" @click="runPrediction">
              Run prediction
            </UButton>
            <UBadge variant="soft" color="neutral">Model: capsule (fixed)</UBadge>
            <UBadge variant="soft" color="neutral">Threshold: {{ threshold.toFixed(1) }} (fixed)</UBadge>
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
            <div v-if="result" class="space-y-2 text-sm">
              <p><span class="font-medium">Model:</span> {{ result.model }}</p>
              <p class="flex items-center gap-2">
                <span class="font-medium">Label:</span>
                <UBadge :color="labelColor" variant="soft">{{ result.label }}</UBadge>
              </p>
              <p class="flex items-center gap-2">
                <span class="font-medium">Defect type:</span>
                <UBadge :color="defectTypeColor" variant="soft">
                  {{ defectTypeText }}
                </UBadge>
              </p>
              <p><span class="font-medium">Defect confidence:</span> {{ result.defect_confidence ?? 'N/A' }}</p>
              <p><span class="font-medium">Score:</span> {{ result.score ?? 'N/A' }}</p>
              <p><span class="font-medium">Pred label:</span> {{ result.pred_label ?? 'N/A' }}</p>
              <p><span class="font-medium">Threshold:</span> {{ result.threshold.toFixed(2) }}</p>
            </div>
            <p v-else class="text-sm text-muted">Run a prediction to see results.</p>
          </UCard>
        </div>

        <div class="grid gap-6 md:grid-cols-3">
          <UCard class="rounded-2xl border border-default/70 shadow-sm">
            <template #header>
              <h2 class="font-semibold">Anomaly Map</h2>
            </template>
            <img v-if="heatmapSrc" :src="heatmapSrc" alt="Anomaly map" class="h-[320px] w-full rounded-md border border-default object-contain bg-elevated">
            <p v-else class="text-sm text-muted">No anomaly map available.</p>
          </UCard>

          <UCard class="rounded-2xl border border-default/70 shadow-sm">
            <template #header>
              <h2 class="font-semibold">Image + Anomaly Map</h2>
            </template>
            <img v-if="anomalyOverlaySrc" :src="anomalyOverlaySrc" alt="Image and anomaly map overlay" class="h-[320px] w-full rounded-md border border-default object-contain bg-elevated">
            <p v-else class="text-sm text-muted">No anomaly overlay available.</p>
          </UCard>

          <UCard class="rounded-2xl border border-default/70 shadow-sm">
            <template #header>
              <h2 class="font-semibold">Image + Pred Mask</h2>
            </template>
            <img v-if="predMaskOverlaySrc" :src="predMaskOverlaySrc" alt="Image and prediction mask overlay" class="h-[320px] w-full rounded-md border border-default object-contain bg-elevated">
            <p v-else class="text-sm text-muted">No prediction mask overlay available.</p>
          </UCard>
        </div>
      </div>
    </UContainer>
  </UApp>
</template>
