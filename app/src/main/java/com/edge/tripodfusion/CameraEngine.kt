package com.edge.tripodfusion

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.SurfaceTexture
import android.hardware.camera2.*
import android.hardware.camera2.params.BlackLevelPattern
import android.hardware.camera2.params.ColorSpaceTransform
import android.hardware.camera2.params.MeteringRectangle
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.RggbChannelVector
import android.hardware.camera2.params.SessionConfiguration
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Range
import android.util.Rational
import android.util.Size
import android.view.Surface
import android.view.TextureView
import kotlinx.coroutines.CompletableDeferred
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

data class LensInfo(
    val openId: String,
    val physicalId: String?,
    val label: String,
    val focalMm: Float,
    val rawSize: Size,
    val previewSize: Size,
    val cfa: Int,
    val whiteLevel: Float,
    val staticBlackLevel: FloatArray,
    val exposureRange: Range<Long>,
    val isoRange: Range<Int>,
    val sensorOrientation: Int,
    val hasOis: Boolean,
    val activeArray: Rect
)

data class CapturedRawFrame(
    val buffer: ByteBuffer,
    val exposureNs: Long,
    val iso: Int,
    val frameType: Int,
    val blackLevels: FloatArray,
    val wbGains: FloatArray,
    val ccm: FloatArray
)

class CameraEngine(private val context: Context) {

    private val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private val bgThread = HandlerThread("Camera2Bg").apply { start() }
    private val bgHandler = Handler(bgThread.looper)
    private val bgExecutor = Executors.newSingleThreadExecutor()

    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var previewSurface: Surface? = null
    private var rawReader: ImageReader? = null

    var currentLens: LensInfo? = null
        private set
    var probeSummary: String = ""
        private set

    private var currentZoomCrop = 1.0f
    private var oisDisabledOnTripod = true

    @Volatile private var lastAeExposureNs: Long = 16_000_000L
    @Volatile private var lastAeIso: Int = 200
    @Volatile private var lastWbGains: FloatArray = floatArrayOf(1.8f, 1.0f, 1.6f)
    @Volatile private var lastCcm: FloatArray = floatArrayOf(
        1f, 0f, 0f,
        0f, 1f, 0f,
        0f, 0f, 1f
    )

    private class PendingPair(
        var buffer: ByteBuffer? = null,
        var result: TotalCaptureResult? = null,
        var frameType: Int = 0
    )

    private val pendingMap = ConcurrentHashMap<Long, PendingPair>()
    private var burstExpectedCount = 0
    private var burstCapturedList = mutableListOf<CapturedRawFrame>()
    private var burstDeferred: CompletableDeferred<List<CapturedRawFrame>>? = null
    private val burstLock = Any()

    fun discoverRawLenses(): List<LensInfo> {
        val rawCandidates = mutableListOf<Pair<String, String?>>()
        val seenIds = HashSet<String>()
        val probeLog = StringBuilder()

        val idsToProbe = LinkedHashSet<String>()
        idsToProbe.addAll(cameraManager.cameraIdList)
        for (n in 0..63) idsToProbe.add(n.toString())
        for (n in 100..105) idsToProbe.add(n.toString())

        for (id in idsToProbe) {
            val chars = try {
                cameraManager.getCameraCharacteristics(id)
            } catch (_: Throwable) {
                continue
            }
            if (chars.get(CameraCharacteristics.LENS_FACING) != CameraCharacteristics.LENS_FACING_BACK) continue

            val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            val rawSizes = map?.getOutputSizes(ImageFormat.RAW_SENSOR)
            val foc = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull() ?: 0f
            val hasRaw = rawSizes != null && rawSizes.isNotEmpty()
            probeLog.append("id$id:${String.format("%.1f", foc)}mm${if (hasRaw) "*" else ""} ")

            if (hasRaw && seenIds.add(id)) {
                rawCandidates.add(id to null)
            }

            for (physId in chars.physicalCameraIds) {
                if (!seenIds.add(physId)) continue
                val pChars = try {
                    cameraManager.getCameraCharacteristics(physId)
                } catch (_: Throwable) {
                    continue
                }
                val pMap = pChars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                val pRaw = pMap?.getOutputSizes(ImageFormat.RAW_SENSOR)
                if (pRaw != null && pRaw.isNotEmpty()) {
                    rawCandidates.add(id to physId)
                }
            }
        }
        probeSummary = probeLog.toString().trim()

        val parsed = mutableListOf<LensInfo>()
        for ((openId, physId) in rawCandidates) {
            val targetId = physId ?: openId
            val chars = cameraManager.getCameraCharacteristics(targetId)
            val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: continue
            val rawSizes = map.getOutputSizes(ImageFormat.RAW_SENSOR) ?: continue
            if (rawSizes.isEmpty()) continue
            val rawSize = rawSizes.maxByOrNull { it.width * it.height } ?: continue

            val previewSizes = map.getOutputSizes(SurfaceTexture::class.java) ?: continue
            val previewSize = choosePreviewSize(previewSizes, rawSize)

            val focals = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
            val focalMm = focals?.firstOrNull() ?: 4.7f
            val physSize = chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
            val eqFocal = if (physSize != null && physSize.width > 0f) {
                focalMm * 36.0f / physSize.width
            } else {
                focalMm * 5.0f
            }

            val cfa = chars.get(CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT) ?: 0
            val whiteLevel = (chars.get(CameraCharacteristics.SENSOR_INFO_WHITE_LEVEL) ?: 1023).toFloat()
            val blPattern: BlackLevelPattern? = chars.get(CameraCharacteristics.SENSOR_BLACK_LEVEL_PATTERN)
            val blInt = IntArray(4)
            blPattern?.copyTo(blInt, 0)
            val blFloat = floatArrayOf(
                blInt[0].toFloat(), blInt[1].toFloat(), blInt[2].toFloat(), blInt[3].toFloat()
            )
            val expRange = chars.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
                ?: Range(100_000L, 400_000_000L)
            val isoRange = chars.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
                ?: Range(100, 3200)
            val orient = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
            val oisModes = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION)
            val hasOis = oisModes?.contains(CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_ON) == true
            val activeArray = chars.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
                ?: Rect(0, 0, rawSize.width, rawSize.height)

            parsed.add(
                LensInfo(
                    openId = openId,
                    physicalId = physId,
                    label = String.format("%.1fmm", eqFocal),
                    focalMm = eqFocal,
                    rawSize = rawSize,
                    previewSize = previewSize,
                    cfa = cfa,
                    whiteLevel = whiteLevel,
                    staticBlackLevel = blFloat,
                    exposureRange = expRange,
                    isoRange = isoRange,
                    sensorOrientation = orient,
                    hasOis = hasOis,
                    activeArray = activeArray
                )
            )
        }

        val deduped = parsed.distinctBy { "${it.rawSize.width}x${it.rawSize.height}_${it.focalMm.roundToInt()}" }
            .sortedBy { it.focalMm }
        val mainFocal = deduped.minByOrNull { abs(it.focalMm - 24f) }?.focalMm ?: 24f

        return deduped.map { lens ->
            val ratio = lens.focalMm / mainFocal
            val cleanLabel = when {
                ratio < 0.75f -> String.format("%.1fx", ratio)
                ratio <= 1.25f -> "1x"
                else -> String.format("%.1fx", ratio)
            }
            lens.copy(label = cleanLabel)
        }
    }

    private fun choosePreviewSize(choices: Array<Size>, rawSize: Size): Size {
        val targetRatio = rawSize.width.toFloat() / rawSize.height.toFloat()
        return choices.filter { it.width <= 1920 && it.height <= 1440 }
            .minByOrNull { abs(it.width.toFloat() / it.height.toFloat() - targetRatio) * 10f - it.width / 1000f }
            ?: choices.first()
    }

    @SuppressLint("MissingPermission")
    fun openLens(
        lens: LensInfo,
        textureView: TextureView,
        onReady: () -> Unit,
        onError: (String) -> Unit
    ) {
        close()
        currentLens = lens
        val st = textureView.surfaceTexture ?: run {
            onError("Preview texture not ready")
            return
        }
        st.setDefaultBufferSize(lens.previewSize.width, lens.previewSize.height)
        val pSurface = Surface(st)
        previewSurface = pSurface

        val reader = ImageReader.newInstance(
            lens.rawSize.width,
            lens.rawSize.height,
            ImageFormat.RAW_SENSOR,
            4
        )
        reader.setOnImageAvailableListener({ r ->
            val img = r.acquireNextImage() ?: return@setOnImageAvailableListener
            try {
                val ts = img.timestamp
                val plane = img.planes[0]
                val srcBuf = plane.buffer
                val rowStride = plane.rowStride
                val w = lens.rawSize.width
                val h = lens.rawSize.height
                val tightlyPacked = ByteBuffer.allocateDirect(w * h * 2).order(ByteOrder.nativeOrder())

                if (rowStride == w * 2) {
                    srcBuf.rewind()
                    tightlyPacked.put(srcBuf)
                } else {
                    val rowBytes = ByteArray(w * 2)
                    for (row in 0 until h) {
                        srcBuf.position(row * rowStride)
                        srcBuf.get(rowBytes, 0, w * 2)
                        tightlyPacked.put(rowBytes)
                    }
                }
                tightlyPacked.rewind()
                completePairImage(ts, tightlyPacked)
            } catch (_: Throwable) {
            } finally {
                img.close()
            }
        }, bgHandler)
        rawReader = reader

        cameraManager.openCamera(lens.openId, object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                cameraDevice = camera
                createSession(camera, lens, pSurface, reader.surface, onReady, onError)
            }

            override fun onDisconnected(camera: CameraDevice) {
                camera.close()
                cameraDevice = null
            }

            override fun onError(camera: CameraDevice, error: Int) {
                camera.close()
                cameraDevice = null
                onError("Camera2 open error: $error")
            }
        }, bgHandler)
    }

    private fun createSession(
        camera: CameraDevice,
        lens: LensInfo,
        pSurface: Surface,
        rSurface: Surface,
        onReady: () -> Unit,
        onError: (String) -> Unit
    ) {
        val outPreview = OutputConfiguration(pSurface).apply {
            if (lens.physicalId != null) setPhysicalCameraId(lens.physicalId)
        }
        val outRaw = OutputConfiguration(rSurface).apply {
            if (lens.physicalId != null) setPhysicalCameraId(lens.physicalId)
        }

        val sessionConfig = SessionConfiguration(
            SessionConfiguration.SESSION_REGULAR,
            listOf(outPreview, outRaw),
            bgExecutor,
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    captureSession = session
                    startPreviewRepeating()
                    onReady()
                }

                override fun onConfigureFailed(session: CameraCaptureSession) {
                    onError("Session config failed on ${lens.label} (id=${lens.openId})")
                }
            }
        )
        camera.createCaptureSession(sessionConfig)
    }

    private fun buildCropRect(lens: LensInfo, zoom: Float): Rect {
        val arr = lens.activeArray
        val z = max(1.0f, zoom)
        val cw = (arr.width() / z).roundToInt()
        val ch = (arr.height() / z).roundToInt()
        val left = arr.left + (arr.width() - cw) / 2
        val top = arr.top + (arr.height() - ch) / 2
        return Rect(left, top, left + cw, top + ch)
    }

    fun setZoomCrop(zoom: Float) {
        currentZoomCrop = zoom
        startPreviewRepeating()
    }

    fun setOisDisabledOnTripod(disabled: Boolean) {
        oisDisabledOnTripod = disabled
        startPreviewRepeating()
    }

    private fun startPreviewRepeating() {
        val cam = cameraDevice ?: return
        val session = captureSession ?: return
        val surface = previewSurface ?: return
        val lens = currentLens ?: return
        try {
            val req = cam.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                addTarget(surface)
                set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)
                set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)
                set(CaptureRequest.SCALER_CROP_REGION, buildCropRect(lens, currentZoomCrop))
                if (lens.hasOis) {
                    set(
                        CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                        if (oisDisabledOnTripod) CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF
                        else CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_ON
                    )
                }
            }
            session.setRepeatingRequest(req.build(), previewCallback, bgHandler)
        } catch (_: Throwable) {
        }
    }

    private val previewCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(
            session: CameraCaptureSession,
            request: CaptureRequest,
            result: TotalCaptureResult
        ) {
            result.get(CaptureResult.SENSOR_EXPOSURE_TIME)?.let { lastAeExposureNs = it }
            result.get(CaptureResult.SENSOR_SENSITIVITY)?.let { lastAeIso = it }
            extractColorMetadata(result)?.let { (wb, ccm) ->
                lastWbGains = wb
                lastCcm = ccm
            }
        }
    }

    fun triggerTapFocus(normX: Float, normY: Float) {
        val cam = cameraDevice ?: return
        val session = captureSession ?: return
        val surface = previewSurface ?: return
        val lens = currentLens ?: return
        try {
            val crop = buildCropRect(lens, currentZoomCrop)
            val cx = (crop.left + normX * crop.width()).roundToInt()
            val cy = (crop.top + normY * crop.height()).roundToInt()
            val halfW = max(80, crop.width() / 10)
            val halfH = max(80, crop.height() / 10)
            val rect = Rect(
                max(crop.left, cx - halfW),
                max(crop.top, cy - halfH),
                min(crop.right, cx + halfW),
                min(crop.bottom, cy + halfH)
            )
            val meter = arrayOf(MeteringRectangle(rect, MeteringRectangle.METERING_WEIGHT_MAX))
            val req = cam.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                addTarget(surface)
                set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)
                set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_AUTO)
                set(CaptureRequest.CONTROL_AF_REGIONS, meter)
                set(CaptureRequest.CONTROL_AE_REGIONS, meter)
                set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_START)
                set(CaptureRequest.SCALER_CROP_REGION, crop)
            }
            session.capture(req.build(), previewCallback, bgHandler)
        } catch (_: Throwable) {
        }
    }

    private fun extractColorMetadata(result: CaptureResult): Pair<FloatArray, FloatArray>? {
        val vec: RggbChannelVector = result.get(CaptureResult.COLOR_CORRECTION_GAINS) ?: return null
        val g = 0.5f * (vec.greenEven + vec.greenOdd)
        val wb = floatArrayOf(vec.red, max(0.01f, g), vec.blue)

        val transform: ColorSpaceTransform? = result.get(CaptureResult.COLOR_CORRECTION_TRANSFORM)
        val ccm = FloatArray(9)
        if (transform != null) {
            val rats = Array(9) { Rational(0, 1) }
            transform.copyElements(rats, 0)
            for (i in 0 until 9) {
                ccm[i] = if (rats[i].denominator != 0) rats[i].toFloat() else 0f
            }
        } else {
            ccm[0] = 1f; ccm[4] = 1f; ccm[8] = 1f
        }
        return wb to ccm
    }

    private fun completePairImage(ts: Long, buf: ByteBuffer) {
        synchronized(burstLock) {
            val pair = pendingMap.getOrPut(ts) { PendingPair() }
            pair.buffer = buf
            tryFinalizePair(ts, pair)
        }
    }

    private fun completePairResult(ts: Long, res: TotalCaptureResult, frameType: Int) {
        synchronized(burstLock) {
            val pair = pendingMap.getOrPut(ts) { PendingPair() }
            pair.result = res
            pair.frameType = frameType
            tryFinalizePair(ts, pair)
        }
    }

    private fun tryFinalizePair(ts: Long, pair: PendingPair) {
        val buf = pair.buffer ?: return
        val res = pair.result ?: return
        val lens = currentLens ?: return
        pendingMap.remove(ts)

        val exp = res.get(CaptureResult.SENSOR_EXPOSURE_TIME) ?: lastAeExposureNs
        val iso = res.get(CaptureResult.SENSOR_SENSITIVITY) ?: lastAeIso
        val dynBl = res.get(CaptureResult.SENSOR_DYNAMIC_BLACK_LEVEL)
        val bl = if (dynBl != null && dynBl.size >= 4) dynBl.copyOf(4) else lens.staticBlackLevel.clone()
        val (wb, ccm) = extractColorMetadata(res) ?: (lastWbGains to lastCcm)

        burstCapturedList.add(
            CapturedRawFrame(
                buffer = buf,
                exposureNs = exp,
                iso = iso,
                frameType = pair.frameType,
                blackLevels = bl,
                wbGains = wb,
                ccm = ccm
            )
        )
        if (burstCapturedList.size >= burstExpectedCount) {
            burstDeferred?.complete(burstCapturedList.toList())
        }
    }

    suspend fun captureHybridBurst(fastCount: Int, longCount: Int): List<CapturedRawFrame> {
        val cam = cameraDevice ?: throw IllegalStateException("Camera not open")
        val session = captureSession ?: throw IllegalStateException("Session not ready")
        val rSurface = rawReader?.surface ?: throw IllegalStateException("RAW reader not ready")
        val pSurface = previewSurface ?: throw IllegalStateException("Preview not ready")
        val lens = currentLens ?: throw IllegalStateException("No
