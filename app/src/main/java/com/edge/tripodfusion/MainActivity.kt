package com.edge.tripodfusion

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.os.Bundle
import android.view.MotionEvent
import android.view.TextureView
import android.view.View
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.edge.tripodfusion.databinding.ActivityMainBinding
import com.google.android.material.button.MaterialButton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraEngine: CameraEngine

    private var lenses: List<LensInfo> = emptyList()
    private var activeLensIdx = 0

    private var timerSeconds = 3
    private var oisDisabled = true
    private var superResScale = 2.0f
    private var zoomCrop = 1.0f

    // Modes: 0 = Hybrid (8 Fast + 6 Long), 1 = Deep Tripod (4 Fast + 10 Long), 2 = Fast Burst (12 Fast)
    private var captureMode = 0
    private var isBusy = false

    private val requestCamPerm = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) initCameraFlow()
        else Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraEngine = CameraEngine(this)

        binding.btnTimer.setOnClickListener {
            timerSeconds = when (timerSeconds) {
                3 -> 5
                5 -> 0
                else -> 3
            }
            binding.btnTimer.text = if (timerSeconds == 0) "Timer OFF" else "Timer ${timerSeconds}s"
        }

        binding.btnOis.setOnClickListener {
            oisDisabled = !oisDisabled
            cameraEngine.setOisDisabledOnTripod(oisDisabled)
            binding.btnOis.text = if (oisDisabled) "OIS: OFF" else "OIS: ON"
        }

        binding.btnScale.setOnClickListener {
            superResScale = when (superResScale) {
                2.0f -> 1.5f
                1.5f -> 1.0f
                else -> 2.0f
            }
            binding.btnScale.text = when (superResScale) {
                2.0f -> "SuperRes 2x"
                1.5f -> "SuperRes 1.5x"
                else -> "Native 1x"
            }
        }

        binding.btnMode.setOnClickListener {
            captureMode = (captureMode + 1) % 3
            binding.btnMode.text = when (captureMode) {
                0 -> "Hybrid\n8F+6L"
                1 -> "Deep\n4F+10L"
                else -> "Action\n12F"
            }
        }

        binding.zoomSeek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                zoomCrop = 1.0f + progress * 0.1f // 1.0x .. 4.0x
                val lens = lenses.getOrNull(activeLensIdx)
                val baseFactor = lens?.label?.removeSuffix("x")?.toFloatOrNull() ?: 1.0f
                val totalZoom = baseFactor * zoomCrop
                binding.zoomLabel.text = String.format("Crop %.1fx\n(%.1fx tot)", zoomCrop, totalZoom)
                cameraEngine.setZoomCrop(zoomCrop)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        binding.previewView.setOnTouchListener { v, event ->
            if (event.action == MotionEvent.ACTION_UP && !isBusy) {
                val nx = (event.x / v.width.toFloat()).coerceIn(0f, 1f)
                val ny = (event.y / v.height.toFloat()).coerceIn(0f, 1f)
                cameraEngine.triggerTapFocus(nx, ny)
                Toast.makeText(this, "Focusing...", Toast.LENGTH_SHORT).show()
            }
            true
        }

        binding.btnCapture.setOnClickListener {
            if (!isBusy) startCapturePipeline()
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            initCameraFlow()
        } else {
            requestCamPerm.launch(Manifest.permission.CAMERA)
        }
    }

    private fun initCameraFlow() {
        lenses = cameraEngine.discoverRawLenses()
        if (lenses.isEmpty()) {
            binding.infoBadge.text = "No RAW16 back lenses found!"
            return
        }

        // Default to 1x lens if found
        activeLensIdx = lenses.indexOfFirst { it.label == "1x" }.takeIf { it >= 0 } ?: 0
        buildLensChips()

        if (binding.previewView.isAvailable) {
            openActiveLens()
        } else {
            binding.previewView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
                override fun onSurfaceTextureAvailable(st: SurfaceTexture, w: Int, h: Int) {
                    openActiveLens()
                }
                override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, w: Int, h: Int) {
                    configurePreviewTransform(w, h)
                }
                override fun onSurfaceTextureDestroyed(st: SurfaceTexture) = true
                override fun onSurfaceTextureUpdated(st: SurfaceTexture) {}
            }
        }
    }

    private fun buildLensChips() {
        binding.lensContainer.removeAllViews()
        lenses.forEachIndexed { idx, lens ->
            val btn = MaterialButton(
                this,
                null,
                com.google.android.material.R.attr.materialButtonOutlinedStyle
            ).apply {
                text = lens.label
                textSize = 12f
                val lp = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    92
                ).apply { setMargins(8, 0, 8, 0) }
                layoutParams = lp
                setOnClickListener {
                    if (!isBusy && idx != activeLensIdx) {
                        activeLensIdx = idx
                        updateLensChipStyles()
                        openActiveLens()
                    }
                }
            }
            binding.lensContainer.addView(btn)
        }
        updateLensChipStyles()
    }

    private fun updateLensChipStyles() {
        for (i in 0 until binding.lensContainer.childCount) {
            val btn = binding.lensContainer.getChildAt(i) as? MaterialButton ?: continue
            val selected = (i == activeLensIdx)
            btn.setTextColor(if (selected) 0xFF000000.toInt() else 0xFFFFFFFF.toInt())
            btn.setBackgroundColor(if (selected) 0xFF00E5FF.toInt() else 0x00000000)
        }
    }

    private fun openActiveLens() {
        val lens = lenses.getOrNull(activeLensIdx) ?: return
        binding.infoBadge.text = "Opening ${lens.label}..."
        cameraEngine.setOisDisabledOnTripod(oisDisabled)
        cameraEngine.setZoomCrop(zoomCrop)
        cameraEngine.openLens(
            lens = lens,
            textureView = binding.previewView,
            onReady = {
                runOnUiThread {
                    configurePreviewTransform(binding.previewView.width, binding.previewView.height)
                    val maxS = lens.exposureRange.upper / 1_000_000_000.0
                    binding.infoBadge.text = String.format(
                        "%s RAW16 %dx%d | Max %.2fs | ISO %d-%d",
                        lens.label,
                        lens.rawSize.width,
                        lens.rawSize.height,
                        maxS,
                        lens.isoRange.lower,
                        lens.isoRange.upper
                    )
                    binding.zoomSeek.progress = binding.zoomSeek.progress // refresh total label
                    val baseFactor = lens.label.removeSuffix("x").toFloatOrNull() ?: 1.0f
                    binding.zoomLabel.text = String.format("Crop %.1fx\n(%.1fx tot)", zoomCrop, baseFactor * zoomCrop)
                }
            },
            onError = { err ->
                runOnUiThread {
                    binding.infoBadge.text = err
                }
            }
        )
    }

    private fun configurePreviewTransform(viewW: Int, viewH: Int) {
        val lens = lenses.getOrNull(activeLensIdx) ?: return
        if (viewW == 0 || viewH == 0) return
        val matrix = Matrix()
        val viewRect = RectF(0f, 0f, viewW.toFloat(), viewH.toFloat())
        // Sensor preview sizes are landscape (w > h) for portrait sensor
        val bufRect = RectF(0f, 0f, lens.previewSize.height.toFloat(), lens.previewSize.width.toFloat())
        val centerX = viewRect.centerX()
        val centerY = viewRect.centerY()
        bufRect.offset(centerX - bufRect.centerX(), centerY - bufRect.centerY())
        matrix.setRectToRect(viewRect, bufRect, Matrix.ScaleToFit.FILL)
        val scale = max(
            viewH.toFloat() / lens.previewSize.width.toFloat(),
            viewW.toFloat() / lens.previewSize.height.toFloat()
        )
        matrix.postScale(scale, scale, centerX, centerY)
        binding.previewView.setTransform(matrix)
    }

    private fun startCapturePipeline() {
        val lens = lenses.getOrNull(activeLensIdx) ?: return
        isBusy = true
        binding.btnCapture.isEnabled = false

        lifecycleScope.launch {
            try {
                // Step 1: Countdown timer
                if (timerSeconds > 0) {
                    binding.countdownText.visibility = View.VISIBLE
                    for (t in timerSeconds downTo 1) {
                        binding.countdownText.text = t.toString()
                        delay(1000L)
                    }
                    binding.countdownText.visibility = View.GONE
                }

                val (fastCount, longCount) = when (captureMode) {
                    0 -> 8 to 6
                    1 -> 4 to 10
                    else -> 12 to 0
                }

                binding.progressOverlay.visibility = View.VISIBLE
                binding.statusText.text = "Shooting ${fastCount} Fast + ${longCount} Long RAW16..."

                val frames = withContext(Dispatchers.IO) {
                    cameraEngine.captureHybridBurst(fastCount, longCount)
                }

                if (frames.isEmpty()) {
                    throw IllegalStateException("No RAW frames captured")
                }

                val t0 = System.currentTimeMillis()
                binding.statusText.text = "Aligning & Steerable SuperRes merging\n(${frames.size} RAW16 frames)..."

                val (finalBmp, elapsedSec) = withContext(Dispatchers.Default) {
                    // Pick reference EV from median/first long or fast frame
                    val refFrame = frames.firstOrNull { it.frameType == 1 } ?: frames.first()
                    val refTotal = max(1.0, refFrame.exposureNs.toDouble() * refFrame.iso.toDouble())

                    val evMults = FloatArray(frames.size) { i ->
                        val fTotal = max(1.0, frames[i].exposureNs.toDouble() * frames[i].iso.toDouble())
                        (refTotal / fTotal).toFloat()
                    }
                    val frameTypes = IntArray(frames.size) { i -> frames[i].frameType }
                    val rawBuffers = Array(frames.size) { i -> frames[i].buffer }

                    val roiW = ((lens.rawSize.width / zoomCrop).toInt() and 1.inv()).coerceIn(128, lens.rawSize.width)
                    val roiH = ((lens.rawSize.height / zoomCrop).toInt() and 1.inv()).coerceIn(128, lens.rawSize.height)

                    // Cap single axis at 6400px so Bitmap fits comfortably in RAM
                    var outW = (roiW * superResScale).roundToInt() and 1.inv()
                    var outH = (roiH * superResScale).roundToInt() and 1.inv()
                    val maxAxis = 6400
                    if (outW > maxAxis || outH > maxAxis) {
                        val s = min(maxAxis.toFloat() / outW, maxAxis.toFloat() / outH)
                        outW = ((outW * s).roundToInt() and 1.inv()).coerceAtLeast(256)
                        outH = ((outH * s).roundToInt() and 1.inv()).coerceAtLeast(256)
                    }

                    val outBitmap = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
                    val status = NativeFusion.processBurst(
                        rawFrames = rawBuffers,
                        width = lens.rawSize.width,
                        height = lens.rawSize.height,
                        cfaPattern = lens.cfa,
                        blackLevels = refFrame.blackLevels,
                        whiteLevel = lens.whiteLevel,
                        evMultipliers = evMults,
                        frameTypes = frameTypes,
                        wbGains = refFrame.wbGains,
                        ccm3x3 = refFrame.ccm,
                        zoomFactor = zoomCrop,
                        superResScale = superResScale,
                        sharpenAmount = 0.25f,
                        outBitmap = outBitmap
                    )

                    if (status != 0) {
                        throw IllegalStateException("Native fusion error: $status")
                    }

                    val rotated = NativeFusion.rotateBitmap(outBitmap, lens.sensorOrientation)
                    val savedUri = NativeFusion.saveJpegToGallery(
                        context = this@MainActivity,
                        bitmap = rotated,
                        focalLengthMm = lens.focalMm,
                        baseExposureNs = refFrame.exposureNs,
                        baseIso = refFrame.iso,
                        lensLabel = lens.label
                    )
                    if (savedUri == null) {
                        throw IllegalStateException("Failed saving JPEG to Pictures/TripodFusion")
                    }
                    val dt = (System.currentTimeMillis() - t0) / 1000.0
                    rotated to dt
                }

                binding.lastThumb.setImageBitmap(finalBmp)
                Toast.makeText(
                    this@MainActivity,
                    String.format("Saved %dx%d in %.1fs → Pictures/TripodFusion", finalBmp.width, finalBmp.height, elapsedSec),
                    Toast.LENGTH_LONG
                ).show()
            } catch (t: Throwable) {
                Toast.makeText(this@MainActivity, "Error: ${t.message}", Toast.LENGTH_LONG).show()
            } finally {
                binding.countdownText.visibility = View.GONE
                binding.progressOverlay.visibility = View.GONE
                binding.btnCapture.isEnabled = true
                isBusy = false
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraEngine.shutdown()
    }
}
