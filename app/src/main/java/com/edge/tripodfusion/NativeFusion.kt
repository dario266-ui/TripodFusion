package com.edge.tripodfusion

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Environment
import android.provider.MediaStore
import androidx.exifinterface.media.ExifInterface
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object NativeFusion {
    init {
        System.loadLibrary("tripodfusion")
    }

    external fun processBurst(
        rawFrames: Array<ByteBuffer>,
        width: Int,
        height: Int,
        cfaPattern: Int,
        blackLevels: FloatArray,
        whiteLevel: Float,
        evMultipliers: FloatArray,
        frameTypes: IntArray,
        wbGains: FloatArray,
        ccm3x3: FloatArray,
        zoomFactor: Float,
        superResScale: Float,
        sharpenAmount: Float,
        outBitmap: Bitmap
    ): Int

    fun rotateBitmap(src: Bitmap, degrees: Int): Bitmap {
        if (degrees % 360 == 0) return src
        val m = Matrix().apply { postRotate(degrees.toFloat()) }
        val rotated = Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
        if (rotated !== src) src.recycle()
        return rotated
    }

    fun saveJpegToGallery(
        context: Context,
        bitmap: Bitmap,
        focalLengthMm: Float,
        baseExposureNs: Long,
        baseIso: Int,
        lensLabel: String
    ): android.net.Uri? {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val name = "TF_${lensLabel.replace(".", "p")}_$stamp.jpg"

        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, name)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(
                MediaStore.Images.Media.RELATIVE_PATH,
                "${Environment.DIRECTORY_PICTURES}/TripodFusion"
            )
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }

        val resolver = context.contentResolver
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return null

        val jpegBytes = ByteArrayOutputStream().use { bos ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 97, bos)
            bos.toByteArray()
        }

        resolver.openOutputStream(uri)?.use { out ->
            out.write(jpegBytes)
        }

        // Write clean EXIF metadata
        try {
            resolver.openFileDescriptor(uri, "rw")?.use { pfd ->
                val exif = ExifInterface(pfd.fileDescriptor)
                exif.setAttribute(ExifInterface.TAG_MAKE, android.os.Build.MANUFACTURER)
                exif.setAttribute(ExifInterface.TAG_MODEL, android.os.Build.MODEL)
                exif.setAttribute(
                    ExifInterface.TAG_FOCAL_LENGTH,
                    "${(focalLengthMm * 100).toInt()}/100"
                )
                exif.setAttribute(
                    ExifInterface.TAG_EXPOSURE_TIME,
                    String.format(Locale.US, "%.6f", baseExposureNs / 1_000_000_000.0)
                )
                exif.setAttribute(ExifInterface.TAG_PHOTOGRAPHIC_SENSITIVITY, baseIso.toString())
                exif.setAttribute(
                    ExifInterface.TAG_SOFTWARE,
                    "TripodFusion RAW16 HybridMFSR"
                )
                exif.saveAttributes()
            }
        } catch (_: Throwable) {
            // Exif failure should not discard saved JPEG
        }

        values.clear()
        values.put(MediaStore.Images.Media.IS_PENDING, 0)
        resolver.update(uri, values, null, null)
        return uri
    }
}
