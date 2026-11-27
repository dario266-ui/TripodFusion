#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <thread>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "TripodFusionCPP", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "TripodFusionCPP", __VA_ARGS__)

namespace {

inline int clampInt(int v, int lo, int hi) {
    return std::max(lo, std::min(v, hi));
}

inline float clampFloat(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
}

// Returns 0=R, 1=G, 2=B for Bayer pixel (x, y)
// CFA constants match CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT:
// 0=RGGB, 1=GRBG, 2=GBRG, 3=BGGR
inline int bayerColor(int x, int y, int cfa) {
    int p = ((y & 1) << 1) | (x & 1);
    switch (cfa) {
        case 0: // RGGB
            return (p == 0) ? 0 : (p == 3 ? 2 : 1);
        case 1: // GRBG
            return (p == 1) ? 0 : (p == 2 ? 2 : 1);
        case 2: // GBRG
            return (p == 2) ? 0 : (p == 1 ? 2 : 1);
        case 3: // BGGR
        default:
            return (p == 3) ? 0 : (p == 0 ? 2 : 1);
    }
}

inline int blIndex(int x, int y) {
    return ((y & 1) << 1) | (x & 1);
}

inline float srgbEncode(float lin) {
    lin = clampFloat(lin, 0.0f, 1.0f);
    if (lin <= 0.0031308f) return 12.92f * lin;
    return 1.055f * std::pow(lin, 1.0f / 2.4f) - 0.055f;
}

// Smooth Narkowicz ACES filmic curve adapted for clean natural dynamic range
inline float filmicToneMap(float x) {
    x = std::max(0.0f, x);
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return clampFloat((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

struct FrameData {
    const uint16_t* raw = nullptr;
    float evMult = 1.0f;
    int type = 0; // 0 = FAST, 1 = LONG
    float sharpness = 0.0f;
    std::vector<float> lumaHalf; // (roiW/2) * (roiH/2)
    int tilesX = 0;
    int tilesY = 0;
    std::vector<float> dx;
    std::vector<float> dy;
};

constexpr int TILE_HALF = 16; // 16x16 in half-res == 32x32 Bayer pixels

float computeSharpness(const std::vector<float>& luma, int hw, int hh) {
    int x0 = hw / 4, x1 = (hw * 3) / 4;
    int y0 = hh / 4, y1 = (hh * 3) / 4;
    double sumGrad = 0.0;
    double sumEnergy = 0.0;
    for (int y = y0 + 1; y < y1 - 1; ++y) {
        const float* row = &luma[y * hw];
        for (int x = x0 + 1; x < x1 - 1; ++x) {
            float gx = row[x + 1] - row[x - 1];
            float gy = luma[(y + 1) * hw + x] - luma[(y - 1) * hw + x];
            sumGrad += (gx * gx + gy * gy);
            sumEnergy += row[x] + 1e-4f;
        }
    }
    return static_cast<float>(sumGrad / std::max(1.0, sumEnergy));
}

void buildPyramidLevel(const std::vector<float>& src, int sw, int sh,
                       std::vector<float>& dst, int& dw, int& dh) {
    dw = std::max(1, sw / 2);
    dh = std::max(1, sh / 2);
    dst.resize(dw * dh);
    for (int y = 0; y < dh; ++y) {
        int sy = std::min(sh - 2, y * 2);
        for (int x = 0; x < dw; ++x) {
            int sx = std::min(sw - 2, x * 2);
            dst[y * dw + x] = 0.25f * (
                src[sy * sw + sx] + src[sy * sw + sx + 1] +
                src[(sy + 1) * sw + sx] + src[(sy + 1) * sw + sx + 1]
            );
        }
    }
}

void alignFrameToReference(
    const FrameData& ref,
    const std::vector<float>& refL1, int r1w, int r1h,
    const std::vector<float>& refL2, int r2w, int r2h,
    FrameData& frame,
    int hw, int hh
) {
    int txCount = std::max(1, (hw + TILE_HALF - 1) / TILE_HALF);
    int tyCount = std::max(1, (hh + TILE_HALF - 1) / TILE_HALF);
    frame.tilesX = txCount;
    frame.tilesY = tyCount;
    frame.dx.assign(txCount * tyCount, 0.0f);
    frame.dy.assign(txCount * tyCount, 0.0f);

    std::vector<float> curL1, curL2;
    int c1w, c1h, c2w, c2h;
    buildPyramidLevel(frame.lumaHalf, hw, hh, curL1, c1w, c1h);
    buildPyramidLevel(curL1, c1w, c1h, curL2, c2w, c2h);

    auto tileSad = [](const std::vector<float>& A, const std::vector<float>& B,
                      int w, int h, int x0, int y0, int tw, int th, int sx, int sy) -> float {
        float err = 0.0f;
        int cnt = 0;
        for (int y = 0; y < th; ++y) {
            int ay = y0 + y;
            int by = ay + sy;
            if (ay < 0 || ay >= h || by < 0 || by >= h) continue;
            for (int x = 0; x < tw; ++x) {
                int ax = x0 + x;
                int bx = ax + sx;
                if (ax < 0 || ax >= w || bx < 0 || bx >= w) continue;
                err += std::fabs(A[ay * w + ax] - B[by * w + bx]);
                ++cnt;
            }
        }
        return (cnt > 4) ? (err / cnt) : 1e9f;
    };

    for (int ty = 0; ty < tyCount; ++ty) {
        for (int tx = 0; tx < txCount; ++tx) {
            // Level 2 coarse search (+-3 at 4x == +-12 half-res == +-24 Bayer pixels)
            int x2 = (tx * TILE_HALF) / 4;
            int y2 = (ty * TILE_HALF) / 4;
            int tw2 = std::max(4, TILE_HALF / 4);
            int bestSx = 0, bestSy = 0;
            float bestErr = 1e9f;
            for (int sy = -3; sy <= 3; ++sy) {
                for (int sx = -3; sx <= 3; ++sx) {
                    float e = tileSad(refL2, curL2, r2w, r2h, x2, y2, tw2, tw2, sx, sy);
                    if (e < bestErr) {
                        bestErr = e;
                        bestSx = sx;
                        bestSy = sy;
                    }
                }
            }

            // Level 1 refine
            int x1 = (tx * TILE_HALF) / 2;
            int y1 = (ty * TILE_HALF) / 2;
            int tw1 = std::max(6, TILE_HALF / 2);
            int cx1 = bestSx * 2, cy1 = bestSy * 2;
            bestErr = 1e9f;
            for (int sy = cy1 - 2; sy <= cy1 + 2; ++sy) {
                for (int sx = cx1 - 2; sx <= cx1 + 2; ++sx) {
                    float e = tileSad(refL1, curL1, r1w, r1h, x1, y1, tw1, tw1, sx, sy);
                    if (e < bestErr) {
                        bestErr = e;
                        bestSx = sx;
                        bestSy = sy;
                    }
                }
            }

            // Level 0 refine + subpixel parabolic fit
            int x0 = tx * TILE_HALF;
            int y0 = ty * TILE_HALF;
            int cx0 = bestSx * 2, cy0 = bestSy * 2;
            float errGrid[3][3];
            bestErr = 1e9f;
            int optX = cx0, optY = cy0;
            for (int sy = cy0 - 2; sy <= cy0 + 2; ++sy) {
                for (int sx = cx0 - 2; sx <= cx0 + 2; ++sx) {
                    float e = tileSad(ref.lumaHalf, frame.lumaHalf, hw, hh, x0, y0, TILE_HALF, TILE_HALF, sx, sy);
                    if (e < bestErr) {
                        bestErr = e;
                        optX = sx;
                        optY = sy;
                    }
                }
            }

            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    errGrid[dy + 1][dx + 1] = tileSad(
                        ref.lumaHalf, frame.lumaHalf, hw, hh,
                        x0, y0, TILE_HALF, TILE_HALF, optX + dx, optY + dy
                    );
                }
            }

            float subX = 0.0f, subY = 0.0f;
            float denomX = errGrid[1][0] - 2.0f * errGrid[1][1] + errGrid[1][2];
            if (denomX > 1e-6f) {
                subX = clampFloat(0.5f * (errGrid[1][0] - errGrid[1][2]) / denomX, -0.5f, 0.5f);
            }
            float denomY = errGrid[0][1] - 2.0f * errGrid[1][1] + errGrid[2][1];
            if (denomY > 1e-6f) {
                subY = clampFloat(0.5f * (errGrid[0][1] - errGrid[2][1]) / denomY, -0.5f, 0.5f);
            }

            // Convert half-res shift to full Bayer pixel units
            frame.dx[ty * txCount + tx] = (optX + subX) * 2.0f;
            frame.dy[ty * txCount + tx] = (optY + subY) * 2.0f;
        }
    }
}

inline void sampleShift(const FrameData& f, float rx, float ry, int roiW, int roiH, float& outDx, float& outDy) {
    float hx = clampFloat(rx * 0.5f / TILE_HALF - 0.5f, 0.0f, (float)(f.tilesX - 1));
    float hy = clampFloat(ry * 0.5f / TILE_HALF - 0.5f, 0.0f, (float)(f.tilesY - 1));
    int x0 = (int)hx, y0 = (int)hy;
    int x1 = std::min(x0 + 1, f.tilesX - 1);
    int y1 = std::min(y0 + 1, f.tilesY - 1);
    float fx = hx - x0, fy = hy - y0;
    float d00x = f.dx[y0 * f.tilesX + x0], d10x = f.dx[y0 * f.tilesX + x1];
    float d01x = f.dx[y1 * f.tilesX + x0], d11x = f.dx[y1 * f.tilesX + x1];
    float d00y = f.dy[y0 * f.tilesX + x0], d10y = f.dy[y0 * f.tilesX + x1];
    float d01y = f.dy[y1 * f.tilesX + x0], d11y = f.dy[y1 * f.tilesX + x1];
    outDx = (1 - fy) * ((1 - fx) * d00x + fx * d10x) + fy * ((1 - fx) * d01x + fx * d11x);
    outDy = (1 - fy) * ((1 - fx) * d00y + fx * d10y) + fy * ((1 - fx) * d01y + fx * d11y);
}

inline float sampleLumaBilinear(const std::vector<float>& luma, int hw, int hh, float rx, float ry) {
    float hx = clampFloat(rx * 0.5f - 0.5f, 0.0f, (float)(hw - 1));
    float hy = clampFloat(ry * 0.5f - 0.5f, 0.0f, (float)(hh - 1));
    int x0 = (int)hx, y0 = (int)hy;
    int x1 = std::min(x0 + 1, hw - 1);
    int y1 = std::min(y0 + 1, hh - 1);
    float fx = hx - x0, fy = hy - y0;
    return (1 - fy) * ((1 - fx) * luma[y0 * hw + x0] + fx * luma[y0 * hw + x1])
         + fy       * ((1 - fx) * luma[y1 * hw + x0] + fx * luma[y1 * hw + x1]);
}

} // namespace

extern "C" JNIEXPORT jint JNICALL
Java_com_edge_tripodfusion_NativeFusion_processBurst(
    JNIEnv* env,
    jobject /* thiz */,
    jobjectArray rawFrames,
    jint width,
    jint height,
    jint cfaPattern,
    jfloatArray blackLevelsArr,
    jfloat whiteLevel,
    jfloatArray evMultipliersArr,
    jintArray frameTypesArr,
    jfloatArray wbGainsArr,
    jfloatArray ccmArr,
    jfloat zoomFactor,
    jfloat superResScale,
    jfloat sharpenAmount,
    jobject outBitmap
) {
    const int numFrames = env->GetArrayLength(rawFrames);
    if (numFrames <= 0 || width <= 0 || height <= 0) {
        LOGE("Invalid input dimensions or zero frames");
        return -1;
    }

    jfloat bl[4];
    env->GetFloatArrayRegion(blackLevelsArr, 0, 4, bl);

    std::vector<float> evMults(numFrames);
    env->GetFloatArrayRegion(evMultipliersArr, 0, numFrames, evMults.data());

    std::vector<jint> frameTypes(numFrames);
    env->GetIntArrayRegion(frameTypesArr, 0, numFrames, frameTypes.data());

    jfloat wbRaw[3];
    env->GetFloatArrayRegion(wbGainsArr, 0, 3, wbRaw);
    float gNorm = std::max(0.01f, wbRaw[1]);
    float wb[3] = {
        clampFloat(wbRaw[0] / gNorm, 0.5f, 5.0f),
        1.0f,
        clampFloat(wbRaw[2] / gNorm, 0.5f, 5.0f)
    };

    jfloat ccm[9];
    env->GetFloatArrayRegion(ccmArr, 0, 9, ccm);
    float ccmSum = std::fabs(ccm[0]) + std::fabs(ccm[4]) + std::fabs(ccm[8]);
    if (ccmSum < 0.2f) {
        ccm[0] = 1.0f; ccm[1] = 0.0f; ccm[2] = 0.0f;
        ccm[3] = 0.0f; ccm[4] = 1.0f; ccm[5] = 0.0f;
        ccm[6] = 0.0f; ccm[7] = 0.0f; ccm[8] = 1.0f;
    }

    AndroidBitmapInfo bmpInfo;
    if (AndroidBitmap_getInfo(env, outBitmap, &bmpInfo) != ANDROID_BITMAP_RESULT_SUCCESS) {
        LOGE("Failed to get Bitmap info");
        return -2;
    }
    const int outW = static_cast<int>(bmpInfo.width);
    const int outH = static_cast<int>(bmpInfo.height);

    void* bmpPixelsVoid = nullptr;
    if (AndroidBitmap_lockPixels(env, outBitmap, &bmpPixelsVoid) != ANDROID_BITMAP_RESULT_SUCCESS) {
        LOGE("Failed to lock Bitmap pixels");
        return -3;
    }
    uint32_t* outPixels = static_cast<uint32_t*>(bmpPixelsVoid);

    // Compute Bayer-aligned ROI crop around sensor center (even coordinates preserve CFA)
    float z = std::max(1.0f, zoomFactor);
    int roiW = clampInt(((int)(width / z)) & ~1, 128, width);
    int roiH = clampInt(((int)(height / z)) & ~1, 128, height);
    int roiX0 = ((width - roiW) / 2) & ~1;
    int roiY0 = ((height - roiH) / 2) & ~1;

    // Gather pointers & auto-check MediaTek RAW16 scaling vs whiteLevel
    std::vector<FrameData> frames(numFrames);
    for (int i = 0; i < numFrames; ++i) {
        jobject bufObj = env->GetObjectArrayElement(rawFrames, i);
        frames[i].raw = static_cast<const uint16_t*>(env->GetDirectBufferAddress(bufObj));
        frames[i].evMult = evMults[i];
        frames[i].type = frameTypes[i];
        env->DeleteLocalRef(bufObj);
        if (!frames[i].raw) {
            AndroidBitmap_unlockPixels(env, outBitmap);
            return -4;
        }
    }

    uint16_t sampleMax = 0;
    for (int y = roiY0; y < roiY0 + roiH; y += 16) {
        const uint16_t* row = frames[0].raw + y * width;
        for (int x = roiX0; x < roiX0 + roiW; x += 16) {
            if (row[x] > sampleMax) sampleMax = row[x];
        }
    }
    float effectiveWhite = whiteLevel;
    float blScale = 1.0f;
    if (sampleMax > whiteLevel * 1.5f) {
        // Handle HALs that left-shift 10/12-bit RAW inside 16-bit container
        float ratio = (sampleMax > 16384) ? (65535.0f / std::max(1.0f, whiteLevel)) : 4.0f;
        blScale = ratio;
        effectiveWhite = whiteLevel * ratio;
    }
    float invRange[4];
    float blAdj[4];
    for (int c = 0; c < 4; ++c) {
        blAdj[c] = bl[c] * blScale;
        invRange[c] = 1.0f / std::max(1.0f, effectiveWhite - blAdj[c]);
    }

    const int hw = roiW / 2;
    const int hh = roiH / 2;

    const int numThreads = std::max(2, std::min(8, (int)std::thread::hardware_concurrency()));

    // Step 1: Build normalized half-res luma proxies & score sharpness
    for (int f = 0; f < numFrames; ++f) {
        frames[f].lumaHalf.resize(hw * hh);
        const uint16_t* rawPtr = frames[f].raw;
        float mult = frames[f].evMult;
        std::vector<float>& luma = frames[f].lumaHalf;

        std::vector<std::thread> workers;
        for (int t = 0; t < numThreads; ++t) {
            workers.emplace_back([&, t]() {
                int yStart = (hh * t) / numThreads;
                int yEnd = (hh * (t + 1)) / numThreads;
                for (int hy = yStart; hy < yEnd; ++hy) {
                    int y = roiY0 + hy * 2;
                    const uint16_t* r0 = rawPtr + y * width;
                    const uint16_t* r1 = rawPtr + (y + 1) * width;
                    for (int hx = 0; hx < hw; ++hx) {
                        int x = roiX0 + hx * 2;
                        float p00 = std::max(0.0f, (r0[x] - blAdj[0]) * invRange[0]);
                        float p10 = std::max(0.0f, (r0[x + 1] - blAdj[1]) * invRange[1]);
                        float p01 = std::max(0.0f, (r1[x] - blAdj[2]) * invRange[2]);
                        float p11 = std::max(0.0f, (r1[x + 1] - blAdj[3]) * invRange[3]);
                        luma[hy * hw + hx] = 0.25f * (p00 + p10 + p01 + p11) * mult;
                    }
                }
            });
        }
        for (auto& w : workers) w.join();
        frames[f].sharpness = computeSharpness(frames[f].lumaHalf, hw, hh);
    }

    // Pick best FAST frame (motion anchor) and best LONG frame
    int bestFastIdx = 0;
    float bestFastSharp = -1.0f;
    int bestLongIdx = -1;
    float bestLongSharp = -1.0f;
    for (int f = 0; f < numFrames; ++f) {
        if (frames[f].type == 0) {
            if (frames[f].sharpness > bestFastSharp) {
                bestFastSharp = frames[f].sharpness;
                bestFastIdx = f;
            }
        } else {
            if (frames[f].sharpness > bestLongSharp) {
                bestLongSharp = frames[f].sharpness;
                bestLongIdx = f;
            }
        }
    }

    int refIdx = bestFastIdx;
    if (bestLongIdx >= 0 && bestLongSharp >= bestFastSharp * 0.68f) {
        refIdx = bestLongIdx;
    }
    LOGI("Selected refIdx=%d (fastBest=%d sharp=%.4f, longBest=%d sharp=%.4f)",
         refIdx, bestFastIdx, bestFastSharp, bestLongIdx, bestLongSharp);

    // Build reference pyramid once & align all frames in parallel
    std::vector<float> refL1, refL2;
    int r1w, r1h, r2w, r2h;
    buildPyramidLevel(frames[refIdx].lumaHalf, hw, hh, refL1, r1w, r1h);
    buildPyramidLevel(refL1, r1w, r1h, refL2, r2w, r2h);

    {
        std::vector<std::thread> workers;
        for (int f = 0; f < numFrames; ++f) {
            if (f == refIdx) {
                frames[f].tilesX = std::max(1, (hw + TILE_HALF - 1) / TILE_HALF);
                frames[f].tilesY = std::max(1, (hh + TILE_HALF - 1) / TILE_HALF);
                frames[f].dx.assign(frames[f].tilesX * frames[f].tilesY, 0.0f);
                frames[f].dy.assign(frames[f].tilesX * frames[f].tilesY, 0.0f);
                continue;
            }
            workers.emplace_back([&, f]() {
                alignFrameToReference(frames[refIdx], refL1, r1w, r1h, refL2, r2w, r2h, frames[f], hw, hh);
            });
        }
        for (auto& w : workers) w.join();
    }

    // Pre-estimate median scene linear brightness to calibrate exposure lift cleanly
    std::vector<int> hist(256, 0);
    int totalSamples = 0;
    for (int y = 0; y < hh; y += 2) {
        for (int x = 0; x < hw; x += 2) {
            float v = frames[refIdx].lumaHalf[y * hw + x];
            int bin = clampInt((int)(clampFloat(v, 0.0f, 1.0f) * 255.0f), 0, 255);
            hist[bin]++;
            totalSamples++;
        }
    }
    int running = 0, medBin = 32;
    for (int b = 0; b < 256; ++b) {
        running += hist[b];
        if (running >= totalSamples / 2) {
            medBin = b;
            break;
        }
    }
    float medianLuma = std::max(0.015f, medBin / 255.0f);
    float exposureLift = clampFloat(0.18f / medianLuma, 0.85f, 3.6f);

    // Precompute frame weights (reject blurry frames)
    std::vector<float> frameBaseWeight(numFrames, 1.0f);
    for (int f = 0; f < numFrames; ++f) {
        if (frames[f].type == 1) {
            if (bestLongSharp > 0.0f && frames[f].sharpness < bestLongSharp * 0.55f) {
                frameBaseWeight[f] = 0.0f;
            } else {
                frameBaseWeight[f] = 3.5f; // Low-ISO clean SNR boost
            }
        } else {
            if (bestFastSharp > 0.0f && frames[f].sharpness < bestFastSharp * 0.55f) {
                frameBaseWeight[f] = 0.1f;
            } else {
                frameBaseWeight[f] = 1.0f;
            }
        }
    }

    // Step 2: Strip-by-strip Steerable Anisotropic Kernel Regression + Hybrid Motion/HDR Fusion
    const float invScaleX = (float)roiW / (float)outW;
    const float invScaleY = (float)roiH / (float)outH;
    const std::vector<float>& refLuma = frames[refIdx].lumaHalf;
    const std::vector<float>& fastLuma = frames[bestFastIdx].lumaHalf;

    // Temporary float RGB output buffer if sharpening is enabled, else direct write
    std::vector<float> sharpBuf((size_t)outW * outH * 3);

    {
        std::vector<std::thread> workers;
        for (int t = 0; t < numThreads; ++t) {
            workers.emplace_back([&, t]() {
                int oyStart = (outH * t) / numThreads;
                int oyEnd = (outH * (t + 1)) / numThreads;

                for (int oy = oyStart; oy < oyEnd; ++oy) {
                    float ry = (oy + 0.5f) * invScaleY;
                    int hy = clampInt((int)(ry * 0.5f), 2, hh - 3);

                    for (int ox = 0; ox < outW; ++ox) {
                        float rx = (ox + 0.5f) * invScaleX;
                        int hx = clampInt((int)(rx * 0.5f), 2, hw - 3);

                        // Local structure tensor on half-res reference luma
                        float Jxx = 0.0f, Jxy = 0.0f, Jyy = 0.0f;
                        for (int ky = -1; ky <= 1; ++ky) {
                            for (int kx = -1; kx <= 1; ++kx) {
                                int cx = hx + kx, cy = hy + ky;
                                float gx = 0.5f * (refLuma[cy * hw + cx + 1] - refLuma[cy * hw + cx - 1]);
                                float gy = 0.5f * (refLuma[(cy + 1) * hw + cx] - refLuma[(cy - 1) * hw + cx]);
                                Jxx += gx * gx;
                                Jxy += gx * gy;
                                Jyy += gy * gy;
                            }
                        }
                        float trace = Jxx + Jyy;
                        float diff = Jxx - Jyy;
                        float disc = std::sqrt(std::max(0.0f, diff * diff + 4.0f * Jxy * Jxy));
                        float l1 = 0.5f * (trace + disc);
                        float l2 = 0.5f * (trace - disc);
                        float coh = (l1 - l2) / (l1 + l2 + 1e-6f);

                        // Gradient normal (nx, ny) and edge tangent (ex, ey)
                        float nx = 1.0f, ny = 0.0f;
                        if (disc > 1e-7f) {
                            float vx = l1 - Jyy;
                            float vy = Jxy;
                            float len = std::sqrt(vx * vx + vy * vy);
                            if (len > 1e-7f) {
                                nx = vx / len;
                                ny = vy / len;
                            }
                        }
                        float ex = -ny, ey = nx;

                        // Steerable kernel radii in Bayer pixel units
                        float sigPar = 0.62f * (1.0f + 1.35f * coh);
                        float sigPerp = 0.62f * (1.0f - 0.45f * coh);
                        float inv2Par = 0.5f / (sigPar * sigPar);
                        float inv2Perp = 0.5f / (sigPerp * sigPerp);

                        float sumC[3] = {0.0f, 0.0f, 0.0f};
                        float sumW[3] = {0.0f, 0.0f, 0.0f};

                        float lumaFastAnchor = sampleLumaBilinear(fastLuma, hw, hh,
                            rx + (bestFastIdx == refIdx ? 0.0f : frames[bestFastIdx].dx[0]),
                            ry + (bestFastIdx == refIdx ? 0.0f : frames[bestFastIdx].dy[0]));
                        float dxF, dyF;
                        sampleShift(frames[bestFastIdx], rx, ry, roiW, roiH, dxF, dyF);
                        lumaFastAnchor = sampleLumaBilinear(fastLuma, hw, hh, rx + dxF, ry + dyF);

                        for (int f = 0; f < numFrames; ++f) {
                            float bw = frameBaseWeight[f];
                            if (bw <= 0.0f) continue;

                            float dx, dy;
                            sampleShift(frames[f], rx, ry, roiW, roiH, dx, dy);
                            float fx = rx + dx;
                            float fy = ry + dy;

                            // Hybrid motion check vs fast motion-anchor
                            float fLuma = sampleLumaBilinear(frames[f].lumaHalf, hw, hh, fx, fy);
                            float lumaDiff = std::fabs(fLuma - lumaFastAnchor);
                            float tol = 0.025f + 0.16f * std::max(fLuma, lumaFastAnchor);
                            float motionWeight = 1.0f;
                            if (frames[f].type == 1) {
                                // Long exposure: if it moved relative to fast freeze frame, reject it smoothly
                                if (lumaDiff > tol) {
                                    float excess = (lumaDiff - tol) / std::max(0.01f, tol);
                                    motionWeight = std::exp(-3.5f * excess * excess);
                                }
                            } else {
                                // Fast frame: mild outlier rejection
                                if (lumaDiff > tol * 1.4f) {
                                    motionWeight = 0.2f;
                                }
                            }
                            float fw = bw * motionWeight;
                            if (fw < 0.02f) continue;

                            int ix0 = clampInt((int)std::floor(fx) - 1, 0, roiW - 1);
                            int ix1 = clampInt((int)std::floor(fx) + 2, 0, roiW - 1);
                            int iy0 = clampInt((int)std::floor(fy) - 1, 0, roiH - 1);
                            int iy1 = clampInt((int)std::floor(fy) + 2, 0, roiH - 1);

                            const uint16_t* rawPtr = frames[f].raw;
                            float evM = frames[f].evMult;

                            for (int py = iy0; py <= iy1; ++py) {
                                int sy = roiY0 + py;
                                float ddy = (float)py + 0.5f - fy;
                                const uint16_t* rowPtr = rawPtr + sy * width;

                                for (int px = ix0; px <= ix1; ++px) {
                                    int sx = roiX0 + px;
                                    uint16_t rawVal = rowPtr[sx];
                                    int bi = blIndex(sx, sy);
                                    float normVal = std::max(0.0f, (rawVal - blAdj[bi]) * invRange[bi]);

                                    // Highlight recovery: ignore clipped pixels in long frames
                                    if (frames[f].type == 1 && normVal > 0.93f) continue;

                                    float ddx = (float)px + 0.5f - fx;
                                    float dAlong = ddx * ex + ddy * ey;
                                    float dAcross = ddx * nx + ddy * ny;
                                    float wk = std::exp(-(dAlong * dAlong * inv2Par + dAcross * dAcross * inv2Perp));
                                    float w = fw * wk;

                                    int col = bayerColor(sx, sy, cfaPattern);
                                    sumC[col] += (normVal * evM) * w;
                                    sumW[col] += w;
                                }
                            }
                        }

                        // Fallback to fast anchor if any channel clipped across all long frames
                        float rgb[3];
                        for (int c = 0; c < 3; ++c) {
                            if (sumW[c] > 1e-6f) {
                                rgb[c] = sumC[c] / sumW[c];
                            } else {
                                rgb[c] = lumaFastAnchor;
                            }
                        }

                        // Apply White Balance & CCM (Camera RGB -> Linear sRGB)
                        float wr = rgb[0] * wb[0];
                        float wg = rgb[1] * wb[1];
                        float wbVal = rgb[2] * wb[2];

                        float sr = (ccm[0] * wr + ccm[1] * wg + ccm[2] * wbVal) * exposureLift;
                        float sg = (ccm[3] * wr + ccm[4] * wg + ccm[5] * wbVal) * exposureLift;
                        float sb = (ccm[6] * wr + ccm[7] * wg + ccm[8] * wbVal) * exposureLift;

                        // Luma-preserving filmic tone mapping (keeps colors natural, prevents clipping)
                        float linY = std::max(1e-5f, 0.2126f * sr + 0.7152f * sg + 0.0722f * sb);
                        float mappedY = filmicToneMap(linY);
                        float toneScale = mappedY / linY;

                        size_t idx = ((size_t)oy * outW + ox) * 3;
                        sharpBuf[idx + 0] = srgbEncode(sr * toneScale);
                        sharpBuf[idx + 1] = srgbEncode(sg * toneScale);
                        sharpBuf[idx + 2] = srgbEncode(sb * toneScale);
                    }
                }
            });
        }
        for (auto& w : workers) w.join();
    }

    // Step 3: Gentle thresholded micro-sharpening & pack ARGB_8888 Bitmap (ABGR in little-endian)
    const float shp = clampFloat(sharpenAmount, 0.0f, 0.8f);
    {
        std::vector<std::thread> workers;
        for (int t = 0; t < numThreads; ++t) {
            workers.emplace_back([&, t]() {
                int oyStart = (outH * t) / numThreads;
                int oyEnd = (outH * (t + 1)) / numThreads;
                for (int y = oyStart; y < oyEnd; ++y) {
                    int ym = std::max(0, y - 1);
                    int yp = std::min(outH - 1, y + 1);
                    for (int x = 0; x < outW; ++x) {
                        int xm = std::max(0, x - 1);
                        int xp = std::min(outW - 1, x + 1);
                        size_t cIdx = ((size_t)y * outW + x) * 3;
                        uint8_t rgba[3];
                        for (int c = 0; c < 3; ++c) {
                            float center = sharpBuf[cIdx + c];
                            float blur = 0.25f * (
                                sharpBuf[((size_t)y * outW + xm) * 3 + c] +
                                sharpBuf[((size_t)y * outW + xp) * 3 + c] +
                                sharpBuf[((size_t)ym * outW + x) * 3 + c] +
                                sharpBuf[((size_t)yp * outW + x) * 3 + c]
                            );
                            float detail = center - blur;
                            // Soft threshold so flat sky/noise isn't amplified
                            float edgeGate = std::min(1.0f, std::fabs(detail) / 0.015f);
                            float outVal = clampFloat(center + shp * edgeGate * detail, 0.0f, 1.0f);
                            rgba[c] = static_cast<uint8_t>(outVal * 255.0f + 0.5f);
                        }
                        // Android ARGB_8888 memory layout is R, G, B, A bytes in little-endian uint32
                        outPixels[(size_t)y * outW + x] =
                            (0xFFu << 24) |
                            ((uint32_t)rgba[2] << 16) |
                            ((uint32_t)rgba[1] << 8) |
                            (uint32_t)rgba[0];
                    }
                }
            });
        }
        for (auto& w : workers) w.join();
    }

    AndroidBitmap_unlockPixels(env, outBitmap);
    LOGI("Fusion complete: %dx%d", outW, outH);
    return 0;
}
