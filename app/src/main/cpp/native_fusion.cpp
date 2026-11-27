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

inline int bayerColor(int x, int y, int cfa) {
    int p = ((y & 1) << 1) | (x & 1);
    switch (cfa) {
        case 0: return (p == 0) ? 0 : (p == 3 ? 2 : 1);
        case 1: return (p == 1) ? 0 : (p == 2 ? 2 : 1);
        case 2: return (p == 2) ? 0 : (p == 1 ? 2 : 1);
        case 3:
        default: return (p == 3) ? 0 : (p == 0 ? 2 : 1);
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
    int type = 0;
    float sharpness = 0.0f;
    std::vector<float> lumaHalf;
    int tilesX = 0;
    int tilesY = 0;
    std::vector<float> dx;
    std::vector<float> dy;
    std::vector<float> tileErr;
};

constexpr int TILE_HALF = 16;

float computeSharpness(const std::vector<float>& luma, int hw, int hh) {
    int x0 = hw / 4, x1 = (hw * 3) / 4;
    int y0 = hh / 4, y1 = (hh * 3) / 4;
    double sumGrad = 0.0;
    double sumEnergy = 0.0;
    for (int y = y0 + 1; y < y1 - 1; ++y) {
        const float* row = &luma[(size_t)y * hw];
        for (int x = x0 + 1; x < x1 - 1; ++x) {
            float gx = row[x + 1] - row[x - 1];
            float gy = luma[(size_t)(y + 1) * hw + x] - luma[(size_t)(y - 1) * hw + x];
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
    dst.resize((size_t)dw * dh);
    for (int y = 0; y < dh; ++y) {
        int sy = std::min(sh - 2, y * 2);
        for (int x = 0; x < dw; ++x) {
            int sx = std::min(sw - 2, x * 2);
            dst[(size_t)y * dw + x] = 0.25f * (
                src[(size_t)sy * sw + sx] + src[(size_t)sy * sw + sx + 1] +
                src[(size_t)(sy + 1) * sw + sx] + src[(size_t)(sy + 1) * sw + sx + 1]
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
    frame.dx.assign((size_t)txCount * tyCount, 0.0f);
    frame.dy.assign((size_t)txCount * tyCount, 0.0f);
    frame.tileErr.assign((size_t)txCount * tyCount, 0.0f);

    std::vector<float> curL1, curL2;
    int c1w, c1h, c2w, c2h;
    buildPyramidLevel(frame.lumaHalf, hw, hh, curL1, c1w, c1h);
    buildPyramidLevel(curL1, c1w, c1h, curL2, c2w, c2h);

    auto regionSad = [](const std::vector<float>& A, const std::vector<float>& B,
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
                err += std::fabs(A[(size_t)ay * w + ax] - B[(size_t)by * w + bx]);
                ++cnt;
            }
        }
        return (cnt > 16) ? (err / cnt) : 1e9f;
    };

    // 1. Global coarse-to-fine frame alignment over central 60% (+-64 Bayer pixels range!)
    int gx2 = r2w / 5, gy2 = r2h / 5, gw2 = (r2w * 3) / 5, gh2 = (r2h * 3) / 5;
    int gSx = 0, gSy = 0;
    float gBest = 1e9f;
    for (int sy = -8; sy <= 8; ++sy) {
        for (int sx = -8; sx <= 8; ++sx) {
            float e = regionSad(refL2, curL2, r2w, r2h, gx2, gy2, gw2, gh2, sx, sy);
            if (e < gBest) { gBest = e; gSx = sx; gSy = sy; }
        }
    }

    int gx1 = r1w / 5, gy1 = r1h / 5, gw1 = (r1w * 3) / 5, gh1 = (r1h * 3) / 5;
    int cx1 = gSx * 2, cy1 = gSy * 2;
    gBest = 1e9f;
    for (int sy = cy1 - 2; sy <= cy1 + 2; ++sy) {
        for (int sx = cx1 - 2; sx <= cx1 + 2; ++sx) {
            float e = regionSad(refL1, curL1, r1w, r1h, gx1, gy1, gw1, gh1, sx, sy);
            if (e < gBest) { gBest = e; gSx = sx; gSy = sy; }
        }
    }

    int gx0 = hw / 5, gy0 = hh / 5, gw0 = (hw * 3) / 5, gh0 = (hh * 3) / 5;
    int cx0 = gSx * 2, cy0 = gSy * 2;
    gBest = 1e9f;
    for (int sy = cy0 - 2; sy <= cy0 + 2; ++sy) {
        for (int sx = cx0 - 2; sx <= cx0 + 2; ++sx) {
            float e = regionSad(ref.lumaHalf, frame.lumaHalf, hw, hh, gx0, gy0, gw0, gh0, sx, sy);
            if (e < gBest) { gBest = e; gSx = sx; gSy = sy; }
        }
    }

    // 2. Constrained local refinement ONLY on textured tiles (+-3 half-res around global)
    std::vector<float> rawDx((size_t)txCount * tyCount, (float)gSx * 2.0f);
    std::vector<float> rawDy((size_t)txCount * tyCount, (float)gSy * 2.0f);

    for (int ty = 0; ty < tyCount; ++ty) {
        for (int tx = 0; tx < txCount; ++tx) {
            int x0 = tx * TILE_HALF;
            int y0 = ty * TILE_HALF;

            // Measure tile texture energy so smooth walls stay locked to global shift
            float texEnergy = 0.0f;
            int tCnt = 0;
            int yMax = std::min(hh - 1, y0 + TILE_HALF);
            int xMax = std::min(hw - 1, x0 + TILE_HALF);
            for (int y = std::max(1, y0); y < yMax; ++y) {
                for (int x = std::max(1, x0); x < xMax; ++x) {
                    texEnergy += std::fabs(ref.lumaHalf[(size_t)y * hw + x + 1] - ref.lumaHalf[(size_t)y * hw + x - 1])
                               + std::fabs(ref.lumaHalf[(size_t)(y + 1) * hw + x] - ref.lumaHalf[(size_t)(y - 1) * hw + x]);
                    tCnt++;
                }
            }
            float avgTex = (tCnt > 0) ? (texEnergy / tCnt) : 0.0f;
            float globalErr = regionSad(ref.lumaHalf, frame.lumaHalf, hw, hh, x0, y0, TILE_HALF, TILE_HALF, gSx, gSy);

            int optX = gSx, optY = gSy;
            float bestErr = globalErr;

            if (avgTex > 0.012f) {
                for (int sy = gSy - 3; sy <= gSy + 3; ++sy) {
                    for (int sx = gSx - 3; sx <= gSx + 3; ++sx) {
                        float e = regionSad(ref.lumaHalf, frame.lumaHalf, hw, hh, x0, y0, TILE_HALF, TILE_HALF, sx, sy);
                        if (e < bestErr) {
                            bestErr = e;
                            optX = sx;
                            optY = sy;
                        }
                    }
                }
                // Only keep local deviation if it meaningfully beats global shift
                if (bestErr > globalErr * 0.90f) {
                    optX = gSx;
                    optY = gSy;
                    bestErr = globalErr;
                }
            }

            float errGrid[3][3];
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    errGrid[dy + 1][dx + 1] = regionSad(
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

            size_t idx = (size_t)ty * txCount + tx;
            rawDx[idx] = (optX + subX) * 2.0f;
            rawDy[idx] = (optY + subY) * 2.0f;
            frame.tileErr[idx] = bestErr;
        }
    }

    // 3. Light 3x3 spatial smoothing on tile shifts so borders/walls never show blocky seams
    for (int ty = 0; ty < tyCount; ++ty) {
        for (int tx = 0; tx < txCount; ++tx) {
            float sx = 0.0f, sy = 0.0f, wSum = 0.0f;
            for (int ky = -1; ky <= 1; ++ky) {
                int ny = clampInt(ty + ky, 0, tyCount - 1);
                for (int kx = -1; kx <= 1; ++kx) {
                    int nx = clampInt(tx + kx, 0, txCount - 1);
                    float w = (kx == 0 && ky == 0) ? 4.0f : 1.0f;
                    size_t nIdx = (size_t)ny * txCount + nx;
                    sx += rawDx[nIdx] * w;
                    sy += rawDy[nIdx] * w;
                    wSum += w;
                }
            }
            size_t idx = (size_t)ty * txCount + tx;
            frame.dx[idx] = sx / wSum;
            frame.dy[idx] = sy / wSum;
        }
    }
}

inline void sampleShiftAndErr(const FrameData& f, float rx, float ry, float& outDx, float& outDy, float& outErr) {
    float hx = clampFloat(rx * 0.5f / TILE_HALF - 0.5f, 0.0f, (float)(f.tilesX - 1));
    float hy = clampFloat(ry * 0.5f / TILE_HALF - 0.5f, 0.0f, (float)(f.tilesY - 1));
    int x0 = (int)hx, y0 = (int)hy;
    int x1 = std::min(x0 + 1, f.tilesX - 1);
    int y1 = std::min(y0 + 1, f.tilesY - 1);
    float fx = hx - x0, fy = hy - y0;
    size_t i00 = (size_t)y0 * f.tilesX + x0;
    size_t i10 = (size_t)y0 * f.tilesX + x1;
    size_t i01 = (size_t)y1 * f.tilesX + x0;
    size_t i11 = (size_t)y1 * f.tilesX + x1;
    outDx = (1 - fy) * ((1 - fx) * f.dx[i00] + fx * f.dx[i10]) + fy * ((1 - fx) * f.dx[i01] + fx * f.dx[i11]);
    outDy = (1 - fy) * ((1 - fx) * f.dy[i00] + fx * f.dy[i10]) + fy * ((1 - fx) * f.dy[i01] + fx * f.dy[i11]);
    outErr = (1 - fy) * ((1 - fx) * f.tileErr[i00] + fx * f.tileErr[i10]) + fy * ((1 - fx) * f.tileErr[i01] + fx * f.tileErr[i11]);
}

inline float sampleLumaBilinear(const std::vector<float>& luma, int hw, int hh, float rx, float ry) {
    float hx = clampFloat(rx * 0.5f - 0.5f, 0.0f, (float)(hw - 1));
    float hy = clampFloat(ry * 0.5f - 0.5f, 0.0f, (float)(hh - 1));
    int x0 = (int)hx, y0 = (int)hy;
    int x1 = std::min(x0 + 1, hw - 1);
    int y1 = std::min(y0 + 1, hh - 1);
    float fx = hx - x0, fy = hy - y0;
    return (1 - fy) * ((1 - fx) * luma[(size_t)y0 * hw + x0] + fx * luma[(size_t)y0 * hw + x1])
         + fy       * ((1 - fx) * luma[(size_t)y1 * hw + x0] + fx * luma[(size_t)y1 * hw + x1]);
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
    jfloat /* superResScale */,
    jfloat sharpenAmount,
    jobject outBitmap
) {
    const int numFrames = env->GetArrayLength(rawFrames);
    if (numFrames <= 0 || width <= 0 || height <= 0) return -1;

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
    if (std::fabs(ccm[0]) + std::fabs(ccm[4]) + std::fabs(ccm[8]) < 0.2f) {
        ccm[0] = 1.0f; ccm[1] = 0.0f; ccm[2] = 0.0f;
        ccm[3] = 0.0f; ccm[4] = 1.0f; ccm[5] = 0.0f;
        ccm[6] = 0.0f; ccm[7] = 0.0f; ccm[8] = 1.0f;
    }

    AndroidBitmapInfo bmpInfo;
    if (AndroidBitmap_getInfo(env, outBitmap, &bmpInfo) != ANDROID_BITMAP_RESULT_SUCCESS) return -2;
    const int outW = static_cast<int>(bmpInfo.width);
    const int outH = static_cast<int>(bmpInfo.height);
    const int outStridePx = static_cast<int>(bmpInfo.stride / 4);

    void* bmpPixelsVoid = nullptr;
    if (AndroidBitmap_lockPixels(env, outBitmap, &bmpPixelsVoid) != ANDROID_BITMAP_RESULT_SUCCESS) return -3;
    uint32_t* outPixels = static_cast<uint32_t*>(bmpPixelsVoid);

    float z = std::max(1.0f, zoomFactor);
    int roiW = clampInt(((int)(width / z)) & ~1, 128, width);
    int roiH = clampInt(((int)(height / z)) & ~1, 128, height);
    int roiX0 = ((width - roiW) / 2) & ~1;
    int roiY0 = ((height - roiH) / 2) & ~1;

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
        const uint16_t* row = frames[0].raw + (size_t)y * width;
        for (int x = roiX0; x < roiX0 + roiW; x += 16) {
            if (row[x] > sampleMax) sampleMax = row[x];
        }
    }
    float effectiveWhite = whiteLevel;
    float blScale = 1.0f;
    if (sampleMax > whiteLevel * 1.5f) {
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
    const int numThreads = std::max(2, std::min(6, (int)std::thread::hardware_concurrency()));

    for (int f = 0; f < numFrames; ++f) {
        frames[f].lumaHalf.resize((size_t)hw * hh);
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
                    const uint16_t* r0 = rawPtr + (size_t)y * width;
                    const uint16_t* r1 = rawPtr + (size_t)(y + 1) * width;
                    for (int hx = 0; hx < hw; ++hx) {
                        int x = roiX0 + hx * 2;
                        float p00 = std::max(0.0f, (r0[x] - blAdj[0]) * invRange[0]);
                        float p10 = std::max(0.0f, (r0[x + 1] - blAdj[1]) * invRange[1]);
                        float p01 = std::max(0.0f, (r1[x] - blAdj[2]) * invRange[2]);
                        float p11 = std::max(0.0f, (r1[x + 1] - blAdj[3]) * invRange[3]);
                        luma[(size_t)hy * hw + hx] = 0.25f * (p00 + p10 + p01 + p11) * mult;
                    }
                }
            });
        }
        for (auto& w : workers) w.join();
        frames[f].sharpness = computeSharpness(frames[f].lumaHalf, hw, hh);
    }

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
    if (bestLongIdx >= 0 && bestLongSharp >= bestFastSharp * 0.72f) {
        refIdx = bestLongIdx;
    }

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
                frames[f].dx.assign((size_t)frames[f].tilesX * frames[f].tilesY, 0.0f);
                frames[f].dy.assign((size_t)frames[f].tilesX * frames[f].tilesY, 0.0f);
                frames[f].tileErr.assign((size_t)frames[f].tilesX * frames[f].tilesY, 0.0f);
                continue;
            }
            workers.emplace_back([&, f]() {
                alignFrameToReference(frames[refIdx], refL1, r1w, r1h, refL2, r2w, r2h, frames[f], hw, hh);
            });
        }
        for (auto& w : workers) w.join();
    }

    std::vector<float>().swap(refL1);
    std::vector<float>().swap(refL2);

    std::vector<int> hist(256, 0);
    int totalSamples = 0;
    for (int y = 0; y < hh; y += 2) {
        for (int x = 0; x < hw; x += 2) {
            float v = frames[refIdx].lumaHalf[(size_t)y * hw + x];
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

    // Reject shaky LONG frames if shot handheld!
    float longFloor = std::max(bestLongSharp * 0.65f, bestFastSharp * 0.68f);
    float fastFloor = bestFastSharp * 0.65f;
    std::vector<float> frameBaseWeight(numFrames, 1.0f);
    for (int f = 0; f < numFrames; ++f) {
        if (frames[f].type == 1) {
            frameBaseWeight[f] = (frames[f].sharpness < longFloor) ? 0.0f : 3.5f;
        } else {
            frameBaseWeight[f] = (frames[f].sharpness < fastFloor) ? 0.0f : 1.0f;
        }
    }
    frameBaseWeight[refIdx] = std::max(1.0f, frameBaseWeight[refIdx]);

    const float invScaleX = (float)roiW / (float)outW;
    const float invScaleY = (float)roiH / (float)outH;
    const std::vector<float>& refLuma = frames[refIdx].lumaHalf;
    const std::vector<float>& fastLuma = frames[bestFastIdx].lumaHalf;
    const float shp = clampFloat(sharpenAmount
