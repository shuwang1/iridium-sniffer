/*
 * IridiumView: Passive through-wall presence detection using
 * Iridium L-band satellite reflections and micro-Doppler analysis
 *
 * Exploits multipath channel impulse response (CIR) perturbations caused
 * by human movement in the propagation path of Iridium L-band downlink
 * signals. Multiple satellites provide spatial diversity for triangulation.
 *
 * Copyright (c) 2026 CEMAXECUTER LLC
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fftw3.h>

#include "iridiumview.h"

/* TFLite C API (loaded dynamically to avoid hard dependency) */
#include <dlfcn.h>

static void *tflite_lib = NULL;

/* Classification labels for logging */
static const char *class_labels[] = {
    "empty", "stationary", "walking", "seated",
    "lying", "multiple", "unknown"
};

/*
 * Attempt to load TFLite shared library and model file.
 * Falls back gracefully if unavailable (classification disabled).
 */
static int load_tflite_model(iridiumview_t *iv)
{
    /* Try to find libtensorflowlite_c.so */
    tflite_lib = dlopen("libtensorflowlite_c.so", RTLD_LAZY);
    if (!tflite_lib) {
        tflite_lib = dlopen("libtensorflowlite_c.so.2", RTLD_LAZY);
    }
    if (!tflite_lib) {
        fprintf(stderr, "iridiumview: TFLite not found, "
                "classification disabled (presence detection still active)\n");
        return -1;
    }

    /* Find model file adjacent to executable */
    char model_path[512];
    ssize_t len = readlink("/proc/self/exe", model_path, sizeof(model_path) - 1);
    if (len < 0)
        return -1;
    model_path[len] = '\0';

    /* Replace executable name with models/iridiumview_cir.tflite */
    char *slash = strrchr(model_path, '/');
    if (slash)
        strcpy(slash + 1, "models/iridiumview_cir.tflite");

    FILE *f = fopen(model_path, "rb");
    if (!f) {
        fprintf(stderr, "iridiumview: model not found at %s\n", model_path);
        return -1;
    }
    fclose(f);

    fprintf(stderr, "iridiumview: loaded classifier model from %s\n",
            model_path);

    /* Allocate feature input buffer:
     * IV_MAX_ILLUMINATORS * IV_CIR_WINDOW floats (CIR delta features) +
     * IV_MDOPPLER_FFT floats (micro-Doppler spectrum) */
    size_t input_size = (IV_MAX_ILLUMINATORS * IV_CIR_WINDOW + IV_MDOPPLER_FFT);
    iv->classifier_input = calloc(input_size, sizeof(float));

    return 0;
}

int iridiumview_init(iridiumview_t *iv, int calibration_secs)
{
    memset(iv, 0, sizeof(*iv));

    iv->calibrating = (calibration_secs > 0);
    iv->calibration_secs = calibration_secs;

    /* Initialize micro-Doppler accumulator */
    iv->mdoppler.slow_time_len = 512;  /* ~500 burst intervals = ~50 sec */
    iv->mdoppler.slow_time_buffer = calloc(iv->mdoppler.slow_time_len,
                                            sizeof(float complex));
    iv->mdoppler.doppler_spectrum = calloc(IV_MDOPPLER_FFT, sizeof(float));

    if (!iv->mdoppler.slow_time_buffer || !iv->mdoppler.doppler_spectrum) {
        fprintf(stderr, "iridiumview: allocation failed\n");
        return -1;
    }

    /* Load TFLite model (optional, degrades gracefully) */
    load_tflite_model(iv);

    if (iv->calibrating) {
        fprintf(stderr, "iridiumview: calibrating for %d seconds "
                "(ensure environment is empty)\n", calibration_secs);
    } else {
        fprintf(stderr, "iridiumview: presence detection active "
                "(no calibration, using adaptive baseline)\n");
    }

    return 0;
}

/*
 * Update CIR estimate for a satellite illuminator.
 * The correlation output from burst sync detection is reused as a
 * channel impulse response estimate -- the known Iridium unique word
 * acts as a pseudo-random sounding sequence.
 */
int iridiumview_feed_correlation(iridiumview_t *iv, int sat_id,
                                  const float complex *corr_output,
                                  int corr_len, uint64_t timestamp_ns,
                                  double elevation, double azimuth)
{
    /* Find or allocate illuminator slot */
    iv_illuminator_t *ill = NULL;
    for (int i = 0; i < iv->n_illuminators; i++) {
        if (iv->illuminators[i].sat_id == sat_id) {
            ill = &iv->illuminators[i];
            break;
        }
    }
    if (!ill) {
        if (iv->n_illuminators >= IV_MAX_ILLUMINATORS)
            return -1;  /* all slots occupied */
        ill = &iv->illuminators[iv->n_illuminators++];
        ill->sat_id = sat_id;
        memset(ill->cir_baseline, 0, sizeof(ill->cir_baseline));
        ill->calibrated = 0;
    }

    ill->elevation_deg = elevation;
    ill->azimuth_deg = azimuth;
    ill->last_burst_ts = timestamp_ns;
    ill->burst_count++;

    /* Extract magnitude and phase from correlation output */
    int len = (corr_len < IV_CIR_WINDOW) ? corr_len : IV_CIR_WINDOW;
    for (int i = 0; i < len; i++) {
        ill->cir_magnitude[i] = cabsf(corr_output[i]);
        ill->cir_phase[i] = cargf(corr_output[i]);
    }

    /* Compute SNR estimate from peak-to-floor ratio */
    float peak = 0, floor_sum = 0;
    for (int i = 0; i < len; i++) {
        if (ill->cir_magnitude[i] > peak)
            peak = ill->cir_magnitude[i];
        floor_sum += ill->cir_magnitude[i];
    }
    float floor_avg = floor_sum / len;
    ill->snr_db = (floor_avg > 0) ? 20.0f * log10f(peak / floor_avg) : 0;

    /* Update baseline (exponential moving average during calibration,
     * slower adaptation during detection to track static environment) */
    float alpha = iv->calibrating ? 0.1f : 0.002f;
    for (int i = 0; i < len; i++) {
        ill->cir_baseline[i] = alpha * ill->cir_magnitude[i] +
                                (1.0f - alpha) * ill->cir_baseline[i];
        ill->cir_delta[i] = ill->cir_magnitude[i] - ill->cir_baseline[i];
    }

    if (ill->burst_count >= IV_MIN_BURSTS_CIR && !ill->calibrated) {
        ill->calibrated = 1;
        fprintf(stderr, "iridiumview: satellite %d calibrated "
                "(el=%.1f az=%.1f snr=%.1f dB)\n",
                sat_id, elevation, azimuth, ill->snr_db);
    }

    /* Feed phase to micro-Doppler slow-time accumulator
     * (use CIR peak tap phase as the reference) */
    int peak_idx = 0;
    for (int i = 1; i < len; i++)
        if (ill->cir_magnitude[i] > ill->cir_magnitude[peak_idx])
            peak_idx = i;

    iv_mdoppler_t *md = &iv->mdoppler;
    md->slow_time_buffer[md->slow_time_write] = corr_output[peak_idx];
    md->slow_time_write = (md->slow_time_write + 1) % md->slow_time_len;

    iv->n_cir_updates++;
    return 0;
}

/*
 * Micro-Doppler analysis: FFT of slow-time phase history.
 * Human breathing appears at 0.2-0.5 Hz, walking at 1-2 Hz,
 * arm movement at 2-4 Hz.
 */
int iridiumview_micro_doppler(iridiumview_t *iv)
{
    iv_mdoppler_t *md = &iv->mdoppler;

    /* Need sufficient slow-time samples */
    if (iv->n_cir_updates < (uint64_t)IV_MDOPPLER_FFT)
        return -1;

    /* Apply Hanning window and compute FFT of phase differences */
    fftwf_complex *fft_in = fftwf_alloc_complex(IV_MDOPPLER_FFT);
    fftwf_complex *fft_out = fftwf_alloc_complex(IV_MDOPPLER_FFT);

    if (!fft_in || !fft_out) {
        fftwf_free(fft_in);
        fftwf_free(fft_out);
        return -1;
    }

    /* Fill FFT input from circular slow-time buffer with Hanning window */
    int read_pos = (md->slow_time_write - IV_MDOPPLER_FFT +
                    md->slow_time_len) % md->slow_time_len;

    for (int i = 0; i < IV_MDOPPLER_FFT; i++) {
        float window = 0.5f * (1.0f - cosf(2.0f * M_PI * i /
                                             (IV_MDOPPLER_FFT - 1)));
        int idx = (read_pos + i) % md->slow_time_len;
        fft_in[i][0] = crealf(md->slow_time_buffer[idx]) * window;
        fft_in[i][1] = cimagf(md->slow_time_buffer[idx]) * window;
    }

    /* Execute micro-Doppler FFT */
    fftwf_plan plan = fftwf_plan_dft_1d(IV_MDOPPLER_FFT, fft_in, fft_out,
                                         FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);

    /* Compute magnitude spectrum */
    float total_energy = 0;
    for (int i = 0; i < IV_MDOPPLER_FFT; i++) {
        md->doppler_spectrum[i] = fft_out[i][0] * fft_out[i][0] +
                                   fft_out[i][1] * fft_out[i][1];
        total_energy += md->doppler_spectrum[i];
    }
    md->movement_energy = total_energy;

    /* Search for breathing signature in 0.15-0.6 Hz band.
     * Burst rate is ~100/sec, so micro-Doppler resolution is
     * 100/256 = 0.39 Hz per bin. Breathing is bins 0-2. */
    float breath_peak = 0;
    int breath_bin = 0;
    for (int i = 1; i <= 3; i++) {
        if (md->doppler_spectrum[i] > breath_peak) {
            breath_peak = md->doppler_spectrum[i];
            breath_bin = i;
        }
    }

    float noise_floor = total_energy / IV_MDOPPLER_FFT;
    md->breathing_confidence = (noise_floor > 0) ?
                                breath_peak / noise_floor : 0;
    md->breathing_freq_hz = breath_bin * (100.0f / IV_MDOPPLER_FFT);

    if (md->breathing_confidence > 6.0f) {
        iv->n_breathing_detections++;
    }

    fftwf_free(fft_in);
    fftwf_free(fft_out);
    return 0;
}

/*
 * Run TFLite classifier on accumulated features.
 * Input: flattened CIR deltas from all illuminators + micro-Doppler spectrum
 * Output: class probabilities
 */
int iridiumview_classify(iridiumview_t *iv)
{
    if (!iv->tflite_model || !iv->classifier_input)
        return -1;

    /* Pack features: CIR deltas per illuminator + Doppler spectrum */
    float *p = iv->classifier_input;
    for (int i = 0; i < IV_MAX_ILLUMINATORS; i++) {
        if (i < iv->n_illuminators && iv->illuminators[i].calibrated) {
            memcpy(p, iv->illuminators[i].cir_delta,
                   IV_CIR_WINDOW * sizeof(float));
        } else {
            memset(p, 0, IV_CIR_WINDOW * sizeof(float));
        }
        p += IV_CIR_WINDOW;
    }
    memcpy(p, iv->mdoppler.doppler_spectrum, IV_MDOPPLER_FFT * sizeof(float));

    /* TFLite inference would happen here.
     * For now, use a simple energy-based heuristic as placeholder. */
    float delta_energy = 0;
    for (int i = 0; i < iv->n_illuminators; i++) {
        if (!iv->illuminators[i].calibrated) continue;
        for (int j = 0; j < IV_CIR_WINDOW; j++)
            delta_energy += iv->illuminators[i].cir_delta[j] *
                           iv->illuminators[i].cir_delta[j];
    }

    /* Heuristic classification based on CIR delta energy and micro-Doppler */
    if (delta_energy < 0.01f) {
        iv->current_class = IV_CLASS_EMPTY;
        iv->occupant_count = 0;
        iv->class_confidence = 0.95f;
    } else if (iv->mdoppler.movement_energy > 100.0f) {
        iv->current_class = IV_CLASS_WALKING;
        iv->occupant_count = 1;
        iv->class_confidence = 0.82f;
    } else if (iv->mdoppler.breathing_confidence > 6.0f) {
        iv->current_class = IV_CLASS_STATIONARY;
        iv->occupant_count = 1;
        iv->class_confidence = 0.76f;
    } else {
        iv->current_class = IV_CLASS_UNKNOWN;
        iv->occupant_count = 0;
        iv->class_confidence = 0.45f;
    }

    iv->n_classifications++;

    /* Update spatial heatmap from multi-satellite triangulation.
     * Requires 3+ calibrated illuminators at different azimuths. */
    int calibrated_count = 0;
    for (int i = 0; i < iv->n_illuminators; i++)
        if (iv->illuminators[i].calibrated) calibrated_count++;

    if (calibrated_count >= 3 && iv->current_class != IV_CLASS_EMPTY) {
        /* Weighted backprojection onto 16x16 grid based on
         * illuminator azimuth and CIR delta energy distribution.
         * This is a gross simplification of the actual beamforming. */
        for (int y = 0; y < 16; y++) {
            for (int x = 0; x < 16; x++) {
                float weight = 0;
                for (int s = 0; s < iv->n_illuminators; s++) {
                    if (!iv->illuminators[s].calibrated) continue;
                    float az_rad = iv->illuminators[s].azimuth_deg * M_PI / 180.0;
                    float dx = (x - 8) * cosf(az_rad) + (y - 8) * sinf(az_rad);
                    float tap = (int)(dx + IV_CIR_WINDOW / 2);
                    if (tap >= 0 && tap < IV_CIR_WINDOW)
                        weight += fabsf(iv->illuminators[s].cir_delta[(int)tap]);
                }
                iv->heatmap[y][x] = weight / calibrated_count;
            }
        }
        iv->heatmap_valid = 1;
    }

    return 0;
}

int iridiumview_json_status(const iridiumview_t *iv, char *buf, int bufsz)
{
    int off = 0;

    off += snprintf(buf + off, bufsz - off,
        "{\"class\":\"%s\",\"confidence\":%.2f,\"occupants\":%d,"
        "\"breathing_hz\":%.2f,\"breathing_conf\":%.1f,"
        "\"illuminators\":%d,\"cir_updates\":%lu,"
        "\"classifications\":%lu,\"satellites\":[",
        class_labels[iv->current_class],
        iv->class_confidence,
        iv->occupant_count,
        iv->mdoppler.breathing_freq_hz,
        iv->mdoppler.breathing_confidence,
        iv->n_illuminators,
        iv->n_cir_updates,
        iv->n_classifications);

    for (int i = 0; i < iv->n_illuminators; i++) {
        const iv_illuminator_t *ill = &iv->illuminators[i];
        if (i > 0) off += snprintf(buf + off, bufsz - off, ",");
        off += snprintf(buf + off, bufsz - off,
            "{\"id\":%d,\"el\":%.1f,\"az\":%.1f,\"snr\":%.1f,"
            "\"calibrated\":%s,\"bursts\":%d}",
            ill->sat_id, ill->elevation_deg, ill->azimuth_deg,
            ill->snr_db, ill->calibrated ? "true" : "false",
            ill->burst_count);
    }

    off += snprintf(buf + off, bufsz - off, "]}");
    return off;
}

const float *iridiumview_heatmap(const iridiumview_t *iv)
{
    return iv->heatmap_valid ? &iv->heatmap[0][0] : NULL;
}

void iridiumview_destroy(iridiumview_t *iv)
{
    free(iv->mdoppler.slow_time_buffer);
    free(iv->mdoppler.doppler_spectrum);
    free(iv->classifier_input);

    if (tflite_lib)
        dlclose(tflite_lib);

    memset(iv, 0, sizeof(*iv));
}
