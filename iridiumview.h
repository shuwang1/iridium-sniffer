/*
 * IridiumView: Passive through-wall presence detection using
 * Iridium L-band satellite reflections and micro-Doppler analysis
 *
 * Copyright (c) 2026 CEMAXECUTER LLC
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef __IRIDIUMVIEW_H__
#define __IRIDIUMVIEW_H__

#include <complex.h>
#include <stdint.h>

/* Maximum simultaneous satellite illuminators tracked */
#define IV_MAX_ILLUMINATORS     8

/* CIR estimation window (samples) */
#define IV_CIR_WINDOW           4096

/* Micro-Doppler FFT size for breathing/movement detection */
#define IV_MDOPPLER_FFT         256

/* Minimum bursts per satellite for valid CIR estimate */
#define IV_MIN_BURSTS_CIR       16

/* Classifier output labels */
typedef enum {
    IV_CLASS_EMPTY = 0,
    IV_CLASS_STATIONARY,
    IV_CLASS_WALKING,
    IV_CLASS_SEATED,
    IV_CLASS_LYING,
    IV_CLASS_MULTIPLE,
    IV_CLASS_UNKNOWN,
    IV_NUM_CLASSES
} iv_classification_t;

/* Per-satellite illuminator state */
typedef struct {
    int      sat_id;
    double   elevation_deg;
    double   azimuth_deg;
    float    cir_magnitude[IV_CIR_WINDOW];
    float    cir_phase[IV_CIR_WINDOW];
    float    cir_baseline[IV_CIR_WINDOW];
    float    cir_delta[IV_CIR_WINDOW];
    int      burst_count;
    uint64_t last_burst_ts;
    float    snr_db;
    int      calibrated;
} iv_illuminator_t;

/* Micro-Doppler accumulator */
typedef struct {
    float complex *slow_time_buffer;    /* burst-to-burst phase history */
    int            slow_time_len;
    int            slow_time_write;
    float         *doppler_spectrum;    /* output of micro-Doppler FFT */
    float          breathing_freq_hz;   /* detected respiratory rate */
    float          breathing_confidence;
    float          movement_energy;     /* total sub-Nyquist energy */
} iv_mdoppler_t;

/* Main IridiumView context */
typedef struct {
    iv_illuminator_t illuminators[IV_MAX_ILLUMINATORS];
    int              n_illuminators;
    iv_mdoppler_t    mdoppler;

    /* TFLite inference */
    void            *tflite_model;
    void            *tflite_interpreter;
    float           *classifier_input;   /* flattened CIR delta features */
    float            classifier_output[IV_NUM_CLASSES];
    iv_classification_t current_class;
    float            class_confidence;
    int              occupant_count;

    /* Calibration state */
    int              calibrating;
    int              calibration_secs;
    int              calibration_elapsed;

    /* Stats */
    uint64_t         n_cir_updates;
    uint64_t         n_classifications;
    uint64_t         n_presence_events;
    uint64_t         n_breathing_detections;

    /* Web dashboard state */
    float            heatmap[16][16];    /* coarse spatial grid */
    int              heatmap_valid;
} iridiumview_t;

/*
 * Initialize IridiumView context. Loads TFLite model from the models/
 * directory adjacent to the executable.
 * Returns 0 on success, -1 on failure.
 */
int iridiumview_init(iridiumview_t *iv, int calibration_secs);

/*
 * Feed a decoded burst's raw correlation output for CIR estimation.
 * Called from the burst downmix pipeline after correlation peak detection.
 *
 * sat_id:       satellite identifier from IRA frame
 * corr_output:  complex correlation output (already computed for frame sync)
 * corr_len:     length of correlation buffer
 * timestamp_ns: burst detection timestamp
 * elevation:    satellite elevation angle (from IRA orbital data)
 * azimuth:      satellite azimuth angle
 */
int iridiumview_feed_correlation(iridiumview_t *iv, int sat_id,
                                  const float complex *corr_output,
                                  int corr_len, uint64_t timestamp_ns,
                                  double elevation, double azimuth);

/*
 * Run micro-Doppler analysis on accumulated slow-time phase history.
 * Should be called periodically (every 1-2 seconds).
 */
int iridiumview_micro_doppler(iridiumview_t *iv);

/*
 * Run TFLite classifier on current CIR delta features.
 * Updates iv->current_class, iv->class_confidence, iv->occupant_count.
 */
int iridiumview_classify(iridiumview_t *iv);

/*
 * Get JSON-formatted presence status for web dashboard.
 * Caller provides buffer. Returns bytes written.
 */
int iridiumview_json_status(const iridiumview_t *iv, char *buf, int bufsz);

/*
 * Get heatmap data for web dashboard spatial overlay.
 * Returns pointer to 16x16 float grid (0.0 = no presence, 1.0 = high confidence).
 */
const float *iridiumview_heatmap(const iridiumview_t *iv);

/*
 * Destroy context and free TFLite resources.
 */
void iridiumview_destroy(iridiumview_t *iv);

#endif
