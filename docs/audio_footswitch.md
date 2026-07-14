# Audio and footswitch processing

Milestone 4 adds interpretable behavioral audio and foot-contact baselines after the explicit clock
alignment created in Milestone 3.

## Audio loading

WAV files are read directly, downmixed to mono, normalized to floating point, and polyphase
resampled to the configured rate. Other containers require both `ffmpeg` and `ffprobe`; the former
decodes mono float samples and the latter records the original sampling rate.

QC includes original/output rates, sample and frame counts, duration, clipping ratio, VAD threshold,
speech interval count, speech duration, and speech ratio.

## Energy VAD

The waveform is divided into fixed frames. Each frame is speech-active when:

```text
RMS >= 10 ** (energy_threshold_dbfs / 20)
```

Speech runs shorter than `minimum_speech_duration_s` are removed. Internal silence gaps shorter
than `minimum_pause_duration_s` are merged into speech. Both raw frame energy/labels and merged
speech/silence intervals are saved on native and IMU-reference clocks.

This is deliberately simple and inspectable. It is not ASR, speaker diarization, language
understanding, or proof that an interval contains human speech.

`first_response_latency` is calculated only when `audio.task_start_seconds` is explicitly set.
Response count, correctness, and accuracy remain blank without transcript or manual labels.

## Footswitch stabilization

Synchronized left/right channels are thresholded into binary contacts. Contact pulses shorter than
`minimum_contact_duration_s` are removed. Internal zero gaps shorter than
`minimum_swing_duration_s` are filled. The process runs twice to stabilize neighboring short runs
and records the number of changed samples.

Rising transitions are named `contact`; falling transitions are named `toe_off`. These are signal
transition labels, not validated biomechanical heel-strike/toe-off claims.

## Timing features

Contact-to-next-toe-off pairs estimate stance. Toe-off-to-next-contact pairs estimate swing.
Consecutive contacts across both sides estimate step intervals. Pairing is performed independently
within each trial/phase window, so boundary-crossing partial cycles are not fabricated.

## IMU event agreement

IMU vertical-acceleration peaks are treated as predictions and footswitch contacts as the reference
for this software baseline. Candidate pairs within `event_matching_tolerance_s` are sorted by
absolute error and greedily assigned one to one. The pipeline reports:

- matched count;
- precision = matches / IMU events;
- recall = matches / footswitch events;
- F1;
- mean absolute matched timing error.

Trial agreement combines annotated straight-walking phases only. Real-data interpretation requires
reporting peak prominence, minimum IMU event interval, debounce durations, threshold, and matching
tolerance.
