from dataclasses import asdict

from app.audio_utils import resample_audio
from app.vad_segmenter import VADSegmenter


def asr_worker_thread(state, settings, transcriber, diarizer):
    def new_segmenter():
        return VADSegmenter(
            settings.target_sr,
            settings.frame_ms,
            settings.vad_aggressiveness,
            settings.min_speech_sec,
            settings.max_segment_sec,
            settings.end_silence_sec,
            settings.pre_roll_sec,
        )

    segmenter = new_segmenter()
    state.worker_running = True
    print("ASR worker thread started.")

    while True:
        item = state.audio_queue.get()

        try:
            state.server_state["worker_items_processed"] += 1

            if state.reset_requested:
                segmenter = new_segmenter()
                state.reset_requested = False
                print("ASR worker reset: new VAD segmenter created.")

            raw_audio = item["audio"]
            source_sr = item["source_sr"]

            audio_16k = resample_audio(raw_audio, source_sr, settings.target_sr)

            diarizer.append_audio(audio_16k)

            speech_segments = segmenter.process_audio(audio_16k)

            if len(speech_segments) > 0:
                state.server_state["vad_segments_created"] += len(speech_segments)
                print("VAD created segments:", len(speech_segments))

            for segment in speech_segments:
                diarizer.maybe_update_diarization()

                record = transcriber.transcribe_speech_segment(segment, diarizer)

                if record is None:
                    state.server_state["segments_filtered"] += 1
                    print("Segment filtered or empty Whisper output.")
                else:
                    state.server_state["segments_transcribed"] += 1
                    state.transcripts.append(record)
                    state.metrics_rows.append(asdict(record))

                    with open("transcription_log.txt", "a", encoding="utf-8") as f:
                        f.write(
                            f"[{record.timestamp}] {record.speaker} "
                            f"{record.start_sec:.2f}-{record.end_sec:.2f}s | "
                            f"latency={record.latency_sec:.3f}s | "
                            f"RTF={record.rtf:.3f} | {record.text}\n"
                        )

                    print(
                        f"[{record.timestamp}] "
                        f"{record.speaker}: {record.text} | "
                        f"{record.start_sec:.2f}-{record.end_sec:.2f}s | "
                        f"latency={record.latency_sec:.3f}s | "
                        f"RTF={record.rtf:.3f}"
                    )

        except Exception as e:
            state.server_state["last_error"] = str(e)
            print("ASR worker error:", e)

        finally:
            state.audio_queue.task_done()
