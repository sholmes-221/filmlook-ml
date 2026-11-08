# Serving benchmark for the MVP (development host now; instance pending)

## Development host, one pinned core (`taskset -c 0`), 2026-11-08

Realistic inputs: twelve originals re-encoded like the website does (long edge 2048 px, JPEG q90).

| measure | value |
|---|---|
| model load (both ONNX sessions, one thread each) | 0.16 s |
| per photo, v1 statistics + attributes + student heads | p50 259 ms, p95 303 ms, max 303 ms |
| peak RSS of the Python process (ORT + NumPy + Pillow + both models) | 165 MB |
| ONNX files | `best/model.onnx` 0.62 MB, `student_sup/model.onnx` 4.34 MB fp32 (the int8 export is unusable) |
| calibration and prototype artifacts | 3 KB |

Where the time goes: about 80 % is the 28 statistics at 1280 px (Pillow and NumPy), 10 % image decoding, the two
ONNX runs together under 5 ms. Memory is dominated by the decoded 1280 px image and the texture crops, not by the
models.

## Expected on VM.Standard.E2.1.Micro (1/8 OCPU of an E2 core, ~1 GB RAM)

The service benchmark of the style student on the same host versus the published E2 Micro experience in this
repository (`service/README.md`) suggests a factor of 4 to 6: plan for **1.0 to 1.8 s per photo** and the same
~165 MB RSS inside the API process (the API itself idles at ~35 MB; the container limit is 640 MB with swap). The
website's work gate already serialises image jobs on the Micro (`FILMLOOK_API_MAX_CONCURRENT=1`), so concurrency
cannot stack memory.

## What must be measured on the instance before any cutover

1. p50 / p95 latency over 20 real uploads through the API (`/api/film/analyze` in retrieval mode on a staging
   `.env`), with the work gate at 1.
2. Peak RSS of the app container (`docker stats`) during those requests and after an hour idle.
3. Cold start time of the API with the models loaded at startup.
4. Disk: the artifacts add ~5.2 MB to the image.
5. CPU: sysstat's 10-minute samples before and after the trial (the monitor already records them).

Gate: p95 < 3 s, peak RSS < 200 MB for the app process, no OOM kill, container healthy. The procedure is read-only
on production data and uses the owner's test account, whose analyses are not persisted.

## Not done yet

No measurement on the instance has been made (deploy of a staging mode needs the adapter in the website first).
Numbers above are from the development host and are labelled as such everywhere they are quoted.
