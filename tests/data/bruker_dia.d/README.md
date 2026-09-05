# Bruker DIA reader fixture

The unmodified `analysis.tdf` and `analysis.tdf_bin` files are copied from
[`MannLabs/timsrust/tests/dia_test.d`](https://github.com/MannLabs/timsrust/tree/5e09572fa3f1aace86e6ba64244a01fd0850fbf3/tests/dia_test.d),
at commit `5e09572fa3f1aace86e6ba64244a01fd0850fbf3`, under the upstream MIT license
reproduced in `LICENSE`.

This small upstream regression fixture exercises actual TDF binary decompression,
isolation-window metadata and ion-mobility metadata without an external Bruker SDK.
It contains six acquisition frames: two MS1 frames and four MS2 frames. The four
DIA isolation windows each occur twice, so the default pyOpenMS DIA export returns
ten spectra: two MS1 spectra and eight MS2 window spectra. The four m/z centers are
400, 600, 800 and 1000, each with a 50 m/z isolation width.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `analysis.tdf` | 24576 | `e8628dd512350c9f760a38bf31be5aaf502d9e293a0246e9c5b457a42df169df` |
| `analysis.tdf_bin` | 91204 | `663fc5e676ff98769670a56389108bd9fc1d2ebb06d98a3b09539713d984d8ad` |

Peak patterns in this fixture are for reader regression tests, not a biological
benchmark. Although the binary is small, it expands to several million peaks.
