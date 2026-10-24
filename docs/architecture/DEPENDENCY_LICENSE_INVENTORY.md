# Dependency license inventory

Release-readiness snapshot, reviewed 2026-10-01. This is an inventory of declared
package and asset terms, not legal advice or a complete third-party notice file.
Package versions below are the resolutions in [`uv.lock`](../../uv.lock), not the
open-ended minimums in [`pyproject.toml`](../../pyproject.toml). A package's
metadata declaration does not establish licenses for bundled third-party code,
model weights, or training data.

## Install paths

- Native base install: six mandatory Python dependencies.
- Default Docker/Compose image: the base plus `corpus`, `vectors`, and `mcp`
  extras. Docker explicitly uses `--no-dev` and does not install `nlp` by default.
- Optional Docker NLP image: adds `spacy` and `stanza`; on Linux, the locked
  Stanza/PyTorch closure includes CUDA runtime wheels even though the image does
  not promise GPU execution.
- `secrets` is a native opt-in extra. `dev` is not installed by the Dockerfile.
- OCR is a separate Docker build argument, off by default; it adds unpinned
  Debian `tesseract-ocr`, `tesseract-ocr-eng`, and `tesseract-ocr-spa` packages.

## Direct Python dependencies

“Declared” means the exact PyPI release metadata was checked. “Corroborated” also
means an upstream license file, model record, or vendor documentation was
checked. Neither status is a compatibility opinion.

| Package and resolved version | Distribution path | Declared terms and verification | Release implication |
| --- | --- | --- | --- |
| [`click` 8.5.0](https://pypi.org/project/click/8.5.0/) | Required base; CLI | BSD-3-Clause; declared | Carry its license and notices in binary/package distributions. |
| [`defusedxml` 0.7.1](https://pypi.org/project/defusedxml/0.7.1/) | Required base; XML parsing | PSF License; corroborated by the [upstream license](https://github.com/tiran/defusedxml/blob/main/LICENSE) | Retain PSF terms and notice. |
| [`pydantic` 2.13.5](https://pypi.org/project/pydantic/2.13.5/) | Required base; data models | MIT; declared | Include the MIT text and copyright notice. |
| [`PyMuPDF` 1.28.2](https://pypi.org/project/pymupdf/1.28.2/) | Required base; PDF extraction and optional OCR integration | Package metadata says GNU AGPL 3.0 **or** Artifex commercial license; corroborated by [PyMuPDF's licensing statement](https://github.com/pymupdf) | The public source path uses AGPL-3.0-only for the integrated first-party project. A customer wanting proprietary terms must arrange a separate Artifex commercial license; confirm Artifex coverage before shipping a prebuilt commercial installer/container. |
| [`PyYAML` 6.0.3](https://pypi.org/project/PyYAML/6.0.3/) | Required base; configuration | MIT; declared | Include the MIT text and copyright notice. |
| [`typer` 0.27.2](https://pypi.org/project/typer/0.27.2/) | Required base; CLI | MIT; declared | Include the MIT text and copyright notice. |
| [`duckdb` 1.5.5](https://pypi.org/project/duckdb/1.5.5/) | `corpus` extra; default Docker image | MIT; declared | Include the MIT text; check the selected wheel's notices for bundled components. |
| [`pyarrow` 25.0.1](https://pypi.org/project/pyarrow/25.0.1/) | `corpus` extra; default Docker image | Apache-2.0; declared | Include Apache-2.0 and applicable NOTICE files. |
| [`sqlite-vec` 0.1.9](https://pypi.org/project/sqlite-vec/0.1.9/) | `vectors` extra; default Docker image | MIT and Apache-2.0; corroborated by the [upstream repository](https://github.com/asg017/sqlite-vec) | Select a license option and include its text/notices. |
| [`mcp` 2.2.0](https://pypi.org/project/mcp/2.2.0/) | `mcp` extra; default Docker image | MIT; corroborated by the [official Python SDK](https://github.com/modelcontextprotocol/python-sdk/blob/main/LICENSE) | Include MIT text and notice. |
| [`spacy` 3.8.16](https://pypi.org/project/spacy/3.8.16/) | Optional `nlp` extra; backend library only | MIT; declared | Backend library terms do not cover separately installed language models. |
| [`stanza` 1.14.0](https://pypi.org/project/stanza/1.14.0/) | Optional `nlp` extra; backend library only | Apache-2.0; corroborated by the [upstream license](https://github.com/stanfordnlp/stanza/blob/main/LICENSE) | Stanza's code license does not establish model-weight or training-data terms. |
| [`keyring` 25.7.0](https://pypi.org/project/keyring/25.7.0/) | Optional `secrets` extra; native install | MIT; declared | Not in the default Docker install; include its MIT notice when the extra is distributed. |

## Material transitive dependencies

For CPython 3.12 Linux, the default Docker lock closure is 41 Python packages:
10 direct packages above and 31 transitives. PyPI metadata for those transitives
declares MIT, BSD, Apache-2.0, PSF, ISC, MIT-0, or MPL terms (including
`cryptography` 50.0.1, declared Apache-2.0 OR BSD-3-Clause). This is a metadata
check, not an audit of every wheel's bundled notices; `uv.lock` is the exact
package/version list to use when producing the image SBOM and notice bundle.

The Linux `nlp` image resolves 105 Python packages in that same profile: 12
direct and 93 transitive. The material additions include:

| Transitive package and resolved version | Declared terms / status | Distribution path and action |
| --- | --- | --- |
| [`udapi` 0.5.2](https://pypi.org/project/udapi/0.5.2/) | GPL-3.0-or-later | Installed through Stanza's dependency chain; review before redistributing the NLP image. |
| [`udtools` 0.2.8](https://pypi.org/project/udtools/0.2.8/) | GPL-2.0-or-later | Installed through Stanza's dependency chain; review before redistributing the NLP image. |
| [`torch` 2.14.0](https://pypi.org/project/torch/2.14.0/) | Apache-2.0, Apache-2.0 WITH LLVM-exception, BSD-2-Clause, BSD-3-Clause, BSL-1.0, and MIT; declared | Stanza dependency; preserve all notices in the selected wheel. |
| `cuda-toolkit` 13.0.3.0; `nvidia-cuda-runtime` 13.0.96; `nvidia-cudnn-cu13` 9.24.0.43; `nvidia-nccl-cu13` 2.30.7; `nvidia-nvshmem-cu13` 3.4.5 | PyPI license metadata unspecified for these exact releases | Present in the locked Linux NLP image. Confirm each wheel's license files and redistribution terms before publishing that image. |
| `nvidia-cublas` 13.1.1.3; `nvidia-cuda-cupti` 13.0.85; `nvidia-cuda-nvrtc` 13.0.88; `nvidia-cufft` 12.0.0.61; `nvidia-cufile` 1.15.1.6; `nvidia-curand` 10.4.0.35; `nvidia-cusolver` 12.0.4.66; `nvidia-cusparse` 12.6.3.3; `nvidia-cusparselt-cu13` 0.8.1; `nvidia-nvjitlink` 13.4.92 | PyPI metadata identifies NVIDIA proprietary terms; `nvidia-nvtx` 13.0.85 declares Apache-2.0 | Review the [NVIDIA CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html) and exact wheel notices for this opt-in image. The lock resolves these wheels even though GPU execution is not configured. |

Other transitive packages in the NLP closure are pinned in `uv.lock`; their
exact license declarations still need to be included in a generated SBOM/notice
review for the target platform. Do not infer those terms from the spaCy or
Stanza library license.

## Models, data, and OCR assets

The README states that the optional NLP image installs backend libraries only:
it does not download or embed model weights. Model selection and Stanza resource
directories are supplied locally. The following terms therefore describe
documented runtime assets, not files currently shipped in the repository or
default image.

| Asset and documented version | Distribution path | Declared terms and verification | Release implication |
| --- | --- | --- | --- |
| [`en_core_web_sm` 3.8.0](https://github.com/explosion/spacy-models/blob/master/meta/en_core_web_sm-3.8.0.json) | Optional local spaCy model; mentioned by language capability docs | MIT in exact upstream model metadata; corroborated | If bundled, include its license and verify training-data notices separately. |
| [`es_core_news_sm` 3.8.0](https://github.com/explosion/spacy-models/blob/master/meta/es_core_news_sm-3.8.0.json) | Optional local spaCy model; README documents Spanish parsing with this package | GNU GPL 3.0 in exact upstream model metadata; corroborated | Do not treat spaCy's MIT license as covering this model. Review redistribution and product compatibility before bundling or making it a managed download; model/data lineage was not independently audited. |
| Stanza Arabic `ar-padt` resource package, used with Stanza 1.14.0 | Optional local Stanza resource directory; model bytes are not in the repository | Stanza's [1.14.0 resource manifest](https://github.com/stanfordnlp/stanza-resources/blob/main/resources_1.14.0.json) identifies the package, but does not declare model-weight terms. The upstream [UD Arabic-PADT corpus](https://github.com/UniversalDependencies/UD_Arabic-PADT) declares CC BY-NC-SA 3.0. | **Unresolved:** record exact resource manifest revision, model files and hashes; establish weight provenance/terms independently. The corpus terms are a lineage signal, not a conclusion about the weights' license. Review before redistributing the weights or relying on them in a commercial offering. |
| Tesseract engine and English/Spanish trained data | Optional OCR Docker build argument; apt package versions are not pinned in the Dockerfile | Tesseract upstream declares Apache-2.0 for the engine and [`tessdata_best`](https://github.com/tesseract-ocr/tessdata_best) declares Apache-2.0 for its models; exact Debian package versions/files in the image are not independently verified | Keep OCR off by default as configured. For an OCR image, capture the resolved Debian packages and notices in its image SBOM; `apt-get update` makes these versions build-time dependent. |

## Container and release follow-up

The final Docker image is based on digest-pinned `python:3.12-slim` and copies
`uv`/`uvx` 0.12.19 binaries from the digest-pinned Astral image. Astral declares
[`uv`](https://github.com/astral-sh/uv) under MIT OR Apache-2.0. The final image
also contains the Python/Debian base package closure, which is outside the
Python lock; copying only the `uv` executables means their license and bundled
component notices must be checked in the produced image. Compose adds no new
third-party runtime packages of its own.

Before release, preserve the mandatory PyMuPDF AGPL notices, produce an
image-specific SBOM and notices for each shipped Docker profile (including
NLP/OCR only if those profiles will ship), and record exact model artifact
versions, hashes, license declarations, and data lineage for any models that
will be distributed. The repository applies SPDX `AGPL-3.0-only` to
DiscourseLens-authored software; this inventory does not make a legal
determination about every release artifact or confirm that all rights holders
can offer a separate commercial license. See the [Architecture Freeze
Candidate](ARCHITECTURE_FREEZE_CANDIDATE.md) and [licensing guide](../LICENSING.md).
