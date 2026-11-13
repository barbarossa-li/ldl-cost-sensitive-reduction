# SCUT-FBP5500 source and cache policy

**Reuse existing data first.** The official archive was downloaded and verified on 13 November 2026. Its unchanged bytes are retained in the author's private Google Drive folder `ASC_LDL_实验数据缓存（私有）`. Because the Drive transfer connector limits individual files to 100 MiB, this private cache contains two byte-concatenated parts plus `CACHE_MANIFEST.json` and `restore_cache.py`. Reassembly was verified against the original ZIP hash.

## Future work in this project

1. Check the current workspace for the ZIP and verify its size/SHA-256.
2. If absent, locate the named private Drive folder, read its manifest, retrieve the two parts and reassemble. This is the default reusable source for the author's connected sessions.
3. Download again from the official source only if no valid local/private cache is accessible. The source and data identity below are already known; do not ask the author to identify the same download again.
4. Keep raw images/ratings in the private research cache. GitHub contains code, protocols, derived result summaries, source references and checksums. Public redistribution of the original archive requires an explicit upstream grant; the repository's MIT code license does not supply that grant.

## Identity and upstream source

- Filename: `SCUT-FBP5500_v2.1.zip`.
- Bytes: `179937702`.
- SHA-256: `b8a756d0634a514d3f2f956c3a9363bb8f5a727c7652cbf9d4a87d80bb55a23c`.
- Official repository: https://github.com/HCIILAB/SCUT-FBP5500-Database-Release .
- Official download: https://drive.google.com/file/d/1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf/view .
- Contents: 5,500 images, final 60-rater scores, image-source metadata, facial landmarks and official split files.

`python datasets/SCUT_FBP5500/fetch.py --cache-dir /path/to/cache` verifies an existing ZIP or reassembles the two cached parts without a network request. The opt-in `--download-official` path requires `gdown==5.2.0` and obtains data directly from the upstream provider; it never uploads or mirrors it.

The upstream README restricts use to non-commercial research and does not explicitly grant public redistribution of the whole image archive. Retain the official README and cite Liang et al., *SCUT-FBP5500: A Diverse Benchmark Dataset for Multi-Paradigm Facial Beauty Prediction*, ICPR 2018, DOI 10.1109/ICPR.2018.8546038. No raw image archive is hosted in this public repository or its releases.
