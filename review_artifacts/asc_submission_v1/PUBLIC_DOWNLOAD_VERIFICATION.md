# Public download verification — BLOCKED

Verification date: 2026-11-12 (Asia/Shanghai).

The published reviewer ZIP at commit `4931e17476da032d7578166a6f8808ac09251ccd` failed integrity and ZIP-format checks. It must not be treated as a verified reproducibility packet.

## Observed evidence

- File: `ASC_REVIEW_REPRODUCIBILITY_v1.zip` (786,444 bytes).
- Expected SHA-256 in that commit's `SHA256SUMS`: `a83d0540120159933c04d8ba964512f5963eae7fe048178e6a1127051857af14`.
- Actual downloaded SHA-256: `ebcab658b287b4b7a485c7568220ff0664f96203930de3a9e35bf47647af6387`.
- Both the commit-pinned raw.githubusercontent.com download and the base64-decoded GitHub REST Blob API response produced identical bytes.
- The downloaded Git blob SHA-1 is `c95575f246b5f69285e3a7a5689a436fbac4d550`, matching GitHub's recorded blob ID. This confirms the retrieved bytes match the stored Git blob.
- Python 3.12 `zipfile.is_zipfile` returned `False`; opening with `zipfile.ZipFile` raised `BadZipFile: File is not a zip file`.

## Scope and next action

Extraction and `python run_checks.py --figures` could not proceed. No new numerical checks or figure regeneration succeeded in this verification attempt. Earlier preparation-audit claims of 13 passing checks are not evidence that this public ZIP works.

Recover the original prepared ZIP (ideally matching the expected SHA-256 above), or the complete source directory used to build it. Then publish the recovered packet, verify its checksum and clean extraction, and run the documented checks on the downloaded public copy before recording a successful verification. Changing SHA256SUMS to match the current invalid ZIP would not fix the problem.

This report adds evidence only; it does not modify the packet, frozen scientific records, or existing checksum.
