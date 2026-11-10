# Dataset licence register

A source may be used only when it is recorded here, its licence permits commercial use, and its provenance record (`templates/PROVENANCE.md`) is complete. The machine-readable registry is `datasets/sources.json`, validated by `urbiva-ai validate-manifest`; its schema is shown in `examples/sources.example.json`.

## Register state

As of 2026-11-10 **no source is approved**, so `datasets/sources.json` does not exist and training cannot start. This is the intended state until the data audit milestone in [docs/roadmap.md](../docs/roadmap.md).

## Accepted licences

| Licence | Accepted | Condition |
| --- | --- | --- |
| CC0-1.0 | Yes | None |
| CC-BY-4.0 | Yes | Attribution text recorded |
| CC-BY-SA-4.0 | Yes | Attribution recorded; share-alike reviewed for derived data |
| ODbL-1.0 | Yes | For map data; attribution and share-alike for the database |
| own-capture-consented | Yes | Written consent and blurring documented |
| partner-agreement | Yes | Signed agreement referenced in the provenance record |
| Any non-commercial licence (for example CC-BY-NC) | No | Incompatible with product use |
| Unknown or missing licence | No | Never used |

## How to add a source

1. Complete a provenance record from `templates/PROVENANCE.md` in `datasets/provenance/<source_id>.md`.
2. Add the source to `datasets/sources.json` with `license` and `commercial_use: true`.
3. Add a row to the approved sources table below and open a pull request; a second person reviews the licence.

## Approved sources

| Source id | Licence | Attribution text | Provenance record | Reviewed by | Date |
| --- | --- | --- | --- | --- | --- |
| none approved | n/a | n/a | n/a | n/a | 2026-11-10 |
