# IEEE bibliography style audit

Date: 2026-10-05

Canonical style source: `../refs.bib` from the latest 13-page IEEE TSP manuscript.

## Result

- PASS: the common manuscript body cites 16 keys, and `refs.bib` contains exactly those 16 entries.
- PASS: no missing key, orphan entry, duplicate key, or duplicate title was found.
- PASS: initialized author names, sentence-case titles, IEEE journal abbreviations, `Proc.` conference forms, page ranges, months, identifiers, and field ordering follow the project bibliography.
- PASS: the new queueing reference `Celik2012DynamicServer` is rendered as an IEEE journal article with verified volume, issue, pages, month, year, and DOI.
- PASS: algorithm citations remain attached to the claims they support: Balduzzi for rotational game geometry, Mokhtari for the proximal-point interpretation, Korpelevich for EGM, Rockafellar for PPM/PPM-3, and Schulman for PPO.

## Judgment calls

- `Lu2022ODE` retains the 2022 volume year; Crossref also reports a 2021 online-publication date, which is not a metadata conflict.
- `Korpelevich1976Extragradient` retains the verified translated `Matecon`, vol. 12, pp. 747--756 record.
- The PPO technical report remains an `@misc` arXiv record, consistent with the canonical TSP bibliography.
