# Triton TVM Backend Devlog TOC

Source: `triton_tvm_backend_devlog.md`

Generated: 2026-05-23

This is a line-numbered table of contents for the current devlog snapshot.
Regenerate it after editing the devlog, because line numbers are intentionally exact.

Regeneration command:

```bash
awk '/^#/ { line = $0; level = 0; while (substr(line, level + 1, 1) == "#") level++; title = substr(line, level + 2); print NR, level, title }' docs/arch/triton_tvm_backend_devlog.md
```

| Line | Level | Section | Link |
| ---: | :---: | --- | --- |
| 1 | H1 | Triton to TVM Backend Development Log | [L1](triton_tvm_backend_devlog.md#L1) |
| 7 | H2 | 2026-05-22: M0-M1.5 Prototype | [L7](triton_tvm_backend_devlog.md#L7) |
| 9 | H3 | Environment | [L9](triton_tvm_backend_devlog.md#L9) |
| 18 | H3 | Implemented Scope | [L18](triton_tvm_backend_devlog.md#L18) |
| 28 | H3 | Public API Added | [L28](triton_tvm_backend_devlog.md#L28) |
| 38 | H3 | TTIR Subset Covered | [L38](triton_tvm_backend_devlog.md#L38) |
| 51 | H3 | Translation Contract | [L51](triton_tvm_backend_devlog.md#L51) |
| 63 | H3 | Runtime and Metadata | [L63](triton_tvm_backend_devlog.md#L63) |
| 71 | H3 | Tests Added | [L71](triton_tvm_backend_devlog.md#L71) |
| 84 | H3 | Validation | [L84](triton_tvm_backend_devlog.md#L84) |
| 103 | H3 | Known Gaps | [L103](triton_tvm_backend_devlog.md#L103) |
| 113 | H3 | Suggested Next Checkpoint | [L113](triton_tvm_backend_devlog.md#L113) |
| 123 | H2 | 2026-05-22: Pre-M2 Hardening | [L123](triton_tvm_backend_devlog.md#L123) |
| 125 | H3 | Motivation | [L125](triton_tvm_backend_devlog.md#L125) |
| 131 | H3 | Implementation Changes | [L131](triton_tvm_backend_devlog.md#L131) |
| 147 | H3 | Tests Added | [L147](triton_tvm_backend_devlog.md#L147) |
| 159 | H3 | Validation | [L159](triton_tvm_backend_devlog.md#L159) |
| 173 | H3 | Remaining M2 Entry Conditions | [L173](triton_tvm_backend_devlog.md#L173) |
| 183 | H2 | 2026-05-22: M2A Standalone Pointwise Expansion | [L183](triton_tvm_backend_devlog.md#L183) |
| 185 | H3 | Implemented Scope | [L185](triton_tvm_backend_devlog.md#L185) |
| 198 | H3 | Tests Added | [L198](triton_tvm_backend_devlog.md#L198) |
| 206 | H3 | Remaining Gaps | [L206](triton_tvm_backend_devlog.md#L206) |
| 212 | H3 | Validation | [L212](triton_tvm_backend_devlog.md#L212) |
| 226 | H2 | 2026-05-22: M2.5 Standalone Pointwise Corpus | [L226](triton_tvm_backend_devlog.md#L226) |
| 228 | H3 | Implemented Scope | [L228](triton_tvm_backend_devlog.md#L228) |
| 241 | H3 | Contract and Runtime Changes | [L241](triton_tvm_backend_devlog.md#L241) |
| 256 | H3 | Tests Added | [L256](triton_tvm_backend_devlog.md#L256) |
| 272 | H3 | Supported TTIR Subset at M2.5 | [L272](triton_tvm_backend_devlog.md#L272) |
| 288 | H3 | Remaining Gaps | [L288](triton_tvm_backend_devlog.md#L288) |
| 299 | H3 | Validation | [L299](triton_tvm_backend_devlog.md#L299) |
| 313 | H2 | 2026-05-22: M2.5 Hardening: Pointwise Contract Cleanup | [L313](triton_tvm_backend_devlog.md#L313) |
| 315 | H3 | Implemented Scope | [L315](triton_tvm_backend_devlog.md#L315) |
| 336 | H3 | Reader Cleanup | [L336](triton_tvm_backend_devlog.md#L336) |
| 345 | H3 | Contract Boundary | [L345](triton_tvm_backend_devlog.md#L345) |
| 353 | H3 | Validation | [L353](triton_tvm_backend_devlog.md#L353) |
| 367 | H2 | 2026-05-22: M2.5 Hardening: Backend Decoupling / CUDA Policy Boundary | [L367](triton_tvm_backend_devlog.md#L367) |
| 369 | H3 | Motivation | [L369](triton_tvm_backend_devlog.md#L369) |
| 375 | H3 | Implementation Changes | [L375](triton_tvm_backend_devlog.md#L375) |
| 421 | H3 | Static Coverage Added | [L421](triton_tvm_backend_devlog.md#L421) |
| 437 | H3 | Remaining Bounds | [L437](triton_tvm_backend_devlog.md#L437) |
| 444 | H3 | Validation | [L444](triton_tvm_backend_devlog.md#L444) |
| 462 | H2 | 2026-05-22: M2.5 Performance Baseline / Regression Guard | [L462](triton_tvm_backend_devlog.md#L462) |
| 464 | H3 | Positioning | [L464](triton_tvm_backend_devlog.md#L464) |
| 470 | H3 | Implementation Changes | [L470](triton_tvm_backend_devlog.md#L470) |
| 490 | H3 | Commands | [L490](triton_tvm_backend_devlog.md#L490) |
| 514 | H3 | Local Baseline | [L514](triton_tvm_backend_devlog.md#L514) |
| 531 | H3 | Validation | [L531](triton_tvm_backend_devlog.md#L531) |
| 551 | H2 | 2026-05-22: M3 Inductor Pointwise Audit | [L551](triton_tvm_backend_devlog.md#L551) |
| 553 | H3 | Implemented Scope | [L553](triton_tvm_backend_devlog.md#L553) |
| 566 | H3 | CLI Added | [L566](triton_tvm_backend_devlog.md#L566) |
| 584 | H3 | Builtin Corpus | [L584](triton_tvm_backend_devlog.md#L584) |
| 613 | H3 | Local Audit Result | [L613](triton_tvm_backend_devlog.md#L613) |
| 644 | H3 | Tests Added | [L644](triton_tvm_backend_devlog.md#L644) |
| 656 | H3 | Remaining Bounds | [L656](triton_tvm_backend_devlog.md#L656) |
| 664 | H2 | 2026-05-22: M3.5 Inductor Pointwise Coverage | [L664](triton_tvm_backend_devlog.md#L664) |
| 666 | H3 | Implemented Scope | [L666](triton_tvm_backend_devlog.md#L666) |
| 696 | H3 | Audit Result | [L696](triton_tvm_backend_devlog.md#L696) |
| 721 | H3 | Tests Added | [L721](triton_tvm_backend_devlog.md#L721) |
| 736 | H3 | Validation | [L736](triton_tvm_backend_devlog.md#L736) |
| 754 | H2 | 2026-05-22: M3.5 Hardening / M4 Entry Gate | [L754](triton_tvm_backend_devlog.md#L754) |
| 756 | H3 | Positioning | [L756](triton_tvm_backend_devlog.md#L756) |
| 762 | H3 | Implementation Changes | [L762](triton_tvm_backend_devlog.md#L762) |
| 785 | H3 | Tests Added | [L785](triton_tvm_backend_devlog.md#L785) |
| 795 | H3 | M4 Entry Criteria | [L795](triton_tvm_backend_devlog.md#L795) |
| 804 | H3 | Validation | [L804](triton_tvm_backend_devlog.md#L804) |
| 846 | H2 | 2026-05-22: M3.5 Pointwise Performance Baseline | [L846](triton_tvm_backend_devlog.md#L846) |
| 848 | H3 | Positioning | [L848](triton_tvm_backend_devlog.md#L848) |
| 854 | H3 | Test Changes | [L854](triton_tvm_backend_devlog.md#L854) |
| 867 | H3 | Commands | [L867](triton_tvm_backend_devlog.md#L867) |
| 883 | H3 | Local Baseline | [L883](triton_tvm_backend_devlog.md#L883) |
| 903 | H3 | Validation | [L903](triton_tvm_backend_devlog.md#L903) |
| 914 | H2 | 2026-05-22: M4 Reduction Subset | [L914](triton_tvm_backend_devlog.md#L914) |
| 916 | H3 | Positioning | [L916](triton_tvm_backend_devlog.md#L916) |
| 924 | H3 | Implementation Changes | [L924](triton_tvm_backend_devlog.md#L924) |
| 944 | H3 | Tests Added | [L944](triton_tvm_backend_devlog.md#L944) |
| 953 | H3 | Validation | [L953](triton_tvm_backend_devlog.md#L953) |
| 971 | H2 | 2026-05-22: M4 Full LN/RMS Closure | [L971](triton_tvm_backend_devlog.md#L971) |
| 973 | H3 | Positioning | [L973](triton_tvm_backend_devlog.md#L973) |
| 979 | H3 | Implementation Changes | [L979](triton_tvm_backend_devlog.md#L979) |
| 988 | H3 | Tests Added | [L988](triton_tvm_backend_devlog.md#L988) |
| 996 | H3 | Validation | [L996](triton_tvm_backend_devlog.md#L996) |
| 1006 | H2 | 2026-05-22: Pre-M5 Debt Sprint Workflows 1-4 | [L1006](triton_tvm_backend_devlog.md#L1006) |
| 1008 | H3 | Scope | [L1008](triton_tvm_backend_devlog.md#L1008) |
| 1013 | H3 | Workflow 1: Pass Surface Cleanup | [L1013](triton_tvm_backend_devlog.md#L1013) |
| 1022 | H3 | Workflow 2: Legacy Contract Alias Cleanup | [L1022](triton_tvm_backend_devlog.md#L1022) |
| 1034 | H3 | Workflow 3: Contract Matrix | [L1034](triton_tvm_backend_devlog.md#L1034) |
| 1043 | H3 | Workflow 4: Runtime, Cache, Stream Boundary | [L1043](triton_tvm_backend_devlog.md#L1043) |
| 1057 | H3 | Validation | [L1057](triton_tvm_backend_devlog.md#L1057) |
| 1078 | H2 | 2026-05-22: Pre-M5 Debt Sprint Workflows 5-8 | [L1078](triton_tvm_backend_devlog.md#L1078) |
| 1080 | H3 | Workflow 5: TTIRReader and Builder Debt | [L1080](triton_tvm_backend_devlog.md#L1080) |
| 1093 | H3 | Workflow 6: Unified Reporting | [L1093](triton_tvm_backend_devlog.md#L1093) |
| 1105 | H3 | Workflow 7: Test Structure | [L1105](triton_tvm_backend_devlog.md#L1105) |
| 1116 | H3 | Workflow 8: Public API Freeze | [L1116](triton_tvm_backend_devlog.md#L1116) |
| 1127 | H3 | Validation | [L1127](triton_tvm_backend_devlog.md#L1127) |
| 1149 | H2 | 2026-05-23: M5 Inductor Integration Prototype | [L1149](triton_tvm_backend_devlog.md#L1149) |
| 1151 | H3 | Scope | [L1151](triton_tvm_backend_devlog.md#L1151) |
| 1158 | H3 | Implementation Changes | [L1158](triton_tvm_backend_devlog.md#L1158) |
| 1176 | H3 | Tests Added | [L1176](triton_tvm_backend_devlog.md#L1176) |
| 1189 | H3 | Validation | [L1189](triton_tvm_backend_devlog.md#L1189) |
| 1207 | H2 | 2026-05-23: M5.5 Inductor Hook Hardening Gate | [L1207](triton_tvm_backend_devlog.md#L1207) |
| 1209 | H3 | Scope | [L1209](triton_tvm_backend_devlog.md#L1209) |
| 1216 | H3 | Implementation Changes | [L1216](triton_tvm_backend_devlog.md#L1216) |
| 1235 | H3 | Tests Added | [L1235](triton_tvm_backend_devlog.md#L1235) |
| 1247 | H3 | Documentation | [L1247](triton_tvm_backend_devlog.md#L1247) |
| 1255 | H3 | Validation | [L1255](triton_tvm_backend_devlog.md#L1255) |
| 1277 | H2 | 2026-05-23: Pre-M6 Integration Contract Cleanup | [L1277](triton_tvm_backend_devlog.md#L1277) |
| 1279 | H3 | Scope | [L1279](triton_tvm_backend_devlog.md#L1279) |
| 1286 | H3 | Implementation Changes | [L1286](triton_tvm_backend_devlog.md#L1286) |
| 1309 | H3 | Tests Added | [L1309](triton_tvm_backend_devlog.md#L1309) |
| 1322 | H3 | Documentation | [L1322](triton_tvm_backend_devlog.md#L1322) |
| 1329 | H3 | Validation | [L1329](triton_tvm_backend_devlog.md#L1329) |
| 1358 | H2 | 2026-05-23: M6/M6.5 External Model Corpus Audit | [L1358](triton_tvm_backend_devlog.md#L1358) |
| 1360 | H3 | Scope | [L1360](triton_tvm_backend_devlog.md#L1360) |
| 1371 | H3 | Dependency Setup | [L1371](triton_tvm_backend_devlog.md#L1371) |
| 1390 | H3 | Implementation Changes | [L1390](triton_tvm_backend_devlog.md#L1390) |
| 1419 | H3 | Tests Added | [L1419](triton_tvm_backend_devlog.md#L1419) |
| 1434 | H3 | Documentation | [L1434](triton_tvm_backend_devlog.md#L1434) |
| 1442 | H3 | Validation | [L1442](triton_tvm_backend_devlog.md#L1442) |
| 1466 | H2 | 2026-05-23: Pre-M7 Reader/Builder Debt Gate | [L1466](triton_tvm_backend_devlog.md#L1466) |
| 1468 | H3 | Scope | [L1468](triton_tvm_backend_devlog.md#L1468) |
| 1476 | H3 | Implementation Changes | [L1476](triton_tvm_backend_devlog.md#L1476) |
| 1498 | H3 | Tests Added | [L1498](triton_tvm_backend_devlog.md#L1498) |
| 1511 | H3 | Documentation | [L1511](triton_tvm_backend_devlog.md#L1511) |
| 1517 | H3 | Validation | [L1517](triton_tvm_backend_devlog.md#L1517) |
