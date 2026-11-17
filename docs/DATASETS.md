# Datasets

None are redistributed. Obtain each from its original source and place it under
`data/raw/<dataset>/`.

| Dataset | Source | Eval split | n | Labels |
|---|---|---|---:|---|
| AVeriTeC | AVeriTeC shared task | public **dev** | 500 | SUPPORTED / REFUTED / NOT ENOUGH EVIDENCE / CONFLICTING |
| PubHealth | PUBHEALTH (Kotonya & Toni) | test | 1,231 | true / false / unproven / mixture |
| RAGuard | RAGuard release | test | 266 | TRUE / FALSE |
| LIAR-RAW | LIAR-RAW | test | 1,251 | TRUE / HALF-TRUE / FALSE |

Evidence budget is **k = 10** throughout.

## AVeriTeC
Label-level evaluation uses the **public development split**, because official
test labels are not available for local scoring. These numbers are therefore not
comparable to shared-task leaderboard scores, which use a different metric
emphasising retrieval and question quality.

## PubHealth
The verifier emits an NLI-style label space while the dataset stores its own
vocabulary. The mapping is 1-to-1 and is verified at load time:

```
false -> contradiction    true    -> entailment
unproven -> neutral       mixture -> mixture
```

## RAGuard
Released split, HerO2-style top-10 retrieval over the RAGuard corpus. Evidence
comes from a single platform, so there is little source variation for F2–F4 to
exploit — worth knowing when interpreting this column.

## LIAR-RAW
Six-way labels collapsed to three, applied **identically to gold and
predictions**:

```
FALSE     <- pants-fire, false, barely-true
HALF-TRUE <- half-true
TRUE      <- mostly-true, true
```

REFLEX predicts 3-way natively; the HerO2 track predicts 6-way and is collapsed.
