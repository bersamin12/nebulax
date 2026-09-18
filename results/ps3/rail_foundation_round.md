# Rail foundation-model and teacher round

All results below use the 272 labelled Train recordings only. Each supervised probe or
fine-tune is fitted within duplicate-grouped 5-fold CV repeated with seeds 0, 1 and 2.
The 64 vibration sensors are grouped as eight cars of eight axle boxes; odd box
positions represent Side I and even positions Side II. Held-out recordings are never
used to update encoders or classification heads. No Test files or labels were used.

The shipped W7 model has historical selection CV **0.8365** (Side I F1 **0.5975**)
and a separate original nested estimate of **0.7571 ± 0.1232**. Its refit with the
current duplicate-group correction on the transfer splits scores **0.8341** (Side I
**0.5930**). The frozen promotion gate is **0.8365** selection macro F1 plus
**0.7471** nested macro F1.

| Trial | Macro F1 | Side I F1 | W7 blend macro F1 |
|---|---:|---:|---:|
| Frozen MOMENT long | 0.6702 | 0.4858 | 0.8350 at 25% |
| Frozen UniTS long | 0.4934 | 0.1721 | 0.8243 at 25% |
| Frozen SimMTM short | 0.4384 | 0.1815 | 0.8266 at 25% |
| Fold-local Ti-MAE-style masked autoencoder | 0.4027 | 0.0807 | 0.8238 at 25% |
| Frozen Chronos-2 small, all sensors jointly | 0.6021 | 0.2224 | 0.8252 at 25% |
| Frozen Moirai-2 small, eight boxes per car jointly | 0.7451 | 0.4811 | 0.8198 at 25% |
| MantisV2 first block fine-tuned | 0.4810 | 0.0000 | 0.8237 at 25% |
| MantisV2 last block fine-tuned | 0.4643 | 0.0000 | 0.8237 at 25% |
| MantisV2 all three blocks fine-tuned | 0.4686 | 0.0267 | 0.8258 at 25% |
| Moirai-2 last block fine-tuned | 0.5459 | 0.1013 | 0.8197 at 25% |
| Moirai-2 last block fine-tuned, fault loss weight 4 | 0.5967 | 0.2020 | 0.8176 at 25% |

The strongest frozen multiscale row, **75% W7 + 12.5% short-window MantisV2 +
12.5% long-window MOMENT**, scored **0.8381 ± 0.0988** with Side I F1 **0.6038**.
It corrected two of 816 repeated held-out predictions and introduced no new errors.
A conditional 13-row nested transfer audit on reused outer splits scored
**0.8363 ± 0.0995**. Contiguous and held-speed-range stress scores were **0.7446**
and **0.6997**, versus **0.7436** and **0.6865** for the current-code W7 refit.
These scores are local validation, not an organiser Test estimate. The fusion's
encoder weights exceed the 5 MB deployed-model limit, while its observed gain is
only two decisions.

**Decision: retain W7.** Neither supervised large-model adaptation beat the
selection gate, so no fine-tuned teacher qualified for a student distillation run.
The W7 artifact remains 790,652 bytes. Its stored prediction CSV is unchanged.

Detailed results: [Mantis frozen](rail_mantis_transfer_round.md),
[frozen encoders and fusion](rail_transfer_compare.md),
[nested transfer audit](rail_transfer_nested.md),
[transfer stress](rail_transfer_stress.md),
[Chronos and Moirai](rail_forecast_transfer_compare.md),
[Ti-MAE](rail_timae_ssl.md),
[Mantis fine-tuning](rail_mantis_finetune.md),
[Moirai-2 fine-tuning](rail_moirai2_finetune.md), and
[balanced Moirai-2 fine-tuning](rail_moirai2_finetune_balanced.md).
