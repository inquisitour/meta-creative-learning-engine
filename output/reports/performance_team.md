# Next test plan for the Performance team

Daily budget: EUR 400, assumed CPM EUR 8.94.

## A_incumbent (control, 20% of budget)
- Creative: hook=social_proof, visual=unboxing, cta=gift_now, format=video
- Hypothesis: Baseline: best existing ad (AD-042).
- Run at least 45,000 impressions and 6 days before reading CTR results.
- Purchase-level confirmation alone would need ~103 days at this budget, so purchases are tracked through the pooled model instead.

## B_champion (exploit, 30% of budget)
- Creative: hook=social_proof, visual=gift_scene, cta=gift_now, format=carousel
- Hypothesis: Stacking all proven winners beats the incumbent (checks that effects add up instead of interfering).
- Run at least 45,000 impressions and 4 days before reading CTR results.
- Purchase-level confirmation alone would need ~69 days at this budget, so purchases are tracked through the pooled model instead.

## C_explore_hook_offer_led (explore, 50% of budget)
- Creative: hook=offer_led, visual=gift_scene, cta=gift_now, format=carousel
- Hypothesis: hook=offer_led could beat social_proof (current estimate +21.6%, upper bound +50.1%).
- Run at least 45,000 impressions and 3 days before reading CTR results.
- Purchase-level confirmation alone would need ~42 days at this budget, so purchases are tracked through the pooled model instead.

**Stopping rule:** Do not read results before every cell reaches min_impressions and min_days. Decide on CTR: adopt a variant only if its 95% CI vs the champion excludes 0; otherwise keep the champion. Purchase effects are not tested per cell (see purchase_test_days_if_run_alone); they stay estimated by the pooled model, which is refit whenever new results arrive.

**Data quality:** ad-to-ad variance is 18.17x (CTR) and 1.36x (CVR) the level pure chance would give; intervals are widened accordingly.
