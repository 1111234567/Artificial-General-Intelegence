# 唯一中间tick FINISH事后补测

Unique post-hoc mechanistic supplement, selected after seeing the first-versus-original-teacher FINISH tradeoff. Static floor((first_legal_tick + original_teacher_FINISH_tick)/2); same final-eligible WAIT. Frozen before any interior autonomous outcome. No sweep, tuning or later position search.

21/21独立绑定，所有比率为3seed等权均值。原始自由行动、回执、合法窗口和成熟请求身份已重建审计。

|快照|组|B普通|B严格|idle动作/千|
|---|---|---:|---:|---:|
|B000|interior|17.65%|16.06%|13.182|
|B000|first|17.65%|16.06%|13.182|
|B000|midpoint|17.65%|16.06%|13.182|
|B000|no_review|17.65%|16.06%|13.182|
|B001|interior|83.99%|46.56%|4.284|
|B001|first|89.21%|49.97%|1.326|
|B001|midpoint|71.30%|42.27%|12.837|
|B001|no_review|75.07%|57.51%|2.379|
|B002|interior|96.22%|62.80%|1.171|
|B002|first|88.05%|56.36%|3.286|
|B002|midpoint|60.11%|42.25%|21.202|
|B002|no_review|90.46%|84.70%|0.678|
|B004|interior|96.40%|77.68%|1.604|
|B004|first|93.59%|79.04%|1.154|
|B004|midpoint|56.15%|45.05%|22.297|
|B004|no_review|86.95%|84.75%|1.561|
|B008|interior|93.26%|83.10%|1.174|
|B008|first|94.21%|85.44%|0.489|
|B008|midpoint|64.24%|60.47%|15.354|
|B008|no_review|99.21%|98.02%|0.299|

|B8随机交错|A冲突普通|A冲突严格|B冲突普通|B冲突严格|
|---|---:|---:|---:|---:|
|interior|84.83%|31.67%|87.96%|71.11%|
|first|61.59%|12.52%|93.70%|84.07%|
|midpoint|94.81%|51.71%|65.74%|58.89%|
|last|94.81%|51.71%|65.74%|58.89%|
|no_review|8.66%|0.00%|99.26%|98.33%|

|组|seed|家族|成熟|普通|严格|首错实际early|未开始|已开始无窗口|首次legal对|首次legal错|窗口未发|
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|interior|0|A|36|34|10|6|0|0|15|21|0|
|interior|0|B|45|40|36|0|0|0|36|9|0|
|interior|1|A|38|28|10|13|0|0|16|21|1|
|interior|1|B|36|29|22|0|0|0|22|12|2|
|interior|2|A|44|38|18|18|0|0|28|15|1|
|interior|2|B|36|34|26|0|0|0|26|10|0|
|first|0|A|36|21|4|25|0|0|12|21|3|
|first|0|B|45|44|41|4|0|0|44|1|0|
|last|0|A|36|35|22|5|0|0|27|9|0|
|last|0|B|45|45|42|3|0|0|43|2|0|
|midpoint|0|A|36|35|22|5|0|0|27|9|0|
|midpoint|0|B|45|45|42|3|0|0|43|2|0|
|no_review|0|A|36|5|0|3|0|0|1|35|0|
|no_review|0|B|45|44|44|0|0|0|44|0|1|
|first|1|A|38|23|4|28|0|0|12|22|4|
|first|1|B|36|34|31|0|0|0|31|5|0|
|last|1|A|38|34|15|0|0|0|15|22|1|
|last|1|B|36|18|14|0|0|0|14|6|16|
|midpoint|1|A|38|34|15|0|0|0|15|22|1|
|midpoint|1|B|36|18|14|0|0|0|14|6|16|
|no_review|1|A|38|2|0|14|0|0|0|37|1|
|no_review|1|B|36|36|35|0|0|0|35|1|0|
|first|2|A|44|29|7|31|0|0|22|15|7|
|first|2|B|36|32|27|0|0|0|27|9|0|
|last|2|A|44|43|24|4|0|0|27|16|1|
|last|2|B|36|17|16|0|0|0|16|3|17|
|midpoint|2|A|44|43|24|4|0|0|27|16|1|
|midpoint|2|B|36|17|16|0|0|0|16|3|17|
|no_review|2|A|44|3|0|31|0|0|1|36|7|
|no_review|2|B|36|36|36|0|0|0|36|0|0|

Actual early FINISH requires original world active & started & now < ready_time-eps, irrespective of action version/content. Earliest active rejection is stored with all overlapping flags; primary exclusive category puts this actual-early flag first. Thus wrong-version early FINISH cannot disappear into wrong-version bins. Report both all matured and completed-but-nonstrict cohorts, early lead=ready_time-now in seconds.

Original pre-action world active & started & ready_time-eps <= now <= deadline+eps. First FINISH inside this actual window is classified correct version/content, other-context content, or other error. Retain no-BEGIN, no-window, and window-but-no-FINISH separately.

Per-seed integer counts and equal-seed mean/sample SD; B learning gains relative to B0 and early first2/first8 snapshots. No pooled A/B rate, no causal rate or general meta-learning inference.
