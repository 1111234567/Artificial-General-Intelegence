# 静态最后WAIT位置：中途观察后的机制补充

Post-interim mechanistic supplement, after observing that over92percent of first-round hard choices used the final eligible WAIT tick and after interim B002 timing diagnostics. Not a prospective fourth arm; main210 bindings and frozen protocol remain unchanged. Supplementary scope frozen before reading last autonomous outcomes.

21/21独立绑定，使用与主线完全一致的公开暖身、外部未来、成熟请求身份和11动作argmax；主210条绑定没有增加或改写。所有比率按3seed等权。

|快照|策略|长B普通|长B严格|idle/千|
|---|---|---:|---:|---:|
|B000|last|17.65%|16.06%|13.182|
|B000|midpoint|17.65%|16.06%|13.182|
|B000|hard|17.65%|16.06%|13.182|
|B000|random|17.65%|16.06%|13.182|
|B000|uniform|17.65%|16.06%|13.182|
|B000|no_review|17.65%|16.06%|13.182|
|B001|last|71.30%|42.27%|12.837|
|B001|midpoint|89.79%|47.39%|2.903|
|B001|hard|72.96%|40.10%|11.516|
|B001|random|92.03%|42.62%|2.413|
|B001|uniform|89.79%|47.39%|2.903|
|B001|no_review|75.07%|57.51%|2.379|
|B002|last|60.11%|42.25%|21.202|
|B002|midpoint|70.99%|33.13%|22.068|
|B002|hard|66.91%|52.78%|19.377|
|B002|random|85.92%|52.14%|9.589|
|B002|uniform|70.99%|33.13%|22.068|
|B002|no_review|90.46%|84.70%|0.678|
|B004|last|56.15%|45.05%|22.297|
|B004|midpoint|85.53%|54.91%|12.461|
|B004|hard|56.95%|44.55%|21.743|
|B004|random|60.69%|42.06%|28.138|
|B004|uniform|85.53%|54.91%|12.461|
|B004|no_review|86.95%|84.75%|1.561|
|B008|last|64.24%|60.47%|15.354|
|B008|midpoint|85.35%|66.55%|9.315|
|B008|hard|64.28%|56.47%|13.481|
|B008|random|63.15%|46.18%|15.966|
|B008|uniform|85.35%|66.55%|9.315|
|B008|no_review|99.21%|98.02%|0.299|

|策略|B8混合A冲突普通|A严格|B冲突普通|B严格|
|---|---:|---:|---:|---:|
|last|94.81%|51.71%|65.74%|58.89%|
|midpoint|61.12%|6.47%|82.78%|62.96%|
|hard|94.05%|46.29%|64.26%|49.07%|
|random|84.18%|37.75%|49.81%|35.37%|
|uniform|61.12%|6.47%|82.78%|62.96%|
|no_review|8.66%|0.00%|99.26%|98.33%|

|seed|last家族|成熟|普通|严格|首错实际早发|首次legal对|首次legal错|legal窗口未发|提前秒均值|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|0|A|36|35|22|5|27|9|0|0.234|
|0|B|45|45|42|3|43|2|0|0.084|
|1|A|38|34|15|0|15|22|1|NA|
|1|B|36|18|14|0|14|6|16|NA|
|2|A|44|43|24|4|27|16|1|0.472|
|2|B|36|17|16|0|16|3|17|NA|

Actual early FINISH requires original world active & started & now < ready_time-eps, irrespective of action version/content. Earliest active rejection is stored with all overlapping flags; primary exclusive category puts this actual-early flag first. Thus wrong-version early FINISH cannot disappear into wrong-version bins. Report both all matured and completed-but-nonstrict cohorts, early lead=ready_time-now in seconds.

Original pre-action world active & started & ready_time-eps <= now <= deadline+eps. First FINISH inside this actual window is classified correct version/content, other-context content, or other error. Retain no-BEGIN, no-window, and window-but-no-FINISH separately.
