# First propagation tick supplement

All original576 anchors, duration8, all four raw payloads. Original controller weights remain unchanged. Cue state24 and scores11 reproduced bitwise for every row before taking the same locked BEGIN. Exactly one0.25s tick is then observed and stepped.

|seed|cue invariant groups|nonzero at+0.25|four distinct at+0.25|raw probe correct/768|mapped probe correct/768|
|---:|---:|---:|---:|---:|---:|
|0|192/192|192/192|192/192|644/768|332/768|
|1|192/192|192/192|192/192|712/768|364/768|
|2|192/192|192/192|192/192|689/768|362/768|

Probe uses unchanged probe.fit_scores and within-job actor4,5 versus6,7 folds in both directions; all anchor variants remain together. Fixed within-anchor-mean null is192/768=25% for every seed and label. No hyperparameter selection or original-controller training.

Exact nonzero detects propagation at the observation grid; scale is saved per anchor. Offline decodability is separate from original readout use and autonomous behavior.
