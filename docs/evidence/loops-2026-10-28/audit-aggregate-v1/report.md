# Existing-model loop audit

Complete: 9 existing checkpoints, 27 raw files, 540 frozen forks.
Protocol SHA256: `d36569f56ebe612a3838a4a332a93668b6e677c486736a660307f50cb2872e6d`.

Every file, checkpoint, frozen source, inherited boundary, exogenous cohort and rate denominator passed validation.
Masks are averaged within trained seed; displayed mean ± sample SD uses three trained seeds.
Strict and completion columns are percentages. Idle actions are per 1,000 idle opportunities.

| Mode | Condition | Intervention | Strict clean completion | Completion | Idle actions |
|---|---|---|---:|---:|---:|
| A | iid | none | 93.2077 ± 0.1617 (n=3) | 99.6354 ± 0.0455 (n=3) | 1.5454 ± 0.1372 (n=3) |
| A | iid | diagonal_only | 0.0182 ± 0.0126 (n=3) | 6.8073 ± 1.7789 (n=3) | 127.7715 ± 6.8196 (n=3) |
| A | iid | dag_cut | 2.6375 ± 0.8354 (n=3) | 20.2088 ± 3.1769 (n=3) | 104.1846 ± 22.1626 (n=3) |
| A | iid | random_cut | 7.0279 ± 0.7556 (n=3) | 33.3960 ± 6.2627 (n=3) | 118.7388 ± 4.9335 (n=3) |
| A | iid | dag_norm_matched | 2.2081 ± 0.8441 (n=3) | 19.1121 ± 2.1487 (n=3) | 100.5871 ± 18.2897 (n=3) |
| A | iid | random_norm_matched | 6.1572 ± 0.4471 (n=3) | 32.1305 ± 6.8213 (n=3) | 116.7703 ± 4.8772 (n=3) |
| A | iid | dag_quarter | 85.2238 ± 3.3856 (n=3) | 92.6235 ± 3.0623 (n=3) | 17.5206 ± 6.4822 (n=3) |
| A | iid | random_quarter | 81.4135 ± 1.4629 (n=3) | 89.6987 ± 1.9084 (n=3) | 20.6073 ± 2.5030 (n=3) |
| A | long_sparse | none | 7.7511 ± 1.4965 (n=3) | 44.1801 ± 1.7931 (n=3) | 282.9553 ± 22.5646 (n=3) |
| A | long_sparse | diagonal_only | 0.0000 ± 0.0000 (n=3) | 16.8455 ± 4.6496 (n=3) | 212.9116 ± 44.3085 (n=3) |
| A | long_sparse | dag_cut | 0.0276 ± 0.0121 (n=3) | 16.6521 ± 3.4600 (n=3) | 155.5426 ± 32.2117 (n=3) |
| A | long_sparse | random_cut | 0.0966 ± 0.0610 (n=3) | 26.5337 ± 0.8925 (n=3) | 224.3429 ± 12.6696 (n=3) |
| A | long_sparse | dag_norm_matched | 0.0380 ± 0.0318 (n=3) | 15.8209 ± 3.0322 (n=3) | 148.3242 ± 27.8406 (n=3) |
| A | long_sparse | random_norm_matched | 0.0828 ± 0.0578 (n=3) | 26.1042 ± 0.7003 (n=3) | 214.2765 ± 15.4091 (n=3) |
| A | long_sparse | dag_quarter | 8.2626 ± 0.7410 (n=3) | 40.0027 ± 1.1379 (n=3) | 220.5880 ± 17.9650 (n=3) |
| A | long_sparse | random_quarter | 6.9803 ± 0.2787 (n=3) | 39.9061 ± 1.1612 (n=3) | 232.6433 ± 17.4265 (n=3) |
| A | longer | none | 0.0000 ± 0.0000 (n=3) | 34.9629 ± 1.0313 (n=3) | 186.5144 ± 15.1752 (n=3) |
| A | longer | diagonal_only | 0.0000 ± 0.0000 (n=3) | 20.3468 ± 2.7662 (n=3) | 189.0570 ± 6.0074 (n=3) |
| A | longer | dag_cut | 0.0000 ± 0.0000 (n=3) | 17.1229 ± 2.0129 (n=3) | 152.1042 ± 12.1743 (n=3) |
| A | longer | random_cut | 0.0034 ± 0.0059 (n=3) | 23.4569 ± 1.8265 (n=3) | 181.8419 ± 17.5385 (n=3) |
| A | longer | dag_norm_matched | 0.0000 ± 0.0000 (n=3) | 16.3514 ± 1.6349 (n=3) | 150.1369 ± 13.8397 (n=3) |
| A | longer | random_norm_matched | 0.0034 ± 0.0059 (n=3) | 22.2790 ± 1.6521 (n=3) | 178.9927 ± 13.3712 (n=3) |
| A | longer | dag_quarter | 0.0034 ± 0.0059 (n=3) | 28.4105 ± 2.4604 (n=3) | 172.7528 ± 14.3603 (n=3) |
| A | longer | random_quarter | 0.0034 ± 0.0059 (n=3) | 29.4016 ± 0.5358 (n=3) | 174.0409 ± 9.5365 (n=3) |
| bp64 | iid | none | 56.3443 ± 3.1448 (n=3) | 78.0727 ± 2.6524 (n=3) | 106.1916 ± 13.5006 (n=3) |
| bp64 | iid | diagonal_only | 0.4293 ± 0.1855 (n=3) | 1.4173 ± 0.8259 (n=3) | 64.5392 ± 14.2581 (n=3) |
| bp64 | iid | dag_cut | 2.3797 ± 0.0460 (n=3) | 3.0113 ± 0.0862 (n=3) | 41.1857 ± 1.3148 (n=3) |
| bp64 | iid | random_cut | 6.4423 ± 4.1057 (n=3) | 9.2234 ± 7.2742 (n=3) | 61.9547 ± 28.4344 (n=3) |
| bp64 | iid | dag_norm_matched | 2.0661 ± 0.1818 (n=3) | 2.6414 ± 0.3443 (n=3) | 40.0207 ± 2.7462 (n=3) |
| bp64 | iid | random_norm_matched | 5.9845 ± 4.1624 (n=3) | 8.5037 ± 7.1234 (n=3) | 61.1756 ± 30.4184 (n=3) |
| bp64 | iid | dag_quarter | 49.0674 ± 5.1326 (n=3) | 59.5565 ± 5.4274 (n=3) | 98.0471 ± 10.0113 (n=3) |
| bp64 | iid | random_quarter | 48.1490 ± 4.5480 (n=3) | 59.1264 ± 5.2820 (n=3) | 99.6587 ± 12.4724 (n=3) |
| bp64 | long_sparse | none | 76.1501 ± 2.2435 (n=3) | 94.5564 ± 2.0252 (n=3) | 1.1797 ± 0.2786 (n=3) |
| bp64 | long_sparse | diagonal_only | 0.0000 ± 0.0000 (n=3) | 0.5059 ± 0.4045 (n=3) | 15.6138 ± 7.2932 (n=3) |
| bp64 | long_sparse | dag_cut | 0.1653 ± 0.0445 (n=3) | 2.2104 ± 0.1561 (n=3) | 26.7843 ± 2.1509 (n=3) |
| bp64 | long_sparse | random_cut | 0.8997 ± 0.2373 (n=3) | 9.5250 ± 4.4989 (n=3) | 57.6877 ± 24.6458 (n=3) |
| bp64 | long_sparse | dag_norm_matched | 0.1301 ± 0.0327 (n=3) | 1.9431 ± 0.1598 (n=3) | 25.2122 ± 2.4937 (n=3) |
| bp64 | long_sparse | random_norm_matched | 0.7274 ± 0.1390 (n=3) | 8.7018 ± 4.0973 (n=3) | 53.6202 ± 23.9129 (n=3) |
| bp64 | long_sparse | dag_quarter | 49.4144 ± 12.9361 (n=3) | 53.3354 ± 14.8794 (n=3) | 5.5990 ± 0.5481 (n=3) |
| bp64 | long_sparse | random_quarter | 43.4456 ± 5.5402 (n=3) | 50.1989 ± 7.8738 (n=3) | 7.2301 ± 0.7851 (n=3) |
| bp64 | longer | none | 2.5129 ± 0.9316 (n=3) | 11.0512 ± 1.5809 (n=3) | 8.2666 ± 1.7542 (n=3) |
| bp64 | longer | diagonal_only | 0.0000 ± 0.0000 (n=3) | 0.7042 ± 0.5572 (n=3) | 42.6348 ± 7.2482 (n=3) |
| bp64 | longer | dag_cut | 0.0245 ± 0.0219 (n=3) | 2.3657 ± 0.4774 (n=3) | 55.3368 ± 9.0308 (n=3) |
| bp64 | longer | random_cut | 0.0140 ± 0.0243 (n=3) | 6.1265 ± 2.9042 (n=3) | 71.8558 ± 23.8829 (n=3) |
| bp64 | longer | dag_norm_matched | 0.0210 ± 0.0211 (n=3) | 2.1519 ± 0.4981 (n=3) | 53.7826 ± 9.0785 (n=3) |
| bp64 | longer | random_norm_matched | 0.0140 ± 0.0161 (n=3) | 5.5096 ± 2.6795 (n=3) | 69.0583 ± 23.1674 (n=3) |
| bp64 | longer | dag_quarter | 1.1811 ± 0.1669 (n=3) | 4.4681 ± 2.4910 (n=3) | 7.1852 ± 4.5825 (n=3) |
| bp64 | longer | random_quarter | 1.2094 ± 0.2199 (n=3) | 4.7036 ± 2.2078 (n=3) | 7.4878 ± 3.8718 (n=3) |
| bp_replay | iid | none | 89.5514 ± 2.3045 (n=3) | 97.2845 ± 1.1730 (n=3) | 4.7268 ± 1.6964 (n=3) |
| bp_replay | iid | diagonal_only | 0.0112 ± 0.0195 (n=3) | 0.0560 ± 0.0337 (n=3) | 39.8852 ± 4.4436 (n=3) |
| bp_replay | iid | dag_cut | 3.8353 ± 0.5021 (n=3) | 7.8112 ± 1.3772 (n=3) | 59.9185 ± 14.7897 (n=3) |
| bp_replay | iid | random_cut | 7.3609 ± 1.3485 (n=3) | 14.1235 ± 5.9113 (n=3) | 73.6543 ± 22.9522 (n=3) |
| bp_replay | iid | dag_norm_matched | 3.1299 ± 0.4647 (n=3) | 6.4396 ± 0.7833 (n=3) | 53.2717 ± 10.6974 (n=3) |
| bp_replay | iid | random_norm_matched | 6.4752 ± 1.8151 (n=3) | 12.8159 ± 6.6046 (n=3) | 68.4712 ± 24.1792 (n=3) |
| bp_replay | iid | dag_quarter | 73.1540 ± 4.2115 (n=3) | 77.0074 ± 4.4605 (n=3) | 19.4556 ± 1.6900 (n=3) |
| bp_replay | iid | random_quarter | 68.8433 ± 0.9256 (n=3) | 73.2668 ± 1.9087 (n=3) | 20.7471 ± 2.3558 (n=3) |
| bp_replay | long_sparse | none | 65.3514 ± 0.4386 (n=3) | 90.9640 ± 3.6534 (n=3) | 1.2080 ± 0.4561 (n=3) |
| bp_replay | long_sparse | diagonal_only | 0.0000 ± 0.0000 (n=3) | 0.0844 ± 0.0186 (n=3) | 7.4654 ± 0.5727 (n=3) |
| bp_replay | long_sparse | dag_cut | 0.3620 ± 0.0449 (n=3) | 4.6886 ± 1.0121 (n=3) | 42.2502 ± 10.4520 (n=3) |
| bp_replay | long_sparse | random_cut | 1.0430 ± 0.2242 (n=3) | 13.5764 ± 5.6861 (n=3) | 102.9239 ± 57.6681 (n=3) |
| bp_replay | long_sparse | dag_norm_matched | 0.2742 ± 0.0328 (n=3) | 3.5056 ± 0.9144 (n=3) | 36.3368 ± 8.2892 (n=3) |
| bp_replay | long_sparse | random_norm_matched | 0.8746 ± 0.1121 (n=3) | 12.1674 ± 5.6934 (n=3) | 90.5332 ± 53.7224 (n=3) |
| bp_replay | long_sparse | dag_quarter | 36.7170 ± 11.0093 (n=3) | 44.1957 ± 13.3937 (n=3) | 3.3887 ± 1.2779 (n=3) |
| bp_replay | long_sparse | random_quarter | 32.0373 ± 4.5141 (n=3) | 42.8731 ± 5.4890 (n=3) | 7.0034 ± 1.9270 (n=3) |
| bp_replay | longer | none | 1.2725 ± 0.5108 (n=3) | 13.3537 ± 1.9844 (n=3) | 7.2244 ± 1.3234 (n=3) |
| bp_replay | longer | diagonal_only | 0.0000 ± 0.0000 (n=3) | 0.5992 ± 0.3937 (n=3) | 34.0606 ± 1.8556 (n=3) |
| bp_replay | longer | dag_cut | 0.0491 ± 0.0338 (n=3) | 4.7980 ± 1.0912 (n=3) | 55.7101 ± 8.5717 (n=3) |
| bp_replay | longer | random_cut | 0.1192 ± 0.0369 (n=3) | 10.2173 ± 5.5102 (n=3) | 88.5715 ± 41.8323 (n=3) |
| bp_replay | longer | dag_norm_matched | 0.0386 ± 0.0338 (n=3) | 4.2723 ± 0.9784 (n=3) | 52.4135 ± 9.0039 (n=3) |
| bp_replay | longer | random_norm_matched | 0.1122 ± 0.0399 (n=3) | 9.2322 ± 5.3608 (n=3) | 80.6187 ± 39.7895 (n=3) |
| bp_replay | longer | dag_quarter | 1.3038 ± 0.2895 (n=3) | 4.9765 ± 1.5907 (n=3) | 7.2244 ± 3.6414 (n=3) |
| bp_replay | longer | random_quarter | 1.1006 ± 0.3154 (n=3) | 6.2598 ± 0.8359 (n=3) | 10.9836 ± 2.2485 (n=3) |

## Paired differences

Differences are left minus right, paired before mask averaging.
The completion differences are percentage points; idle differences are per 1,000 idle opportunities.

| Mode | Condition | Left | Right | Strict difference | Completion difference | Idle difference |
|---|---|---|---|---:|---:|---:|
| A | iid | dag_norm_matched | random_norm_matched | -3.9491 ± 1.0519 (n=3) | -13.0184 ± 8.6249 (n=3) | -16.1832 ± 22.0616 (n=3) |
| A | iid | dag_quarter | random_quarter | 3.8103 ± 3.8167 (n=3) | 2.9248 ± 3.5113 (n=3) | -3.0868 ± 6.5439 (n=3) |
| A | iid | diagonal_only | none | -93.1895 ± 0.1537 (n=3) | -92.8281 ± 1.7739 (n=3) | 126.2261 ± 6.9567 (n=3) |
| A | iid | dag_cut | none | -90.5702 ± 0.8092 (n=3) | -79.4266 ± 3.2099 (n=3) | 102.6392 ± 22.0877 (n=3) |
| A | iid | random_cut | none | -86.1798 ± 0.9114 (n=3) | -66.2395 ± 6.2175 (n=3) | 117.1934 ± 4.9220 (n=3) |
| A | iid | dag_norm_matched | none | -90.9996 ± 0.7728 (n=3) | -80.5234 ± 2.1875 (n=3) | 99.0417 ± 18.2128 (n=3) |
| A | iid | random_norm_matched | none | -87.0505 ± 0.6040 (n=3) | -67.5049 ± 6.7759 (n=3) | 115.2249 ± 4.8561 (n=3) |
| A | iid | dag_quarter | none | -7.9839 ± 3.5227 (n=3) | -7.0120 ± 3.0244 (n=3) | 15.9752 ± 6.5512 (n=3) |
| A | iid | random_quarter | none | -11.7942 ± 1.5436 (n=3) | -9.9367 ± 1.9316 (n=3) | 19.0619 ± 2.6323 (n=3) |
| A | long_sparse | dag_norm_matched | random_norm_matched | -0.0449 ± 0.0488 (n=3) | -10.2833 ± 3.6024 (n=3) | -65.9523 ± 36.0969 (n=3) |
| A | long_sparse | dag_quarter | random_quarter | 1.2823 ± 0.4840 (n=3) | 0.0967 ± 2.0936 (n=3) | -12.0553 ± 21.5980 (n=3) |
| A | long_sparse | diagonal_only | none | -7.7511 ± 1.4965 (n=3) | -27.3346 ± 3.4335 (n=3) | -70.0437 ± 47.1880 (n=3) |
| A | long_sparse | dag_cut | none | -7.7235 ± 1.4957 (n=3) | -27.5279 ± 2.0030 (n=3) | -127.4127 ± 24.6167 (n=3) |
| A | long_sparse | random_cut | none | -7.6545 ± 1.5037 (n=3) | -17.6463 ± 2.6553 (n=3) | -58.6124 ± 34.9906 (n=3) |
| A | long_sparse | dag_norm_matched | none | -7.7132 ± 1.4843 (n=3) | -28.3591 ± 1.5721 (n=3) | -134.6311 ± 22.6995 (n=3) |
| A | long_sparse | random_norm_matched | none | -7.6683 ± 1.5300 (n=3) | -18.0759 ± 2.2003 (n=3) | -68.6788 ± 37.5254 (n=3) |
| A | long_sparse | dag_quarter | none | 0.5115 ± 1.0144 (n=3) | -4.1773 ± 2.4279 (n=3) | -62.3673 ± 8.4207 (n=3) |
| A | long_sparse | random_quarter | none | -0.7708 ± 1.2370 (n=3) | -4.2740 ± 0.8277 (n=3) | -50.3120 ± 19.0266 (n=3) |
| A | longer | dag_norm_matched | random_norm_matched | -0.0034 ± 0.0059 (n=3) | -5.9276 ± 2.4545 (n=3) | -28.8557 ± 18.9145 (n=3) |
| A | longer | dag_quarter | random_quarter | 0.0000 ± 0.0000 (n=3) | -0.9911 ± 2.7891 (n=3) | -1.2881 ± 8.6438 (n=3) |
| A | longer | diagonal_only | none | 0.0000 ± 0.0000 (n=3) | -14.6161 ± 3.6940 (n=3) | 2.5426 ± 20.5168 (n=3) |
| A | longer | dag_cut | none | 0.0000 ± 0.0000 (n=3) | -17.8400 ± 2.0392 (n=3) | -34.4102 ± 3.0187 (n=3) |
| A | longer | random_cut | none | 0.0034 ± 0.0059 (n=3) | -11.5061 ± 0.8826 (n=3) | -4.6726 ± 21.6622 (n=3) |
| A | longer | dag_norm_matched | none | 0.0000 ± 0.0000 (n=3) | -18.6116 ± 1.6680 (n=3) | -36.3775 ± 4.2144 (n=3) |
| A | longer | random_norm_matched | none | 0.0034 ± 0.0059 (n=3) | -12.6839 ± 0.8109 (n=3) | -7.5218 ± 22.5082 (n=3) |
| A | longer | dag_quarter | none | 0.0034 ± 0.0059 (n=3) | -6.5525 ± 1.6042 (n=3) | -13.7616 ± 8.0053 (n=3) |
| A | longer | random_quarter | none | 0.0034 ± 0.0059 (n=3) | -5.5613 ± 1.2157 (n=3) | -12.4736 ± 5.7137 (n=3) |
| bp64 | iid | dag_norm_matched | random_norm_matched | -3.9185 ± 3.9957 (n=3) | -5.8622 ± 6.7931 (n=3) | -21.1549 ± 27.6845 (n=3) |
| bp64 | iid | dag_quarter | random_quarter | 0.9184 ± 1.3169 (n=3) | 0.4301 ± 1.4277 (n=3) | -1.6116 ± 4.6232 (n=3) |
| bp64 | iid | diagonal_only | none | -55.9150 ± 3.1450 (n=3) | -76.6554 ± 1.9236 (n=3) | -41.6524 ± 27.7587 (n=3) |
| bp64 | iid | dag_cut | none | -53.9646 ± 3.1120 (n=3) | -75.0613 ± 2.5830 (n=3) | -65.0059 ± 14.7471 (n=3) |
| bp64 | iid | random_cut | none | -49.9020 ± 1.0631 (n=3) | -68.8493 ± 4.6680 (n=3) | -44.2369 ± 41.1609 (n=3) |
| bp64 | iid | dag_norm_matched | none | -54.2782 ± 2.9711 (n=3) | -75.4312 ± 2.3443 (n=3) | -66.1709 ± 16.0436 (n=3) |
| bp64 | iid | random_norm_matched | none | -50.3597 ± 1.0937 (n=3) | -69.5690 ± 4.5203 (n=3) | -45.0160 ± 43.4006 (n=3) |
| bp64 | iid | dag_quarter | none | -7.2769 ± 2.2152 (n=3) | -18.5162 ± 3.2121 (n=3) | -8.1445 ± 4.5251 (n=3) |
| bp64 | iid | random_quarter | none | -8.1953 ± 1.4032 (n=3) | -18.9463 ± 2.6998 (n=3) | -6.5329 ± 1.6919 (n=3) |
| bp64 | long_sparse | dag_norm_matched | random_norm_matched | -0.5973 ± 0.1396 (n=3) | -6.7587 ± 3.9456 (n=3) | -28.4080 ± 22.0998 (n=3) |
| bp64 | long_sparse | dag_quarter | random_quarter | 5.9688 ± 7.6726 (n=3) | 3.1365 ± 8.0938 (n=3) | -1.6311 ± 1.0757 (n=3) |
| bp64 | long_sparse | diagonal_only | none | -76.1501 ± 2.2435 (n=3) | -94.0505 ± 1.9768 (n=3) | 14.4341 ± 7.3773 (n=3) |
| bp64 | long_sparse | dag_cut | none | -75.9848 ± 2.2873 (n=3) | -92.3460 ± 1.8935 (n=3) | 25.6046 ± 2.3307 (n=3) |
| bp64 | long_sparse | random_cut | none | -75.2504 ± 2.3644 (n=3) | -85.0314 ± 3.3834 (n=3) | 56.5080 ± 24.9190 (n=3) |
| bp64 | long_sparse | dag_norm_matched | none | -76.0200 ± 2.2738 (n=3) | -92.6133 ± 1.9631 (n=3) | 24.0325 ± 2.6554 (n=3) |
| bp64 | long_sparse | random_norm_matched | none | -75.4227 ± 2.3126 (n=3) | -85.8546 ± 3.0960 (n=3) | 52.4405 ± 24.1810 (n=3) |
| bp64 | long_sparse | dag_quarter | none | -26.7357 ± 13.6442 (n=3) | -41.2210 ± 13.5925 (n=3) | 4.4193 ± 0.4032 (n=3) |
| bp64 | long_sparse | random_quarter | none | -32.7045 ± 6.8733 (n=3) | -44.3575 ± 6.1160 (n=3) | 6.0504 ± 1.0382 (n=3) |
| bp64 | longer | dag_norm_matched | random_norm_matched | 0.0070 ± 0.0161 (n=3) | -3.3577 ± 2.7678 (n=3) | -15.2758 ± 20.0731 (n=3) |
| bp64 | longer | dag_quarter | random_quarter | -0.0282 ± 0.3560 (n=3) | -0.2355 ± 1.2345 (n=3) | -0.3026 ± 1.1373 (n=3) |
| bp64 | longer | diagonal_only | none | -2.5129 ± 0.9316 (n=3) | -10.3469 ± 1.2870 (n=3) | 34.3682 ± 6.5715 (n=3) |
| bp64 | longer | dag_cut | none | -2.4884 ± 0.9500 (n=3) | -8.6855 ± 1.6299 (n=3) | 47.0703 ± 8.4342 (n=3) |
| bp64 | longer | random_cut | none | -2.4989 ± 0.9366 (n=3) | -4.9247 ± 1.3763 (n=3) | 63.5892 ± 22.1320 (n=3) |
| bp64 | longer | dag_norm_matched | none | -2.4919 ± 0.9455 (n=3) | -8.8993 ± 1.6222 (n=3) | 45.5160 ± 8.4657 (n=3) |
| bp64 | longer | random_norm_matched | none | -2.4989 ± 0.9295 (n=3) | -5.5416 ± 1.1474 (n=3) | 60.7918 ± 21.4210 (n=3) |
| bp64 | longer | dag_quarter | none | -1.3318 ± 0.8008 (n=3) | -6.5831 ± 1.0422 (n=3) | -1.0814 ± 2.9171 (n=3) |
| bp64 | longer | random_quarter | none | -1.3036 ± 0.9881 (n=3) | -6.3475 ± 0.7940 (n=3) | -0.7788 ± 2.1204 (n=3) |
| bp_replay | iid | dag_norm_matched | random_norm_matched | -3.3453 ± 2.2760 (n=3) | -6.3763 ± 7.3298 (n=3) | -15.1995 ± 34.8031 (n=3) |
| bp_replay | iid | dag_quarter | random_quarter | 4.3107 ± 4.2286 (n=3) | 3.7407 ± 4.6257 (n=3) | -1.2915 ± 0.7687 (n=3) |
| bp_replay | iid | diagonal_only | none | -89.5402 ± 2.3163 (n=3) | -97.2285 ± 1.1462 (n=3) | 35.1584 ± 4.0907 (n=3) |
| bp_replay | iid | dag_cut | none | -85.7161 ± 2.7994 (n=3) | -89.4733 ± 1.9198 (n=3) | 55.1917 ± 13.3029 (n=3) |
| bp_replay | iid | random_cut | none | -82.1906 ± 1.1545 (n=3) | -83.1610 ± 5.2131 (n=3) | 68.9274 ± 24.6318 (n=3) |
| bp_replay | iid | dag_norm_matched | none | -86.4215 ± 2.6966 (n=3) | -90.8449 ± 1.6181 (n=3) | 48.5449 ± 9.0813 (n=3) |
| bp_replay | iid | random_norm_matched | none | -83.0762 ± 1.0769 (n=3) | -84.4686 ± 5.8598 (n=3) | 63.7444 ± 25.8594 (n=3) |
| bp_replay | iid | dag_quarter | none | -16.3974 ± 3.4110 (n=3) | -20.2770 ± 3.2905 (n=3) | 14.7288 ± 0.6403 (n=3) |
| bp_replay | iid | random_quarter | none | -20.7082 ± 3.0618 (n=3) | -24.0177 ± 2.1734 (n=3) | 16.0203 ± 1.3087 (n=3) |
| bp_replay | long_sparse | dag_norm_matched | random_norm_matched | -0.6004 ± 0.1449 (n=3) | -8.6619 ± 6.4870 (n=3) | -54.1965 ± 62.0105 (n=3) |
| bp_replay | long_sparse | dag_quarter | random_quarter | 4.6796 ± 6.7917 (n=3) | 1.3226 ± 8.5199 (n=3) | -3.6147 ± 0.8944 (n=3) |
| bp_replay | long_sparse | diagonal_only | none | -65.3514 ± 0.4386 (n=3) | -90.8797 ± 3.6420 (n=3) | 6.2574 ± 0.7211 (n=3) |
| bp_replay | long_sparse | dag_cut | none | -64.9894 ± 0.4785 (n=3) | -86.2755 ± 4.6631 (n=3) | 41.0423 ± 10.0026 (n=3) |
| bp_replay | long_sparse | random_cut | none | -64.3084 ± 0.3827 (n=3) | -77.3876 ± 2.1290 (n=3) | 101.7159 ± 58.1231 (n=3) |
| bp_replay | long_sparse | dag_norm_matched | none | -65.0772 ± 0.4623 (n=3) | -87.4585 ± 4.5012 (n=3) | 35.1288 ± 7.8333 (n=3) |
| bp_replay | long_sparse | random_norm_matched | none | -64.4768 ± 0.3711 (n=3) | -78.7966 ± 2.1244 (n=3) | 89.3253 ± 54.1784 (n=3) |
| bp_replay | long_sparse | dag_quarter | none | -28.6344 ± 10.9751 (n=3) | -46.7684 ± 10.1210 (n=3) | 2.1807 ± 1.2771 (n=3) |
| bp_replay | long_sparse | random_quarter | none | -33.3140 ± 4.3703 (n=3) | -48.0909 ± 3.8266 (n=3) | 5.7954 ± 1.7216 (n=3) |
| bp_replay | longer | dag_norm_matched | random_norm_matched | -0.0736 ± 0.0364 (n=3) | -4.9599 ± 6.0824 (n=3) | -28.2052 ± 44.1000 (n=3) |
| bp_replay | longer | dag_quarter | random_quarter | 0.2032 ± 0.2370 (n=3) | -1.2833 ± 0.8021 (n=3) | -3.7592 ± 1.4819 (n=3) |
| bp_replay | longer | diagonal_only | none | -1.2725 ± 0.5108 (n=3) | -12.7546 ± 1.8010 (n=3) | 26.8362 ± 1.0280 (n=3) |
| bp_replay | longer | dag_cut | none | -1.2234 ± 0.5113 (n=3) | -8.5557 ± 2.3951 (n=3) | 48.4857 ± 8.3882 (n=3) |
| bp_replay | longer | random_cut | none | -1.1534 ± 0.4913 (n=3) | -3.1365 ± 4.0480 (n=3) | 81.3471 ± 40.8178 (n=3) |
| bp_replay | longer | dag_norm_matched | none | -1.2340 ± 0.5113 (n=3) | -9.0814 ± 2.3764 (n=3) | 45.1891 ± 8.7423 (n=3) |
| bp_replay | longer | random_norm_matched | none | -1.1603 ± 0.4768 (n=3) | -4.1215 ± 3.8494 (n=3) | 73.3943 ± 38.7642 (n=3) |
| bp_replay | longer | dag_quarter | none | 0.0312 ± 0.6104 (n=3) | -8.3772 ± 0.5924 (n=3) | -0.0000 ± 2.3438 (n=3) |
| bp_replay | longer | random_quarter | none | -0.1720 ± 0.3889 (n=3) | -7.0939 ± 1.1487 (n=3) | 3.7592 ± 1.0895 (n=3) |

## Structural summaries

The nine first-draw summaries are retained in report.json; individual state samples remain in the linked raw files.
| Source | Width | Exact off-diagonal edges | Cyclic SCC sizes | Guaranteed simple-cycle lengths |
|---|---:|---:|---|---|
| seed_000_A | 128 | 16256 | [128] | [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128] |
| seed_000_bp64 | 128 | 16256 | [128] | [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128] |
| seed_000_bp_replay | 128 | 16256 | [128] | [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128] |
| seed_001_A | 128 | 16256 | [128] | [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128] |
| seed_001_bp64 | 128 | 16256 | [128] | [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128] |
| seed_001_bp_replay | 128 | 16256 | [128] | [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128] |
| seed_002_A | 128 | 16256 | [128] | [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128] |
| seed_002_bp64 | 128 | 16256 | [128] | [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128] |
| seed_002_bp_replay | 128 | 16256 | [128] | [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128] |

## Scope of interpretation

- Structural directed cycles establish possible feedback, not a causal share of behavior.
- Frozen-state weighted closed walks can revisit neurons and repeat shorter cycles; they do not count simple cycles or measure multi-tick feedback.
- DAG/random deletions match incoming counts and matched endpoints also match row L2 norms; outgoing degree, spectrum and state distributions remain unmatched.
- Quarter interventions retain cycles and their interpolated row norms need not match.
- Parameter intervention scores are counterfactuals. Only none is the unmodified existing model.
- Three trained seeds provide descriptive means and sample SD, not a definitive causal estimate.

## Raw provenance

| Raw result | SHA256 |
|---|---|
| [seed_000_A_mask_11.json](<F:/agi/data/loops/audit-v1/seed_000_A_mask_11.json>) | `f1d64f87e0202f8f430a71fa5e6fb0bd4370d9a7043f359cd38ed1cdfa52ecb1` |
| [seed_000_A_mask_29.json](<F:/agi/data/loops/audit-v1/seed_000_A_mask_29.json>) | `29d25c5534d470a32cf4b736a588b7ee3f6b4e990518d5833f6e4307679a2145` |
| [seed_000_A_mask_47.json](<F:/agi/data/loops/audit-v1/seed_000_A_mask_47.json>) | `47d906bb6e261f42cb62b9bcfd10df81e8b4ed2c18a5170de94b30856f2e1add` |
| [seed_000_bp64_mask_11.json](<F:/agi/data/loops/audit-v1/seed_000_bp64_mask_11.json>) | `88bb0b91454c238b376223473200c6fa95f98163380661f7f39d69f904f8ae03` |
| [seed_000_bp64_mask_29.json](<F:/agi/data/loops/audit-v1/seed_000_bp64_mask_29.json>) | `dc8331bc7f421ca558fb40aeb6cb9d67f31e30576120ea246a6d538096f48b42` |
| [seed_000_bp64_mask_47.json](<F:/agi/data/loops/audit-v1/seed_000_bp64_mask_47.json>) | `26ae943391710f67b861b058996f21fd638a509c8cea89484c89868b26db93d1` |
| [seed_000_bp_replay_mask_11.json](<F:/agi/data/loops/audit-v1/seed_000_bp_replay_mask_11.json>) | `07d799d9c584590256d6fcdcbc701c6c2b7031a015ea511839de0309c761408b` |
| [seed_000_bp_replay_mask_29.json](<F:/agi/data/loops/audit-v1/seed_000_bp_replay_mask_29.json>) | `129feae9aa31cdee8c29a42bfd3aac29b276baa4d69225db9558d118dd3dc345` |
| [seed_000_bp_replay_mask_47.json](<F:/agi/data/loops/audit-v1/seed_000_bp_replay_mask_47.json>) | `1ea658d83e27e37d3f0877165cb4c87e728a62cc666c11ca6483d2ef1f944f43` |
| [seed_001_A_mask_11.json](<F:/agi/data/loops/audit-v1/seed_001_A_mask_11.json>) | `434cf8f9972360a3c7ffb6871b661f99e4cf8c98efd307b42db1c47ddb6f400f` |
| [seed_001_A_mask_29.json](<F:/agi/data/loops/audit-v1/seed_001_A_mask_29.json>) | `99e6c96ab45bd49b6dd7c149affb43e7b4578bc0779e1f431763790c22d622b8` |
| [seed_001_A_mask_47.json](<F:/agi/data/loops/audit-v1/seed_001_A_mask_47.json>) | `e8e53ed2a6fea00cedb42f50a72e39871e90a0c434df25e93d451a2475b9d9da` |
| [seed_001_bp64_mask_11.json](<F:/agi/data/loops/audit-v1/seed_001_bp64_mask_11.json>) | `a8a0689797de3ad785ce63b2d67cb686aa7138a881c5cc8988c88f7f09f1659c` |
| [seed_001_bp64_mask_29.json](<F:/agi/data/loops/audit-v1/seed_001_bp64_mask_29.json>) | `4508bcdf034674a2052a29ba8901c24f3c453d3fe33740557abc4e88cc6e40f4` |
| [seed_001_bp64_mask_47.json](<F:/agi/data/loops/audit-v1/seed_001_bp64_mask_47.json>) | `ab04e2d4ba5fd074d6245abee60faa7ac1333fe14ab36e1c0dfc219d8201110f` |
| [seed_001_bp_replay_mask_11.json](<F:/agi/data/loops/audit-v1/seed_001_bp_replay_mask_11.json>) | `fba6f154b379ee25d16e228d3f030413f46c05b6deb85d11a2a9819a94c490b8` |
| [seed_001_bp_replay_mask_29.json](<F:/agi/data/loops/audit-v1/seed_001_bp_replay_mask_29.json>) | `1352883ec0df5fff9fc9378ee52e90dcfba0e3b789febd7835d2d928cf30f985` |
| [seed_001_bp_replay_mask_47.json](<F:/agi/data/loops/audit-v1/seed_001_bp_replay_mask_47.json>) | `933015a227ab6626fdcb36dec8090ee7e3dc99f74be45bbc4e13ac61e9b53ec7` |
| [seed_002_A_mask_11.json](<F:/agi/data/loops/audit-v1/seed_002_A_mask_11.json>) | `ab6977d617bf96df87d5fce8099f60d9d9d2890e602ff81c72d4c8fbb196a51d` |
| [seed_002_A_mask_29.json](<F:/agi/data/loops/audit-v1/seed_002_A_mask_29.json>) | `cb5357519d4b76ae7f879b45d73d200929df921b532c250314b23a7fa12145cd` |
| [seed_002_A_mask_47.json](<F:/agi/data/loops/audit-v1/seed_002_A_mask_47.json>) | `89ed3ed9f5d840926242550c66cfc36a6285aecdcd58261801450c0630b9ca3d` |
| [seed_002_bp64_mask_11.json](<F:/agi/data/loops/audit-v1/seed_002_bp64_mask_11.json>) | `7805c56ce99bb0949bc648e3f6a6e1b746017586f96932a0c4a4e85b539c807f` |
| [seed_002_bp64_mask_29.json](<F:/agi/data/loops/audit-v1/seed_002_bp64_mask_29.json>) | `f9ca3cce61c92acd39274f97525ea89b1411992f09976c5cc05ad13357ed18d1` |
| [seed_002_bp64_mask_47.json](<F:/agi/data/loops/audit-v1/seed_002_bp64_mask_47.json>) | `9cb6affcfed8e33063b5619eea378d981e21766874260464e516742592ca784f` |
| [seed_002_bp_replay_mask_11.json](<F:/agi/data/loops/audit-v1/seed_002_bp_replay_mask_11.json>) | `11015919e8f5594390aae9f5f90169b88f0813a7a6ca74e73f6ec390d03dd0c3` |
| [seed_002_bp_replay_mask_29.json](<F:/agi/data/loops/audit-v1/seed_002_bp_replay_mask_29.json>) | `fe8bcd96560c5ad0af7a62e1f05d6a545444870f039350d325b05a122d4a34ee` |
| [seed_002_bp_replay_mask_47.json](<F:/agi/data/loops/audit-v1/seed_002_bp_replay_mask_47.json>) | `51d773699664ad2c5ad9bbf53204b28c3e1b2b8adcdfe4fe39637869f5a4a61b` |
