# iGenVS-ultra cold speed benchmark

Completed: 2026-09-06T16:15:26.444778+00:00

Each row is one cold timing sample on Arrhenius GH200 120GB GPUs. No warm-up run and no repeated timing sample were used. All public-wrapper performance controls remained `auto`; docking used score-only output and screening used only the count argument with cross-batch overlap.

## Docking

| Engine | Mode | GPUs | Input | Finite success | Yield | Complete wall (s) | Input/h | Successful/h | Engine-only successful/h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unidock | fast | 1 | 20,000 | 19,679 | 98.395% | 302.967 | 237,649 | 233,835 | 242,218 |
| unidock | fast | 2 | 40,000 | 39,338 | 98.345% | 301.827 | 477,095 | 469,199 | 485,652 |
| unidock | fast | 4 | 80,000 | 78,692 | 98.365% | 319.341 | 901,859 | 887,113 | 931,350 |
| unidock | balance | 1 | 20,000 | 19,676 | 98.380% | 1,002.441 | 71,825 | 70,661 | 71,401 |
| unidock | balance | 2 | 40,000 | 39,322 | 98.305% | 1,020.373 | 141,125 | 138,733 | 140,307 |
| unidock | balance | 4 | 80,000 | 78,699 | 98.374% | 1,011.047 | 284,853 | 280,221 | 283,773 |
| unidock | detail | 1 | 20,000 | 19,650 | 98.250% | 1,262.779 | 57,017 | 56,019 | 56,491 |
| unidock | detail | 2 | 40,000 | 39,324 | 98.310% | 1,283.153 | 112,224 | 110,327 | 111,373 |
| unidock | detail | 4 | 80,000 | 78,684 | 98.355% | 1,308.415 | 220,114 | 216,493 | 218,568 |
| autodock-gpu | fast | 1 | 20,000 | 19,818 | 99.090% | 1,993.594 | 36,116 | 35,787 | 36,013 |
| autodock-gpu | fast | 2 | 40,000 | 39,615 | 99.037% | 2,021.262 | 71,243 | 70,557 | 71,097 |
| autodock-gpu | fast | 4 | 80,000 | 79,253 | 99.066% | 2,044.646 | 140,856 | 139,540 | 140,864 |

Docking timings are seconds. Preparation CPU is summed across workers; engine worker time is summed across GPU shards. Critical values are the slowest concurrent shard.

| Case | Target | Validate | Prep critical | Prep CPU sum | Engine critical | Engine worker sum | Results critical | Cleanup critical | Wrapper docking stage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unidock-fast-1gpu-cold | 0.079 | 0.595 | 5.271 | 1,388.956 | 292.482 | 291.378 | 0.136 | 0.415 | 302.781 |
| unidock-fast-2gpu-cold | 0.080 | 1.838 | 5.552 | 2,598.954 | 291.601 | 579.349 | 0.129 | 1.224 | 301.592 |
| unidock-fast-4gpu-cold | 0.074 | 4.986 | 5.740 | 5,338.802 | 304.173 | 1,198.472 | 0.137 | 0.508 | 319.158 |
| unidock-balance-1gpu-cold | 0.081 | 0.609 | 5.210 | 1,339.639 | 992.052 | 990.924 | 0.132 | 0.406 | 1,002.244 |
| unidock-balance-2gpu-cold | 0.079 | 1.676 | 5.650 | 2,625.097 | 1,008.923 | 2,012.045 | 0.136 | 0.397 | 1,020.157 |
| unidock-balance-4gpu-cold | 0.082 | 3.091 | 5.656 | 5,232.423 | 998.390 | 3,948.401 | 0.132 | 0.471 | 1,010.857 |
| unidock-detail-1gpu-cold | 0.079 | 0.602 | 5.307 | 1,398.093 | 1,252.235 | 1,251.073 | 0.142 | 0.388 | 1,262.600 |
| unidock-detail-2gpu-cold | 0.085 | 2.163 | 6.146 | 2,731.260 | 1,271.097 | 2,505.717 | 0.138 | 0.394 | 1,282.853 |
| unidock-detail-4gpu-cold | 0.073 | 2.888 | 5.569 | 5,265.287 | 1,295.993 | 5,052.078 | 0.136 | 0.401 | 1,308.265 |
| autodock-gpu-fast-1gpu-cold | 0.076 | 0.572 | 6.458 | 1,333.481 | 1,981.107 | 11,522.194 | 0.225 | 0.777 | 1,993.354 |
| autodock-gpu-fast-2gpu-cold | 0.075 | 4.477 | 5.898 | 2,654.604 | 2,005.902 | 23,048.115 | 0.317 | 1.094 | 2,021.076 |
| autodock-gpu-fast-4gpu-cold | 0.075 | 5.089 | 5.885 | 5,341.972 | 2,025.428 | 46,734.744 | 0.282 | 1.356 | 2,044.428 |

## Ultra screening

| GPUs | Committed finite scores | Candidate yield | Encoding yield | Complete wall (s) | Screening/h | Candidate generation/h | Encoding/h | Head inference/h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10,000,000 | 91.182% | 100.000% | 1,151.134 | 31,273,519 | 42,740,588 | 64,589,878 | 17,348,807,030 |
| 2 | 10,000,000 | 91.182% | 100.000% | 716.952 | 50,212,570 | 83,966,683 | 153,059,150 | 29,285,556,779 |
| 4 | 10,000,000 | 91.182% | 100.000% | 506.403 | 71,089,653 | 163,066,224 | 430,357,414 | 157,640,140,384 |

Screening timings are seconds and stage sums may exceed complete wall time because generation and scoring overlap across batches. GPU-sharded stage values use each batch's slowest shard before summing batches.

| GPUs | Batches | Startup | Generate | Validate | Admit | Policy | Encode | Heads | Score stage | Batch wall | Shutdown | Finalize |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 11 | 31.695 | 923.744 | 5.393 | 122.122 | 68.194 | 557.363 | 2.075 | 784.202 | 1,103.340 | 10.284 | 4.741 |
| 2 | 9 | 32.019 | 470.202 | 5.626 | 120.669 | 35.846 | 235.203 | 1.229 | 361.338 | 665.252 | 12.691 | 5.962 |
| 4 | 8 | 24.703 | 242.118 | 6.134 | 121.345 | 17.658 | 83.651 | 0.228 | 163.451 | 450.041 | 23.687 | 6.517 |

## Audit

- Protocol SHA-256: `151c50c694da4dd58b819d6104831a9ae41825eb390f92799f533bbb8f930a9b`
- Locked at: `2026-09-06T15:12:44.830322+00:00`
- Full machine snapshots, resolved automatic plans, commands, hashes, counts, and   unrounded measurements are retained in each case's `summary.json`.
