# Readme.md

An implementation of [Deep contextualized word representations](https://arxiv.org/abs/1802.05365) of Peters et al. (2018).
The module includes character embedding of [Kim et al. (2015)](https://arxiv.org/pdf/1508.06615.pdf).

# Usage



# Experimental

## Result

The model achieved best score at Epoch 07 which is early of the training (valid score).

```
Epoch: 07 | Time: 6m 40s
	Train Loss: 0.717 | Train PPL:   2.048
	 Val. Loss: 1.428 |  Val. PPL:   4.171
```

The entire training/validation losses are shown in below.

```
Epoch: 01 | Time: 6m 42s
	Train Loss: 3.745 | Train PPL:  42.303
	 Val. Loss: 2.486 |  Val. PPL:  12.014
Epoch: 02 | Time: 6m 36s
	Train Loss: 2.140 | Train PPL:   8.503
	 Val. Loss: 2.020 |  Val. PPL:   7.540
Epoch: 03 | Time: 6m 30s
	Train Loss: 1.612 | Train PPL:   5.013
	 Val. Loss: 1.711 |  Val. PPL:   5.533
Epoch: 04 | Time: 6m 33s
	Train Loss: 1.290 | Train PPL:   3.634
	 Val. Loss: 1.590 |  Val. PPL:   4.906
Epoch: 05 | Time: 6m 33s
	Train Loss: 1.047 | Train PPL:   2.849
	 Val. Loss: 1.536 |  Val. PPL:   4.646
Epoch: 06 | Time: 6m 39s
	Train Loss: 0.859 | Train PPL:   2.360
	 Val. Loss: 1.477 |  Val. PPL:   4.381
Epoch: 07 | Time: 6m 40s
	Train Loss: 0.717 | Train PPL:   2.048
	 Val. Loss: 1.428 |  Val. PPL:   4.171
Epoch: 08 | Time: 6m 44s
	Train Loss: 0.606 | Train PPL:   1.832
	 Val. Loss: 1.453 |  Val. PPL:   4.276
Epoch: 09 | Time: 6m 45s
	Train Loss: 0.519 | Train PPL:   1.680
	 Val. Loss: 1.477 |  Val. PPL:   4.379
Epoch: 10 | Time: 6m 41s
	Train Loss: 0.446 | Train PPL:   1.562
	 Val. Loss: 1.441 |  Val. PPL:   4.226
Epoch: 11 | Time: 6m 57s
	Train Loss: 0.389 | Train PPL:   1.475
	 Val. Loss: 1.464 |  Val. PPL:   4.323
Epoch: 12 | Time: 6m 57s
	Train Loss: 0.343 | Train PPL:   1.410
	 Val. Loss: 1.471 |  Val. PPL:   4.353
Epoch: 13 | Time: 7m 2s
	Train Loss: 0.306 | Train PPL:   1.358
	 Val. Loss: 1.520 |  Val. PPL:   4.572
Epoch: 14 | Time: 7m 4s
	Train Loss: 0.274 | Train PPL:   1.315
	 Val. Loss: 1.505 |  Val. PPL:   4.506
Epoch: 15 | Time: 7m 0s
	Train Loss: 0.249 | Train PPL:   1.283
	 Val. Loss: 1.530 |  Val. PPL:   4.618
Epoch: 16 | Time: 6m 41s
	Train Loss: 0.227 | Train PPL:   1.255
	 Val. Loss: 1.555 |  Val. PPL:   4.737
Epoch: 17 | Time: 6m 41s
	Train Loss: 0.209 | Train PPL:   1.232
	 Val. Loss: 1.581 |  Val. PPL:   4.858
Epoch: 18 | Time: 6m 48s
	Train Loss: 0.192 | Train PPL:   1.212
	 Val. Loss: 1.600 |  Val. PPL:   4.951
Epoch: 19 | Time: 7m 7s
	Train Loss: 0.178 | Train PPL:   1.195
	 Val. Loss: 1.610 |  Val. PPL:   5.004
Epoch: 20 | Time: 7m 2s
	Train Loss: 0.167 | Train PPL:   1.181
	 Val. Loss: 1.639 |  Val. PPL:   5.150
Epoch: 21 | Time: 7m 14s
	Train Loss: 0.158 | Train PPL:   1.171
	 Val. Loss: 1.668 |  Val. PPL:   5.300
Epoch: 22 | Time: 7m 22s
	Train Loss: 0.149 | Train PPL:   1.161
	 Val. Loss: 1.671 |  Val. PPL:   5.319
Epoch: 23 | Time: 7m 28s
	Train Loss: 0.141 | Train PPL:   1.151
	 Val. Loss: 1.705 |  Val. PPL:   5.499
Epoch: 24 | Time: 7m 31s
	Train Loss: 0.135 | Train PPL:   1.144
	 Val. Loss: 1.723 |  Val. PPL:   5.600
Epoch: 25 | Time: 7m 11s
	Train Loss: 0.128 | Train PPL:   1.136
	 Val. Loss: 1.740 |  Val. PPL:   5.697
Epoch: 26 | Time: 7m 1s
	Train Loss: 0.123 | Train PPL:   1.130
	 Val. Loss: 1.781 |  Val. PPL:   5.938
Epoch: 27 | Time: 7m 23s
	Train Loss: 0.118 | Train PPL:   1.125
	 Val. Loss: 1.771 |  Val. PPL:   5.874
Epoch: 28 | Time: 7m 29s
	Train Loss: 0.115 | Train PPL:   1.122
	 Val. Loss: 1.813 |  Val. PPL:   6.129
Epoch: 29 | Time: 7m 33s
	Train Loss: 0.110 | Train PPL:   1.116
	 Val. Loss: 1.826 |  Val. PPL:   6.207
Epoch: 30 | Time: 7m 28s
	Train Loss: 0.107 | Train PPL:   1.113
	 Val. Loss: 1.863 |  Val. PPL:   6.446
Epoch: 31 | Time: 7m 33s
	Train Loss: 0.103 | Train PPL:   1.109
	 Val. Loss: 1.848 |  Val. PPL:   6.350
Epoch: 32 | Time: 7m 39s
	Train Loss: 0.101 | Train PPL:   1.106
	 Val. Loss: 1.867 |  Val. PPL:   6.471
Epoch: 33 | Time: 7m 35s
	Train Loss: 0.097 | Train PPL:   1.102
	 Val. Loss: 1.881 |  Val. PPL:   6.558
Epoch: 34 | Time: 7m 34s
	Train Loss: 0.095 | Train PPL:   1.099
	 Val. Loss: 1.891 |  Val. PPL:   6.628
Epoch: 35 | Time: 7m 33s
	Train Loss: 0.092 | Train PPL:   1.097
	 Val. Loss: 1.913 |  Val. PPL:   6.775
Epoch: 36 | Time: 7m 36s
	Train Loss: 0.090 | Train PPL:   1.094
	 Val. Loss: 1.920 |  Val. PPL:   6.822
Epoch: 37 | Time: 7m 30s
	Train Loss: 0.088 | Train PPL:   1.092
	 Val. Loss: 1.929 |  Val. PPL:   6.884
Epoch: 38 | Time: 7m 30s
	Train Loss: 0.088 | Train PPL:   1.092
	 Val. Loss: 1.960 |  Val. PPL:   7.098
Epoch: 39 | Time: 7m 29s
	Train Loss: 0.086 | Train PPL:   1.090
	 Val. Loss: 1.960 |  Val. PPL:   7.101
Epoch: 40 | Time: 7m 29s
	Train Loss: 0.084 | Train PPL:   1.087
	 Val. Loss: 1.965 |  Val. PPL:   7.138
Epoch: 41 | Time: 7m 25s
	Train Loss: 0.082 | Train PPL:   1.085
	 Val. Loss: 1.985 |  Val. PPL:   7.277
Epoch: 42 | Time: 7m 26s
	Train Loss: 0.080 | Train PPL:   1.084
	 Val. Loss: 1.995 |  Val. PPL:   7.354
Epoch: 43 | Time: 7m 30s
	Train Loss: 0.079 | Train PPL:   1.082
	 Val. Loss: 1.996 |  Val. PPL:   7.360
Epoch: 44 | Time: 7m 34s
	Train Loss: 0.078 | Train PPL:   1.081
	 Val. Loss: 2.044 |  Val. PPL:   7.724
Epoch: 45 | Time: 7m 37s
	Train Loss: 0.077 | Train PPL:   1.080
	 Val. Loss: 2.044 |  Val. PPL:   7.721
Epoch: 46 | Time: 7m 31s
	Train Loss: 0.076 | Train PPL:   1.079
	 Val. Loss: 2.041 |  Val. PPL:   7.699
Epoch: 47 | Time: 7m 32s
	Train Loss: 0.074 | Train PPL:   1.077
	 Val. Loss: 2.049 |  Val. PPL:   7.757
Epoch: 48 | Time: 7m 30s
	Train Loss: 0.073 | Train PPL:   1.075
	 Val. Loss: 2.069 |  Val. PPL:   7.919
Epoch: 49 | Time: 7m 30s
	Train Loss: 0.072 | Train PPL:   1.075
	 Val. Loss: 2.076 |  Val. PPL:   7.970
Epoch: 50 | Time: 7m 25s
	Train Loss: 0.071 | Train PPL:   1.074
	 Val. Loss: 2.085 |  Val. PPL:   8.046
Epoch: 51 | Time: 7m 29s
	Train Loss: 0.070 | Train PPL:   1.073
	 Val. Loss: 2.102 |  Val. PPL:   8.185
Epoch: 52 | Time: 7m 22s
	Train Loss: 0.069 | Train PPL:   1.072
	 Val. Loss: 2.110 |  Val. PPL:   8.250
Epoch: 53 | Time: 7m 11s
	Train Loss: 0.068 | Train PPL:   1.070
	 Val. Loss: 2.113 |  Val. PPL:   8.273
Epoch: 54 | Time: 7m 14s
	Train Loss: 0.067 | Train PPL:   1.069
	 Val. Loss: 2.128 |  Val. PPL:   8.395
Epoch: 55 | Time: 7m 20s
	Train Loss: 0.066 | Train PPL:   1.068
	 Val. Loss: 2.137 |  Val. PPL:   8.476
Epoch: 56 | Time: 7m 32s
	Train Loss: 0.067 | Train PPL:   1.069
	 Val. Loss: 2.169 |  Val. PPL:   8.749
Epoch: 57 | Time: 7m 34s
	Train Loss: 0.065 | Train PPL:   1.068
	 Val. Loss: 2.168 |  Val. PPL:   8.737
Epoch: 58 | Time: 7m 35s
	Train Loss: 0.064 | Train PPL:   1.066
	 Val. Loss: 2.169 |  Val. PPL:   8.746
Epoch: 59 | Time: 7m 35s
	Train Loss: 0.064 | Train PPL:   1.066
	 Val. Loss: 2.214 |  Val. PPL:   9.157
Epoch: 60 | Time: 7m 32s
	Train Loss: 0.063 | Train PPL:   1.065
	 Val. Loss: 2.201 |  Val. PPL:   9.032
Epoch: 61 | Time: 7m 36s
	Train Loss: 0.062 | Train PPL:   1.064
	 Val. Loss: 2.203 |  Val. PPL:   9.054
Epoch: 62 | Time: 7m 33s
	Train Loss: 0.061 | Train PPL:   1.063
	 Val. Loss: 2.217 |  Val. PPL:   9.178
Epoch: 63 | Time: 7m 29s
	Train Loss: 0.061 | Train PPL:   1.063
	 Val. Loss: 2.228 |  Val. PPL:   9.281
Epoch: 64 | Time: 7m 29s
	Train Loss: 0.063 | Train PPL:   1.065
	 Val. Loss: 2.257 |  Val. PPL:   9.552
Epoch: 65 | Time: 7m 34s
	Train Loss: 0.061 | Train PPL:   1.063
	 Val. Loss: 2.283 |  Val. PPL:   9.804
Epoch: 66 | Time: 7m 28s
	Train Loss: 0.060 | Train PPL:   1.062
	 Val. Loss: 2.256 |  Val. PPL:   9.545
Epoch: 67 | Time: 7m 32s
	Train Loss: 0.060 | Train PPL:   1.062
	 Val. Loss: 2.253 |  Val. PPL:   9.518
Epoch: 68 | Time: 7m 22s
	Train Loss: 0.060 | Train PPL:   1.062
	 Val. Loss: 2.282 |  Val. PPL:   9.798
Epoch: 69 | Time: 7m 8s
	Train Loss: 0.059 | Train PPL:   1.061
	 Val. Loss: 2.293 |  Val. PPL:   9.900
Epoch: 70 | Time: 7m 12s
	Train Loss: 0.059 | Train PPL:   1.061
	 Val. Loss: 2.290 |  Val. PPL:   9.878
Epoch: 71 | Time: 7m 16s
	Train Loss: 0.058 | Train PPL:   1.060
	 Val. Loss: 2.296 |  Val. PPL:   9.936
Epoch: 72 | Time: 7m 24s
	Train Loss: 0.058 | Train PPL:   1.060
	 Val. Loss: 2.323 |  Val. PPL:  10.208
Epoch: 73 | Time: 7m 24s
	Train Loss: 0.057 | Train PPL:   1.059
	 Val. Loss: 2.321 |  Val. PPL:  10.190
Epoch: 74 | Time: 7m 30s
	Train Loss: 0.058 | Train PPL:   1.060
	 Val. Loss: 2.317 |  Val. PPL:  10.146
Epoch: 75 | Time: 7m 36s
	Train Loss: 0.058 | Train PPL:   1.059
	 Val. Loss: 2.324 |  Val. PPL:  10.217
Epoch: 76 | Time: 7m 39s
	Train Loss: 0.057 | Train PPL:   1.059
	 Val. Loss: 2.313 |  Val. PPL:  10.103
Epoch: 77 | Time: 7m 36s
	Train Loss: 0.056 | Train PPL:   1.058
	 Val. Loss: 2.349 |  Val. PPL:  10.480
Epoch: 78 | Time: 7m 32s
	Train Loss: 0.056 | Train PPL:   1.058
	 Val. Loss: 2.356 |  Val. PPL:  10.554
Epoch: 79 | Time: 7m 30s
	Train Loss: 0.056 | Train PPL:   1.057
	 Val. Loss: 2.380 |  Val. PPL:  10.810
Epoch: 80 | Time: 7m 34s
	Train Loss: 0.056 | Train PPL:   1.057
	 Val. Loss: 2.357 |  Val. PPL:  10.556
Epoch: 81 | Time: 7m 23s
	Train Loss: 0.057 | Train PPL:   1.059
	 Val. Loss: 2.384 |  Val. PPL:  10.846
Epoch: 82 | Time: 7m 8s
	Train Loss: 0.055 | Train PPL:   1.057
	 Val. Loss: 2.402 |  Val. PPL:  11.048
Epoch: 83 | Time: 7m 5s
	Train Loss: 0.055 | Train PPL:   1.057
	 Val. Loss: 2.401 |  Val. PPL:  11.036
Epoch: 84 | Time: 6m 54s
	Train Loss: 0.055 | Train PPL:   1.056
	 Val. Loss: 2.404 |  Val. PPL:  11.062
Epoch: 85 | Time: 6m 49s
	Train Loss: 0.055 | Train PPL:   1.057
	 Val. Loss: 2.426 |  Val. PPL:  11.313
Epoch: 86 | Time: 6m 50s
	Train Loss: 0.055 | Train PPL:   1.056
	 Val. Loss: 2.411 |  Val. PPL:  11.147
Epoch: 87 | Time: 6m 56s
	Train Loss: 0.055 | Train PPL:   1.056
	 Val. Loss: 2.438 |  Val. PPL:  11.445
Epoch: 88 | Time: 7m 5s
	Train Loss: 0.055 | Train PPL:   1.057
	 Val. Loss: 2.420 |  Val. PPL:  11.241
Epoch: 89 | Time: 7m 7s
	Train Loss: 0.055 | Train PPL:   1.057
	 Val. Loss: 2.457 |  Val. PPL:  11.672
Epoch: 90 | Time: 6m 56s
	Train Loss: 0.054 | Train PPL:   1.056
	 Val. Loss: 2.462 |  Val. PPL:  11.730
Epoch: 91 | Time: 6m 57s
	Train Loss: 0.055 | Train PPL:   1.056
	 Val. Loss: 2.455 |  Val. PPL:  11.650
Epoch: 92 | Time: 7m 0s
	Train Loss: 0.054 | Train PPL:   1.055
	 Val. Loss: 2.471 |  Val. PPL:  11.838
Epoch: 93 | Time: 6m 58s
	Train Loss: 0.054 | Train PPL:   1.055
	 Val. Loss: 2.453 |  Val. PPL:  11.623
Epoch: 94 | Time: 7m 2s
	Train Loss: 0.054 | Train PPL:   1.055
	 Val. Loss: 2.480 |  Val. PPL:  11.944
Epoch: 95 | Time: 7m 2s
	Train Loss: 0.054 | Train PPL:   1.056
	 Val. Loss: 2.476 |  Val. PPL:  11.898
Epoch: 96 | Time: 6m 57s
	Train Loss: 0.054 | Train PPL:   1.056
	 Val. Loss: 2.483 |  Val. PPL:  11.981
Epoch: 97 | Time: 7m 1s
	Train Loss: 0.054 | Train PPL:   1.056
	 Val. Loss: 2.492 |  Val. PPL:  12.085
Epoch: 98 | Time: 7m 2s
	Train Loss: 0.054 | Train PPL:   1.055
	 Val. Loss: 2.498 |  Val. PPL:  12.163
Epoch: 99 | Time: 6m 59s
	Train Loss: 0.054 | Train PPL:   1.056
	 Val. Loss: 2.501 |  Val. PPL:  12.191
Epoch: 100 | Time: 6m 56s
	Train Loss: 0.054 | Train PPL:   1.055
	 Val. Loss: 2.520 |  Val. PPL:  12.430
```