
## SMOKE 04:18

### seed 0 [smoke]  Ndb=40,000  LM-only=3.3446  brute=+17.06% (λ=0.70)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +17.01% | 100% |
| dodeca-uniform P2 probe2 | 1x | +17.08% | 100% |
| dodeca-WHITENED P2 probe1 | 124x | +10.53% | 62% |
| dodeca-WHITENED P2 probe2 | 33x | +15.34% | 90% |
| dodeca-WHITENED P3 probe1 | 920x | +4.37% | 26% |
| dodeca-WHITENED P3 probe2 | 153x | +11.08% | 65% |
| IVF nlist256 nprobe1 | 216x | +16.29% | 95% |
| IVF nlist256 nprobe2 | 112x | +16.39% | 96% |
| IVF nlist256 nprobe4 | 58x | +16.33% | 96% |
| IVF nlist256 nprobe8 | 30x | +16.45% | 96% |
| IVF nlist256 nprobe16 | 15x | +16.67% | 98% |
| IVF nlist256 nprobe32 | 7x | +17.01% | 100% |
## SMOKE OK

# INDEX-DONE-RIGHT autonomous run — start 2026-05-30 04:26 (budget 9.5h)
Frontier: gain-retained vs speedup. brute ref | uniform-dodeca | whitened-dodeca | IVF(nprobe).
Honest Q: does whitened substrate addressing match IVF on learned-float keys, or does IVF win?

### seed 0 [tiny->prid ds300K s0]  Ndb=300,000  LM-only=2.7768  brute=+10.66% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 3x | +10.13% | 95% |
| dodeca-uniform P2 probe2 | 2x | +10.55% | 99% |
| dodeca-WHITENED P2 probe1 | 122x | +7.41% | 69% |
| dodeca-WHITENED P2 probe2 | 33x | +9.70% | 91% |
| dodeca-WHITENED P3 probe1 | 814x | +5.36% | 50% |
| dodeca-WHITENED P3 probe2 | 143x | +9.36% | 88% |
| IVF nlist256 nprobe1 | 193x | +10.54% | 99% |
| IVF nlist256 nprobe2 | 105x | +10.59% | 99% |
| IVF nlist256 nprobe4 | 56x | +10.61% | 99% |
| IVF nlist256 nprobe8 | 29x | +10.64% | 100% |
| IVF nlist256 nprobe16 | 15x | +10.65% | 100% |
| IVF nlist256 nprobe32 | 8x | +10.68% | 100% |
| IVF nlist1024 nprobe1 | 741x | +9.84% | 92% |
| IVF nlist1024 nprobe2 | 412x | +10.37% | 97% |
| IVF nlist1024 nprobe4 | 217x | +10.41% | 98% |
| IVF nlist1024 nprobe8 | 113x | +10.41% | 98% |
| IVF nlist1024 nprobe16 | 58x | +10.41% | 98% |
| IVF nlist1024 nprobe32 | 30x | +10.65% | 100% |
_(tiny->prid ds300K s0 done in 1043s | 1 runs | 0.3h/9.5h elapsed)_

### seed 0 [prid->tiny ds300K s0]  Ndb=300,000  LM-only=3.1073  brute=+17.63% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 3x | +16.86% | 96% |
| dodeca-uniform P2 probe2 | 2x | +17.55% | 100% |
| dodeca-WHITENED P2 probe1 | 122x | +14.28% | 81% |
| dodeca-WHITENED P2 probe2 | 33x | +16.91% | 96% |
| dodeca-WHITENED P3 probe1 | 886x | +9.78% | 55% |
| dodeca-WHITENED P3 probe2 | 148x | +16.18% | 92% |
| IVF nlist256 nprobe1 | 206x | +17.68% | 100% |
| IVF nlist256 nprobe2 | 109x | +17.65% | 100% |
| IVF nlist256 nprobe4 | 56x | +17.63% | 100% |
| IVF nlist256 nprobe8 | 29x | +17.63% | 100% |
| IVF nlist256 nprobe16 | 14x | +17.62% | 100% |
| IVF nlist256 nprobe32 | 7x | +17.62% | 100% |
| IVF nlist1024 nprobe1 | 760x | +17.45% | 99% |
| IVF nlist1024 nprobe2 | 415x | +17.49% | 99% |
| IVF nlist1024 nprobe4 | 222x | +17.60% | 100% |
| IVF nlist1024 nprobe8 | 117x | +17.62% | 100% |
| IVF nlist1024 nprobe16 | 60x | +17.62% | 100% |
| IVF nlist1024 nprobe32 | 31x | +17.62% | 100% |
_(prid->tiny ds300K s0 done in 752s | 2 runs | 0.5h/9.5h elapsed)_

### seed 0 [tiny->omc_ ds300K s0]  Ndb=300,000  LM-only=5.7216  brute=+49.36% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +48.93% | 99% |
| dodeca-uniform P2 probe2 | 1x | +49.26% | 100% |
| dodeca-WHITENED P2 probe1 | 122x | +47.33% | 96% |
| dodeca-WHITENED P2 probe2 | 33x | +47.91% | 97% |
| dodeca-WHITENED P3 probe1 | 997x | +41.95% | 85% |
| dodeca-WHITENED P3 probe2 | 154x | +47.59% | 96% |
| IVF nlist256 nprobe1 | 203x | +49.22% | 100% |
| IVF nlist256 nprobe2 | 106x | +49.31% | 100% |
| IVF nlist256 nprobe4 | 56x | +49.29% | 100% |
| IVF nlist256 nprobe8 | 29x | +49.33% | 100% |
| IVF nlist256 nprobe16 | 14x | +49.33% | 100% |
| IVF nlist256 nprobe32 | 7x | +49.34% | 100% |
| IVF nlist1024 nprobe1 | 841x | +48.17% | 98% |
| IVF nlist1024 nprobe2 | 439x | +49.07% | 99% |
| IVF nlist1024 nprobe4 | 229x | +49.18% | 100% |
| IVF nlist1024 nprobe8 | 119x | +49.21% | 100% |
| IVF nlist1024 nprobe16 | 61x | +49.22% | 100% |
| IVF nlist1024 nprobe32 | 31x | +49.22% | 100% |
_(tiny->omc_ ds300K s0 done in 104s | 3 runs | 0.5h/9.5h elapsed)_

### seed 0 [tiny->prid ds600K s0]  Ndb=600,000  LM-only=2.7768  brute=+10.82% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 3x | +10.17% | 94% |
| dodeca-uniform P2 probe2 | 2x | +10.80% | 100% |
| dodeca-WHITENED P2 probe1 | 125x | +9.07% | 84% |
| dodeca-WHITENED P2 probe2 | 33x | +10.84% | 100% |
| dodeca-WHITENED P3 probe1 | 867x | +6.50% | 60% |
| dodeca-WHITENED P3 probe2 | 142x | +10.32% | 95% |
| IVF nlist256 nprobe1 | 198x | +10.28% | 95% |
| IVF nlist256 nprobe2 | 105x | +10.44% | 96% |
| IVF nlist256 nprobe4 | 56x | +10.74% | 99% |
| IVF nlist256 nprobe8 | 29x | +10.77% | 99% |
| IVF nlist256 nprobe16 | 14x | +10.79% | 100% |
| IVF nlist256 nprobe32 | 7x | +10.81% | 100% |
| IVF nlist1024 nprobe1 | 775x | +10.33% | 95% |
| IVF nlist1024 nprobe2 | 422x | +10.53% | 97% |
| IVF nlist1024 nprobe4 | 218x | +10.77% | 99% |
| IVF nlist1024 nprobe8 | 114x | +10.79% | 100% |
| IVF nlist1024 nprobe16 | 58x | +10.79% | 100% |
| IVF nlist1024 nprobe32 | 30x | +10.80% | 100% |
_(tiny->prid ds600K s0 done in 139s | 4 runs | 0.6h/9.5h elapsed)_

### seed 0 [prid->tiny ds600K s0]  Ndb=600,000  LM-only=3.1073  brute=+17.50% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 3x | +17.17% | 98% |
| dodeca-uniform P2 probe2 | 2x | +17.57% | 100% |
| dodeca-WHITENED P2 probe1 | 124x | +15.59% | 89% |
| dodeca-WHITENED P2 probe2 | 33x | +17.56% | 100% |
| dodeca-WHITENED P3 probe1 | 860x | +12.73% | 73% |
| dodeca-WHITENED P3 probe2 | 143x | +17.45% | 100% |
| IVF nlist256 nprobe1 | 207x | +17.44% | 100% |
| IVF nlist256 nprobe2 | 108x | +17.49% | 100% |
| IVF nlist256 nprobe4 | 56x | +17.51% | 100% |
| IVF nlist256 nprobe8 | 29x | +17.51% | 100% |
| IVF nlist256 nprobe16 | 15x | +17.51% | 100% |
| IVF nlist256 nprobe32 | 8x | +17.51% | 100% |
| IVF nlist1024 nprobe1 | 787x | +17.31% | 99% |
| IVF nlist1024 nprobe2 | 428x | +17.43% | 100% |
| IVF nlist1024 nprobe4 | 226x | +17.50% | 100% |
| IVF nlist1024 nprobe8 | 118x | +17.51% | 100% |
| IVF nlist1024 nprobe16 | 60x | +17.51% | 100% |
| IVF nlist1024 nprobe32 | 31x | +17.51% | 100% |
_(prid->tiny ds600K s0 done in 133s | 5 runs | 0.6h/9.5h elapsed)_

### seed 0 [tiny->omc_ ds600K s0]  Ndb=600,000  LM-only=5.7216  brute=+49.64% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +49.53% | 100% |
| dodeca-uniform P2 probe2 | 1x | +49.56% | 100% |
| dodeca-WHITENED P2 probe1 | 121x | +46.98% | 95% |
| dodeca-WHITENED P2 probe2 | 32x | +48.84% | 98% |
| dodeca-WHITENED P3 probe1 | 898x | +45.87% | 92% |
| dodeca-WHITENED P3 probe2 | 151x | +49.58% | 100% |
| IVF nlist256 nprobe1 | 210x | +49.22% | 99% |
| IVF nlist256 nprobe2 | 109x | +49.74% | 100% |
| IVF nlist256 nprobe4 | 56x | +49.71% | 100% |
| IVF nlist256 nprobe8 | 29x | +49.73% | 100% |
| IVF nlist256 nprobe16 | 15x | +49.74% | 100% |
| IVF nlist256 nprobe32 | 8x | +49.75% | 100% |
| IVF nlist1024 nprobe1 | 838x | +49.78% | 100% |
| IVF nlist1024 nprobe2 | 444x | +49.85% | 100% |
| IVF nlist1024 nprobe4 | 229x | +49.71% | 100% |
| IVF nlist1024 nprobe8 | 118x | +49.73% | 100% |
| IVF nlist1024 nprobe16 | 60x | +49.73% | 100% |
| IVF nlist1024 nprobe32 | 31x | +49.73% | 100% |
_(tiny->omc_ ds600K s0 done in 138s | 6 runs | 0.6h/9.5h elapsed)_

### seed 1 [tiny->prid ds300K s1]  Ndb=300,000  LM-only=2.8068  brute=+10.69% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +10.60% | 99% |
| dodeca-uniform P2 probe2 | 1x | +10.70% | 100% |
| dodeca-WHITENED P2 probe1 | 117x | +8.65% | 81% |
| dodeca-WHITENED P2 probe2 | 31x | +10.55% | 99% |
| dodeca-WHITENED P3 probe1 | 903x | +6.93% | 65% |
| dodeca-WHITENED P3 probe2 | 150x | +9.81% | 92% |
| IVF nlist256 nprobe1 | 189x | +10.31% | 96% |
| IVF nlist256 nprobe2 | 104x | +10.44% | 98% |
| IVF nlist256 nprobe4 | 56x | +10.66% | 100% |
| IVF nlist256 nprobe8 | 29x | +10.67% | 100% |
| IVF nlist256 nprobe16 | 14x | +10.67% | 100% |
| IVF nlist256 nprobe32 | 7x | +10.69% | 100% |
| IVF nlist1024 nprobe1 | 769x | +9.92% | 93% |
| IVF nlist1024 nprobe2 | 419x | +10.70% | 100% |
| IVF nlist1024 nprobe4 | 220x | +10.64% | 99% |
| IVF nlist1024 nprobe8 | 116x | +10.66% | 100% |
| IVF nlist1024 nprobe16 | 60x | +10.67% | 100% |
| IVF nlist1024 nprobe32 | 30x | +10.67% | 100% |
_(tiny->prid ds300K s1 done in 100s | 7 runs | 0.7h/9.5h elapsed)_

### seed 1 [prid->tiny ds300K s1]  Ndb=300,000  LM-only=3.0177  brute=+15.42% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +15.42% | 100% |
| dodeca-uniform P2 probe2 | 1x | +15.42% | 100% |
| dodeca-WHITENED P2 probe1 | 115x | +13.87% | 90% |
| dodeca-WHITENED P2 probe2 | 31x | +15.34% | 100% |
| dodeca-WHITENED P3 probe1 | 827x | +9.27% | 60% |
| dodeca-WHITENED P3 probe2 | 141x | +13.59% | 88% |
| IVF nlist256 nprobe1 | 198x | +15.43% | 100% |
| IVF nlist256 nprobe2 | 106x | +15.42% | 100% |
| IVF nlist256 nprobe4 | 55x | +15.41% | 100% |
| IVF nlist256 nprobe8 | 29x | +15.41% | 100% |
| IVF nlist256 nprobe16 | 14x | +15.42% | 100% |
| IVF nlist256 nprobe32 | 7x | +15.42% | 100% |
| IVF nlist1024 nprobe1 | 761x | +14.99% | 97% |
| IVF nlist1024 nprobe2 | 416x | +15.24% | 99% |
| IVF nlist1024 nprobe4 | 220x | +15.41% | 100% |
| IVF nlist1024 nprobe8 | 114x | +15.42% | 100% |
| IVF nlist1024 nprobe16 | 58x | +15.42% | 100% |
| IVF nlist1024 nprobe32 | 30x | +15.42% | 100% |
_(prid->tiny ds300K s1 done in 105s | 8 runs | 0.7h/9.5h elapsed)_

### seed 1 [tiny->omc_ ds300K s1]  Ndb=300,000  LM-only=5.4816  brute=+48.07% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +48.07% | 100% |
| dodeca-uniform P2 probe2 | 1x | +48.07% | 100% |
| dodeca-WHITENED P2 probe1 | 120x | +46.42% | 97% |
| dodeca-WHITENED P2 probe2 | 32x | +47.71% | 99% |
| dodeca-WHITENED P3 probe1 | 819x | +40.67% | 85% |
| dodeca-WHITENED P3 probe2 | 140x | +46.87% | 97% |
| IVF nlist256 nprobe1 | 198x | +47.69% | 99% |
| IVF nlist256 nprobe2 | 105x | +47.91% | 100% |
| IVF nlist256 nprobe4 | 55x | +48.00% | 100% |
| IVF nlist256 nprobe8 | 29x | +48.04% | 100% |
| IVF nlist256 nprobe16 | 15x | +48.06% | 100% |
| IVF nlist256 nprobe32 | 8x | +48.07% | 100% |
| IVF nlist1024 nprobe1 | 830x | +48.00% | 100% |
| IVF nlist1024 nprobe2 | 431x | +48.15% | 100% |
| IVF nlist1024 nprobe4 | 222x | +48.03% | 100% |
| IVF nlist1024 nprobe8 | 114x | +48.07% | 100% |
| IVF nlist1024 nprobe16 | 59x | +48.07% | 100% |
| IVF nlist1024 nprobe32 | 30x | +48.07% | 100% |
_(tiny->omc_ ds300K s1 done in 105s | 9 runs | 0.7h/9.5h elapsed)_

### seed 1 [tiny->prid ds600K s1]  Ndb=600,000  LM-only=2.8068  brute=+11.13% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +11.18% | 100% |
| dodeca-uniform P2 probe2 | 1x | +11.14% | 100% |
| dodeca-WHITENED P2 probe1 | 105x | +10.57% | 95% |
| dodeca-WHITENED P2 probe2 | 29x | +11.36% | 102% |
| dodeca-WHITENED P3 probe1 | 845x | +8.08% | 73% |
| dodeca-WHITENED P3 probe2 | 144x | +10.85% | 97% |
| IVF nlist256 nprobe1 | 206x | +11.03% | 99% |
| IVF nlist256 nprobe2 | 107x | +11.10% | 100% |
| IVF nlist256 nprobe4 | 55x | +11.12% | 100% |
| IVF nlist256 nprobe8 | 29x | +11.14% | 100% |
| IVF nlist256 nprobe16 | 15x | +11.14% | 100% |
| IVF nlist256 nprobe32 | 7x | +11.14% | 100% |
| IVF nlist1024 nprobe1 | 792x | +10.92% | 98% |
| IVF nlist1024 nprobe2 | 430x | +11.11% | 100% |
| IVF nlist1024 nprobe4 | 222x | +11.13% | 100% |
| IVF nlist1024 nprobe8 | 115x | +11.12% | 100% |
| IVF nlist1024 nprobe16 | 60x | +11.12% | 100% |
| IVF nlist1024 nprobe32 | 30x | +11.12% | 100% |
_(tiny->prid ds600K s1 done in 137s | 10 runs | 0.8h/9.5h elapsed)_

### seed 1 [prid->tiny ds600K s1]  Ndb=600,000  LM-only=3.0177  brute=+15.44% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +15.44% | 100% |
| dodeca-uniform P2 probe2 | 1x | +15.44% | 100% |
| dodeca-WHITENED P2 probe1 | 117x | +14.06% | 91% |
| dodeca-WHITENED P2 probe2 | 31x | +15.60% | 101% |
| dodeca-WHITENED P3 probe1 | 808x | +10.84% | 70% |
| dodeca-WHITENED P3 probe2 | 142x | +14.95% | 97% |
| IVF nlist256 nprobe1 | 195x | +15.43% | 100% |
| IVF nlist256 nprobe2 | 104x | +15.44% | 100% |
| IVF nlist256 nprobe4 | 54x | +15.44% | 100% |
| IVF nlist256 nprobe8 | 28x | +15.44% | 100% |
| IVF nlist256 nprobe16 | 15x | +15.44% | 100% |
| IVF nlist256 nprobe32 | 7x | +15.44% | 100% |
| IVF nlist1024 nprobe1 | 778x | +15.47% | 100% |
| IVF nlist1024 nprobe2 | 425x | +15.45% | 100% |
| IVF nlist1024 nprobe4 | 225x | +15.46% | 100% |
| IVF nlist1024 nprobe8 | 116x | +15.46% | 100% |
| IVF nlist1024 nprobe16 | 59x | +15.46% | 100% |
| IVF nlist1024 nprobe32 | 30x | +15.46% | 100% |
_(prid->tiny ds600K s1 done in 145s | 11 runs | 0.8h/9.5h elapsed)_

### seed 1 [tiny->omc_ ds600K s1]  Ndb=600,000  LM-only=5.4816  brute=+48.25% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +48.27% | 100% |
| dodeca-uniform P2 probe2 | 1x | +48.25% | 100% |
| dodeca-WHITENED P2 probe1 | 119x | +46.61% | 97% |
| dodeca-WHITENED P2 probe2 | 32x | +47.98% | 99% |
| dodeca-WHITENED P3 probe1 | 756x | +44.29% | 92% |
| dodeca-WHITENED P3 probe2 | 133x | +47.58% | 99% |
| IVF nlist256 nprobe1 | 207x | +48.06% | 100% |
| IVF nlist256 nprobe2 | 106x | +48.19% | 100% |
| IVF nlist256 nprobe4 | 56x | +48.21% | 100% |
| IVF nlist256 nprobe8 | 30x | +48.22% | 100% |
| IVF nlist256 nprobe16 | 15x | +48.24% | 100% |
| IVF nlist256 nprobe32 | 8x | +48.26% | 100% |
| IVF nlist1024 nprobe1 | 844x | +48.04% | 100% |
| IVF nlist1024 nprobe2 | 443x | +48.26% | 100% |
| IVF nlist1024 nprobe4 | 228x | +48.27% | 100% |
| IVF nlist1024 nprobe8 | 117x | +48.26% | 100% |
| IVF nlist1024 nprobe16 | 60x | +48.26% | 100% |
| IVF nlist1024 nprobe32 | 31x | +48.25% | 100% |
_(tiny->omc_ ds600K s1 done in 147s | 12 runs | 0.8h/9.5h elapsed)_

### seed 2 [tiny->prid ds300K s2]  Ndb=300,000  LM-only=2.7590  brute=+10.04% (λ=0.75)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +9.66% | 96% |
| dodeca-uniform P2 probe2 | 1x | +10.05% | 100% |
| dodeca-WHITENED P2 probe1 | 126x | +7.72% | 77% |
| dodeca-WHITENED P2 probe2 | 33x | +9.79% | 98% |
| dodeca-WHITENED P3 probe1 | 957x | +5.78% | 58% |
| dodeca-WHITENED P3 probe2 | 158x | +8.85% | 88% |
| IVF nlist256 nprobe1 | 203x | +9.63% | 96% |
| IVF nlist256 nprobe2 | 109x | +9.76% | 97% |
| IVF nlist256 nprobe4 | 58x | +9.80% | 98% |
| IVF nlist256 nprobe8 | 30x | +9.83% | 98% |
| IVF nlist256 nprobe16 | 15x | +9.84% | 98% |
| IVF nlist256 nprobe32 | 8x | +10.02% | 100% |
| IVF nlist1024 nprobe1 | 760x | +9.53% | 95% |
| IVF nlist1024 nprobe2 | 413x | +9.88% | 98% |
| IVF nlist1024 nprobe4 | 218x | +9.82% | 98% |
| IVF nlist1024 nprobe8 | 114x | +9.84% | 98% |
| IVF nlist1024 nprobe16 | 58x | +9.84% | 98% |
| IVF nlist1024 nprobe32 | 30x | +10.01% | 100% |
_(tiny->prid ds300K s2 done in 103s | 13 runs | 0.9h/9.5h elapsed)_

### seed 2 [prid->tiny ds300K s2]  Ndb=300,000  LM-only=3.0671  brute=+16.82% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +16.82% | 100% |
| dodeca-uniform P2 probe2 | 1x | +16.82% | 100% |
| dodeca-WHITENED P2 probe1 | 123x | +14.02% | 83% |
| dodeca-WHITENED P2 probe2 | 33x | +16.87% | 100% |
| dodeca-WHITENED P3 probe1 | 1017x | +10.17% | 60% |
| dodeca-WHITENED P3 probe2 | 164x | +15.58% | 93% |
| IVF nlist256 nprobe1 | 199x | +16.98% | 101% |
| IVF nlist256 nprobe2 | 106x | +16.81% | 100% |
| IVF nlist256 nprobe4 | 54x | +16.83% | 100% |
| IVF nlist256 nprobe8 | 27x | +16.82% | 100% |
| IVF nlist256 nprobe16 | 14x | +16.82% | 100% |
| IVF nlist256 nprobe32 | 7x | +16.82% | 100% |
| IVF nlist1024 nprobe1 | 747x | +16.85% | 100% |
| IVF nlist1024 nprobe2 | 409x | +16.91% | 101% |
| IVF nlist1024 nprobe4 | 217x | +16.92% | 101% |
| IVF nlist1024 nprobe8 | 113x | +16.82% | 100% |
| IVF nlist1024 nprobe16 | 58x | +16.82% | 100% |
| IVF nlist1024 nprobe32 | 30x | +16.82% | 100% |
_(prid->tiny ds300K s2 done in 104s | 14 runs | 0.9h/9.5h elapsed)_

### seed 2 [tiny->omc_ ds300K s2]  Ndb=300,000  LM-only=5.6946  brute=+49.58% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +49.58% | 100% |
| dodeca-uniform P2 probe2 | 1x | +49.58% | 100% |
| dodeca-WHITENED P2 probe1 | 122x | +47.98% | 97% |
| dodeca-WHITENED P2 probe2 | 32x | +48.89% | 99% |
| dodeca-WHITENED P3 probe1 | 801x | +41.63% | 84% |
| dodeca-WHITENED P3 probe2 | 142x | +48.03% | 97% |
| IVF nlist256 nprobe1 | 212x | +48.69% | 98% |
| IVF nlist256 nprobe2 | 107x | +49.01% | 99% |
| IVF nlist256 nprobe4 | 56x | +49.45% | 100% |
| IVF nlist256 nprobe8 | 29x | +49.56% | 100% |
| IVF nlist256 nprobe16 | 15x | +49.57% | 100% |
| IVF nlist256 nprobe32 | 8x | +49.58% | 100% |
| IVF nlist1024 nprobe1 | 879x | +48.88% | 99% |
| IVF nlist1024 nprobe2 | 460x | +49.24% | 99% |
| IVF nlist1024 nprobe4 | 236x | +49.41% | 100% |
| IVF nlist1024 nprobe8 | 120x | +49.30% | 99% |
| IVF nlist1024 nprobe16 | 61x | +49.56% | 100% |
| IVF nlist1024 nprobe32 | 31x | +49.58% | 100% |
_(tiny->omc_ ds300K s2 done in 106s | 15 runs | 0.9h/9.5h elapsed)_

### seed 2 [tiny->prid ds600K s2]  Ndb=600,000  LM-only=2.7590  brute=+10.05% (λ=0.75)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +9.79% | 97% |
| dodeca-uniform P2 probe2 | 1x | +10.05% | 100% |
| dodeca-WHITENED P2 probe1 | 120x | +7.67% | 76% |
| dodeca-WHITENED P2 probe2 | 32x | +9.47% | 94% |
| dodeca-WHITENED P3 probe1 | 982x | +6.61% | 66% |
| dodeca-WHITENED P3 probe2 | 155x | +9.36% | 93% |
| IVF nlist256 nprobe1 | 200x | +9.55% | 95% |
| IVF nlist256 nprobe2 | 109x | +9.80% | 97% |
| IVF nlist256 nprobe4 | 57x | +9.80% | 97% |
| IVF nlist256 nprobe8 | 29x | +9.99% | 99% |
| IVF nlist256 nprobe16 | 15x | +10.01% | 100% |
| IVF nlist256 nprobe32 | 8x | +10.03% | 100% |
| IVF nlist1024 nprobe1 | 764x | +9.50% | 94% |
| IVF nlist1024 nprobe2 | 414x | +9.84% | 98% |
| IVF nlist1024 nprobe4 | 219x | +9.83% | 98% |
| IVF nlist1024 nprobe8 | 113x | +9.84% | 98% |
| IVF nlist1024 nprobe16 | 59x | +10.03% | 100% |
| IVF nlist1024 nprobe32 | 30x | +10.03% | 100% |
_(tiny->prid ds600K s2 done in 142s | 16 runs | 1.0h/9.5h elapsed)_

### seed 2 [prid->tiny ds600K s2]  Ndb=600,000  LM-only=3.0671  brute=+17.07% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +17.07% | 100% |
| dodeca-uniform P2 probe2 | 1x | +17.08% | 100% |
| dodeca-WHITENED P2 probe1 | 121x | +15.52% | 91% |
| dodeca-WHITENED P2 probe2 | 32x | +17.03% | 100% |
| dodeca-WHITENED P3 probe1 | 975x | +11.31% | 66% |
| dodeca-WHITENED P3 probe2 | 160x | +16.36% | 96% |
| IVF nlist256 nprobe1 | 192x | +17.08% | 100% |
| IVF nlist256 nprobe2 | 102x | +17.06% | 100% |
| IVF nlist256 nprobe4 | 54x | +17.06% | 100% |
| IVF nlist256 nprobe8 | 28x | +17.06% | 100% |
| IVF nlist256 nprobe16 | 15x | +17.06% | 100% |
| IVF nlist256 nprobe32 | 7x | +17.06% | 100% |
| IVF nlist1024 nprobe1 | 787x | +17.19% | 101% |
| IVF nlist1024 nprobe2 | 431x | +17.12% | 100% |
| IVF nlist1024 nprobe4 | 227x | +17.05% | 100% |
| IVF nlist1024 nprobe8 | 118x | +17.06% | 100% |
| IVF nlist1024 nprobe16 | 60x | +17.06% | 100% |
| IVF nlist1024 nprobe32 | 30x | +17.06% | 100% |
_(prid->tiny ds600K s2 done in 146s | 17 runs | 1.0h/9.5h elapsed)_

### seed 2 [tiny->omc_ ds600K s2]  Ndb=600,000  LM-only=5.6946  brute=+48.84% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +48.84% | 100% |
| dodeca-uniform P2 probe2 | 1x | +48.84% | 100% |
| dodeca-WHITENED P2 probe1 | 113x | +47.96% | 98% |
| dodeca-WHITENED P2 probe2 | 31x | +49.47% | 101% |
| dodeca-WHITENED P3 probe1 | 907x | +44.40% | 91% |
| dodeca-WHITENED P3 probe2 | 153x | +49.34% | 101% |
| IVF nlist256 nprobe1 | 204x | +48.37% | 99% |
| IVF nlist256 nprobe2 | 105x | +48.68% | 100% |
| IVF nlist256 nprobe4 | 55x | +48.85% | 100% |
| IVF nlist256 nprobe8 | 28x | +48.84% | 100% |
| IVF nlist256 nprobe16 | 15x | +48.84% | 100% |
| IVF nlist256 nprobe32 | 8x | +48.84% | 100% |
| IVF nlist1024 nprobe1 | 855x | +48.71% | 100% |
| IVF nlist1024 nprobe2 | 446x | +49.01% | 100% |
| IVF nlist1024 nprobe4 | 228x | +48.77% | 100% |
| IVF nlist1024 nprobe8 | 116x | +48.82% | 100% |
| IVF nlist1024 nprobe16 | 59x | +48.84% | 100% |
| IVF nlist1024 nprobe32 | 30x | +48.84% | 100% |
_(tiny->omc_ ds600K s2 done in 139s | 18 runs | 1.1h/9.5h elapsed)_

### seed 3 [tiny->prid ds300K s3]  Ndb=300,000  LM-only=2.7713  brute=+9.56% (λ=0.75)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +9.52% | 100% |
| dodeca-uniform P2 probe2 | 1x | +9.56% | 100% |
| dodeca-WHITENED P2 probe1 | 125x | +7.54% | 79% |
| dodeca-WHITENED P2 probe2 | 33x | +9.54% | 100% |
| dodeca-WHITENED P3 probe1 | 941x | +6.10% | 64% |
| dodeca-WHITENED P3 probe2 | 155x | +9.35% | 98% |
| IVF nlist256 nprobe1 | 200x | +9.24% | 97% |
| IVF nlist256 nprobe2 | 109x | +9.26% | 97% |
| IVF nlist256 nprobe4 | 56x | +9.31% | 97% |
| IVF nlist256 nprobe8 | 29x | +9.55% | 100% |
| IVF nlist256 nprobe16 | 15x | +9.56% | 100% |
| IVF nlist256 nprobe32 | 7x | +9.56% | 100% |
| IVF nlist1024 nprobe1 | 762x | +8.65% | 91% |
| IVF nlist1024 nprobe2 | 411x | +8.96% | 94% |
| IVF nlist1024 nprobe4 | 218x | +9.10% | 95% |
| IVF nlist1024 nprobe8 | 115x | +9.31% | 97% |
| IVF nlist1024 nprobe16 | 60x | +9.54% | 100% |
| IVF nlist1024 nprobe32 | 30x | +9.55% | 100% |
_(tiny->prid ds300K s3 done in 109s | 19 runs | 1.1h/9.5h elapsed)_

### seed 3 [prid->tiny ds300K s3]  Ndb=300,000  LM-only=3.0817  brute=+17.00% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +17.00% | 100% |
| dodeca-uniform P2 probe2 | 1x | +17.00% | 100% |
| dodeca-WHITENED P2 probe1 | 118x | +14.60% | 86% |
| dodeca-WHITENED P2 probe2 | 32x | +17.19% | 101% |
| dodeca-WHITENED P3 probe1 | 856x | +10.50% | 62% |
| dodeca-WHITENED P3 probe2 | 151x | +16.20% | 95% |
| IVF nlist256 nprobe1 | 202x | +16.65% | 98% |
| IVF nlist256 nprobe2 | 108x | +16.91% | 100% |
| IVF nlist256 nprobe4 | 56x | +16.98% | 100% |
| IVF nlist256 nprobe8 | 28x | +16.99% | 100% |
| IVF nlist256 nprobe16 | 14x | +16.98% | 100% |
| IVF nlist256 nprobe32 | 7x | +16.98% | 100% |
| IVF nlist1024 nprobe1 | 803x | +17.03% | 100% |
| IVF nlist1024 nprobe2 | 435x | +16.71% | 98% |
| IVF nlist1024 nprobe4 | 230x | +16.77% | 99% |
| IVF nlist1024 nprobe8 | 119x | +16.99% | 100% |
| IVF nlist1024 nprobe16 | 60x | +16.99% | 100% |
| IVF nlist1024 nprobe32 | 31x | +16.99% | 100% |
_(prid->tiny ds300K s3 done in 101s | 20 runs | 1.1h/9.5h elapsed)_

### seed 3 [tiny->omc_ ds300K s3]  Ndb=300,000  LM-only=5.7216  brute=+49.72% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +49.71% | 100% |
| dodeca-uniform P2 probe2 | 1x | +49.69% | 100% |
| dodeca-WHITENED P2 probe1 | 117x | +47.09% | 95% |
| dodeca-WHITENED P2 probe2 | 32x | +49.16% | 99% |
| dodeca-WHITENED P3 probe1 | 807x | +43.53% | 88% |
| dodeca-WHITENED P3 probe2 | 142x | +48.75% | 98% |
| IVF nlist256 nprobe1 | 192x | +49.65% | 100% |
| IVF nlist256 nprobe2 | 103x | +49.64% | 100% |
| IVF nlist256 nprobe4 | 55x | +49.71% | 100% |
| IVF nlist256 nprobe8 | 29x | +49.72% | 100% |
| IVF nlist256 nprobe16 | 15x | +49.72% | 100% |
| IVF nlist256 nprobe32 | 7x | +49.72% | 100% |
| IVF nlist1024 nprobe1 | 761x | +48.88% | 98% |
| IVF nlist1024 nprobe2 | 403x | +49.43% | 99% |
| IVF nlist1024 nprobe4 | 211x | +49.56% | 100% |
| IVF nlist1024 nprobe8 | 112x | +49.67% | 100% |
| IVF nlist1024 nprobe16 | 58x | +49.72% | 100% |
| IVF nlist1024 nprobe32 | 30x | +49.72% | 100% |
_(tiny->omc_ ds300K s3 done in 100s | 21 runs | 1.1h/9.5h elapsed)_

### seed 3 [tiny->prid ds600K s3]  Ndb=600,000  LM-only=2.7713  brute=+9.70% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +9.68% | 100% |
| dodeca-uniform P2 probe2 | 1x | +9.70% | 100% |
| dodeca-WHITENED P2 probe1 | 128x | +8.60% | 89% |
| dodeca-WHITENED P2 probe2 | 33x | +10.37% | 107% |
| dodeca-WHITENED P3 probe1 | 957x | +6.14% | 63% |
| dodeca-WHITENED P3 probe2 | 160x | +9.59% | 99% |
| IVF nlist256 nprobe1 | 207x | +9.69% | 100% |
| IVF nlist256 nprobe2 | 109x | +9.67% | 100% |
| IVF nlist256 nprobe4 | 56x | +9.68% | 100% |
| IVF nlist256 nprobe8 | 29x | +9.69% | 100% |
| IVF nlist256 nprobe16 | 15x | +9.70% | 100% |
| IVF nlist256 nprobe32 | 8x | +9.70% | 100% |
| IVF nlist1024 nprobe1 | 787x | +8.95% | 92% |
| IVF nlist1024 nprobe2 | 428x | +9.40% | 97% |
| IVF nlist1024 nprobe4 | 227x | +9.44% | 97% |
| IVF nlist1024 nprobe8 | 118x | +9.46% | 98% |
| IVF nlist1024 nprobe16 | 60x | +9.71% | 100% |
| IVF nlist1024 nprobe32 | 31x | +9.71% | 100% |
_(tiny->prid ds600K s3 done in 140s | 22 runs | 1.2h/9.5h elapsed)_

### seed 3 [prid->tiny ds600K s3]  Ndb=600,000  LM-only=3.0817  brute=+17.60% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +17.61% | 100% |
| dodeca-uniform P2 probe2 | 1x | +17.61% | 100% |
| dodeca-WHITENED P2 probe1 | 122x | +16.01% | 91% |
| dodeca-WHITENED P2 probe2 | 33x | +17.53% | 100% |
| dodeca-WHITENED P3 probe1 | 878x | +11.94% | 68% |
| dodeca-WHITENED P3 probe2 | 149x | +16.36% | 93% |
| IVF nlist256 nprobe1 | 195x | +17.57% | 100% |
| IVF nlist256 nprobe2 | 104x | +17.63% | 100% |
| IVF nlist256 nprobe4 | 54x | +17.62% | 100% |
| IVF nlist256 nprobe8 | 28x | +17.61% | 100% |
| IVF nlist256 nprobe16 | 14x | +17.61% | 100% |
| IVF nlist256 nprobe32 | 7x | +17.61% | 100% |
| IVF nlist1024 nprobe1 | 744x | +17.69% | 101% |
| IVF nlist1024 nprobe2 | 416x | +17.63% | 100% |
| IVF nlist1024 nprobe4 | 225x | +17.62% | 100% |
| IVF nlist1024 nprobe8 | 116x | +17.62% | 100% |
| IVF nlist1024 nprobe16 | 59x | +17.61% | 100% |
| IVF nlist1024 nprobe32 | 30x | +17.61% | 100% |
_(prid->tiny ds600K s3 done in 143s | 23 runs | 1.2h/9.5h elapsed)_

### seed 3 [tiny->omc_ ds600K s3]  Ndb=600,000  LM-only=5.7216  brute=+49.55% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +49.49% | 100% |
| dodeca-uniform P2 probe2 | 1x | +49.53% | 100% |
| dodeca-WHITENED P2 probe1 | 117x | +47.64% | 96% |
| dodeca-WHITENED P2 probe2 | 32x | +48.96% | 99% |
| dodeca-WHITENED P3 probe1 | 760x | +46.68% | 94% |
| dodeca-WHITENED P3 probe2 | 137x | +49.39% | 100% |
| IVF nlist256 nprobe1 | 193x | +49.37% | 100% |
| IVF nlist256 nprobe2 | 105x | +49.57% | 100% |
| IVF nlist256 nprobe4 | 55x | +49.55% | 100% |
| IVF nlist256 nprobe8 | 28x | +49.55% | 100% |
| IVF nlist256 nprobe16 | 14x | +49.55% | 100% |
| IVF nlist256 nprobe32 | 7x | +49.55% | 100% |
| IVF nlist1024 nprobe1 | 794x | +49.63% | 100% |
| IVF nlist1024 nprobe2 | 429x | +49.54% | 100% |
| IVF nlist1024 nprobe4 | 221x | +49.55% | 100% |
| IVF nlist1024 nprobe8 | 115x | +49.55% | 100% |
| IVF nlist1024 nprobe16 | 60x | +49.55% | 100% |
| IVF nlist1024 nprobe32 | 30x | +49.55% | 100% |
_(tiny->omc_ ds600K s3 done in 136s | 24 runs | 1.3h/9.5h elapsed)_

### seed 4 [tiny->prid ds300K s4]  Ndb=300,000  LM-only=2.7905  brute=+11.30% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +11.27% | 100% |
| dodeca-uniform P2 probe2 | 1x | +11.29% | 100% |
| dodeca-WHITENED P2 probe1 | 123x | +8.84% | 78% |
| dodeca-WHITENED P2 probe2 | 33x | +10.74% | 95% |
| dodeca-WHITENED P3 probe1 | 871x | +6.80% | 60% |
| dodeca-WHITENED P3 probe2 | 141x | +9.80% | 87% |
| IVF nlist256 nprobe1 | 210x | +10.96% | 97% |
| IVF nlist256 nprobe2 | 111x | +11.26% | 100% |
| IVF nlist256 nprobe4 | 57x | +11.28% | 100% |
| IVF nlist256 nprobe8 | 29x | +11.29% | 100% |
| IVF nlist256 nprobe16 | 15x | +11.30% | 100% |
| IVF nlist256 nprobe32 | 8x | +11.30% | 100% |
| IVF nlist1024 nprobe1 | 792x | +10.56% | 93% |
| IVF nlist1024 nprobe2 | 433x | +11.01% | 97% |
| IVF nlist1024 nprobe4 | 229x | +11.01% | 97% |
| IVF nlist1024 nprobe8 | 117x | +11.05% | 98% |
| IVF nlist1024 nprobe16 | 59x | +11.04% | 98% |
| IVF nlist1024 nprobe32 | 30x | +11.30% | 100% |
_(tiny->prid ds300K s4 done in 105s | 25 runs | 1.3h/9.5h elapsed)_

### seed 4 [prid->tiny ds300K s4]  Ndb=300,000  LM-only=3.0189  brute=+15.74% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +15.67% | 100% |
| dodeca-uniform P2 probe2 | 1x | +15.71% | 100% |
| dodeca-WHITENED P2 probe1 | 121x | +12.86% | 82% |
| dodeca-WHITENED P2 probe2 | 33x | +14.89% | 95% |
| dodeca-WHITENED P3 probe1 | 838x | +9.40% | 60% |
| dodeca-WHITENED P3 probe2 | 143x | +14.80% | 94% |
| IVF nlist256 nprobe1 | 192x | +15.64% | 99% |
| IVF nlist256 nprobe2 | 103x | +15.72% | 100% |
| IVF nlist256 nprobe4 | 55x | +15.73% | 100% |
| IVF nlist256 nprobe8 | 29x | +15.73% | 100% |
| IVF nlist256 nprobe16 | 14x | +15.73% | 100% |
| IVF nlist256 nprobe32 | 7x | +15.73% | 100% |
| IVF nlist1024 nprobe1 | 769x | +15.15% | 96% |
| IVF nlist1024 nprobe2 | 417x | +15.64% | 99% |
| IVF nlist1024 nprobe4 | 224x | +15.82% | 101% |
| IVF nlist1024 nprobe8 | 116x | +15.73% | 100% |
| IVF nlist1024 nprobe16 | 59x | +15.74% | 100% |
| IVF nlist1024 nprobe32 | 30x | +15.74% | 100% |
_(prid->tiny ds300K s4 done in 101s | 26 runs | 1.3h/9.5h elapsed)_

### seed 4 [tiny->omc_ ds300K s4]  Ndb=300,000  LM-only=5.5723  brute=+48.73% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 3x | +48.34% | 99% |
| dodeca-uniform P2 probe2 | 1x | +48.71% | 100% |
| dodeca-WHITENED P2 probe1 | 118x | +45.65% | 94% |
| dodeca-WHITENED P2 probe2 | 32x | +47.98% | 98% |
| dodeca-WHITENED P3 probe1 | 943x | +40.62% | 83% |
| dodeca-WHITENED P3 probe2 | 150x | +46.87% | 96% |
| IVF nlist256 nprobe1 | 202x | +48.44% | 99% |
| IVF nlist256 nprobe2 | 109x | +48.65% | 100% |
| IVF nlist256 nprobe4 | 58x | +48.70% | 100% |
| IVF nlist256 nprobe8 | 30x | +48.71% | 100% |
| IVF nlist256 nprobe16 | 15x | +48.71% | 100% |
| IVF nlist256 nprobe32 | 8x | +48.72% | 100% |
| IVF nlist1024 nprobe1 | 839x | +47.95% | 98% |
| IVF nlist1024 nprobe2 | 446x | +48.45% | 99% |
| IVF nlist1024 nprobe4 | 233x | +48.62% | 100% |
| IVF nlist1024 nprobe8 | 119x | +48.72% | 100% |
| IVF nlist1024 nprobe16 | 60x | +48.73% | 100% |
| IVF nlist1024 nprobe32 | 31x | +48.73% | 100% |
_(tiny->omc_ ds300K s4 done in 98s | 27 runs | 1.3h/9.5h elapsed)_

### seed 4 [tiny->prid ds600K s4]  Ndb=600,000  LM-only=2.7905  brute=+11.11% (λ=0.75)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +11.15% | 100% |
| dodeca-uniform P2 probe2 | 1x | +11.11% | 100% |
| dodeca-WHITENED P2 probe1 | 125x | +10.18% | 92% |
| dodeca-WHITENED P2 probe2 | 33x | +11.72% | 105% |
| dodeca-WHITENED P3 probe1 | 881x | +7.47% | 67% |
| dodeca-WHITENED P3 probe2 | 143x | +11.01% | 99% |
| IVF nlist256 nprobe1 | 208x | +10.86% | 98% |
| IVF nlist256 nprobe2 | 108x | +10.83% | 97% |
| IVF nlist256 nprobe4 | 56x | +10.85% | 98% |
| IVF nlist256 nprobe8 | 29x | +11.08% | 100% |
| IVF nlist256 nprobe16 | 15x | +11.12% | 100% |
| IVF nlist256 nprobe32 | 7x | +11.13% | 100% |
| IVF nlist1024 nprobe1 | 795x | +10.83% | 97% |
| IVF nlist1024 nprobe2 | 432x | +10.86% | 98% |
| IVF nlist1024 nprobe4 | 225x | +10.86% | 98% |
| IVF nlist1024 nprobe8 | 116x | +10.86% | 98% |
| IVF nlist1024 nprobe16 | 59x | +10.87% | 98% |
| IVF nlist1024 nprobe32 | 30x | +11.13% | 100% |
_(tiny->prid ds600K s4 done in 141s | 28 runs | 1.4h/9.5h elapsed)_

### seed 4 [prid->tiny ds600K s4]  Ndb=600,000  LM-only=3.0189  brute=+15.54% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +15.56% | 100% |
| dodeca-uniform P2 probe2 | 1x | +15.50% | 100% |
| dodeca-WHITENED P2 probe1 | 126x | +13.89% | 89% |
| dodeca-WHITENED P2 probe2 | 33x | +15.31% | 99% |
| dodeca-WHITENED P3 probe1 | 744x | +11.65% | 75% |
| dodeca-WHITENED P3 probe2 | 134x | +15.35% | 99% |
| IVF nlist256 nprobe1 | 193x | +15.59% | 100% |
| IVF nlist256 nprobe2 | 103x | +15.51% | 100% |
| IVF nlist256 nprobe4 | 53x | +15.53% | 100% |
| IVF nlist256 nprobe8 | 28x | +15.52% | 100% |
| IVF nlist256 nprobe16 | 15x | +15.52% | 100% |
| IVF nlist256 nprobe32 | 8x | +15.51% | 100% |
| IVF nlist1024 nprobe1 | 760x | +15.58% | 100% |
| IVF nlist1024 nprobe2 | 407x | +15.58% | 100% |
| IVF nlist1024 nprobe4 | 216x | +15.53% | 100% |
| IVF nlist1024 nprobe8 | 114x | +15.53% | 100% |
| IVF nlist1024 nprobe16 | 57x | +15.53% | 100% |
| IVF nlist1024 nprobe32 | 30x | +15.53% | 100% |
_(prid->tiny ds600K s4 done in 143s | 29 runs | 1.4h/9.5h elapsed)_

### seed 4 [tiny->omc_ ds600K s4]  Ndb=600,000  LM-only=5.5723  brute=+48.95% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 3x | +48.52% | 99% |
| dodeca-uniform P2 probe2 | 1x | +48.92% | 100% |
| dodeca-WHITENED P2 probe1 | 122x | +47.24% | 97% |
| dodeca-WHITENED P2 probe2 | 33x | +48.72% | 100% |
| dodeca-WHITENED P3 probe1 | 816x | +44.52% | 91% |
| dodeca-WHITENED P3 probe2 | 138x | +48.04% | 98% |
| IVF nlist256 nprobe1 | 198x | +48.94% | 100% |
| IVF nlist256 nprobe2 | 107x | +49.03% | 100% |
| IVF nlist256 nprobe4 | 56x | +48.93% | 100% |
| IVF nlist256 nprobe8 | 30x | +48.93% | 100% |
| IVF nlist256 nprobe16 | 15x | +48.94% | 100% |
| IVF nlist256 nprobe32 | 8x | +48.94% | 100% |
| IVF nlist1024 nprobe1 | 814x | +48.90% | 100% |
| IVF nlist1024 nprobe2 | 434x | +48.95% | 100% |
| IVF nlist1024 nprobe4 | 227x | +48.92% | 100% |
| IVF nlist1024 nprobe8 | 118x | +48.94% | 100% |
| IVF nlist1024 nprobe16 | 60x | +48.94% | 100% |
| IVF nlist1024 nprobe32 | 30x | +48.94% | 100% |
_(tiny->omc_ ds600K s4 done in 137s | 30 runs | 1.5h/9.5h elapsed)_

### seed 5 [tiny->prid ds300K s5]  Ndb=300,000  LM-only=2.7924  brute=+11.60% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +11.53% | 99% |
| dodeca-uniform P2 probe2 | 1x | +11.63% | 100% |
| dodeca-WHITENED P2 probe1 | 119x | +9.02% | 78% |
| dodeca-WHITENED P2 probe2 | 32x | +10.49% | 90% |
| dodeca-WHITENED P3 probe1 | 966x | +4.55% | 39% |
| dodeca-WHITENED P3 probe2 | 157x | +9.17% | 79% |
| IVF nlist256 nprobe1 | 204x | +11.06% | 95% |
| IVF nlist256 nprobe2 | 108x | +11.06% | 95% |
| IVF nlist256 nprobe4 | 57x | +11.09% | 96% |
| IVF nlist256 nprobe8 | 29x | +11.32% | 98% |
| IVF nlist256 nprobe16 | 15x | +11.34% | 98% |
| IVF nlist256 nprobe32 | 7x | +11.35% | 98% |
| IVF nlist1024 nprobe1 | 795x | +10.58% | 91% |
| IVF nlist1024 nprobe2 | 432x | +11.19% | 96% |
| IVF nlist1024 nprobe4 | 224x | +11.30% | 97% |
| IVF nlist1024 nprobe8 | 117x | +11.33% | 98% |
| IVF nlist1024 nprobe16 | 60x | +11.32% | 98% |
| IVF nlist1024 nprobe32 | 30x | +11.57% | 100% |
_(tiny->prid ds300K s5 done in 104s | 31 runs | 1.5h/9.5h elapsed)_

### seed 5 [prid->tiny ds300K s5]  Ndb=300,000  LM-only=3.0944  brute=+16.98% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +16.87% | 99% |
| dodeca-uniform P2 probe2 | 1x | +16.98% | 100% |
| dodeca-WHITENED P2 probe1 | 124x | +14.67% | 86% |
| dodeca-WHITENED P2 probe2 | 33x | +16.72% | 99% |
| dodeca-WHITENED P3 probe1 | 948x | +9.75% | 57% |
| dodeca-WHITENED P3 probe2 | 153x | +15.45% | 91% |
| IVF nlist256 nprobe1 | 211x | +17.09% | 101% |
| IVF nlist256 nprobe2 | 109x | +17.03% | 100% |
| IVF nlist256 nprobe4 | 56x | +16.99% | 100% |
| IVF nlist256 nprobe8 | 29x | +16.98% | 100% |
| IVF nlist256 nprobe16 | 15x | +16.98% | 100% |
| IVF nlist256 nprobe32 | 7x | +16.98% | 100% |
| IVF nlist1024 nprobe1 | 823x | +16.83% | 99% |
| IVF nlist1024 nprobe2 | 441x | +16.80% | 99% |
| IVF nlist1024 nprobe4 | 232x | +16.93% | 100% |
| IVF nlist1024 nprobe8 | 119x | +16.98% | 100% |
| IVF nlist1024 nprobe16 | 61x | +16.98% | 100% |
| IVF nlist1024 nprobe32 | 31x | +16.98% | 100% |
_(prid->tiny ds300K s5 done in 100s | 32 runs | 1.5h/9.5h elapsed)_

### seed 5 [tiny->omc_ ds300K s5]  Ndb=300,000  LM-only=5.7640  brute=+50.16% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +49.93% | 100% |
| dodeca-uniform P2 probe2 | 1x | +50.14% | 100% |
| dodeca-WHITENED P2 probe1 | 118x | +48.34% | 96% |
| dodeca-WHITENED P2 probe2 | 32x | +50.12% | 100% |
| dodeca-WHITENED P3 probe1 | 930x | +41.74% | 83% |
| dodeca-WHITENED P3 probe2 | 156x | +48.77% | 97% |
| IVF nlist256 nprobe1 | 188x | +49.63% | 99% |
| IVF nlist256 nprobe2 | 105x | +50.12% | 100% |
| IVF nlist256 nprobe4 | 56x | +50.12% | 100% |
| IVF nlist256 nprobe8 | 30x | +50.12% | 100% |
| IVF nlist256 nprobe16 | 15x | +50.15% | 100% |
| IVF nlist256 nprobe32 | 8x | +50.15% | 100% |
| IVF nlist1024 nprobe1 | 816x | +49.03% | 98% |
| IVF nlist1024 nprobe2 | 435x | +50.33% | 100% |
| IVF nlist1024 nprobe4 | 225x | +50.32% | 100% |
| IVF nlist1024 nprobe8 | 116x | +50.24% | 100% |
| IVF nlist1024 nprobe16 | 59x | +50.11% | 100% |
| IVF nlist1024 nprobe32 | 30x | +50.13% | 100% |
_(tiny->omc_ ds300K s5 done in 102s | 33 runs | 1.5h/9.5h elapsed)_

### seed 5 [tiny->prid ds600K s5]  Ndb=600,000  LM-only=2.7924  brute=+11.34% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +11.32% | 100% |
| dodeca-uniform P2 probe2 | 1x | +11.34% | 100% |
| dodeca-WHITENED P2 probe1 | 117x | +10.27% | 91% |
| dodeca-WHITENED P2 probe2 | 32x | +11.48% | 101% |
| dodeca-WHITENED P3 probe1 | 997x | +8.20% | 72% |
| dodeca-WHITENED P3 probe2 | 158x | +10.97% | 97% |
| IVF nlist256 nprobe1 | 197x | +11.09% | 98% |
| IVF nlist256 nprobe2 | 103x | +11.16% | 98% |
| IVF nlist256 nprobe4 | 55x | +11.33% | 100% |
| IVF nlist256 nprobe8 | 29x | +11.33% | 100% |
| IVF nlist256 nprobe16 | 15x | +11.36% | 100% |
| IVF nlist256 nprobe32 | 7x | +11.36% | 100% |
| IVF nlist1024 nprobe1 | 788x | +10.98% | 97% |
| IVF nlist1024 nprobe2 | 423x | +11.12% | 98% |
| IVF nlist1024 nprobe4 | 222x | +11.13% | 98% |
| IVF nlist1024 nprobe8 | 115x | +11.08% | 98% |
| IVF nlist1024 nprobe16 | 59x | +11.31% | 100% |
| IVF nlist1024 nprobe32 | 30x | +11.33% | 100% |
_(tiny->prid ds600K s5 done in 151s | 34 runs | 1.6h/9.5h elapsed)_

### seed 5 [prid->tiny ds600K s5]  Ndb=600,000  LM-only=3.0944  brute=+17.25% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +17.20% | 100% |
| dodeca-uniform P2 probe2 | 1x | +17.25% | 100% |
| dodeca-WHITENED P2 probe1 | 121x | +15.45% | 90% |
| dodeca-WHITENED P2 probe2 | 32x | +16.57% | 96% |
| dodeca-WHITENED P3 probe1 | 897x | +12.20% | 71% |
| dodeca-WHITENED P3 probe2 | 148x | +16.35% | 95% |
| IVF nlist256 nprobe1 | 204x | +17.21% | 100% |
| IVF nlist256 nprobe2 | 106x | +17.26% | 100% |
| IVF nlist256 nprobe4 | 55x | +17.27% | 100% |
| IVF nlist256 nprobe8 | 28x | +17.26% | 100% |
| IVF nlist256 nprobe16 | 14x | +17.26% | 100% |
| IVF nlist256 nprobe32 | 8x | +17.26% | 100% |
| IVF nlist1024 nprobe1 | 803x | +16.96% | 98% |
| IVF nlist1024 nprobe2 | 432x | +17.35% | 101% |
| IVF nlist1024 nprobe4 | 227x | +17.25% | 100% |
| IVF nlist1024 nprobe8 | 116x | +17.26% | 100% |
| IVF nlist1024 nprobe16 | 59x | +17.26% | 100% |
| IVF nlist1024 nprobe32 | 30x | +17.26% | 100% |
_(prid->tiny ds600K s5 done in 140s | 35 runs | 1.6h/9.5h elapsed)_

### seed 5 [tiny->omc_ ds600K s5]  Ndb=600,000  LM-only=5.7640  brute=+50.14% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +50.12% | 100% |
| dodeca-uniform P2 probe2 | 1x | +50.13% | 100% |
| dodeca-WHITENED P2 probe1 | 120x | +49.04% | 98% |
| dodeca-WHITENED P2 probe2 | 32x | +50.63% | 101% |
| dodeca-WHITENED P3 probe1 | 909x | +46.56% | 93% |
| dodeca-WHITENED P3 probe2 | 154x | +49.51% | 99% |
| IVF nlist256 nprobe1 | 184x | +50.60% | 101% |
| IVF nlist256 nprobe2 | 102x | +50.38% | 100% |
| IVF nlist256 nprobe4 | 54x | +50.11% | 100% |
| IVF nlist256 nprobe8 | 28x | +50.12% | 100% |
| IVF nlist256 nprobe16 | 14x | +50.14% | 100% |
| IVF nlist256 nprobe32 | 7x | +50.14% | 100% |
| IVF nlist1024 nprobe1 | 846x | +50.55% | 101% |
| IVF nlist1024 nprobe2 | 446x | +50.55% | 101% |
| IVF nlist1024 nprobe4 | 228x | +50.32% | 100% |
| IVF nlist1024 nprobe8 | 116x | +50.22% | 100% |
| IVF nlist1024 nprobe16 | 60x | +50.23% | 100% |
| IVF nlist1024 nprobe32 | 30x | +50.24% | 100% |
_(tiny->omc_ ds600K s5 done in 138s | 36 runs | 1.7h/9.5h elapsed)_

### seed 6 [tiny->prid ds300K s6]  Ndb=300,000  LM-only=2.7762  brute=+10.94% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +10.85% | 99% |
| dodeca-uniform P2 probe2 | 1x | +10.92% | 100% |
| dodeca-WHITENED P2 probe1 | 107x | +8.89% | 81% |
| dodeca-WHITENED P2 probe2 | 30x | +10.74% | 98% |
| dodeca-WHITENED P3 probe1 | 995x | +5.93% | 54% |
| dodeca-WHITENED P3 probe2 | 154x | +9.46% | 86% |
| IVF nlist256 nprobe1 | 189x | +10.83% | 99% |
| IVF nlist256 nprobe2 | 102x | +10.83% | 99% |
| IVF nlist256 nprobe4 | 55x | +10.95% | 100% |
| IVF nlist256 nprobe8 | 28x | +10.95% | 100% |
| IVF nlist256 nprobe16 | 15x | +10.96% | 100% |
| IVF nlist256 nprobe32 | 7x | +10.96% | 100% |
| IVF nlist1024 nprobe1 | 787x | +10.57% | 97% |
| IVF nlist1024 nprobe2 | 428x | +10.79% | 99% |
| IVF nlist1024 nprobe4 | 223x | +10.94% | 100% |
| IVF nlist1024 nprobe8 | 115x | +10.94% | 100% |
| IVF nlist1024 nprobe16 | 59x | +10.95% | 100% |
| IVF nlist1024 nprobe32 | 30x | +10.94% | 100% |
_(tiny->prid ds300K s6 done in 104s | 37 runs | 1.7h/9.5h elapsed)_

### seed 6 [prid->tiny ds300K s6]  Ndb=300,000  LM-only=3.0449  brute=+15.84% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +15.81% | 100% |
| dodeca-uniform P2 probe2 | 1x | +15.82% | 100% |
| dodeca-WHITENED P2 probe1 | 102x | +13.88% | 88% |
| dodeca-WHITENED P2 probe2 | 29x | +16.30% | 103% |
| dodeca-WHITENED P3 probe1 | 844x | +9.33% | 59% |
| dodeca-WHITENED P3 probe2 | 144x | +14.59% | 92% |
| IVF nlist256 nprobe1 | 195x | +15.94% | 101% |
| IVF nlist256 nprobe2 | 104x | +15.88% | 100% |
| IVF nlist256 nprobe4 | 53x | +15.83% | 100% |
| IVF nlist256 nprobe8 | 28x | +15.82% | 100% |
| IVF nlist256 nprobe16 | 15x | +15.82% | 100% |
| IVF nlist256 nprobe32 | 7x | +15.82% | 100% |
| IVF nlist1024 nprobe1 | 780x | +15.24% | 96% |
| IVF nlist1024 nprobe2 | 425x | +15.83% | 100% |
| IVF nlist1024 nprobe4 | 225x | +15.81% | 100% |
| IVF nlist1024 nprobe8 | 116x | +15.83% | 100% |
| IVF nlist1024 nprobe16 | 59x | +15.83% | 100% |
| IVF nlist1024 nprobe32 | 30x | +15.83% | 100% |
_(prid->tiny ds300K s6 done in 103s | 38 runs | 1.7h/9.5h elapsed)_

### seed 6 [tiny->omc_ ds300K s6]  Ndb=300,000  LM-only=5.4885  brute=+47.18% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +47.33% | 100% |
| dodeca-uniform P2 probe2 | 1x | +47.19% | 100% |
| dodeca-WHITENED P2 probe1 | 103x | +45.33% | 96% |
| dodeca-WHITENED P2 probe2 | 28x | +47.07% | 100% |
| dodeca-WHITENED P3 probe1 | 796x | +41.05% | 87% |
| dodeca-WHITENED P3 probe2 | 138x | +46.22% | 98% |
| IVF nlist256 nprobe1 | 198x | +47.65% | 101% |
| IVF nlist256 nprobe2 | 107x | +47.18% | 100% |
| IVF nlist256 nprobe4 | 55x | +47.18% | 100% |
| IVF nlist256 nprobe8 | 29x | +47.17% | 100% |
| IVF nlist256 nprobe16 | 15x | +47.17% | 100% |
| IVF nlist256 nprobe32 | 8x | +47.17% | 100% |
| IVF nlist1024 nprobe1 | 789x | +46.45% | 98% |
| IVF nlist1024 nprobe2 | 428x | +46.94% | 99% |
| IVF nlist1024 nprobe4 | 224x | +47.02% | 100% |
| IVF nlist1024 nprobe8 | 116x | +47.17% | 100% |
| IVF nlist1024 nprobe16 | 59x | +47.17% | 100% |
| IVF nlist1024 nprobe32 | 30x | +47.17% | 100% |
_(tiny->omc_ ds300K s6 done in 102s | 39 runs | 1.7h/9.5h elapsed)_

### seed 6 [tiny->prid ds600K s6]  Ndb=600,000  LM-only=2.7762  brute=+11.32% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +11.21% | 99% |
| dodeca-uniform P2 probe2 | 1x | +11.30% | 100% |
| dodeca-WHITENED P2 probe1 | 111x | +9.54% | 84% |
| dodeca-WHITENED P2 probe2 | 30x | +10.94% | 97% |
| dodeca-WHITENED P3 probe1 | 1036x | +6.44% | 57% |
| dodeca-WHITENED P3 probe2 | 159x | +10.84% | 96% |
| IVF nlist256 nprobe1 | 200x | +11.22% | 99% |
| IVF nlist256 nprobe2 | 107x | +11.34% | 100% |
| IVF nlist256 nprobe4 | 57x | +11.35% | 100% |
| IVF nlist256 nprobe8 | 29x | +11.35% | 100% |
| IVF nlist256 nprobe16 | 15x | +11.35% | 100% |
| IVF nlist256 nprobe32 | 8x | +11.35% | 100% |
| IVF nlist1024 nprobe1 | 778x | +11.13% | 98% |
| IVF nlist1024 nprobe2 | 420x | +11.28% | 100% |
| IVF nlist1024 nprobe4 | 219x | +11.34% | 100% |
| IVF nlist1024 nprobe8 | 114x | +11.34% | 100% |
| IVF nlist1024 nprobe16 | 58x | +11.35% | 100% |
| IVF nlist1024 nprobe32 | 30x | +11.35% | 100% |
_(tiny->prid ds600K s6 done in 143s | 40 runs | 1.8h/9.5h elapsed)_

### seed 6 [prid->tiny ds600K s6]  Ndb=600,000  LM-only=3.0449  brute=+15.94% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +15.93% | 100% |
| dodeca-uniform P2 probe2 | 1x | +15.94% | 100% |
| dodeca-WHITENED P2 probe1 | 100x | +14.38% | 90% |
| dodeca-WHITENED P2 probe2 | 29x | +16.17% | 101% |
| dodeca-WHITENED P3 probe1 | 894x | +11.06% | 69% |
| dodeca-WHITENED P3 probe2 | 145x | +14.87% | 93% |
| IVF nlist256 nprobe1 | 201x | +15.86% | 100% |
| IVF nlist256 nprobe2 | 105x | +15.94% | 100% |
| IVF nlist256 nprobe4 | 54x | +15.99% | 100% |
| IVF nlist256 nprobe8 | 28x | +15.96% | 100% |
| IVF nlist256 nprobe16 | 15x | +15.96% | 100% |
| IVF nlist256 nprobe32 | 7x | +15.96% | 100% |
| IVF nlist1024 nprobe1 | 800x | +15.90% | 100% |
| IVF nlist1024 nprobe2 | 437x | +15.94% | 100% |
| IVF nlist1024 nprobe4 | 227x | +15.96% | 100% |
| IVF nlist1024 nprobe8 | 116x | +15.96% | 100% |
| IVF nlist1024 nprobe16 | 59x | +15.96% | 100% |
| IVF nlist1024 nprobe32 | 30x | +15.96% | 100% |
_(prid->tiny ds600K s6 done in 144s | 41 runs | 1.8h/9.5h elapsed)_

### seed 6 [tiny->omc_ ds600K s6]  Ndb=600,000  LM-only=5.4885  brute=+47.93% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +47.78% | 100% |
| dodeca-uniform P2 probe2 | 1x | +47.93% | 100% |
| dodeca-WHITENED P2 probe1 | 102x | +46.83% | 98% |
| dodeca-WHITENED P2 probe2 | 28x | +47.53% | 99% |
| dodeca-WHITENED P3 probe1 | 839x | +42.43% | 89% |
| dodeca-WHITENED P3 probe2 | 143x | +46.85% | 98% |
| IVF nlist256 nprobe1 | 205x | +47.99% | 100% |
| IVF nlist256 nprobe2 | 109x | +47.88% | 100% |
| IVF nlist256 nprobe4 | 56x | +47.89% | 100% |
| IVF nlist256 nprobe8 | 29x | +47.91% | 100% |
| IVF nlist256 nprobe16 | 15x | +47.92% | 100% |
| IVF nlist256 nprobe32 | 8x | +47.92% | 100% |
| IVF nlist1024 nprobe1 | 786x | +46.88% | 98% |
| IVF nlist1024 nprobe2 | 425x | +47.67% | 99% |
| IVF nlist1024 nprobe4 | 224x | +47.80% | 100% |
| IVF nlist1024 nprobe8 | 116x | +47.81% | 100% |
| IVF nlist1024 nprobe16 | 60x | +47.81% | 100% |
| IVF nlist1024 nprobe32 | 30x | +47.80% | 100% |
_(tiny->omc_ ds600K s6 done in 136s | 42 runs | 1.9h/9.5h elapsed)_

### seed 7 [tiny->prid ds300K s7]  Ndb=300,000  LM-only=2.7598  brute=+10.01% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +10.00% | 100% |
| dodeca-uniform P2 probe2 | 1x | +10.01% | 100% |
| dodeca-WHITENED P2 probe1 | 115x | +7.56% | 75% |
| dodeca-WHITENED P2 probe2 | 31x | +9.92% | 99% |
| dodeca-WHITENED P3 probe1 | 981x | +4.68% | 47% |
| dodeca-WHITENED P3 probe2 | 159x | +8.88% | 89% |
| IVF nlist256 nprobe1 | 206x | +10.00% | 100% |
| IVF nlist256 nprobe2 | 109x | +9.92% | 99% |
| IVF nlist256 nprobe4 | 57x | +9.99% | 100% |
| IVF nlist256 nprobe8 | 29x | +10.00% | 100% |
| IVF nlist256 nprobe16 | 15x | +10.00% | 100% |
| IVF nlist256 nprobe32 | 8x | +10.02% | 100% |
| IVF nlist1024 nprobe1 | 765x | +9.71% | 97% |
| IVF nlist1024 nprobe2 | 423x | +9.67% | 97% |
| IVF nlist1024 nprobe4 | 222x | +9.90% | 99% |
| IVF nlist1024 nprobe8 | 116x | +9.93% | 99% |
| IVF nlist1024 nprobe16 | 60x | +9.99% | 100% |
| IVF nlist1024 nprobe32 | 30x | +10.00% | 100% |
_(tiny->prid ds300K s7 done in 104s | 43 runs | 1.9h/9.5h elapsed)_

### seed 7 [prid->tiny ds300K s7]  Ndb=300,000  LM-only=3.0639  brute=+16.78% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +16.73% | 100% |
| dodeca-uniform P2 probe2 | 1x | +16.77% | 100% |
| dodeca-WHITENED P2 probe1 | 117x | +13.42% | 80% |
| dodeca-WHITENED P2 probe2 | 31x | +16.16% | 96% |
| dodeca-WHITENED P3 probe1 | 947x | +9.14% | 54% |
| dodeca-WHITENED P3 probe2 | 155x | +14.87% | 89% |
| IVF nlist256 nprobe1 | 198x | +16.73% | 100% |
| IVF nlist256 nprobe2 | 104x | +16.81% | 100% |
| IVF nlist256 nprobe4 | 53x | +16.78% | 100% |
| IVF nlist256 nprobe8 | 27x | +16.78% | 100% |
| IVF nlist256 nprobe16 | 14x | +16.78% | 100% |
| IVF nlist256 nprobe32 | 8x | +16.78% | 100% |
| IVF nlist1024 nprobe1 | 769x | +16.26% | 97% |
| IVF nlist1024 nprobe2 | 418x | +16.73% | 100% |
| IVF nlist1024 nprobe4 | 221x | +16.78% | 100% |
| IVF nlist1024 nprobe8 | 116x | +16.77% | 100% |
| IVF nlist1024 nprobe16 | 59x | +16.78% | 100% |
| IVF nlist1024 nprobe32 | 30x | +16.78% | 100% |
_(prid->tiny ds300K s7 done in 102s | 44 runs | 1.9h/9.5h elapsed)_

### seed 7 [tiny->omc_ ds300K s7]  Ndb=300,000  LM-only=5.4054  brute=+46.51% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +46.55% | 100% |
| dodeca-uniform P2 probe2 | 1x | +46.51% | 100% |
| dodeca-WHITENED P2 probe1 | 111x | +43.95% | 94% |
| dodeca-WHITENED P2 probe2 | 30x | +45.55% | 98% |
| dodeca-WHITENED P3 probe1 | 897x | +39.44% | 85% |
| dodeca-WHITENED P3 probe2 | 149x | +45.66% | 98% |
| IVF nlist256 nprobe1 | 195x | +46.38% | 100% |
| IVF nlist256 nprobe2 | 104x | +46.49% | 100% |
| IVF nlist256 nprobe4 | 55x | +46.50% | 100% |
| IVF nlist256 nprobe8 | 28x | +46.51% | 100% |
| IVF nlist256 nprobe16 | 14x | +46.51% | 100% |
| IVF nlist256 nprobe32 | 7x | +46.51% | 100% |
| IVF nlist1024 nprobe1 | 824x | +45.58% | 98% |
| IVF nlist1024 nprobe2 | 438x | +46.16% | 99% |
| IVF nlist1024 nprobe4 | 229x | +46.40% | 100% |
| IVF nlist1024 nprobe8 | 118x | +46.52% | 100% |
| IVF nlist1024 nprobe16 | 61x | +46.52% | 100% |
| IVF nlist1024 nprobe32 | 31x | +46.51% | 100% |
_(tiny->omc_ ds300K s7 done in 104s | 45 runs | 2.0h/9.5h elapsed)_

### seed 7 [tiny->prid ds600K s7]  Ndb=600,000  LM-only=2.7598  brute=+10.89% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +10.88% | 100% |
| dodeca-uniform P2 probe2 | 1x | +10.89% | 100% |
| dodeca-WHITENED P2 probe1 | 118x | +9.77% | 90% |
| dodeca-WHITENED P2 probe2 | 31x | +11.18% | 103% |
| dodeca-WHITENED P3 probe1 | 984x | +6.69% | 61% |
| dodeca-WHITENED P3 probe2 | 158x | +10.43% | 96% |
| IVF nlist256 nprobe1 | 200x | +10.73% | 99% |
| IVF nlist256 nprobe2 | 106x | +10.81% | 99% |
| IVF nlist256 nprobe4 | 57x | +10.87% | 100% |
| IVF nlist256 nprobe8 | 29x | +10.87% | 100% |
| IVF nlist256 nprobe16 | 15x | +10.88% | 100% |
| IVF nlist256 nprobe32 | 8x | +10.89% | 100% |
| IVF nlist1024 nprobe1 | 760x | +10.45% | 96% |
| IVF nlist1024 nprobe2 | 419x | +10.71% | 98% |
| IVF nlist1024 nprobe4 | 221x | +10.77% | 99% |
| IVF nlist1024 nprobe8 | 114x | +10.78% | 99% |
| IVF nlist1024 nprobe16 | 58x | +10.79% | 99% |
| IVF nlist1024 nprobe32 | 30x | +10.81% | 99% |
_(tiny->prid ds600K s7 done in 139s | 46 runs | 2.0h/9.5h elapsed)_

### seed 7 [prid->tiny ds600K s7]  Ndb=600,000  LM-only=3.0639  brute=+16.97% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +16.95% | 100% |
| dodeca-uniform P2 probe2 | 1x | +16.97% | 100% |
| dodeca-WHITENED P2 probe1 | 119x | +14.22% | 84% |
| dodeca-WHITENED P2 probe2 | 31x | +16.71% | 98% |
| dodeca-WHITENED P3 probe1 | 917x | +11.40% | 67% |
| dodeca-WHITENED P3 probe2 | 151x | +16.28% | 96% |
| IVF nlist256 nprobe1 | 203x | +17.00% | 100% |
| IVF nlist256 nprobe2 | 107x | +17.00% | 100% |
| IVF nlist256 nprobe4 | 56x | +17.00% | 100% |
| IVF nlist256 nprobe8 | 29x | +16.99% | 100% |
| IVF nlist256 nprobe16 | 15x | +16.99% | 100% |
| IVF nlist256 nprobe32 | 8x | +16.99% | 100% |
| IVF nlist1024 nprobe1 | 797x | +17.05% | 100% |
| IVF nlist1024 nprobe2 | 430x | +16.98% | 100% |
| IVF nlist1024 nprobe4 | 226x | +17.00% | 100% |
| IVF nlist1024 nprobe8 | 117x | +16.99% | 100% |
| IVF nlist1024 nprobe16 | 59x | +16.99% | 100% |
| IVF nlist1024 nprobe32 | 30x | +16.99% | 100% |
_(prid->tiny ds600K s7 done in 143s | 47 runs | 2.0h/9.5h elapsed)_

### seed 7 [tiny->omc_ ds600K s7]  Ndb=600,000  LM-only=5.4054  brute=+46.66% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +46.76% | 100% |
| dodeca-uniform P2 probe2 | 1x | +46.66% | 100% |
| dodeca-WHITENED P2 probe1 | 110x | +45.21% | 97% |
| dodeca-WHITENED P2 probe2 | 30x | +46.56% | 100% |
| dodeca-WHITENED P3 probe1 | 921x | +41.44% | 89% |
| dodeca-WHITENED P3 probe2 | 153x | +45.99% | 99% |
| IVF nlist256 nprobe1 | 200x | +46.35% | 99% |
| IVF nlist256 nprobe2 | 105x | +46.47% | 100% |
| IVF nlist256 nprobe4 | 55x | +46.62% | 100% |
| IVF nlist256 nprobe8 | 28x | +46.63% | 100% |
| IVF nlist256 nprobe16 | 14x | +46.65% | 100% |
| IVF nlist256 nprobe32 | 7x | +46.65% | 100% |
| IVF nlist1024 nprobe1 | 818x | +46.48% | 100% |
| IVF nlist1024 nprobe2 | 435x | +46.52% | 100% |
| IVF nlist1024 nprobe4 | 224x | +46.64% | 100% |
| IVF nlist1024 nprobe8 | 116x | +46.65% | 100% |
| IVF nlist1024 nprobe16 | 59x | +46.66% | 100% |
| IVF nlist1024 nprobe32 | 30x | +46.66% | 100% |
_(tiny->omc_ ds600K s7 done in 139s | 48 runs | 2.1h/9.5h elapsed)_

### seed 8 [tiny->prid ds300K s8]  Ndb=300,000  LM-only=2.7641  brute=+9.60% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +9.10% | 95% |
| dodeca-uniform P2 probe2 | 1x | +9.57% | 100% |
| dodeca-WHITENED P2 probe1 | 116x | +7.18% | 75% |
| dodeca-WHITENED P2 probe2 | 31x | +9.25% | 96% |
| dodeca-WHITENED P3 probe1 | 931x | +4.98% | 52% |
| dodeca-WHITENED P3 probe2 | 154x | +8.59% | 89% |
| IVF nlist256 nprobe1 | 203x | +9.18% | 96% |
| IVF nlist256 nprobe2 | 107x | +9.18% | 96% |
| IVF nlist256 nprobe4 | 55x | +9.36% | 98% |
| IVF nlist256 nprobe8 | 28x | +9.38% | 98% |
| IVF nlist256 nprobe16 | 15x | +9.39% | 98% |
| IVF nlist256 nprobe32 | 7x | +9.40% | 98% |
| IVF nlist1024 nprobe1 | 759x | +9.21% | 96% |
| IVF nlist1024 nprobe2 | 421x | +9.43% | 98% |
| IVF nlist1024 nprobe4 | 221x | +9.44% | 98% |
| IVF nlist1024 nprobe8 | 114x | +9.44% | 98% |
| IVF nlist1024 nprobe16 | 59x | +9.43% | 98% |
| IVF nlist1024 nprobe32 | 30x | +9.43% | 98% |
_(tiny->prid ds300K s8 done in 99s | 49 runs | 2.1h/9.5h elapsed)_

### seed 8 [prid->tiny ds300K s8]  Ndb=300,000  LM-only=3.0660  brute=+16.66% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 3x | +16.45% | 99% |
| dodeca-uniform P2 probe2 | 2x | +16.72% | 100% |
| dodeca-WHITENED P2 probe1 | 115x | +13.88% | 83% |
| dodeca-WHITENED P2 probe2 | 32x | +16.43% | 99% |
| dodeca-WHITENED P3 probe1 | 960x | +8.89% | 53% |
| dodeca-WHITENED P3 probe2 | 153x | +14.29% | 86% |
| IVF nlist256 nprobe1 | 184x | +16.44% | 99% |
| IVF nlist256 nprobe2 | 104x | +16.71% | 100% |
| IVF nlist256 nprobe4 | 55x | +16.65% | 100% |
| IVF nlist256 nprobe8 | 28x | +16.66% | 100% |
| IVF nlist256 nprobe16 | 14x | +16.65% | 100% |
| IVF nlist256 nprobe32 | 7x | +16.65% | 100% |
| IVF nlist1024 nprobe1 | 767x | +16.13% | 97% |
| IVF nlist1024 nprobe2 | 423x | +16.57% | 99% |
| IVF nlist1024 nprobe4 | 224x | +16.66% | 100% |
| IVF nlist1024 nprobe8 | 117x | +16.65% | 100% |
| IVF nlist1024 nprobe16 | 60x | +16.65% | 100% |
| IVF nlist1024 nprobe32 | 30x | +16.65% | 100% |
_(prid->tiny ds300K s8 done in 103s | 50 runs | 2.1h/9.5h elapsed)_

### seed 8 [tiny->omc_ ds300K s8]  Ndb=300,000  LM-only=5.7617  brute=+49.73% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +49.73% | 100% |
| dodeca-uniform P2 probe2 | 1x | +49.73% | 100% |
| dodeca-WHITENED P2 probe1 | 122x | +47.58% | 96% |
| dodeca-WHITENED P2 probe2 | 33x | +49.66% | 100% |
| dodeca-WHITENED P3 probe1 | 905x | +42.69% | 86% |
| dodeca-WHITENED P3 probe2 | 151x | +49.00% | 99% |
| IVF nlist256 nprobe1 | 189x | +49.82% | 100% |
| IVF nlist256 nprobe2 | 100x | +49.81% | 100% |
| IVF nlist256 nprobe4 | 52x | +49.71% | 100% |
| IVF nlist256 nprobe8 | 28x | +49.73% | 100% |
| IVF nlist256 nprobe16 | 14x | +49.73% | 100% |
| IVF nlist256 nprobe32 | 7x | +49.73% | 100% |
| IVF nlist1024 nprobe1 | 792x | +48.19% | 97% |
| IVF nlist1024 nprobe2 | 427x | +49.72% | 100% |
| IVF nlist1024 nprobe4 | 224x | +49.74% | 100% |
| IVF nlist1024 nprobe8 | 117x | +49.85% | 100% |
| IVF nlist1024 nprobe16 | 60x | +49.85% | 100% |
| IVF nlist1024 nprobe32 | 31x | +49.85% | 100% |
_(tiny->omc_ ds300K s8 done in 103s | 51 runs | 2.2h/9.5h elapsed)_

### seed 8 [tiny->prid ds600K s8]  Ndb=600,000  LM-only=2.7641  brute=+9.97% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +9.64% | 97% |
| dodeca-uniform P2 probe2 | 1x | +9.95% | 100% |
| dodeca-WHITENED P2 probe1 | 110x | +8.56% | 86% |
| dodeca-WHITENED P2 probe2 | 31x | +9.32% | 93% |
| dodeca-WHITENED P3 probe1 | 923x | +6.07% | 61% |
| dodeca-WHITENED P3 probe2 | 151x | +9.41% | 94% |
| IVF nlist256 nprobe1 | 194x | +9.63% | 97% |
| IVF nlist256 nprobe2 | 103x | +9.93% | 100% |
| IVF nlist256 nprobe4 | 55x | +9.93% | 100% |
| IVF nlist256 nprobe8 | 29x | +9.93% | 100% |
| IVF nlist256 nprobe16 | 15x | +9.94% | 100% |
| IVF nlist256 nprobe32 | 7x | +9.95% | 100% |
| IVF nlist1024 nprobe1 | 782x | +9.59% | 96% |
| IVF nlist1024 nprobe2 | 427x | +9.71% | 97% |
| IVF nlist1024 nprobe4 | 219x | +9.72% | 97% |
| IVF nlist1024 nprobe8 | 114x | +9.73% | 98% |
| IVF nlist1024 nprobe16 | 59x | +9.73% | 98% |
| IVF nlist1024 nprobe32 | 30x | +9.92% | 99% |
_(tiny->prid ds600K s8 done in 138s | 52 runs | 2.2h/9.5h elapsed)_

### seed 8 [prid->tiny ds600K s8]  Ndb=600,000  LM-only=3.0660  brute=+16.56% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 3x | +16.40% | 99% |
| dodeca-uniform P2 probe2 | 2x | +16.47% | 100% |
| dodeca-WHITENED P2 probe1 | 119x | +15.11% | 91% |
| dodeca-WHITENED P2 probe2 | 32x | +16.79% | 101% |
| dodeca-WHITENED P3 probe1 | 875x | +11.72% | 71% |
| dodeca-WHITENED P3 probe2 | 149x | +16.80% | 101% |
| IVF nlist256 nprobe1 | 208x | +16.54% | 100% |
| IVF nlist256 nprobe2 | 114x | +16.55% | 100% |
| IVF nlist256 nprobe4 | 59x | +16.56% | 100% |
| IVF nlist256 nprobe8 | 30x | +16.56% | 100% |
| IVF nlist256 nprobe16 | 15x | +16.56% | 100% |
| IVF nlist256 nprobe32 | 8x | +16.56% | 100% |
| IVF nlist1024 nprobe1 | 778x | +16.50% | 100% |
| IVF nlist1024 nprobe2 | 421x | +16.50% | 100% |
| IVF nlist1024 nprobe4 | 221x | +16.56% | 100% |
| IVF nlist1024 nprobe8 | 116x | +16.55% | 100% |
| IVF nlist1024 nprobe16 | 60x | +16.55% | 100% |
| IVF nlist1024 nprobe32 | 31x | +16.55% | 100% |
_(prid->tiny ds600K s8 done in 132s | 53 runs | 2.2h/9.5h elapsed)_

### seed 8 [tiny->omc_ ds600K s8]  Ndb=600,000  LM-only=5.7617  brute=+49.29% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +49.29% | 100% |
| dodeca-uniform P2 probe2 | 1x | +49.29% | 100% |
| dodeca-WHITENED P2 probe1 | 128x | +47.91% | 97% |
| dodeca-WHITENED P2 probe2 | 33x | +49.00% | 99% |
| dodeca-WHITENED P3 probe1 | 928x | +44.17% | 90% |
| dodeca-WHITENED P3 probe2 | 156x | +48.89% | 99% |
| IVF nlist256 nprobe1 | 176x | +48.87% | 99% |
| IVF nlist256 nprobe2 | 98x | +49.28% | 100% |
| IVF nlist256 nprobe4 | 51x | +49.28% | 100% |
| IVF nlist256 nprobe8 | 26x | +49.28% | 100% |
| IVF nlist256 nprobe16 | 14x | +49.29% | 100% |
| IVF nlist256 nprobe32 | 7x | +49.29% | 100% |
| IVF nlist1024 nprobe1 | 801x | +49.17% | 100% |
| IVF nlist1024 nprobe2 | 435x | +49.18% | 100% |
| IVF nlist1024 nprobe4 | 227x | +49.29% | 100% |
| IVF nlist1024 nprobe8 | 117x | +49.29% | 100% |
| IVF nlist1024 nprobe16 | 60x | +49.29% | 100% |
| IVF nlist1024 nprobe32 | 30x | +49.30% | 100% |
_(tiny->omc_ ds600K s8 done in 140s | 54 runs | 2.3h/9.5h elapsed)_

### seed 9 [tiny->prid ds300K s9]  Ndb=300,000  LM-only=2.7906  brute=+11.47% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +11.39% | 99% |
| dodeca-uniform P2 probe2 | 1x | +11.47% | 100% |
| dodeca-WHITENED P2 probe1 | 123x | +9.12% | 79% |
| dodeca-WHITENED P2 probe2 | 32x | +10.22% | 89% |
| dodeca-WHITENED P3 probe1 | 974x | +5.16% | 45% |
| dodeca-WHITENED P3 probe2 | 151x | +9.40% | 82% |
| IVF nlist256 nprobe1 | 210x | +11.18% | 97% |
| IVF nlist256 nprobe2 | 111x | +11.19% | 98% |
| IVF nlist256 nprobe4 | 58x | +11.39% | 99% |
| IVF nlist256 nprobe8 | 30x | +11.44% | 100% |
| IVF nlist256 nprobe16 | 15x | +11.45% | 100% |
| IVF nlist256 nprobe32 | 8x | +11.47% | 100% |
| IVF nlist1024 nprobe1 | 794x | +10.62% | 93% |
| IVF nlist1024 nprobe2 | 426x | +10.96% | 96% |
| IVF nlist1024 nprobe4 | 221x | +11.47% | 100% |
| IVF nlist1024 nprobe8 | 114x | +11.46% | 100% |
| IVF nlist1024 nprobe16 | 58x | +11.47% | 100% |
| IVF nlist1024 nprobe32 | 30x | +11.47% | 100% |
_(tiny->prid ds300K s9 done in 104s | 55 runs | 2.3h/9.5h elapsed)_

### seed 9 [prid->tiny ds300K s9]  Ndb=300,000  LM-only=3.0785  brute=+17.44% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +17.44% | 100% |
| dodeca-uniform P2 probe2 | 1x | +17.45% | 100% |
| dodeca-WHITENED P2 probe1 | 125x | +14.15% | 81% |
| dodeca-WHITENED P2 probe2 | 33x | +16.51% | 95% |
| dodeca-WHITENED P3 probe1 | 931x | +10.70% | 61% |
| dodeca-WHITENED P3 probe2 | 148x | +16.33% | 94% |
| IVF nlist256 nprobe1 | 193x | +17.31% | 99% |
| IVF nlist256 nprobe2 | 105x | +17.44% | 100% |
| IVF nlist256 nprobe4 | 55x | +17.45% | 100% |
| IVF nlist256 nprobe8 | 29x | +17.44% | 100% |
| IVF nlist256 nprobe16 | 15x | +17.44% | 100% |
| IVF nlist256 nprobe32 | 7x | +17.44% | 100% |
| IVF nlist1024 nprobe1 | 779x | +16.89% | 97% |
| IVF nlist1024 nprobe2 | 423x | +17.20% | 99% |
| IVF nlist1024 nprobe4 | 223x | +17.48% | 100% |
| IVF nlist1024 nprobe8 | 115x | +17.45% | 100% |
| IVF nlist1024 nprobe16 | 59x | +17.45% | 100% |
| IVF nlist1024 nprobe32 | 30x | +17.45% | 100% |
_(prid->tiny ds300K s9 done in 104s | 56 runs | 2.3h/9.5h elapsed)_

### seed 9 [tiny->omc_ ds300K s9]  Ndb=300,000  LM-only=5.6589  brute=+48.26% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +48.17% | 100% |
| dodeca-uniform P2 probe2 | 1x | +48.26% | 100% |
| dodeca-WHITENED P2 probe1 | 126x | +45.93% | 95% |
| dodeca-WHITENED P2 probe2 | 33x | +48.15% | 100% |
| dodeca-WHITENED P3 probe1 | 813x | +41.82% | 87% |
| dodeca-WHITENED P3 probe2 | 139x | +47.83% | 99% |
| IVF nlist256 nprobe1 | 196x | +48.59% | 101% |
| IVF nlist256 nprobe2 | 104x | +48.38% | 100% |
| IVF nlist256 nprobe4 | 54x | +48.28% | 100% |
| IVF nlist256 nprobe8 | 29x | +48.25% | 100% |
| IVF nlist256 nprobe16 | 15x | +48.26% | 100% |
| IVF nlist256 nprobe32 | 8x | +48.26% | 100% |
| IVF nlist1024 nprobe1 | 818x | +47.56% | 99% |
| IVF nlist1024 nprobe2 | 433x | +48.24% | 100% |
| IVF nlist1024 nprobe4 | 224x | +48.27% | 100% |
| IVF nlist1024 nprobe8 | 116x | +48.27% | 100% |
| IVF nlist1024 nprobe16 | 60x | +48.27% | 100% |
| IVF nlist1024 nprobe32 | 30x | +48.28% | 100% |
_(tiny->omc_ ds300K s9 done in 104s | 57 runs | 2.4h/9.5h elapsed)_

### seed 9 [tiny->prid ds600K s9]  Ndb=600,000  LM-only=2.7906  brute=+11.15% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +11.12% | 100% |
| dodeca-uniform P2 probe2 | 1x | +11.16% | 100% |
| dodeca-WHITENED P2 probe1 | 124x | +10.02% | 90% |
| dodeca-WHITENED P2 probe2 | 33x | +11.08% | 99% |
| dodeca-WHITENED P3 probe1 | 898x | +7.63% | 68% |
| dodeca-WHITENED P3 probe2 | 148x | +11.36% | 102% |
| IVF nlist256 nprobe1 | 194x | +11.09% | 99% |
| IVF nlist256 nprobe2 | 104x | +11.08% | 99% |
| IVF nlist256 nprobe4 | 54x | +11.12% | 100% |
| IVF nlist256 nprobe8 | 28x | +11.15% | 100% |
| IVF nlist256 nprobe16 | 15x | +11.15% | 100% |
| IVF nlist256 nprobe32 | 8x | +11.16% | 100% |
| IVF nlist1024 nprobe1 | 779x | +11.25% | 101% |
| IVF nlist1024 nprobe2 | 427x | +11.12% | 100% |
| IVF nlist1024 nprobe4 | 224x | +11.14% | 100% |
| IVF nlist1024 nprobe8 | 116x | +11.12% | 100% |
| IVF nlist1024 nprobe16 | 59x | +11.15% | 100% |
| IVF nlist1024 nprobe32 | 30x | +11.15% | 100% |
_(tiny->prid ds600K s9 done in 144s | 58 runs | 2.4h/9.5h elapsed)_

### seed 9 [prid->tiny ds600K s9]  Ndb=600,000  LM-only=3.0785  brute=+17.20% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +17.30% | 101% |
| dodeca-uniform P2 probe2 | 1x | +17.34% | 101% |
| dodeca-WHITENED P2 probe1 | 123x | +15.69% | 91% |
| dodeca-WHITENED P2 probe2 | 32x | +16.92% | 98% |
| dodeca-WHITENED P3 probe1 | 844x | +11.69% | 68% |
| dodeca-WHITENED P3 probe2 | 145x | +16.50% | 96% |
| IVF nlist256 nprobe1 | 204x | +17.40% | 101% |
| IVF nlist256 nprobe2 | 108x | +17.21% | 100% |
| IVF nlist256 nprobe4 | 56x | +17.21% | 100% |
| IVF nlist256 nprobe8 | 29x | +17.21% | 100% |
| IVF nlist256 nprobe16 | 14x | +17.20% | 100% |
| IVF nlist256 nprobe32 | 7x | +17.20% | 100% |
| IVF nlist1024 nprobe1 | 770x | +16.94% | 98% |
| IVF nlist1024 nprobe2 | 425x | +17.23% | 100% |
| IVF nlist1024 nprobe4 | 225x | +17.20% | 100% |
| IVF nlist1024 nprobe8 | 117x | +17.21% | 100% |
| IVF nlist1024 nprobe16 | 59x | +17.21% | 100% |
| IVF nlist1024 nprobe32 | 30x | +17.21% | 100% |
_(prid->tiny ds600K s9 done in 141s | 59 runs | 2.4h/9.5h elapsed)_

### seed 9 [tiny->omc_ ds600K s9]  Ndb=600,000  LM-only=5.6589  brute=+48.83% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +48.90% | 100% |
| dodeca-uniform P2 probe2 | 1x | +48.83% | 100% |
| dodeca-WHITENED P2 probe1 | 125x | +47.80% | 98% |
| dodeca-WHITENED P2 probe2 | 33x | +48.70% | 100% |
| dodeca-WHITENED P3 probe1 | 881x | +43.58% | 89% |
| dodeca-WHITENED P3 probe2 | 141x | +48.17% | 99% |
| IVF nlist256 nprobe1 | 188x | +48.45% | 99% |
| IVF nlist256 nprobe2 | 98x | +48.77% | 100% |
| IVF nlist256 nprobe4 | 50x | +48.77% | 100% |
| IVF nlist256 nprobe8 | 26x | +48.78% | 100% |
| IVF nlist256 nprobe16 | 13x | +48.79% | 100% |
| IVF nlist256 nprobe32 | 7x | +48.79% | 100% |
| IVF nlist1024 nprobe1 | 820x | +47.84% | 98% |
| IVF nlist1024 nprobe2 | 431x | +48.56% | 99% |
| IVF nlist1024 nprobe4 | 223x | +48.70% | 100% |
| IVF nlist1024 nprobe8 | 115x | +48.80% | 100% |
| IVF nlist1024 nprobe16 | 58x | +48.82% | 100% |
| IVF nlist1024 nprobe32 | 30x | +48.82% | 100% |
_(tiny->omc_ ds600K s9 done in 143s | 60 runs | 2.5h/9.5h elapsed)_

### seed 10 [tiny->prid ds300K s10]  Ndb=300,000  LM-only=2.7738  brute=+10.31% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +10.31% | 100% |
| dodeca-uniform P2 probe2 | 1x | +10.31% | 100% |
| dodeca-WHITENED P2 probe1 | 123x | +8.73% | 85% |
| dodeca-WHITENED P2 probe2 | 32x | +10.46% | 101% |
| dodeca-WHITENED P3 probe1 | 907x | +5.43% | 53% |
| dodeca-WHITENED P3 probe2 | 149x | +9.96% | 97% |
| IVF nlist256 nprobe1 | 201x | +9.57% | 93% |
| IVF nlist256 nprobe2 | 108x | +10.23% | 99% |
| IVF nlist256 nprobe4 | 57x | +10.28% | 100% |
| IVF nlist256 nprobe8 | 29x | +10.28% | 100% |
| IVF nlist256 nprobe16 | 15x | +10.29% | 100% |
| IVF nlist256 nprobe32 | 7x | +10.29% | 100% |
| IVF nlist1024 nprobe1 | 789x | +9.37% | 91% |
| IVF nlist1024 nprobe2 | 429x | +10.16% | 99% |
| IVF nlist1024 nprobe4 | 225x | +10.05% | 97% |
| IVF nlist1024 nprobe8 | 118x | +10.06% | 98% |
| IVF nlist1024 nprobe16 | 61x | +10.27% | 100% |
| IVF nlist1024 nprobe32 | 31x | +10.27% | 100% |
_(tiny->prid ds300K s10 done in 103s | 61 runs | 2.5h/9.5h elapsed)_

### seed 10 [prid->tiny ds300K s10]  Ndb=300,000  LM-only=3.0384  brute=+15.90% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +15.86% | 100% |
| dodeca-uniform P2 probe2 | 1x | +15.89% | 100% |
| dodeca-WHITENED P2 probe1 | 120x | +13.49% | 85% |
| dodeca-WHITENED P2 probe2 | 32x | +15.76% | 99% |
| dodeca-WHITENED P3 probe1 | 865x | +9.63% | 61% |
| dodeca-WHITENED P3 probe2 | 143x | +14.73% | 93% |
| IVF nlist256 nprobe1 | 191x | +15.81% | 99% |
| IVF nlist256 nprobe2 | 105x | +15.90% | 100% |
| IVF nlist256 nprobe4 | 54x | +15.90% | 100% |
| IVF nlist256 nprobe8 | 28x | +15.90% | 100% |
| IVF nlist256 nprobe16 | 15x | +15.90% | 100% |
| IVF nlist256 nprobe32 | 8x | +15.90% | 100% |
| IVF nlist1024 nprobe1 | 774x | +15.64% | 98% |
| IVF nlist1024 nprobe2 | 421x | +15.88% | 100% |
| IVF nlist1024 nprobe4 | 224x | +15.87% | 100% |
| IVF nlist1024 nprobe8 | 116x | +15.91% | 100% |
| IVF nlist1024 nprobe16 | 59x | +15.90% | 100% |
| IVF nlist1024 nprobe32 | 31x | +15.90% | 100% |
_(prid->tiny ds300K s10 done in 102s | 62 runs | 2.5h/9.5h elapsed)_

### seed 10 [tiny->omc_ ds300K s10]  Ndb=300,000  LM-only=5.7764  brute=+49.91% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +50.28% | 101% |
| dodeca-uniform P2 probe2 | 1x | +49.91% | 100% |
| dodeca-WHITENED P2 probe1 | 126x | +48.63% | 97% |
| dodeca-WHITENED P2 probe2 | 33x | +50.10% | 100% |
| dodeca-WHITENED P3 probe1 | 816x | +43.58% | 87% |
| dodeca-WHITENED P3 probe2 | 139x | +49.88% | 100% |
| IVF nlist256 nprobe1 | 197x | +49.84% | 100% |
| IVF nlist256 nprobe2 | 101x | +50.09% | 100% |
| IVF nlist256 nprobe4 | 53x | +49.87% | 100% |
| IVF nlist256 nprobe8 | 28x | +49.94% | 100% |
| IVF nlist256 nprobe16 | 14x | +49.90% | 100% |
| IVF nlist256 nprobe32 | 7x | +49.90% | 100% |
| IVF nlist1024 nprobe1 | 816x | +49.45% | 99% |
| IVF nlist1024 nprobe2 | 431x | +49.72% | 100% |
| IVF nlist1024 nprobe4 | 223x | +49.75% | 100% |
| IVF nlist1024 nprobe8 | 115x | +49.93% | 100% |
| IVF nlist1024 nprobe16 | 59x | +49.90% | 100% |
| IVF nlist1024 nprobe32 | 30x | +49.90% | 100% |
_(tiny->omc_ ds300K s10 done in 103s | 63 runs | 2.6h/9.5h elapsed)_

### seed 10 [tiny->prid ds600K s10]  Ndb=600,000  LM-only=2.7738  brute=+10.30% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +10.29% | 100% |
| dodeca-uniform P2 probe2 | 1x | +10.30% | 100% |
| dodeca-WHITENED P2 probe1 | 120x | +9.25% | 90% |
| dodeca-WHITENED P2 probe2 | 32x | +10.35% | 100% |
| dodeca-WHITENED P3 probe1 | 944x | +7.06% | 69% |
| dodeca-WHITENED P3 probe2 | 150x | +9.94% | 97% |
| IVF nlist256 nprobe1 | 197x | +10.12% | 98% |
| IVF nlist256 nprobe2 | 107x | +10.18% | 99% |
| IVF nlist256 nprobe4 | 56x | +10.28% | 100% |
| IVF nlist256 nprobe8 | 29x | +10.31% | 100% |
| IVF nlist256 nprobe16 | 14x | +10.31% | 100% |
| IVF nlist256 nprobe32 | 7x | +10.30% | 100% |
| IVF nlist1024 nprobe1 | 797x | +10.04% | 97% |
| IVF nlist1024 nprobe2 | 431x | +10.11% | 98% |
| IVF nlist1024 nprobe4 | 225x | +10.26% | 100% |
| IVF nlist1024 nprobe8 | 116x | +10.27% | 100% |
| IVF nlist1024 nprobe16 | 59x | +10.29% | 100% |
| IVF nlist1024 nprobe32 | 30x | +10.30% | 100% |
_(tiny->prid ds600K s10 done in 142s | 64 runs | 2.6h/9.5h elapsed)_

### seed 10 [prid->tiny ds600K s10]  Ndb=600,000  LM-only=3.0384  brute=+16.18% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +16.18% | 100% |
| dodeca-uniform P2 probe2 | 1x | +16.18% | 100% |
| dodeca-WHITENED P2 probe1 | 125x | +15.26% | 94% |
| dodeca-WHITENED P2 probe2 | 33x | +15.99% | 99% |
| dodeca-WHITENED P3 probe1 | 850x | +12.26% | 76% |
| dodeca-WHITENED P3 probe2 | 145x | +15.76% | 97% |
| IVF nlist256 nprobe1 | 199x | +16.17% | 100% |
| IVF nlist256 nprobe2 | 105x | +16.19% | 100% |
| IVF nlist256 nprobe4 | 54x | +16.19% | 100% |
| IVF nlist256 nprobe8 | 28x | +16.19% | 100% |
| IVF nlist256 nprobe16 | 14x | +16.19% | 100% |
| IVF nlist256 nprobe32 | 7x | +16.19% | 100% |
| IVF nlist1024 nprobe1 | 801x | +16.27% | 101% |
| IVF nlist1024 nprobe2 | 435x | +16.28% | 101% |
| IVF nlist1024 nprobe4 | 228x | +16.18% | 100% |
| IVF nlist1024 nprobe8 | 117x | +16.18% | 100% |
| IVF nlist1024 nprobe16 | 60x | +16.18% | 100% |
| IVF nlist1024 nprobe32 | 30x | +16.18% | 100% |
_(prid->tiny ds600K s10 done in 146s | 65 runs | 2.6h/9.5h elapsed)_

### seed 10 [tiny->omc_ ds600K s10]  Ndb=600,000  LM-only=5.7764  brute=+50.27% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +50.35% | 100% |
| dodeca-uniform P2 probe2 | 1x | +50.27% | 100% |
| dodeca-WHITENED P2 probe1 | 117x | +49.95% | 99% |
| dodeca-WHITENED P2 probe2 | 32x | +50.69% | 101% |
| dodeca-WHITENED P3 probe1 | 858x | +46.89% | 93% |
| dodeca-WHITENED P3 probe2 | 140x | +50.68% | 101% |
| IVF nlist256 nprobe1 | 196x | +49.84% | 99% |
| IVF nlist256 nprobe2 | 107x | +50.01% | 99% |
| IVF nlist256 nprobe4 | 56x | +50.21% | 100% |
| IVF nlist256 nprobe8 | 29x | +50.25% | 100% |
| IVF nlist256 nprobe16 | 15x | +50.26% | 100% |
| IVF nlist256 nprobe32 | 8x | +50.26% | 100% |
| IVF nlist1024 nprobe1 | 830x | +50.48% | 100% |
| IVF nlist1024 nprobe2 | 436x | +50.04% | 100% |
| IVF nlist1024 nprobe4 | 226x | +50.04% | 100% |
| IVF nlist1024 nprobe8 | 116x | +50.22% | 100% |
| IVF nlist1024 nprobe16 | 59x | +50.25% | 100% |
| IVF nlist1024 nprobe32 | 30x | +50.26% | 100% |
_(tiny->omc_ ds600K s10 done in 142s | 66 runs | 2.7h/9.5h elapsed)_

### seed 11 [tiny->prid ds300K s11]  Ndb=300,000  LM-only=2.7340  brute=+10.20% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +10.20% | 100% |
| dodeca-uniform P2 probe2 | 1x | +10.20% | 100% |
| dodeca-WHITENED P2 probe1 | 125x | +8.46% | 83% |
| dodeca-WHITENED P2 probe2 | 33x | +10.48% | 103% |
| dodeca-WHITENED P3 probe1 | 816x | +5.40% | 53% |
| dodeca-WHITENED P3 probe2 | 142x | +9.02% | 88% |
| IVF nlist256 nprobe1 | 200x | +9.63% | 94% |
| IVF nlist256 nprobe2 | 108x | +9.71% | 95% |
| IVF nlist256 nprobe4 | 57x | +9.73% | 95% |
| IVF nlist256 nprobe8 | 29x | +9.95% | 98% |
| IVF nlist256 nprobe16 | 15x | +10.17% | 100% |
| IVF nlist256 nprobe32 | 8x | +10.19% | 100% |
| IVF nlist1024 nprobe1 | 770x | +9.36% | 92% |
| IVF nlist1024 nprobe2 | 416x | +9.71% | 95% |
| IVF nlist1024 nprobe4 | 218x | +9.72% | 95% |
| IVF nlist1024 nprobe8 | 113x | +9.75% | 96% |
| IVF nlist1024 nprobe16 | 59x | +9.76% | 96% |
| IVF nlist1024 nprobe32 | 30x | +9.97% | 98% |
_(tiny->prid ds300K s11 done in 103s | 67 runs | 2.7h/9.5h elapsed)_

### seed 11 [prid->tiny ds300K s11]  Ndb=300,000  LM-only=3.1056  brute=+17.45% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +17.41% | 100% |
| dodeca-uniform P2 probe2 | 1x | +17.41% | 100% |
| dodeca-WHITENED P2 probe1 | 127x | +14.57% | 83% |
| dodeca-WHITENED P2 probe2 | 33x | +16.90% | 97% |
| dodeca-WHITENED P3 probe1 | 957x | +10.49% | 60% |
| dodeca-WHITENED P3 probe2 | 151x | +16.18% | 93% |
| IVF nlist256 nprobe1 | 191x | +17.59% | 101% |
| IVF nlist256 nprobe2 | 105x | +17.39% | 100% |
| IVF nlist256 nprobe4 | 54x | +17.45% | 100% |
| IVF nlist256 nprobe8 | 28x | +17.45% | 100% |
| IVF nlist256 nprobe16 | 14x | +17.45% | 100% |
| IVF nlist256 nprobe32 | 7x | +17.45% | 100% |
| IVF nlist1024 nprobe1 | 781x | +16.86% | 97% |
| IVF nlist1024 nprobe2 | 426x | +17.48% | 100% |
| IVF nlist1024 nprobe4 | 222x | +17.42% | 100% |
| IVF nlist1024 nprobe8 | 115x | +17.45% | 100% |
| IVF nlist1024 nprobe16 | 59x | +17.45% | 100% |
| IVF nlist1024 nprobe32 | 30x | +17.45% | 100% |
_(prid->tiny ds300K s11 done in 103s | 68 runs | 2.7h/9.5h elapsed)_

### seed 11 [tiny->omc_ ds300K s11]  Ndb=300,000  LM-only=5.4262  brute=+47.10% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +47.17% | 100% |
| dodeca-uniform P2 probe2 | 1x | +47.10% | 100% |
| dodeca-WHITENED P2 probe1 | 125x | +44.69% | 95% |
| dodeca-WHITENED P2 probe2 | 33x | +47.09% | 100% |
| dodeca-WHITENED P3 probe1 | 908x | +40.06% | 85% |
| dodeca-WHITENED P3 probe2 | 146x | +45.92% | 97% |
| IVF nlist256 nprobe1 | 196x | +47.16% | 100% |
| IVF nlist256 nprobe2 | 104x | +47.04% | 100% |
| IVF nlist256 nprobe4 | 56x | +47.09% | 100% |
| IVF nlist256 nprobe8 | 30x | +47.10% | 100% |
| IVF nlist256 nprobe16 | 15x | +47.21% | 100% |
| IVF nlist256 nprobe32 | 8x | +47.21% | 100% |
| IVF nlist1024 nprobe1 | 806x | +47.35% | 101% |
| IVF nlist1024 nprobe2 | 426x | +47.55% | 101% |
| IVF nlist1024 nprobe4 | 222x | +47.39% | 101% |
| IVF nlist1024 nprobe8 | 116x | +47.11% | 100% |
| IVF nlist1024 nprobe16 | 61x | +47.11% | 100% |
| IVF nlist1024 nprobe32 | 31x | +47.11% | 100% |
_(tiny->omc_ ds300K s11 done in 107s | 69 runs | 2.8h/9.5h elapsed)_

### seed 11 [tiny->prid ds600K s11]  Ndb=600,000  LM-only=2.7340  brute=+10.80% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +10.80% | 100% |
| dodeca-uniform P2 probe2 | 1x | +10.80% | 100% |
| dodeca-WHITENED P2 probe1 | 121x | +10.19% | 94% |
| dodeca-WHITENED P2 probe2 | 33x | +10.88% | 101% |
| dodeca-WHITENED P3 probe1 | 923x | +6.23% | 58% |
| dodeca-WHITENED P3 probe2 | 150x | +10.64% | 98% |
| IVF nlist256 nprobe1 | 205x | +10.26% | 95% |
| IVF nlist256 nprobe2 | 109x | +10.26% | 95% |
| IVF nlist256 nprobe4 | 57x | +10.48% | 97% |
| IVF nlist256 nprobe8 | 30x | +10.52% | 97% |
| IVF nlist256 nprobe16 | 15x | +10.75% | 100% |
| IVF nlist256 nprobe32 | 8x | +10.76% | 100% |
| IVF nlist1024 nprobe1 | 780x | +9.89% | 92% |
| IVF nlist1024 nprobe2 | 426x | +10.46% | 97% |
| IVF nlist1024 nprobe4 | 224x | +10.66% | 99% |
| IVF nlist1024 nprobe8 | 117x | +10.68% | 99% |
| IVF nlist1024 nprobe16 | 60x | +10.68% | 99% |
| IVF nlist1024 nprobe32 | 31x | +10.71% | 99% |
_(tiny->prid ds600K s11 done in 144s | 70 runs | 2.8h/9.5h elapsed)_

### seed 11 [prid->tiny ds600K s11]  Ndb=600,000  LM-only=3.1056  brute=+17.97% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +17.97% | 100% |
| dodeca-uniform P2 probe2 | 1x | +17.97% | 100% |
| dodeca-WHITENED P2 probe1 | 121x | +16.39% | 91% |
| dodeca-WHITENED P2 probe2 | 33x | +17.72% | 99% |
| dodeca-WHITENED P3 probe1 | 960x | +13.28% | 74% |
| dodeca-WHITENED P3 probe2 | 158x | +17.86% | 99% |
| IVF nlist256 nprobe1 | 196x | +18.00% | 100% |
| IVF nlist256 nprobe2 | 102x | +17.96% | 100% |
| IVF nlist256 nprobe4 | 52x | +17.95% | 100% |
| IVF nlist256 nprobe8 | 26x | +17.95% | 100% |
| IVF nlist256 nprobe16 | 13x | +17.95% | 100% |
| IVF nlist256 nprobe32 | 7x | +17.95% | 100% |
| IVF nlist1024 nprobe1 | 768x | +17.84% | 99% |
| IVF nlist1024 nprobe2 | 424x | +18.00% | 100% |
| IVF nlist1024 nprobe4 | 225x | +17.97% | 100% |
| IVF nlist1024 nprobe8 | 116x | +17.96% | 100% |
| IVF nlist1024 nprobe16 | 59x | +17.97% | 100% |
| IVF nlist1024 nprobe32 | 30x | +17.97% | 100% |
_(prid->tiny ds600K s11 done in 145s | 71 runs | 2.8h/9.5h elapsed)_

### seed 11 [tiny->omc_ ds600K s11]  Ndb=600,000  LM-only=5.4262  brute=+47.52% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +47.51% | 100% |
| dodeca-uniform P2 probe2 | 1x | +47.52% | 100% |
| dodeca-WHITENED P2 probe1 | 123x | +45.57% | 96% |
| dodeca-WHITENED P2 probe2 | 33x | +46.86% | 99% |
| dodeca-WHITENED P3 probe1 | 872x | +43.03% | 91% |
| dodeca-WHITENED P3 probe2 | 147x | +46.90% | 99% |
| IVF nlist256 nprobe1 | 187x | +47.43% | 100% |
| IVF nlist256 nprobe2 | 101x | +47.58% | 100% |
| IVF nlist256 nprobe4 | 53x | +47.52% | 100% |
| IVF nlist256 nprobe8 | 27x | +47.52% | 100% |
| IVF nlist256 nprobe16 | 14x | +47.52% | 100% |
| IVF nlist256 nprobe32 | 7x | +47.52% | 100% |
| IVF nlist1024 nprobe1 | 795x | +47.51% | 100% |
| IVF nlist1024 nprobe2 | 432x | +47.60% | 100% |
| IVF nlist1024 nprobe4 | 224x | +47.56% | 100% |
| IVF nlist1024 nprobe8 | 116x | +47.54% | 100% |
| IVF nlist1024 nprobe16 | 60x | +47.52% | 100% |
| IVF nlist1024 nprobe32 | 30x | +47.52% | 100% |
_(tiny->omc_ ds600K s11 done in 144s | 72 runs | 2.9h/9.5h elapsed)_

### seed 12 [tiny->prid ds300K s12]  Ndb=300,000  LM-only=2.7728  brute=+10.20% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +9.81% | 96% |
| dodeca-uniform P2 probe2 | 1x | +10.19% | 100% |
| dodeca-WHITENED P2 probe1 | 124x | +8.07% | 79% |
| dodeca-WHITENED P2 probe2 | 33x | +9.96% | 98% |
| dodeca-WHITENED P3 probe1 | 842x | +6.31% | 62% |
| dodeca-WHITENED P3 probe2 | 145x | +9.25% | 91% |
| IVF nlist256 nprobe1 | 199x | +9.89% | 97% |
| IVF nlist256 nprobe2 | 106x | +10.03% | 98% |
| IVF nlist256 nprobe4 | 55x | +9.96% | 98% |
| IVF nlist256 nprobe8 | 29x | +9.95% | 98% |
| IVF nlist256 nprobe16 | 15x | +10.19% | 100% |
| IVF nlist256 nprobe32 | 8x | +10.19% | 100% |
| IVF nlist1024 nprobe1 | 801x | +9.41% | 92% |
| IVF nlist1024 nprobe2 | 432x | +9.99% | 98% |
| IVF nlist1024 nprobe4 | 224x | +9.95% | 98% |
| IVF nlist1024 nprobe8 | 116x | +9.96% | 98% |
| IVF nlist1024 nprobe16 | 59x | +9.94% | 97% |
| IVF nlist1024 nprobe32 | 30x | +10.18% | 100% |
_(tiny->prid ds300K s12 done in 104s | 73 runs | 2.9h/9.5h elapsed)_

### seed 12 [prid->tiny ds300K s12]  Ndb=300,000  LM-only=3.0313  brute=+16.06% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +15.99% | 100% |
| dodeca-uniform P2 probe2 | 1x | +16.06% | 100% |
| dodeca-WHITENED P2 probe1 | 126x | +12.99% | 81% |
| dodeca-WHITENED P2 probe2 | 33x | +15.45% | 96% |
| dodeca-WHITENED P3 probe1 | 853x | +9.04% | 56% |
| dodeca-WHITENED P3 probe2 | 144x | +14.92% | 93% |
| IVF nlist256 nprobe1 | 197x | +16.16% | 101% |
| IVF nlist256 nprobe2 | 104x | +16.08% | 100% |
| IVF nlist256 nprobe4 | 55x | +16.07% | 100% |
| IVF nlist256 nprobe8 | 28x | +16.07% | 100% |
| IVF nlist256 nprobe16 | 15x | +16.06% | 100% |
| IVF nlist256 nprobe32 | 7x | +16.06% | 100% |
| IVF nlist1024 nprobe1 | 760x | +16.00% | 100% |
| IVF nlist1024 nprobe2 | 419x | +16.16% | 101% |
| IVF nlist1024 nprobe4 | 226x | +16.03% | 100% |
| IVF nlist1024 nprobe8 | 117x | +16.06% | 100% |
| IVF nlist1024 nprobe16 | 60x | +16.05% | 100% |
| IVF nlist1024 nprobe32 | 30x | +16.05% | 100% |
_(prid->tiny ds300K s12 done in 99s | 74 runs | 2.9h/9.5h elapsed)_

### seed 12 [tiny->omc_ ds300K s12]  Ndb=300,000  LM-only=5.7131  brute=+49.85% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +49.53% | 99% |
| dodeca-uniform P2 probe2 | 1x | +49.85% | 100% |
| dodeca-WHITENED P2 probe1 | 124x | +47.83% | 96% |
| dodeca-WHITENED P2 probe2 | 33x | +49.76% | 100% |
| dodeca-WHITENED P3 probe1 | 814x | +43.21% | 87% |
| dodeca-WHITENED P3 probe2 | 141x | +48.45% | 97% |
| IVF nlist256 nprobe1 | 179x | +49.46% | 99% |
| IVF nlist256 nprobe2 | 99x | +49.73% | 100% |
| IVF nlist256 nprobe4 | 53x | +49.79% | 100% |
| IVF nlist256 nprobe8 | 28x | +49.84% | 100% |
| IVF nlist256 nprobe16 | 14x | +49.85% | 100% |
| IVF nlist256 nprobe32 | 7x | +49.85% | 100% |
| IVF nlist1024 nprobe1 | 829x | +49.16% | 99% |
| IVF nlist1024 nprobe2 | 444x | +49.64% | 100% |
| IVF nlist1024 nprobe4 | 228x | +49.78% | 100% |
| IVF nlist1024 nprobe8 | 118x | +49.85% | 100% |
| IVF nlist1024 nprobe16 | 60x | +49.84% | 100% |
| IVF nlist1024 nprobe32 | 31x | +49.84% | 100% |
_(tiny->omc_ ds300K s12 done in 101s | 75 runs | 3.0h/9.5h elapsed)_

### seed 12 [tiny->prid ds600K s12]  Ndb=600,000  LM-only=2.7728  brute=+10.64% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +10.35% | 97% |
| dodeca-uniform P2 probe2 | 1x | +10.40% | 98% |
| dodeca-WHITENED P2 probe1 | 123x | +9.30% | 87% |
| dodeca-WHITENED P2 probe2 | 33x | +10.51% | 99% |
| dodeca-WHITENED P3 probe1 | 885x | +7.23% | 68% |
| dodeca-WHITENED P3 probe2 | 149x | +10.43% | 98% |
| IVF nlist256 nprobe1 | 208x | +10.38% | 98% |
| IVF nlist256 nprobe2 | 109x | +10.40% | 98% |
| IVF nlist256 nprobe4 | 58x | +10.41% | 98% |
| IVF nlist256 nprobe8 | 29x | +10.41% | 98% |
| IVF nlist256 nprobe16 | 15x | +10.41% | 98% |
| IVF nlist256 nprobe32 | 8x | +10.65% | 100% |
| IVF nlist1024 nprobe1 | 805x | +9.88% | 93% |
| IVF nlist1024 nprobe2 | 433x | +10.11% | 95% |
| IVF nlist1024 nprobe4 | 228x | +10.13% | 95% |
| IVF nlist1024 nprobe8 | 118x | +10.12% | 95% |
| IVF nlist1024 nprobe16 | 60x | +10.35% | 97% |
| IVF nlist1024 nprobe32 | 31x | +10.37% | 97% |
_(tiny->prid ds600K s12 done in 144s | 76 runs | 3.0h/9.5h elapsed)_

### seed 12 [prid->tiny ds600K s12]  Ndb=600,000  LM-only=3.0313  brute=+16.43% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +16.40% | 100% |
| dodeca-uniform P2 probe2 | 1x | +16.43% | 100% |
| dodeca-WHITENED P2 probe1 | 121x | +14.85% | 90% |
| dodeca-WHITENED P2 probe2 | 32x | +15.80% | 96% |
| dodeca-WHITENED P3 probe1 | 844x | +11.41% | 69% |
| dodeca-WHITENED P3 probe2 | 147x | +15.64% | 95% |
| IVF nlist256 nprobe1 | 192x | +16.44% | 100% |
| IVF nlist256 nprobe2 | 103x | +16.47% | 100% |
| IVF nlist256 nprobe4 | 55x | +16.45% | 100% |
| IVF nlist256 nprobe8 | 29x | +16.45% | 100% |
| IVF nlist256 nprobe16 | 15x | +16.44% | 100% |
| IVF nlist256 nprobe32 | 7x | +16.44% | 100% |
| IVF nlist1024 nprobe1 | 787x | +16.27% | 99% |
| IVF nlist1024 nprobe2 | 426x | +16.49% | 100% |
| IVF nlist1024 nprobe4 | 222x | +16.49% | 100% |
| IVF nlist1024 nprobe8 | 114x | +16.45% | 100% |
| IVF nlist1024 nprobe16 | 59x | +16.44% | 100% |
| IVF nlist1024 nprobe32 | 30x | +16.44% | 100% |
_(prid->tiny ds600K s12 done in 138s | 77 runs | 3.0h/9.5h elapsed)_

### seed 12 [tiny->omc_ ds600K s12]  Ndb=600,000  LM-only=5.7131  brute=+50.11% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +49.90% | 100% |
| dodeca-uniform P2 probe2 | 1x | +50.10% | 100% |
| dodeca-WHITENED P2 probe1 | 130x | +48.47% | 97% |
| dodeca-WHITENED P2 probe2 | 34x | +50.04% | 100% |
| dodeca-WHITENED P3 probe1 | 819x | +45.97% | 92% |
| dodeca-WHITENED P3 probe2 | 138x | +49.15% | 98% |
| IVF nlist256 nprobe1 | 211x | +50.02% | 100% |
| IVF nlist256 nprobe2 | 112x | +50.10% | 100% |
| IVF nlist256 nprobe4 | 59x | +50.18% | 100% |
| IVF nlist256 nprobe8 | 30x | +50.11% | 100% |
| IVF nlist256 nprobe16 | 15x | +50.11% | 100% |
| IVF nlist256 nprobe32 | 8x | +50.11% | 100% |
| IVF nlist1024 nprobe1 | 793x | +50.25% | 100% |
| IVF nlist1024 nprobe2 | 421x | +50.29% | 100% |
| IVF nlist1024 nprobe4 | 220x | +50.16% | 100% |
| IVF nlist1024 nprobe8 | 115x | +50.10% | 100% |
| IVF nlist1024 nprobe16 | 59x | +50.10% | 100% |
| IVF nlist1024 nprobe32 | 30x | +50.11% | 100% |
_(tiny->omc_ ds600K s12 done in 130s | 78 runs | 3.1h/9.5h elapsed)_

### seed 13 [tiny->prid ds300K s13]  Ndb=300,000  LM-only=2.7758  brute=+10.83% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +10.81% | 100% |
| dodeca-uniform P2 probe2 | 1x | +10.83% | 100% |
| dodeca-WHITENED P2 probe1 | 120x | +8.31% | 77% |
| dodeca-WHITENED P2 probe2 | 32x | +10.33% | 95% |
| dodeca-WHITENED P3 probe1 | 995x | +5.56% | 51% |
| dodeca-WHITENED P3 probe2 | 157x | +10.53% | 97% |
| IVF nlist256 nprobe1 | 211x | +10.21% | 94% |
| IVF nlist256 nprobe2 | 112x | +10.38% | 96% |
| IVF nlist256 nprobe4 | 59x | +10.42% | 96% |
| IVF nlist256 nprobe8 | 30x | +10.60% | 98% |
| IVF nlist256 nprobe16 | 15x | +10.82% | 100% |
| IVF nlist256 nprobe32 | 8x | +10.83% | 100% |
| IVF nlist1024 nprobe1 | 790x | +10.52% | 97% |
| IVF nlist1024 nprobe2 | 432x | +10.60% | 98% |
| IVF nlist1024 nprobe4 | 225x | +10.64% | 98% |
| IVF nlist1024 nprobe8 | 116x | +10.80% | 100% |
| IVF nlist1024 nprobe16 | 60x | +10.80% | 100% |
| IVF nlist1024 nprobe32 | 30x | +10.80% | 100% |
_(tiny->prid ds300K s13 done in 103s | 79 runs | 3.1h/9.5h elapsed)_

### seed 13 [prid->tiny ds300K s13]  Ndb=300,000  LM-only=3.0398  brute=+15.97% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +15.81% | 99% |
| dodeca-uniform P2 probe2 | 1x | +15.97% | 100% |
| dodeca-WHITENED P2 probe1 | 121x | +13.22% | 83% |
| dodeca-WHITENED P2 probe2 | 33x | +15.16% | 95% |
| dodeca-WHITENED P3 probe1 | 848x | +10.11% | 63% |
| dodeca-WHITENED P3 probe2 | 144x | +14.48% | 91% |
| IVF nlist256 nprobe1 | 199x | +15.89% | 100% |
| IVF nlist256 nprobe2 | 106x | +15.93% | 100% |
| IVF nlist256 nprobe4 | 55x | +15.96% | 100% |
| IVF nlist256 nprobe8 | 28x | +15.96% | 100% |
| IVF nlist256 nprobe16 | 14x | +15.97% | 100% |
| IVF nlist256 nprobe32 | 7x | +15.97% | 100% |
| IVF nlist1024 nprobe1 | 777x | +15.42% | 97% |
| IVF nlist1024 nprobe2 | 425x | +15.81% | 99% |
| IVF nlist1024 nprobe4 | 226x | +15.93% | 100% |
| IVF nlist1024 nprobe8 | 119x | +15.96% | 100% |
| IVF nlist1024 nprobe16 | 61x | +15.96% | 100% |
| IVF nlist1024 nprobe32 | 31x | +15.96% | 100% |
_(prid->tiny ds300K s13 done in 106s | 80 runs | 3.1h/9.5h elapsed)_

### seed 13 [tiny->omc_ ds300K s13]  Ndb=300,000  LM-only=5.2720  brute=+44.57% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +44.59% | 100% |
| dodeca-uniform P2 probe2 | 1x | +44.55% | 100% |
| dodeca-WHITENED P2 probe1 | 109x | +43.33% | 97% |
| dodeca-WHITENED P2 probe2 | 30x | +44.22% | 99% |
| dodeca-WHITENED P3 probe1 | 777x | +40.35% | 91% |
| dodeca-WHITENED P3 probe2 | 133x | +44.60% | 100% |
| IVF nlist256 nprobe1 | 179x | +44.27% | 99% |
| IVF nlist256 nprobe2 | 104x | +44.53% | 100% |
| IVF nlist256 nprobe4 | 56x | +44.52% | 100% |
| IVF nlist256 nprobe8 | 29x | +44.56% | 100% |
| IVF nlist256 nprobe16 | 15x | +44.56% | 100% |
| IVF nlist256 nprobe32 | 8x | +44.56% | 100% |
| IVF nlist1024 nprobe1 | 821x | +44.04% | 99% |
| IVF nlist1024 nprobe2 | 441x | +44.50% | 100% |
| IVF nlist1024 nprobe4 | 230x | +44.53% | 100% |
| IVF nlist1024 nprobe8 | 118x | +44.56% | 100% |
| IVF nlist1024 nprobe16 | 60x | +44.57% | 100% |
| IVF nlist1024 nprobe32 | 30x | +44.57% | 100% |
_(tiny->omc_ ds300K s13 done in 102s | 81 runs | 3.2h/9.5h elapsed)_

### seed 13 [tiny->prid ds600K s13]  Ndb=600,000  LM-only=2.7758  brute=+11.07% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +11.07% | 100% |
| dodeca-uniform P2 probe2 | 1x | +11.07% | 100% |
| dodeca-WHITENED P2 probe1 | 114x | +9.71% | 88% |
| dodeca-WHITENED P2 probe2 | 31x | +11.01% | 99% |
| dodeca-WHITENED P3 probe1 | 944x | +6.89% | 62% |
| dodeca-WHITENED P3 probe2 | 150x | +10.80% | 98% |
| IVF nlist256 nprobe1 | 197x | +10.87% | 98% |
| IVF nlist256 nprobe2 | 104x | +10.83% | 98% |
| IVF nlist256 nprobe4 | 55x | +11.03% | 100% |
| IVF nlist256 nprobe8 | 29x | +11.05% | 100% |
| IVF nlist256 nprobe16 | 15x | +11.06% | 100% |
| IVF nlist256 nprobe32 | 7x | +11.06% | 100% |
| IVF nlist1024 nprobe1 | 781x | +10.60% | 96% |
| IVF nlist1024 nprobe2 | 426x | +10.90% | 98% |
| IVF nlist1024 nprobe4 | 225x | +11.01% | 99% |
| IVF nlist1024 nprobe8 | 117x | +11.03% | 100% |
| IVF nlist1024 nprobe16 | 60x | +11.04% | 100% |
| IVF nlist1024 nprobe32 | 30x | +11.04% | 100% |
_(tiny->prid ds600K s13 done in 141s | 82 runs | 3.2h/9.5h elapsed)_

### seed 13 [prid->tiny ds600K s13]  Ndb=600,000  LM-only=3.0398  brute=+16.13% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +16.05% | 99% |
| dodeca-uniform P2 probe2 | 1x | +16.13% | 100% |
| dodeca-WHITENED P2 probe1 | 124x | +13.94% | 86% |
| dodeca-WHITENED P2 probe2 | 33x | +15.45% | 96% |
| dodeca-WHITENED P3 probe1 | 905x | +10.83% | 67% |
| dodeca-WHITENED P3 probe2 | 147x | +15.23% | 94% |
| IVF nlist256 nprobe1 | 204x | +16.01% | 99% |
| IVF nlist256 nprobe2 | 107x | +16.10% | 100% |
| IVF nlist256 nprobe4 | 55x | +16.10% | 100% |
| IVF nlist256 nprobe8 | 29x | +16.10% | 100% |
| IVF nlist256 nprobe16 | 15x | +16.10% | 100% |
| IVF nlist256 nprobe32 | 7x | +16.10% | 100% |
| IVF nlist1024 nprobe1 | 750x | +15.99% | 99% |
| IVF nlist1024 nprobe2 | 416x | +16.06% | 100% |
| IVF nlist1024 nprobe4 | 222x | +16.12% | 100% |
| IVF nlist1024 nprobe8 | 117x | +16.12% | 100% |
| IVF nlist1024 nprobe16 | 60x | +16.12% | 100% |
| IVF nlist1024 nprobe32 | 31x | +16.12% | 100% |
_(prid->tiny ds600K s13 done in 137s | 83 runs | 3.2h/9.5h elapsed)_

### seed 13 [tiny->omc_ ds600K s13]  Ndb=600,000  LM-only=5.2720  brute=+46.03% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +45.83% | 100% |
| dodeca-uniform P2 probe2 | 1x | +46.03% | 100% |
| dodeca-WHITENED P2 probe1 | 111x | +45.11% | 98% |
| dodeca-WHITENED P2 probe2 | 30x | +45.25% | 98% |
| dodeca-WHITENED P3 probe1 | 881x | +41.37% | 90% |
| dodeca-WHITENED P3 probe2 | 140x | +45.59% | 99% |
| IVF nlist256 nprobe1 | 180x | +45.81% | 100% |
| IVF nlist256 nprobe2 | 100x | +45.92% | 100% |
| IVF nlist256 nprobe4 | 53x | +46.02% | 100% |
| IVF nlist256 nprobe8 | 28x | +46.03% | 100% |
| IVF nlist256 nprobe16 | 15x | +46.03% | 100% |
| IVF nlist256 nprobe32 | 8x | +46.03% | 100% |
| IVF nlist1024 nprobe1 | 786x | +45.57% | 99% |
| IVF nlist1024 nprobe2 | 427x | +45.65% | 99% |
| IVF nlist1024 nprobe4 | 224x | +45.92% | 100% |
| IVF nlist1024 nprobe8 | 117x | +45.93% | 100% |
| IVF nlist1024 nprobe16 | 60x | +45.94% | 100% |
| IVF nlist1024 nprobe32 | 31x | +45.94% | 100% |
_(tiny->omc_ ds600K s13 done in 141s | 84 runs | 3.3h/9.5h elapsed)_

### seed 14 [tiny->prid ds300K s14]  Ndb=300,000  LM-only=2.7911  brute=+10.52% (λ=0.75)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +10.28% | 98% |
| dodeca-uniform P2 probe2 | 1x | +10.52% | 100% |
| dodeca-WHITENED P2 probe1 | 123x | +8.23% | 78% |
| dodeca-WHITENED P2 probe2 | 32x | +10.36% | 99% |
| dodeca-WHITENED P3 probe1 | 882x | +6.39% | 61% |
| dodeca-WHITENED P3 probe2 | 147x | +10.01% | 95% |
| IVF nlist256 nprobe1 | 214x | +9.85% | 94% |
| IVF nlist256 nprobe2 | 111x | +10.20% | 97% |
| IVF nlist256 nprobe4 | 58x | +10.26% | 98% |
| IVF nlist256 nprobe8 | 29x | +10.51% | 100% |
| IVF nlist256 nprobe16 | 15x | +10.53% | 100% |
| IVF nlist256 nprobe32 | 7x | +10.53% | 100% |
| IVF nlist1024 nprobe1 | 776x | +10.24% | 97% |
| IVF nlist1024 nprobe2 | 426x | +10.13% | 96% |
| IVF nlist1024 nprobe4 | 224x | +10.29% | 98% |
| IVF nlist1024 nprobe8 | 116x | +10.31% | 98% |
| IVF nlist1024 nprobe16 | 59x | +10.53% | 100% |
| IVF nlist1024 nprobe32 | 30x | +10.54% | 100% |
_(tiny->prid ds300K s14 done in 103s | 85 runs | 3.3h/9.5h elapsed)_

### seed 14 [prid->tiny ds300K s14]  Ndb=300,000  LM-only=3.1407  brute=+19.07% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +19.02% | 100% |
| dodeca-uniform P2 probe2 | 1x | +19.07% | 100% |
| dodeca-WHITENED P2 probe1 | 122x | +16.51% | 87% |
| dodeca-WHITENED P2 probe2 | 33x | +18.16% | 95% |
| dodeca-WHITENED P3 probe1 | 878x | +11.11% | 58% |
| dodeca-WHITENED P3 probe2 | 143x | +17.06% | 89% |
| IVF nlist256 nprobe1 | 203x | +19.09% | 100% |
| IVF nlist256 nprobe2 | 106x | +19.08% | 100% |
| IVF nlist256 nprobe4 | 54x | +19.08% | 100% |
| IVF nlist256 nprobe8 | 28x | +19.07% | 100% |
| IVF nlist256 nprobe16 | 14x | +19.07% | 100% |
| IVF nlist256 nprobe32 | 7x | +19.07% | 100% |
| IVF nlist1024 nprobe1 | 777x | +18.41% | 97% |
| IVF nlist1024 nprobe2 | 425x | +19.00% | 100% |
| IVF nlist1024 nprobe4 | 225x | +19.08% | 100% |
| IVF nlist1024 nprobe8 | 118x | +19.07% | 100% |
| IVF nlist1024 nprobe16 | 60x | +19.07% | 100% |
| IVF nlist1024 nprobe32 | 30x | +19.07% | 100% |
_(prid->tiny ds300K s14 done in 99s | 86 runs | 3.3h/9.5h elapsed)_

### seed 14 [tiny->omc_ ds300K s14]  Ndb=300,000  LM-only=5.4376  brute=+46.81% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 3x | +46.78% | 100% |
| dodeca-uniform P2 probe2 | 1x | +46.81% | 100% |
| dodeca-WHITENED P2 probe1 | 128x | +45.84% | 98% |
| dodeca-WHITENED P2 probe2 | 33x | +47.56% | 102% |
| dodeca-WHITENED P3 probe1 | 917x | +41.53% | 89% |
| dodeca-WHITENED P3 probe2 | 151x | +46.41% | 99% |
| IVF nlist256 nprobe1 | 204x | +47.14% | 101% |
| IVF nlist256 nprobe2 | 108x | +46.97% | 100% |
| IVF nlist256 nprobe4 | 56x | +46.87% | 100% |
| IVF nlist256 nprobe8 | 29x | +46.88% | 100% |
| IVF nlist256 nprobe16 | 15x | +46.88% | 100% |
| IVF nlist256 nprobe32 | 8x | +46.89% | 100% |
| IVF nlist1024 nprobe1 | 805x | +46.50% | 99% |
| IVF nlist1024 nprobe2 | 429x | +47.09% | 101% |
| IVF nlist1024 nprobe4 | 224x | +47.07% | 101% |
| IVF nlist1024 nprobe8 | 116x | +46.87% | 100% |
| IVF nlist1024 nprobe16 | 59x | +46.89% | 100% |
| IVF nlist1024 nprobe32 | 30x | +46.89% | 100% |
_(tiny->omc_ ds300K s14 done in 94s | 87 runs | 3.4h/9.5h elapsed)_

### seed 14 [tiny->prid ds600K s14]  Ndb=600,000  LM-only=2.7911  brute=+11.63% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +11.60% | 100% |
| dodeca-uniform P2 probe2 | 1x | +11.63% | 100% |
| dodeca-WHITENED P2 probe1 | 124x | +9.73% | 84% |
| dodeca-WHITENED P2 probe2 | 33x | +11.20% | 96% |
| dodeca-WHITENED P3 probe1 | 948x | +7.11% | 61% |
| dodeca-WHITENED P3 probe2 | 148x | +10.99% | 94% |
| IVF nlist256 nprobe1 | 209x | +11.33% | 97% |
| IVF nlist256 nprobe2 | 109x | +11.62% | 100% |
| IVF nlist256 nprobe4 | 56x | +11.63% | 100% |
| IVF nlist256 nprobe8 | 29x | +11.64% | 100% |
| IVF nlist256 nprobe16 | 15x | +11.65% | 100% |
| IVF nlist256 nprobe32 | 8x | +11.66% | 100% |
| IVF nlist1024 nprobe1 | 797x | +10.83% | 93% |
| IVF nlist1024 nprobe2 | 430x | +11.16% | 96% |
| IVF nlist1024 nprobe4 | 228x | +11.57% | 99% |
| IVF nlist1024 nprobe8 | 118x | +11.58% | 100% |
| IVF nlist1024 nprobe16 | 60x | +11.63% | 100% |
| IVF nlist1024 nprobe32 | 30x | +11.64% | 100% |
_(tiny->prid ds600K s14 done in 140s | 88 runs | 3.4h/9.5h elapsed)_

### seed 14 [prid->tiny ds600K s14]  Ndb=600,000  LM-only=3.1407  brute=+18.96% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +19.13% | 101% |
| dodeca-uniform P2 probe2 | 1x | +18.96% | 100% |
| dodeca-WHITENED P2 probe1 | 120x | +17.69% | 93% |
| dodeca-WHITENED P2 probe2 | 32x | +18.80% | 99% |
| dodeca-WHITENED P3 probe1 | 889x | +13.60% | 72% |
| dodeca-WHITENED P3 probe2 | 147x | +18.44% | 97% |
| IVF nlist256 nprobe1 | 191x | +18.66% | 98% |
| IVF nlist256 nprobe2 | 103x | +18.96% | 100% |
| IVF nlist256 nprobe4 | 53x | +18.96% | 100% |
| IVF nlist256 nprobe8 | 27x | +18.96% | 100% |
| IVF nlist256 nprobe16 | 14x | +18.95% | 100% |
| IVF nlist256 nprobe32 | 7x | +18.95% | 100% |
| IVF nlist1024 nprobe1 | 801x | +18.85% | 99% |
| IVF nlist1024 nprobe2 | 425x | +19.02% | 100% |
| IVF nlist1024 nprobe4 | 225x | +18.96% | 100% |
| IVF nlist1024 nprobe8 | 116x | +18.97% | 100% |
| IVF nlist1024 nprobe16 | 58x | +18.97% | 100% |
| IVF nlist1024 nprobe32 | 30x | +18.97% | 100% |
_(prid->tiny ds600K s14 done in 140s | 89 runs | 3.4h/9.5h elapsed)_

### seed 14 [tiny->omc_ ds600K s14]  Ndb=600,000  LM-only=5.4376  brute=+47.44% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 3x | +47.49% | 100% |
| dodeca-uniform P2 probe2 | 1x | +47.44% | 100% |
| dodeca-WHITENED P2 probe1 | 119x | +45.35% | 96% |
| dodeca-WHITENED P2 probe2 | 32x | +47.17% | 99% |
| dodeca-WHITENED P3 probe1 | 836x | +43.88% | 93% |
| dodeca-WHITENED P3 probe2 | 144x | +46.62% | 98% |
| IVF nlist256 nprobe1 | 195x | +47.51% | 100% |
| IVF nlist256 nprobe2 | 104x | +47.57% | 100% |
| IVF nlist256 nprobe4 | 55x | +47.44% | 100% |
| IVF nlist256 nprobe8 | 28x | +47.45% | 100% |
| IVF nlist256 nprobe16 | 15x | +47.45% | 100% |
| IVF nlist256 nprobe32 | 8x | +47.45% | 100% |
| IVF nlist1024 nprobe1 | 798x | +47.45% | 100% |
| IVF nlist1024 nprobe2 | 428x | +47.35% | 100% |
| IVF nlist1024 nprobe4 | 221x | +47.42% | 100% |
| IVF nlist1024 nprobe8 | 114x | +47.43% | 100% |
| IVF nlist1024 nprobe16 | 59x | +47.44% | 100% |
| IVF nlist1024 nprobe32 | 30x | +47.44% | 100% |
_(tiny->omc_ ds600K s14 done in 136s | 90 runs | 3.5h/9.5h elapsed)_

### seed 15 [tiny->prid ds300K s15]  Ndb=300,000  LM-only=2.7288  brute=+9.34% (λ=0.75)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +9.46% | 101% |
| dodeca-uniform P2 probe2 | 1x | +9.31% | 100% |
| dodeca-WHITENED P2 probe1 | 119x | +7.72% | 83% |
| dodeca-WHITENED P2 probe2 | 32x | +9.50% | 102% |
| dodeca-WHITENED P3 probe1 | 797x | +4.83% | 52% |
| dodeca-WHITENED P3 probe2 | 137x | +8.32% | 89% |
| IVF nlist256 nprobe1 | 211x | +8.71% | 93% |
| IVF nlist256 nprobe2 | 110x | +8.84% | 95% |
| IVF nlist256 nprobe4 | 58x | +9.07% | 97% |
| IVF nlist256 nprobe8 | 29x | +9.07% | 97% |
| IVF nlist256 nprobe16 | 15x | +9.10% | 97% |
| IVF nlist256 nprobe32 | 7x | +9.11% | 98% |
| IVF nlist1024 nprobe1 | 768x | +8.85% | 95% |
| IVF nlist1024 nprobe2 | 422x | +9.12% | 98% |
| IVF nlist1024 nprobe4 | 223x | +9.12% | 98% |
| IVF nlist1024 nprobe8 | 115x | +9.11% | 97% |
| IVF nlist1024 nprobe16 | 58x | +9.11% | 97% |
| IVF nlist1024 nprobe32 | 30x | +9.11% | 98% |
_(tiny->prid ds300K s15 done in 101s | 91 runs | 3.5h/9.5h elapsed)_

### seed 15 [prid->tiny ds300K s15]  Ndb=300,000  LM-only=3.0308  brute=+15.49% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 3x | +14.75% | 95% |
| dodeca-uniform P2 probe2 | 1x | +15.18% | 98% |
| dodeca-WHITENED P2 probe1 | 116x | +12.80% | 83% |
| dodeca-WHITENED P2 probe2 | 31x | +15.00% | 97% |
| dodeca-WHITENED P3 probe1 | 890x | +8.04% | 52% |
| dodeca-WHITENED P3 probe2 | 149x | +13.33% | 86% |
| IVF nlist256 nprobe1 | 197x | +15.22% | 98% |
| IVF nlist256 nprobe2 | 102x | +15.48% | 100% |
| IVF nlist256 nprobe4 | 53x | +15.48% | 100% |
| IVF nlist256 nprobe8 | 27x | +15.49% | 100% |
| IVF nlist256 nprobe16 | 14x | +15.49% | 100% |
| IVF nlist256 nprobe32 | 7x | +15.49% | 100% |
| IVF nlist1024 nprobe1 | 795x | +15.09% | 97% |
| IVF nlist1024 nprobe2 | 434x | +15.49% | 100% |
| IVF nlist1024 nprobe4 | 228x | +15.48% | 100% |
| IVF nlist1024 nprobe8 | 118x | +15.53% | 100% |
| IVF nlist1024 nprobe16 | 60x | +15.53% | 100% |
| IVF nlist1024 nprobe32 | 30x | +15.53% | 100% |
_(prid->tiny ds300K s15 done in 99s | 92 runs | 3.5h/9.5h elapsed)_

### seed 15 [tiny->omc_ ds300K s15]  Ndb=300,000  LM-only=5.8743  brute=+49.77% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +49.67% | 100% |
| dodeca-uniform P2 probe2 | 1x | +49.76% | 100% |
| dodeca-WHITENED P2 probe1 | 118x | +48.03% | 97% |
| dodeca-WHITENED P2 probe2 | 32x | +49.39% | 99% |
| dodeca-WHITENED P3 probe1 | 697x | +45.76% | 92% |
| dodeca-WHITENED P3 probe2 | 135x | +48.96% | 98% |
| IVF nlist256 nprobe1 | 203x | +49.79% | 100% |
| IVF nlist256 nprobe2 | 106x | +49.74% | 100% |
| IVF nlist256 nprobe4 | 55x | +49.75% | 100% |
| IVF nlist256 nprobe8 | 29x | +49.75% | 100% |
| IVF nlist256 nprobe16 | 15x | +49.76% | 100% |
| IVF nlist256 nprobe32 | 7x | +49.77% | 100% |
| IVF nlist1024 nprobe1 | 805x | +49.62% | 100% |
| IVF nlist1024 nprobe2 | 433x | +49.87% | 100% |
| IVF nlist1024 nprobe4 | 226x | +49.75% | 100% |
| IVF nlist1024 nprobe8 | 116x | +49.76% | 100% |
| IVF nlist1024 nprobe16 | 59x | +49.77% | 100% |
| IVF nlist1024 nprobe32 | 30x | +49.76% | 100% |
_(tiny->omc_ ds300K s15 done in 106s | 93 runs | 3.6h/9.5h elapsed)_

### seed 15 [tiny->prid ds600K s15]  Ndb=600,000  LM-only=2.7288  brute=+9.71% (λ=0.75)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +9.47% | 97% |
| dodeca-uniform P2 probe2 | 1x | +9.71% | 100% |
| dodeca-WHITENED P2 probe1 | 115x | +7.93% | 82% |
| dodeca-WHITENED P2 probe2 | 31x | +9.57% | 99% |
| dodeca-WHITENED P3 probe1 | 880x | +6.95% | 72% |
| dodeca-WHITENED P3 probe2 | 144x | +9.61% | 99% |
| IVF nlist256 nprobe1 | 197x | +9.38% | 97% |
| IVF nlist256 nprobe2 | 103x | +9.39% | 97% |
| IVF nlist256 nprobe4 | 55x | +9.42% | 97% |
| IVF nlist256 nprobe8 | 29x | +9.67% | 100% |
| IVF nlist256 nprobe16 | 15x | +9.70% | 100% |
| IVF nlist256 nprobe32 | 8x | +9.70% | 100% |
| IVF nlist1024 nprobe1 | 738x | +9.16% | 94% |
| IVF nlist1024 nprobe2 | 412x | +9.42% | 97% |
| IVF nlist1024 nprobe4 | 218x | +9.45% | 97% |
| IVF nlist1024 nprobe8 | 113x | +9.44% | 97% |
| IVF nlist1024 nprobe16 | 58x | +9.44% | 97% |
| IVF nlist1024 nprobe32 | 30x | +9.66% | 99% |
_(tiny->prid ds600K s15 done in 140s | 94 runs | 3.6h/9.5h elapsed)_

### seed 15 [prid->tiny ds600K s15]  Ndb=600,000  LM-only=3.0308  brute=+16.00% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 3x | +15.79% | 99% |
| dodeca-uniform P2 probe2 | 1x | +15.71% | 98% |
| dodeca-WHITENED P2 probe1 | 114x | +13.69% | 86% |
| dodeca-WHITENED P2 probe2 | 31x | +15.20% | 95% |
| dodeca-WHITENED P3 probe1 | 884x | +10.76% | 67% |
| dodeca-WHITENED P3 probe2 | 151x | +14.93% | 93% |
| IVF nlist256 nprobe1 | 191x | +16.03% | 100% |
| IVF nlist256 nprobe2 | 105x | +16.00% | 100% |
| IVF nlist256 nprobe4 | 55x | +16.00% | 100% |
| IVF nlist256 nprobe8 | 28x | +16.00% | 100% |
| IVF nlist256 nprobe16 | 14x | +16.00% | 100% |
| IVF nlist256 nprobe32 | 7x | +16.00% | 100% |
| IVF nlist1024 nprobe1 | 781x | +16.11% | 101% |
| IVF nlist1024 nprobe2 | 421x | +16.04% | 100% |
| IVF nlist1024 nprobe4 | 222x | +16.03% | 100% |
| IVF nlist1024 nprobe8 | 115x | +16.01% | 100% |
| IVF nlist1024 nprobe16 | 58x | +16.01% | 100% |
| IVF nlist1024 nprobe32 | 30x | +16.01% | 100% |
_(prid->tiny ds600K s15 done in 137s | 95 runs | 3.6h/9.5h elapsed)_

### seed 15 [tiny->omc_ ds600K s15]  Ndb=600,000  LM-only=5.8743  brute=+50.38% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +50.45% | 100% |
| dodeca-uniform P2 probe2 | 1x | +50.39% | 100% |
| dodeca-WHITENED P2 probe1 | 115x | +50.14% | 100% |
| dodeca-WHITENED P2 probe2 | 32x | +49.99% | 99% |
| dodeca-WHITENED P3 probe1 | 806x | +47.11% | 93% |
| dodeca-WHITENED P3 probe2 | 144x | +50.03% | 99% |
| IVF nlist256 nprobe1 | 192x | +50.37% | 100% |
| IVF nlist256 nprobe2 | 102x | +50.33% | 100% |
| IVF nlist256 nprobe4 | 54x | +50.37% | 100% |
| IVF nlist256 nprobe8 | 28x | +50.38% | 100% |
| IVF nlist256 nprobe16 | 14x | +50.38% | 100% |
| IVF nlist256 nprobe32 | 8x | +50.39% | 100% |
| IVF nlist1024 nprobe1 | 791x | +49.95% | 99% |
| IVF nlist1024 nprobe2 | 429x | +50.38% | 100% |
| IVF nlist1024 nprobe4 | 223x | +50.36% | 100% |
| IVF nlist1024 nprobe8 | 116x | +50.39% | 100% |
| IVF nlist1024 nprobe16 | 59x | +50.39% | 100% |
| IVF nlist1024 nprobe32 | 30x | +50.40% | 100% |
_(tiny->omc_ ds600K s15 done in 167s | 96 runs | 3.7h/9.5h elapsed)_

### seed 16 [tiny->prid ds300K s16]  Ndb=300,000  LM-only=2.7711  brute=+10.66% (λ=0.75)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +10.46% | 98% |
| dodeca-uniform P2 probe2 | 1x | +10.65% | 100% |
| dodeca-WHITENED P2 probe1 | 117x | +9.04% | 85% |
| dodeca-WHITENED P2 probe2 | 32x | +10.66% | 100% |
| dodeca-WHITENED P3 probe1 | 955x | +5.95% | 56% |
| dodeca-WHITENED P3 probe2 | 152x | +9.20% | 86% |
| IVF nlist256 nprobe1 | 201x | +10.39% | 97% |
| IVF nlist256 nprobe2 | 108x | +10.65% | 100% |
| IVF nlist256 nprobe4 | 56x | +10.65% | 100% |
| IVF nlist256 nprobe8 | 29x | +10.65% | 100% |
| IVF nlist256 nprobe16 | 15x | +10.65% | 100% |
| IVF nlist256 nprobe32 | 7x | +10.64% | 100% |
| IVF nlist1024 nprobe1 | 784x | +9.75% | 92% |
| IVF nlist1024 nprobe2 | 423x | +10.57% | 99% |
| IVF nlist1024 nprobe4 | 222x | +10.65% | 100% |
| IVF nlist1024 nprobe8 | 115x | +10.64% | 100% |
| IVF nlist1024 nprobe16 | 59x | +10.66% | 100% |
| IVF nlist1024 nprobe32 | 30x | +10.65% | 100% |
_(tiny->prid ds300K s16 done in 102s | 97 runs | 3.7h/9.5h elapsed)_

### seed 16 [prid->tiny ds300K s16]  Ndb=300,000  LM-only=3.0009  brute=+15.47% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +15.39% | 99% |
| dodeca-uniform P2 probe2 | 1x | +15.47% | 100% |
| dodeca-WHITENED P2 probe1 | 120x | +12.61% | 82% |
| dodeca-WHITENED P2 probe2 | 32x | +15.39% | 99% |
| dodeca-WHITENED P3 probe1 | 903x | +9.13% | 59% |
| dodeca-WHITENED P3 probe2 | 150x | +14.21% | 92% |
| IVF nlist256 nprobe1 | 195x | +15.46% | 100% |
| IVF nlist256 nprobe2 | 104x | +15.45% | 100% |
| IVF nlist256 nprobe4 | 55x | +15.47% | 100% |
| IVF nlist256 nprobe8 | 28x | +15.47% | 100% |
| IVF nlist256 nprobe16 | 14x | +15.47% | 100% |
| IVF nlist256 nprobe32 | 7x | +15.47% | 100% |
| IVF nlist1024 nprobe1 | 775x | +14.61% | 94% |
| IVF nlist1024 nprobe2 | 419x | +15.31% | 99% |
| IVF nlist1024 nprobe4 | 220x | +15.48% | 100% |
| IVF nlist1024 nprobe8 | 114x | +15.48% | 100% |
| IVF nlist1024 nprobe16 | 59x | +15.48% | 100% |
| IVF nlist1024 nprobe32 | 30x | +15.48% | 100% |
_(prid->tiny ds300K s16 done in 99s | 98 runs | 3.7h/9.5h elapsed)_

### seed 16 [tiny->omc_ ds300K s16]  Ndb=300,000  LM-only=5.6828  brute=+49.37% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +49.40% | 100% |
| dodeca-uniform P2 probe2 | 1x | +49.37% | 100% |
| dodeca-WHITENED P2 probe1 | 119x | +46.89% | 95% |
| dodeca-WHITENED P2 probe2 | 32x | +48.78% | 99% |
| dodeca-WHITENED P3 probe1 | 845x | +43.60% | 88% |
| dodeca-WHITENED P3 probe2 | 141x | +48.75% | 99% |
| IVF nlist256 nprobe1 | 201x | +49.20% | 100% |
| IVF nlist256 nprobe2 | 106x | +49.15% | 100% |
| IVF nlist256 nprobe4 | 55x | +49.20% | 100% |
| IVF nlist256 nprobe8 | 29x | +49.22% | 100% |
| IVF nlist256 nprobe16 | 15x | +49.25% | 100% |
| IVF nlist256 nprobe32 | 8x | +49.25% | 100% |
| IVF nlist1024 nprobe1 | 800x | +48.71% | 99% |
| IVF nlist1024 nprobe2 | 428x | +49.13% | 100% |
| IVF nlist1024 nprobe4 | 224x | +49.02% | 99% |
| IVF nlist1024 nprobe8 | 116x | +49.22% | 100% |
| IVF nlist1024 nprobe16 | 60x | +49.25% | 100% |
| IVF nlist1024 nprobe32 | 31x | +49.26% | 100% |
_(tiny->omc_ ds300K s16 done in 99s | 99 runs | 3.8h/9.5h elapsed)_

### seed 16 [tiny->prid ds600K s16]  Ndb=600,000  LM-only=2.7711  brute=+10.99% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +10.86% | 99% |
| dodeca-uniform P2 probe2 | 1x | +10.99% | 100% |
| dodeca-WHITENED P2 probe1 | 120x | +9.67% | 88% |
| dodeca-WHITENED P2 probe2 | 32x | +10.43% | 95% |
| dodeca-WHITENED P3 probe1 | 937x | +7.05% | 64% |
| dodeca-WHITENED P3 probe2 | 152x | +9.77% | 89% |
| IVF nlist256 nprobe1 | 207x | +10.86% | 99% |
| IVF nlist256 nprobe2 | 108x | +11.19% | 102% |
| IVF nlist256 nprobe4 | 58x | +10.96% | 100% |
| IVF nlist256 nprobe8 | 29x | +10.96% | 100% |
| IVF nlist256 nprobe16 | 15x | +10.98% | 100% |
| IVF nlist256 nprobe32 | 8x | +10.98% | 100% |
| IVF nlist1024 nprobe1 | 801x | +10.63% | 97% |
| IVF nlist1024 nprobe2 | 434x | +10.89% | 99% |
| IVF nlist1024 nprobe4 | 225x | +10.92% | 99% |
| IVF nlist1024 nprobe8 | 116x | +10.94% | 100% |
| IVF nlist1024 nprobe16 | 59x | +10.97% | 100% |
| IVF nlist1024 nprobe32 | 30x | +10.98% | 100% |
_(tiny->prid ds600K s16 done in 139s | 100 runs | 3.8h/9.5h elapsed)_

### seed 16 [prid->tiny ds600K s16]  Ndb=600,000  LM-only=3.0009  brute=+15.66% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +15.70% | 100% |
| dodeca-uniform P2 probe2 | 1x | +15.65% | 100% |
| dodeca-WHITENED P2 probe1 | 118x | +13.49% | 86% |
| dodeca-WHITENED P2 probe2 | 32x | +15.08% | 96% |
| dodeca-WHITENED P3 probe1 | 924x | +10.31% | 66% |
| dodeca-WHITENED P3 probe2 | 148x | +14.76% | 94% |
| IVF nlist256 nprobe1 | 209x | +15.63% | 100% |
| IVF nlist256 nprobe2 | 109x | +15.66% | 100% |
| IVF nlist256 nprobe4 | 55x | +15.65% | 100% |
| IVF nlist256 nprobe8 | 29x | +15.67% | 100% |
| IVF nlist256 nprobe16 | 15x | +15.67% | 100% |
| IVF nlist256 nprobe32 | 8x | +15.67% | 100% |
| IVF nlist1024 nprobe1 | 794x | +15.52% | 99% |
| IVF nlist1024 nprobe2 | 434x | +15.68% | 100% |
| IVF nlist1024 nprobe4 | 225x | +15.65% | 100% |
| IVF nlist1024 nprobe8 | 116x | +15.66% | 100% |
| IVF nlist1024 nprobe16 | 59x | +15.66% | 100% |
| IVF nlist1024 nprobe32 | 30x | +15.66% | 100% |
_(prid->tiny ds600K s16 done in 136s | 101 runs | 3.9h/9.5h elapsed)_

### seed 16 [tiny->omc_ ds600K s16]  Ndb=600,000  LM-only=5.6828  brute=+50.04% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +50.22% | 100% |
| dodeca-uniform P2 probe2 | 1x | +50.04% | 100% |
| dodeca-WHITENED P2 probe1 | 116x | +48.06% | 96% |
| dodeca-WHITENED P2 probe2 | 32x | +50.21% | 100% |
| dodeca-WHITENED P3 probe1 | 826x | +46.43% | 93% |
| dodeca-WHITENED P3 probe2 | 141x | +49.40% | 99% |
| IVF nlist256 nprobe1 | 196x | +49.85% | 100% |
| IVF nlist256 nprobe2 | 104x | +49.74% | 99% |
| IVF nlist256 nprobe4 | 56x | +49.97% | 100% |
| IVF nlist256 nprobe8 | 29x | +50.13% | 100% |
| IVF nlist256 nprobe16 | 15x | +50.15% | 100% |
| IVF nlist256 nprobe32 | 8x | +50.15% | 100% |
| IVF nlist1024 nprobe1 | 837x | +49.66% | 99% |
| IVF nlist1024 nprobe2 | 448x | +50.11% | 100% |
| IVF nlist1024 nprobe4 | 234x | +50.08% | 100% |
| IVF nlist1024 nprobe8 | 120x | +50.03% | 100% |
| IVF nlist1024 nprobe16 | 61x | +50.03% | 100% |
| IVF nlist1024 nprobe32 | 31x | +50.03% | 100% |
_(tiny->omc_ ds600K s16 done in 137s | 102 runs | 3.9h/9.5h elapsed)_

### seed 17 [tiny->prid ds300K s17]  Ndb=300,000  LM-only=2.7228  brute=+10.91% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +10.68% | 98% |
| dodeca-uniform P2 probe2 | 1x | +10.92% | 100% |
| dodeca-WHITENED P2 probe1 | 120x | +7.83% | 72% |
| dodeca-WHITENED P2 probe2 | 32x | +10.59% | 97% |
| dodeca-WHITENED P3 probe1 | 989x | +5.37% | 49% |
| dodeca-WHITENED P3 probe2 | 158x | +8.93% | 82% |
| IVF nlist256 nprobe1 | 208x | +10.92% | 100% |
| IVF nlist256 nprobe2 | 110x | +10.97% | 101% |
| IVF nlist256 nprobe4 | 57x | +10.97% | 101% |
| IVF nlist256 nprobe8 | 29x | +10.93% | 100% |
| IVF nlist256 nprobe16 | 15x | +10.94% | 100% |
| IVF nlist256 nprobe32 | 8x | +10.94% | 100% |
| IVF nlist1024 nprobe1 | 789x | +10.16% | 93% |
| IVF nlist1024 nprobe2 | 427x | +10.96% | 100% |
| IVF nlist1024 nprobe4 | 223x | +11.02% | 101% |
| IVF nlist1024 nprobe8 | 115x | +10.97% | 101% |
| IVF nlist1024 nprobe16 | 59x | +10.94% | 100% |
| IVF nlist1024 nprobe32 | 30x | +10.94% | 100% |
_(tiny->prid ds300K s17 done in 101s | 103 runs | 3.9h/9.5h elapsed)_

### seed 17 [prid->tiny ds300K s17]  Ndb=300,000  LM-only=3.1190  brute=+18.90% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +18.90% | 100% |
| dodeca-uniform P2 probe2 | 1x | +18.90% | 100% |
| dodeca-WHITENED P2 probe1 | 124x | +14.86% | 79% |
| dodeca-WHITENED P2 probe2 | 33x | +19.00% | 101% |
| dodeca-WHITENED P3 probe1 | 884x | +11.56% | 61% |
| dodeca-WHITENED P3 probe2 | 145x | +16.97% | 90% |
| IVF nlist256 nprobe1 | 187x | +18.76% | 99% |
| IVF nlist256 nprobe2 | 96x | +18.86% | 100% |
| IVF nlist256 nprobe4 | 52x | +18.90% | 100% |
| IVF nlist256 nprobe8 | 27x | +18.90% | 100% |
| IVF nlist256 nprobe16 | 14x | +18.90% | 100% |
| IVF nlist256 nprobe32 | 7x | +18.90% | 100% |
| IVF nlist1024 nprobe1 | 787x | +18.48% | 98% |
| IVF nlist1024 nprobe2 | 427x | +18.68% | 99% |
| IVF nlist1024 nprobe4 | 230x | +18.91% | 100% |
| IVF nlist1024 nprobe8 | 118x | +18.91% | 100% |
| IVF nlist1024 nprobe16 | 60x | +18.90% | 100% |
| IVF nlist1024 nprobe32 | 30x | +18.90% | 100% |
_(prid->tiny ds300K s17 done in 103s | 104 runs | 3.9h/9.5h elapsed)_

### seed 17 [tiny->omc_ ds300K s17]  Ndb=300,000  LM-only=5.6232  brute=+47.98% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +47.90% | 100% |
| dodeca-uniform P2 probe2 | 1x | +47.82% | 100% |
| dodeca-WHITENED P2 probe1 | 119x | +46.98% | 98% |
| dodeca-WHITENED P2 probe2 | 32x | +48.25% | 101% |
| dodeca-WHITENED P3 probe1 | 819x | +41.43% | 86% |
| dodeca-WHITENED P3 probe2 | 136x | +47.14% | 98% |
| IVF nlist256 nprobe1 | 205x | +48.05% | 100% |
| IVF nlist256 nprobe2 | 106x | +48.05% | 100% |
| IVF nlist256 nprobe4 | 55x | +47.98% | 100% |
| IVF nlist256 nprobe8 | 29x | +47.98% | 100% |
| IVF nlist256 nprobe16 | 15x | +47.98% | 100% |
| IVF nlist256 nprobe32 | 8x | +47.99% | 100% |
| IVF nlist1024 nprobe1 | 827x | +48.12% | 100% |
| IVF nlist1024 nprobe2 | 433x | +47.81% | 100% |
| IVF nlist1024 nprobe4 | 225x | +47.85% | 100% |
| IVF nlist1024 nprobe8 | 115x | +47.96% | 100% |
| IVF nlist1024 nprobe16 | 59x | +47.97% | 100% |
| IVF nlist1024 nprobe32 | 30x | +47.98% | 100% |
_(tiny->omc_ ds300K s17 done in 101s | 105 runs | 4.0h/9.5h elapsed)_

### seed 17 [tiny->prid ds600K s17]  Ndb=600,000  LM-only=2.7228  brute=+11.89% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +11.83% | 100% |
| dodeca-uniform P2 probe2 | 1x | +11.88% | 100% |
| dodeca-WHITENED P2 probe1 | 119x | +10.08% | 85% |
| dodeca-WHITENED P2 probe2 | 32x | +11.63% | 98% |
| dodeca-WHITENED P3 probe1 | 955x | +6.87% | 58% |
| dodeca-WHITENED P3 probe2 | 153x | +11.42% | 96% |
| IVF nlist256 nprobe1 | 196x | +11.51% | 97% |
| IVF nlist256 nprobe2 | 103x | +11.66% | 98% |
| IVF nlist256 nprobe4 | 54x | +11.85% | 100% |
| IVF nlist256 nprobe8 | 28x | +11.88% | 100% |
| IVF nlist256 nprobe16 | 15x | +11.88% | 100% |
| IVF nlist256 nprobe32 | 7x | +11.88% | 100% |
| IVF nlist1024 nprobe1 | 787x | +11.60% | 98% |
| IVF nlist1024 nprobe2 | 425x | +11.86% | 100% |
| IVF nlist1024 nprobe4 | 221x | +11.80% | 99% |
| IVF nlist1024 nprobe8 | 114x | +11.82% | 99% |
| IVF nlist1024 nprobe16 | 58x | +11.84% | 100% |
| IVF nlist1024 nprobe32 | 30x | +11.86% | 100% |
_(tiny->prid ds600K s17 done in 129s | 106 runs | 4.0h/9.5h elapsed)_

### seed 17 [prid->tiny ds600K s17]  Ndb=600,000  LM-only=3.1190  brute=+18.88% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +18.88% | 100% |
| dodeca-uniform P2 probe2 | 1x | +18.88% | 100% |
| dodeca-WHITENED P2 probe1 | 122x | +16.79% | 89% |
| dodeca-WHITENED P2 probe2 | 33x | +18.69% | 99% |
| dodeca-WHITENED P3 probe1 | 930x | +13.97% | 74% |
| dodeca-WHITENED P3 probe2 | 151x | +18.51% | 98% |
| IVF nlist256 nprobe1 | 200x | +18.96% | 100% |
| IVF nlist256 nprobe2 | 105x | +18.87% | 100% |
| IVF nlist256 nprobe4 | 53x | +18.89% | 100% |
| IVF nlist256 nprobe8 | 28x | +18.89% | 100% |
| IVF nlist256 nprobe16 | 14x | +18.89% | 100% |
| IVF nlist256 nprobe32 | 7x | +18.89% | 100% |
| IVF nlist1024 nprobe1 | 774x | +18.75% | 99% |
| IVF nlist1024 nprobe2 | 422x | +18.98% | 101% |
| IVF nlist1024 nprobe4 | 223x | +18.90% | 100% |
| IVF nlist1024 nprobe8 | 115x | +18.89% | 100% |
| IVF nlist1024 nprobe16 | 59x | +18.89% | 100% |
| IVF nlist1024 nprobe32 | 30x | +18.89% | 100% |
_(prid->tiny ds600K s17 done in 144s | 107 runs | 4.1h/9.5h elapsed)_

### seed 17 [tiny->omc_ ds600K s17]  Ndb=600,000  LM-only=5.6232  brute=+48.44% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +48.40% | 100% |
| dodeca-uniform P2 probe2 | 1x | +48.32% | 100% |
| dodeca-WHITENED P2 probe1 | 123x | +47.19% | 97% |
| dodeca-WHITENED P2 probe2 | 32x | +48.65% | 100% |
| dodeca-WHITENED P3 probe1 | 789x | +44.53% | 92% |
| dodeca-WHITENED P3 probe2 | 140x | +48.46% | 100% |
| IVF nlist256 nprobe1 | 196x | +48.08% | 99% |
| IVF nlist256 nprobe2 | 102x | +48.35% | 100% |
| IVF nlist256 nprobe4 | 53x | +48.42% | 100% |
| IVF nlist256 nprobe8 | 27x | +48.43% | 100% |
| IVF nlist256 nprobe16 | 14x | +48.44% | 100% |
| IVF nlist256 nprobe32 | 7x | +48.44% | 100% |
| IVF nlist1024 nprobe1 | 792x | +48.83% | 101% |
| IVF nlist1024 nprobe2 | 417x | +48.14% | 99% |
| IVF nlist1024 nprobe4 | 216x | +48.29% | 100% |
| IVF nlist1024 nprobe8 | 114x | +48.43% | 100% |
| IVF nlist1024 nprobe16 | 59x | +48.43% | 100% |
| IVF nlist1024 nprobe32 | 30x | +48.44% | 100% |
_(tiny->omc_ ds600K s17 done in 133s | 108 runs | 4.1h/9.5h elapsed)_

### seed 18 [tiny->prid ds300K s18]  Ndb=300,000  LM-only=2.7773  brute=+10.74% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +10.58% | 99% |
| dodeca-uniform P2 probe2 | 1x | +10.74% | 100% |
| dodeca-WHITENED P2 probe1 | 127x | +8.84% | 82% |
| dodeca-WHITENED P2 probe2 | 33x | +10.81% | 101% |
| dodeca-WHITENED P3 probe1 | 906x | +5.87% | 55% |
| dodeca-WHITENED P3 probe2 | 153x | +9.58% | 89% |
| IVF nlist256 nprobe1 | 205x | +10.43% | 97% |
| IVF nlist256 nprobe2 | 107x | +10.64% | 99% |
| IVF nlist256 nprobe4 | 57x | +10.69% | 99% |
| IVF nlist256 nprobe8 | 29x | +10.68% | 99% |
| IVF nlist256 nprobe16 | 15x | +10.72% | 100% |
| IVF nlist256 nprobe32 | 7x | +10.74% | 100% |
| IVF nlist1024 nprobe1 | 782x | +9.79% | 91% |
| IVF nlist1024 nprobe2 | 423x | +10.37% | 97% |
| IVF nlist1024 nprobe4 | 224x | +10.45% | 97% |
| IVF nlist1024 nprobe8 | 116x | +10.46% | 97% |
| IVF nlist1024 nprobe16 | 59x | +10.46% | 97% |
| IVF nlist1024 nprobe32 | 30x | +10.46% | 97% |
_(tiny->prid ds300K s18 done in 104s | 109 runs | 4.1h/9.5h elapsed)_

### seed 18 [prid->tiny ds300K s18]  Ndb=300,000  LM-only=3.0721  brute=+17.29% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +17.29% | 100% |
| dodeca-uniform P2 probe2 | 1x | +17.29% | 100% |
| dodeca-WHITENED P2 probe1 | 122x | +14.50% | 84% |
| dodeca-WHITENED P2 probe2 | 33x | +16.91% | 98% |
| dodeca-WHITENED P3 probe1 | 1017x | +10.60% | 61% |
| dodeca-WHITENED P3 probe2 | 163x | +15.44% | 89% |
| IVF nlist256 nprobe1 | 196x | +17.23% | 100% |
| IVF nlist256 nprobe2 | 104x | +17.26% | 100% |
| IVF nlist256 nprobe4 | 55x | +17.29% | 100% |
| IVF nlist256 nprobe8 | 29x | +17.28% | 100% |
| IVF nlist256 nprobe16 | 15x | +17.29% | 100% |
| IVF nlist256 nprobe32 | 8x | +17.29% | 100% |
| IVF nlist1024 nprobe1 | 790x | +16.78% | 97% |
| IVF nlist1024 nprobe2 | 436x | +17.32% | 100% |
| IVF nlist1024 nprobe4 | 229x | +17.31% | 100% |
| IVF nlist1024 nprobe8 | 118x | +17.28% | 100% |
| IVF nlist1024 nprobe16 | 60x | +17.28% | 100% |
| IVF nlist1024 nprobe32 | 30x | +17.28% | 100% |
_(prid->tiny ds300K s18 done in 103s | 110 runs | 4.1h/9.5h elapsed)_

### seed 18 [tiny->omc_ ds300K s18]  Ndb=300,000  LM-only=5.2373  brute=+45.55% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +45.54% | 100% |
| dodeca-uniform P2 probe2 | 1x | +45.55% | 100% |
| dodeca-WHITENED P2 probe1 | 121x | +43.26% | 95% |
| dodeca-WHITENED P2 probe2 | 33x | +45.13% | 99% |
| dodeca-WHITENED P3 probe1 | 803x | +38.27% | 84% |
| dodeca-WHITENED P3 probe2 | 139x | +45.39% | 100% |
| IVF nlist256 nprobe1 | 182x | +45.26% | 99% |
| IVF nlist256 nprobe2 | 97x | +45.46% | 100% |
| IVF nlist256 nprobe4 | 51x | +45.42% | 100% |
| IVF nlist256 nprobe8 | 26x | +45.42% | 100% |
| IVF nlist256 nprobe16 | 14x | +45.44% | 100% |
| IVF nlist256 nprobe32 | 7x | +45.45% | 100% |
| IVF nlist1024 nprobe1 | 778x | +44.81% | 98% |
| IVF nlist1024 nprobe2 | 423x | +45.58% | 100% |
| IVF nlist1024 nprobe4 | 222x | +45.58% | 100% |
| IVF nlist1024 nprobe8 | 113x | +45.50% | 100% |
| IVF nlist1024 nprobe16 | 58x | +45.54% | 100% |
| IVF nlist1024 nprobe32 | 29x | +45.54% | 100% |
_(tiny->omc_ ds300K s18 done in 104s | 111 runs | 4.2h/9.5h elapsed)_

### seed 18 [tiny->prid ds600K s18]  Ndb=600,000  LM-only=2.7773  brute=+11.05% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +10.95% | 99% |
| dodeca-uniform P2 probe2 | 1x | +11.05% | 100% |
| dodeca-WHITENED P2 probe1 | 128x | +9.08% | 82% |
| dodeca-WHITENED P2 probe2 | 34x | +10.65% | 96% |
| dodeca-WHITENED P3 probe1 | 844x | +7.32% | 66% |
| dodeca-WHITENED P3 probe2 | 146x | +10.91% | 99% |
| IVF nlist256 nprobe1 | 210x | +10.80% | 98% |
| IVF nlist256 nprobe2 | 109x | +10.96% | 99% |
| IVF nlist256 nprobe4 | 57x | +11.01% | 100% |
| IVF nlist256 nprobe8 | 29x | +11.04% | 100% |
| IVF nlist256 nprobe16 | 15x | +11.04% | 100% |
| IVF nlist256 nprobe32 | 8x | +11.05% | 100% |
| IVF nlist1024 nprobe1 | 772x | +10.41% | 94% |
| IVF nlist1024 nprobe2 | 417x | +10.76% | 97% |
| IVF nlist1024 nprobe4 | 218x | +10.75% | 97% |
| IVF nlist1024 nprobe8 | 114x | +10.75% | 97% |
| IVF nlist1024 nprobe16 | 59x | +10.78% | 98% |
| IVF nlist1024 nprobe32 | 30x | +10.78% | 98% |
_(tiny->prid ds600K s18 done in 142s | 112 runs | 4.2h/9.5h elapsed)_

### seed 18 [prid->tiny ds600K s18]  Ndb=600,000  LM-only=3.0721  brute=+17.40% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +17.40% | 100% |
| dodeca-uniform P2 probe2 | 1x | +17.40% | 100% |
| dodeca-WHITENED P2 probe1 | 123x | +15.91% | 91% |
| dodeca-WHITENED P2 probe2 | 33x | +17.08% | 98% |
| dodeca-WHITENED P3 probe1 | 955x | +11.75% | 68% |
| dodeca-WHITENED P3 probe2 | 157x | +16.42% | 94% |
| IVF nlist256 nprobe1 | 213x | +17.35% | 100% |
| IVF nlist256 nprobe2 | 112x | +17.38% | 100% |
| IVF nlist256 nprobe4 | 57x | +17.41% | 100% |
| IVF nlist256 nprobe8 | 29x | +17.40% | 100% |
| IVF nlist256 nprobe16 | 15x | +17.40% | 100% |
| IVF nlist256 nprobe32 | 7x | +17.40% | 100% |
| IVF nlist1024 nprobe1 | 790x | +17.26% | 99% |
| IVF nlist1024 nprobe2 | 431x | +17.40% | 100% |
| IVF nlist1024 nprobe4 | 225x | +17.36% | 100% |
| IVF nlist1024 nprobe8 | 117x | +17.36% | 100% |
| IVF nlist1024 nprobe16 | 59x | +17.37% | 100% |
| IVF nlist1024 nprobe32 | 30x | +17.37% | 100% |
_(prid->tiny ds600K s18 done in 143s | 113 runs | 4.3h/9.5h elapsed)_

### seed 18 [tiny->omc_ ds600K s18]  Ndb=600,000  LM-only=5.2373  brute=+46.48% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +46.38% | 100% |
| dodeca-uniform P2 probe2 | 1x | +46.48% | 100% |
| dodeca-WHITENED P2 probe1 | 122x | +44.43% | 96% |
| dodeca-WHITENED P2 probe2 | 33x | +46.38% | 100% |
| dodeca-WHITENED P3 probe1 | 828x | +41.82% | 90% |
| dodeca-WHITENED P3 probe2 | 145x | +45.18% | 97% |
| IVF nlist256 nprobe1 | 193x | +46.35% | 100% |
| IVF nlist256 nprobe2 | 104x | +46.43% | 100% |
| IVF nlist256 nprobe4 | 55x | +46.42% | 100% |
| IVF nlist256 nprobe8 | 28x | +46.46% | 100% |
| IVF nlist256 nprobe16 | 14x | +46.47% | 100% |
| IVF nlist256 nprobe32 | 7x | +46.48% | 100% |
| IVF nlist1024 nprobe1 | 807x | +46.03% | 99% |
| IVF nlist1024 nprobe2 | 434x | +46.31% | 100% |
| IVF nlist1024 nprobe4 | 228x | +46.46% | 100% |
| IVF nlist1024 nprobe8 | 117x | +46.48% | 100% |
| IVF nlist1024 nprobe16 | 60x | +46.49% | 100% |
| IVF nlist1024 nprobe32 | 31x | +46.48% | 100% |
_(tiny->omc_ ds600K s18 done in 139s | 114 runs | 4.3h/9.5h elapsed)_

### seed 19 [tiny->prid ds300K s19]  Ndb=300,000  LM-only=2.7968  brute=+10.57% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +10.55% | 100% |
| dodeca-uniform P2 probe2 | 1x | +10.57% | 100% |
| dodeca-WHITENED P2 probe1 | 124x | +9.06% | 86% |
| dodeca-WHITENED P2 probe2 | 32x | +10.85% | 103% |
| dodeca-WHITENED P3 probe1 | 958x | +5.76% | 54% |
| dodeca-WHITENED P3 probe2 | 156x | +9.53% | 90% |
| IVF nlist256 nprobe1 | 197x | +9.96% | 94% |
| IVF nlist256 nprobe2 | 108x | +10.25% | 97% |
| IVF nlist256 nprobe4 | 58x | +10.52% | 100% |
| IVF nlist256 nprobe8 | 29x | +10.56% | 100% |
| IVF nlist256 nprobe16 | 15x | +10.57% | 100% |
| IVF nlist256 nprobe32 | 8x | +10.57% | 100% |
| IVF nlist1024 nprobe1 | 760x | +9.79% | 93% |
| IVF nlist1024 nprobe2 | 419x | +10.60% | 100% |
| IVF nlist1024 nprobe4 | 222x | +10.57% | 100% |
| IVF nlist1024 nprobe8 | 116x | +10.56% | 100% |
| IVF nlist1024 nprobe16 | 60x | +10.57% | 100% |
| IVF nlist1024 nprobe32 | 31x | +10.57% | 100% |
_(tiny->prid ds300K s19 done in 103s | 115 runs | 4.3h/9.5h elapsed)_

### seed 19 [prid->tiny ds300K s19]  Ndb=300,000  LM-only=3.0280  brute=+16.32% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 5x | +15.97% | 98% |
| dodeca-uniform P2 probe2 | 1x | +16.38% | 100% |
| dodeca-WHITENED P2 probe1 | 118x | +13.92% | 85% |
| dodeca-WHITENED P2 probe2 | 32x | +16.05% | 98% |
| dodeca-WHITENED P3 probe1 | 965x | +9.24% | 57% |
| dodeca-WHITENED P3 probe2 | 157x | +14.64% | 90% |
| IVF nlist256 nprobe1 | 209x | +16.33% | 100% |
| IVF nlist256 nprobe2 | 109x | +16.32% | 100% |
| IVF nlist256 nprobe4 | 56x | +16.31% | 100% |
| IVF nlist256 nprobe8 | 29x | +16.32% | 100% |
| IVF nlist256 nprobe16 | 15x | +16.32% | 100% |
| IVF nlist256 nprobe32 | 7x | +16.32% | 100% |
| IVF nlist1024 nprobe1 | 752x | +15.82% | 97% |
| IVF nlist1024 nprobe2 | 420x | +16.43% | 101% |
| IVF nlist1024 nprobe4 | 221x | +16.34% | 100% |
| IVF nlist1024 nprobe8 | 113x | +16.32% | 100% |
| IVF nlist1024 nprobe16 | 58x | +16.31% | 100% |
| IVF nlist1024 nprobe32 | 30x | +16.31% | 100% |
_(prid->tiny ds300K s19 done in 95s | 116 runs | 4.3h/9.5h elapsed)_

### seed 19 [tiny->omc_ ds300K s19]  Ndb=300,000  LM-only=5.4436  brute=+47.33% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 16x | +46.03% | 97% |
| dodeca-uniform P2 probe2 | 5x | +47.08% | 99% |
| dodeca-WHITENED P2 probe1 | 123x | +45.35% | 96% |
| dodeca-WHITENED P2 probe2 | 33x | +47.27% | 100% |
| dodeca-WHITENED P3 probe1 | 900x | +41.16% | 87% |
| dodeca-WHITENED P3 probe2 | 150x | +47.07% | 99% |
| IVF nlist256 nprobe1 | 211x | +47.38% | 100% |
| IVF nlist256 nprobe2 | 110x | +47.28% | 100% |
| IVF nlist256 nprobe4 | 56x | +47.30% | 100% |
| IVF nlist256 nprobe8 | 29x | +47.33% | 100% |
| IVF nlist256 nprobe16 | 15x | +47.33% | 100% |
| IVF nlist256 nprobe32 | 7x | +47.33% | 100% |
| IVF nlist1024 nprobe1 | 817x | +47.26% | 100% |
| IVF nlist1024 nprobe2 | 433x | +47.86% | 101% |
| IVF nlist1024 nprobe4 | 222x | +47.51% | 100% |
| IVF nlist1024 nprobe8 | 115x | +47.40% | 100% |
| IVF nlist1024 nprobe16 | 59x | +47.34% | 100% |
| IVF nlist1024 nprobe32 | 30x | +47.33% | 100% |
_(tiny->omc_ ds300K s19 done in 95s | 117 runs | 4.4h/9.5h elapsed)_

### seed 19 [tiny->prid ds600K s19]  Ndb=600,000  LM-only=2.7968  brute=+10.25% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +10.20% | 99% |
| dodeca-uniform P2 probe2 | 1x | +10.25% | 100% |
| dodeca-WHITENED P2 probe1 | 118x | +9.88% | 96% |
| dodeca-WHITENED P2 probe2 | 32x | +10.55% | 103% |
| dodeca-WHITENED P3 probe1 | 1106x | +6.95% | 68% |
| dodeca-WHITENED P3 probe2 | 168x | +10.12% | 99% |
| IVF nlist256 nprobe1 | 191x | +10.01% | 98% |
| IVF nlist256 nprobe2 | 102x | +10.18% | 99% |
| IVF nlist256 nprobe4 | 54x | +10.22% | 100% |
| IVF nlist256 nprobe8 | 28x | +10.24% | 100% |
| IVF nlist256 nprobe16 | 15x | +10.25% | 100% |
| IVF nlist256 nprobe32 | 8x | +10.25% | 100% |
| IVF nlist1024 nprobe1 | 761x | +9.74% | 95% |
| IVF nlist1024 nprobe2 | 415x | +10.13% | 99% |
| IVF nlist1024 nprobe4 | 220x | +10.19% | 99% |
| IVF nlist1024 nprobe8 | 115x | +10.23% | 100% |
| IVF nlist1024 nprobe16 | 59x | +10.24% | 100% |
| IVF nlist1024 nprobe32 | 30x | +10.24% | 100% |
_(tiny->prid ds600K s19 done in 142s | 118 runs | 4.4h/9.5h elapsed)_

### seed 19 [prid->tiny ds600K s19]  Ndb=600,000  LM-only=3.0280  brute=+16.20% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 5x | +16.38% | 101% |
| dodeca-uniform P2 probe2 | 1x | +16.22% | 100% |
| dodeca-WHITENED P2 probe1 | 112x | +14.60% | 90% |
| dodeca-WHITENED P2 probe2 | 31x | +15.98% | 99% |
| dodeca-WHITENED P3 probe1 | 951x | +10.26% | 63% |
| dodeca-WHITENED P3 probe2 | 152x | +15.87% | 98% |
| IVF nlist256 nprobe1 | 200x | +16.15% | 100% |
| IVF nlist256 nprobe2 | 106x | +16.19% | 100% |
| IVF nlist256 nprobe4 | 54x | +16.19% | 100% |
| IVF nlist256 nprobe8 | 28x | +16.18% | 100% |
| IVF nlist256 nprobe16 | 14x | +16.18% | 100% |
| IVF nlist256 nprobe32 | 7x | +16.18% | 100% |
| IVF nlist1024 nprobe1 | 761x | +15.98% | 99% |
| IVF nlist1024 nprobe2 | 413x | +16.04% | 99% |
| IVF nlist1024 nprobe4 | 220x | +16.17% | 100% |
| IVF nlist1024 nprobe8 | 117x | +16.18% | 100% |
| IVF nlist1024 nprobe16 | 60x | +16.18% | 100% |
| IVF nlist1024 nprobe32 | 30x | +16.18% | 100% |
_(prid->tiny ds600K s19 done in 132s | 119 runs | 4.5h/9.5h elapsed)_

### seed 19 [tiny->omc_ ds600K s19]  Ndb=600,000  LM-only=5.4436  brute=+47.72% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 16x | +47.30% | 99% |
| dodeca-uniform P2 probe2 | 5x | +47.60% | 100% |
| dodeca-WHITENED P2 probe1 | 119x | +46.00% | 96% |
| dodeca-WHITENED P2 probe2 | 31x | +47.52% | 100% |
| dodeca-WHITENED P3 probe1 | 957x | +42.95% | 90% |
| dodeca-WHITENED P3 probe2 | 157x | +47.07% | 99% |
| IVF nlist256 nprobe1 | 187x | +47.62% | 100% |
| IVF nlist256 nprobe2 | 102x | +47.62% | 100% |
| IVF nlist256 nprobe4 | 53x | +47.71% | 100% |
| IVF nlist256 nprobe8 | 28x | +47.72% | 100% |
| IVF nlist256 nprobe16 | 14x | +47.72% | 100% |
| IVF nlist256 nprobe32 | 7x | +47.72% | 100% |
| IVF nlist1024 nprobe1 | 830x | +47.86% | 100% |
| IVF nlist1024 nprobe2 | 445x | +47.78% | 100% |
| IVF nlist1024 nprobe4 | 231x | +47.71% | 100% |
| IVF nlist1024 nprobe8 | 118x | +47.72% | 100% |
| IVF nlist1024 nprobe16 | 60x | +47.72% | 100% |
| IVF nlist1024 nprobe32 | 31x | +47.72% | 100% |
_(tiny->omc_ ds600K s19 done in 119s | 120 runs | 4.5h/9.5h elapsed)_

### seed 20 [tiny->prid ds300K s20]  Ndb=300,000  LM-only=2.7612  brute=+11.12% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +10.79% | 97% |
| dodeca-uniform P2 probe2 | 1x | +11.11% | 100% |
| dodeca-WHITENED P2 probe1 | 122x | +8.22% | 74% |
| dodeca-WHITENED P2 probe2 | 33x | +9.87% | 89% |
| dodeca-WHITENED P3 probe1 | 804x | +5.96% | 54% |
| dodeca-WHITENED P3 probe2 | 137x | +9.23% | 83% |
| IVF nlist256 nprobe1 | 200x | +10.52% | 95% |
| IVF nlist256 nprobe2 | 106x | +10.92% | 98% |
| IVF nlist256 nprobe4 | 56x | +10.88% | 98% |
| IVF nlist256 nprobe8 | 29x | +10.88% | 98% |
| IVF nlist256 nprobe16 | 15x | +10.90% | 98% |
| IVF nlist256 nprobe32 | 7x | +10.90% | 98% |
| IVF nlist1024 nprobe1 | 747x | +10.01% | 90% |
| IVF nlist1024 nprobe2 | 410x | +10.83% | 97% |
| IVF nlist1024 nprobe4 | 217x | +10.87% | 98% |
| IVF nlist1024 nprobe8 | 113x | +10.88% | 98% |
| IVF nlist1024 nprobe16 | 58x | +10.89% | 98% |
| IVF nlist1024 nprobe32 | 30x | +10.90% | 98% |
_(tiny->prid ds300K s20 done in 99s | 121 runs | 4.5h/9.5h elapsed)_

### seed 20 [prid->tiny ds300K s20]  Ndb=300,000  LM-only=3.0579  brute=+16.66% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +16.66% | 100% |
| dodeca-uniform P2 probe2 | 1x | +16.66% | 100% |
| dodeca-WHITENED P2 probe1 | 123x | +13.69% | 82% |
| dodeca-WHITENED P2 probe2 | 33x | +16.26% | 98% |
| dodeca-WHITENED P3 probe1 | 935x | +9.97% | 60% |
| dodeca-WHITENED P3 probe2 | 151x | +15.98% | 96% |
| IVF nlist256 nprobe1 | 206x | +16.50% | 99% |
| IVF nlist256 nprobe2 | 107x | +16.61% | 100% |
| IVF nlist256 nprobe4 | 54x | +16.61% | 100% |
| IVF nlist256 nprobe8 | 28x | +16.60% | 100% |
| IVF nlist256 nprobe16 | 14x | +16.60% | 100% |
| IVF nlist256 nprobe32 | 7x | +16.67% | 100% |
| IVF nlist1024 nprobe1 | 800x | +16.35% | 98% |
| IVF nlist1024 nprobe2 | 434x | +16.60% | 100% |
| IVF nlist1024 nprobe4 | 227x | +16.65% | 100% |
| IVF nlist1024 nprobe8 | 116x | +16.67% | 100% |
| IVF nlist1024 nprobe16 | 59x | +16.67% | 100% |
| IVF nlist1024 nprobe32 | 30x | +16.67% | 100% |
_(prid->tiny ds300K s20 done in 105s | 122 runs | 4.5h/9.5h elapsed)_

### seed 20 [tiny->omc_ ds300K s20]  Ndb=300,000  LM-only=5.7815  brute=+50.11% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +49.56% | 99% |
| dodeca-uniform P2 probe2 | 1x | +50.12% | 100% |
| dodeca-WHITENED P2 probe1 | 125x | +47.60% | 95% |
| dodeca-WHITENED P2 probe2 | 33x | +49.88% | 100% |
| dodeca-WHITENED P3 probe1 | 832x | +43.31% | 86% |
| dodeca-WHITENED P3 probe2 | 140x | +49.13% | 98% |
| IVF nlist256 nprobe1 | 210x | +49.78% | 99% |
| IVF nlist256 nprobe2 | 108x | +49.95% | 100% |
| IVF nlist256 nprobe4 | 55x | +50.10% | 100% |
| IVF nlist256 nprobe8 | 28x | +50.10% | 100% |
| IVF nlist256 nprobe16 | 15x | +50.11% | 100% |
| IVF nlist256 nprobe32 | 7x | +50.11% | 100% |
| IVF nlist1024 nprobe1 | 838x | +49.50% | 99% |
| IVF nlist1024 nprobe2 | 440x | +50.46% | 101% |
| IVF nlist1024 nprobe4 | 226x | +50.31% | 100% |
| IVF nlist1024 nprobe8 | 116x | +50.19% | 100% |
| IVF nlist1024 nprobe16 | 59x | +50.11% | 100% |
| IVF nlist1024 nprobe32 | 30x | +50.12% | 100% |
_(tiny->omc_ ds300K s20 done in 106s | 123 runs | 4.6h/9.5h elapsed)_

### seed 20 [tiny->prid ds600K s20]  Ndb=600,000  LM-only=2.7612  brute=+10.98% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +10.95% | 100% |
| dodeca-uniform P2 probe2 | 1x | +10.97% | 100% |
| dodeca-WHITENED P2 probe1 | 125x | +9.19% | 84% |
| dodeca-WHITENED P2 probe2 | 33x | +10.73% | 98% |
| dodeca-WHITENED P3 probe1 | 883x | +7.05% | 64% |
| dodeca-WHITENED P3 probe2 | 146x | +10.65% | 97% |
| IVF nlist256 nprobe1 | 201x | +10.52% | 96% |
| IVF nlist256 nprobe2 | 104x | +10.73% | 98% |
| IVF nlist256 nprobe4 | 53x | +10.75% | 98% |
| IVF nlist256 nprobe8 | 27x | +10.77% | 98% |
| IVF nlist256 nprobe16 | 14x | +10.97% | 100% |
| IVF nlist256 nprobe32 | 7x | +10.98% | 100% |
| IVF nlist1024 nprobe1 | 755x | +10.49% | 96% |
| IVF nlist1024 nprobe2 | 414x | +10.76% | 98% |
| IVF nlist1024 nprobe4 | 216x | +10.79% | 98% |
| IVF nlist1024 nprobe8 | 113x | +10.81% | 98% |
| IVF nlist1024 nprobe16 | 59x | +11.02% | 100% |
| IVF nlist1024 nprobe32 | 30x | +11.03% | 100% |
_(tiny->prid ds600K s20 done in 142s | 124 runs | 4.6h/9.5h elapsed)_

### seed 20 [prid->tiny ds600K s20]  Ndb=600,000  LM-only=3.0579  brute=+16.89% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +16.89% | 100% |
| dodeca-uniform P2 probe2 | 1x | +16.88% | 100% |
| dodeca-WHITENED P2 probe1 | 119x | +15.41% | 91% |
| dodeca-WHITENED P2 probe2 | 32x | +16.64% | 99% |
| dodeca-WHITENED P3 probe1 | 888x | +10.98% | 65% |
| dodeca-WHITENED P3 probe2 | 144x | +15.08% | 89% |
| IVF nlist256 nprobe1 | 193x | +16.84% | 100% |
| IVF nlist256 nprobe2 | 100x | +16.86% | 100% |
| IVF nlist256 nprobe4 | 52x | +16.86% | 100% |
| IVF nlist256 nprobe8 | 27x | +16.86% | 100% |
| IVF nlist256 nprobe16 | 14x | +16.86% | 100% |
| IVF nlist256 nprobe32 | 7x | +16.86% | 100% |
| IVF nlist1024 nprobe1 | 769x | +16.76% | 99% |
| IVF nlist1024 nprobe2 | 421x | +16.90% | 100% |
| IVF nlist1024 nprobe4 | 223x | +16.90% | 100% |
| IVF nlist1024 nprobe8 | 115x | +16.89% | 100% |
| IVF nlist1024 nprobe16 | 58x | +16.89% | 100% |
| IVF nlist1024 nprobe32 | 30x | +16.89% | 100% |
_(prid->tiny ds600K s20 done in 145s | 125 runs | 4.7h/9.5h elapsed)_

### seed 20 [tiny->omc_ ds600K s20]  Ndb=600,000  LM-only=5.7815  brute=+50.91% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +50.71% | 100% |
| dodeca-uniform P2 probe2 | 1x | +50.91% | 100% |
| dodeca-WHITENED P2 probe1 | 115x | +48.68% | 96% |
| dodeca-WHITENED P2 probe2 | 31x | +50.70% | 100% |
| dodeca-WHITENED P3 probe1 | 749x | +46.72% | 92% |
| dodeca-WHITENED P3 probe2 | 128x | +49.95% | 98% |
| IVF nlist256 nprobe1 | 196x | +50.81% | 100% |
| IVF nlist256 nprobe2 | 102x | +51.03% | 100% |
| IVF nlist256 nprobe4 | 53x | +51.01% | 100% |
| IVF nlist256 nprobe8 | 28x | +50.90% | 100% |
| IVF nlist256 nprobe16 | 14x | +50.91% | 100% |
| IVF nlist256 nprobe32 | 7x | +50.91% | 100% |
| IVF nlist1024 nprobe1 | 844x | +51.00% | 100% |
| IVF nlist1024 nprobe2 | 447x | +51.15% | 100% |
| IVF nlist1024 nprobe4 | 229x | +51.03% | 100% |
| IVF nlist1024 nprobe8 | 119x | +51.02% | 100% |
| IVF nlist1024 nprobe16 | 61x | +50.92% | 100% |
| IVF nlist1024 nprobe32 | 30x | +50.91% | 100% |
_(tiny->omc_ ds600K s20 done in 135s | 126 runs | 4.7h/9.5h elapsed)_

### seed 21 [tiny->prid ds300K s21]  Ndb=300,000  LM-only=2.7694  brute=+10.18% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 4x | +9.70% | 95% |
| dodeca-uniform P2 probe2 | 2x | +10.28% | 101% |
| dodeca-WHITENED P2 probe1 | 122x | +8.43% | 83% |
| dodeca-WHITENED P2 probe2 | 32x | +10.35% | 102% |
| dodeca-WHITENED P3 probe1 | 907x | +5.96% | 59% |
| dodeca-WHITENED P3 probe2 | 151x | +9.85% | 97% |
| IVF nlist256 nprobe1 | 194x | +10.18% | 100% |
| IVF nlist256 nprobe2 | 103x | +10.05% | 99% |
| IVF nlist256 nprobe4 | 54x | +10.09% | 99% |
| IVF nlist256 nprobe8 | 28x | +10.14% | 100% |
| IVF nlist256 nprobe16 | 15x | +10.17% | 100% |
| IVF nlist256 nprobe32 | 8x | +10.18% | 100% |
| IVF nlist1024 nprobe1 | 753x | +10.00% | 98% |
| IVF nlist1024 nprobe2 | 416x | +10.16% | 100% |
| IVF nlist1024 nprobe4 | 219x | +10.15% | 100% |
| IVF nlist1024 nprobe8 | 113x | +10.17% | 100% |
| IVF nlist1024 nprobe16 | 58x | +10.18% | 100% |
| IVF nlist1024 nprobe32 | 30x | +10.18% | 100% |
_(tiny->prid ds300K s21 done in 94s | 127 runs | 4.7h/9.5h elapsed)_

### seed 21 [prid->tiny ds300K s21]  Ndb=300,000  LM-only=3.0773  brute=+17.32% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +16.90% | 98% |
| dodeca-uniform P2 probe2 | 1x | +17.33% | 100% |
| dodeca-WHITENED P2 probe1 | 116x | +14.01% | 81% |
| dodeca-WHITENED P2 probe2 | 32x | +16.20% | 94% |
| dodeca-WHITENED P3 probe1 | 880x | +10.57% | 61% |
| dodeca-WHITENED P3 probe2 | 144x | +14.42% | 83% |
| IVF nlist256 nprobe1 | 201x | +17.14% | 99% |
| IVF nlist256 nprobe2 | 106x | +17.25% | 100% |
| IVF nlist256 nprobe4 | 54x | +17.31% | 100% |
| IVF nlist256 nprobe8 | 28x | +17.30% | 100% |
| IVF nlist256 nprobe16 | 14x | +17.31% | 100% |
| IVF nlist256 nprobe32 | 7x | +17.31% | 100% |
| IVF nlist1024 nprobe1 | 781x | +16.92% | 98% |
| IVF nlist1024 nprobe2 | 425x | +17.28% | 100% |
| IVF nlist1024 nprobe4 | 226x | +17.22% | 99% |
| IVF nlist1024 nprobe8 | 116x | +17.32% | 100% |
| IVF nlist1024 nprobe16 | 59x | +17.32% | 100% |
| IVF nlist1024 nprobe32 | 30x | +17.31% | 100% |
_(prid->tiny ds300K s21 done in 100s | 128 runs | 4.7h/9.5h elapsed)_

### seed 21 [tiny->omc_ ds300K s21]  Ndb=300,000  LM-only=5.4087  brute=+46.15% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +46.14% | 100% |
| dodeca-uniform P2 probe2 | 1x | +46.15% | 100% |
| dodeca-WHITENED P2 probe1 | 120x | +43.83% | 95% |
| dodeca-WHITENED P2 probe2 | 32x | +45.92% | 100% |
| dodeca-WHITENED P3 probe1 | 955x | +39.63% | 86% |
| dodeca-WHITENED P3 probe2 | 148x | +44.69% | 97% |
| IVF nlist256 nprobe1 | 203x | +45.77% | 99% |
| IVF nlist256 nprobe2 | 106x | +46.10% | 100% |
| IVF nlist256 nprobe4 | 55x | +46.14% | 100% |
| IVF nlist256 nprobe8 | 28x | +46.15% | 100% |
| IVF nlist256 nprobe16 | 14x | +46.15% | 100% |
| IVF nlist256 nprobe32 | 7x | +46.16% | 100% |
| IVF nlist1024 nprobe1 | 803x | +45.89% | 99% |
| IVF nlist1024 nprobe2 | 426x | +45.96% | 100% |
| IVF nlist1024 nprobe4 | 222x | +45.82% | 99% |
| IVF nlist1024 nprobe8 | 116x | +46.16% | 100% |
| IVF nlist1024 nprobe16 | 59x | +46.15% | 100% |
| IVF nlist1024 nprobe32 | 30x | +46.15% | 100% |
_(tiny->omc_ ds300K s21 done in 106s | 129 runs | 4.8h/9.5h elapsed)_

### seed 21 [tiny->prid ds600K s21]  Ndb=600,000  LM-only=2.7694  brute=+10.70% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 4x | +10.57% | 99% |
| dodeca-uniform P2 probe2 | 2x | +10.68% | 100% |
| dodeca-WHITENED P2 probe1 | 122x | +9.10% | 85% |
| dodeca-WHITENED P2 probe2 | 33x | +10.21% | 95% |
| dodeca-WHITENED P3 probe1 | 866x | +7.38% | 69% |
| dodeca-WHITENED P3 probe2 | 146x | +10.78% | 101% |
| IVF nlist256 nprobe1 | 210x | +10.53% | 98% |
| IVF nlist256 nprobe2 | 110x | +10.68% | 100% |
| IVF nlist256 nprobe4 | 56x | +10.69% | 100% |
| IVF nlist256 nprobe8 | 29x | +10.70% | 100% |
| IVF nlist256 nprobe16 | 15x | +10.70% | 100% |
| IVF nlist256 nprobe32 | 8x | +10.70% | 100% |
| IVF nlist1024 nprobe1 | 789x | +10.25% | 96% |
| IVF nlist1024 nprobe2 | 430x | +10.65% | 100% |
| IVF nlist1024 nprobe4 | 225x | +10.66% | 100% |
| IVF nlist1024 nprobe8 | 116x | +10.67% | 100% |
| IVF nlist1024 nprobe16 | 59x | +10.69% | 100% |
| IVF nlist1024 nprobe32 | 30x | +10.69% | 100% |
_(tiny->prid ds600K s21 done in 124s | 130 runs | 4.8h/9.5h elapsed)_

### seed 21 [prid->tiny ds600K s21]  Ndb=600,000  LM-only=3.0773  brute=+17.31% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +17.17% | 99% |
| dodeca-uniform P2 probe2 | 1x | +17.31% | 100% |
| dodeca-WHITENED P2 probe1 | 119x | +15.29% | 88% |
| dodeca-WHITENED P2 probe2 | 32x | +16.59% | 96% |
| dodeca-WHITENED P3 probe1 | 912x | +11.72% | 68% |
| dodeca-WHITENED P3 probe2 | 149x | +16.83% | 97% |
| IVF nlist256 nprobe1 | 208x | +17.31% | 100% |
| IVF nlist256 nprobe2 | 108x | +17.30% | 100% |
| IVF nlist256 nprobe4 | 56x | +17.29% | 100% |
| IVF nlist256 nprobe8 | 29x | +17.30% | 100% |
| IVF nlist256 nprobe16 | 15x | +17.30% | 100% |
| IVF nlist256 nprobe32 | 8x | +17.30% | 100% |
| IVF nlist1024 nprobe1 | 772x | +17.14% | 99% |
| IVF nlist1024 nprobe2 | 417x | +17.28% | 100% |
| IVF nlist1024 nprobe4 | 220x | +17.29% | 100% |
| IVF nlist1024 nprobe8 | 115x | +17.29% | 100% |
| IVF nlist1024 nprobe16 | 60x | +17.30% | 100% |
| IVF nlist1024 nprobe32 | 31x | +17.30% | 100% |
_(prid->tiny ds600K s21 done in 139s | 131 runs | 4.8h/9.5h elapsed)_

### seed 21 [tiny->omc_ ds600K s21]  Ndb=600,000  LM-only=5.4087  brute=+46.29% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +46.29% | 100% |
| dodeca-uniform P2 probe2 | 1x | +46.29% | 100% |
| dodeca-WHITENED P2 probe1 | 120x | +45.34% | 98% |
| dodeca-WHITENED P2 probe2 | 32x | +46.35% | 100% |
| dodeca-WHITENED P3 probe1 | 913x | +41.66% | 90% |
| dodeca-WHITENED P3 probe2 | 146x | +44.69% | 97% |
| IVF nlist256 nprobe1 | 202x | +46.10% | 100% |
| IVF nlist256 nprobe2 | 107x | +46.26% | 100% |
| IVF nlist256 nprobe4 | 56x | +46.29% | 100% |
| IVF nlist256 nprobe8 | 29x | +46.29% | 100% |
| IVF nlist256 nprobe16 | 15x | +46.29% | 100% |
| IVF nlist256 nprobe32 | 8x | +46.29% | 100% |
| IVF nlist1024 nprobe1 | 779x | +45.91% | 99% |
| IVF nlist1024 nprobe2 | 417x | +46.15% | 100% |
| IVF nlist1024 nprobe4 | 218x | +46.23% | 100% |
| IVF nlist1024 nprobe8 | 114x | +46.30% | 100% |
| IVF nlist1024 nprobe16 | 59x | +46.30% | 100% |
| IVF nlist1024 nprobe32 | 30x | +46.30% | 100% |
_(tiny->omc_ ds600K s21 done in 144s | 132 runs | 4.9h/9.5h elapsed)_

### seed 22 [tiny->prid ds300K s22]  Ndb=300,000  LM-only=2.7807  brute=+10.82% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 4x | +10.33% | 95% |
| dodeca-uniform P2 probe2 | 1x | +10.79% | 100% |
| dodeca-WHITENED P2 probe1 | 112x | +8.09% | 75% |
| dodeca-WHITENED P2 probe2 | 30x | +10.21% | 94% |
| dodeca-WHITENED P3 probe1 | 880x | +5.38% | 50% |
| dodeca-WHITENED P3 probe2 | 148x | +9.07% | 84% |
| IVF nlist256 nprobe1 | 206x | +10.26% | 95% |
| IVF nlist256 nprobe2 | 109x | +10.60% | 98% |
| IVF nlist256 nprobe4 | 57x | +10.81% | 100% |
| IVF nlist256 nprobe8 | 30x | +10.81% | 100% |
| IVF nlist256 nprobe16 | 15x | +10.82% | 100% |
| IVF nlist256 nprobe32 | 8x | +10.82% | 100% |
| IVF nlist1024 nprobe1 | 781x | +10.29% | 95% |
| IVF nlist1024 nprobe2 | 423x | +10.62% | 98% |
| IVF nlist1024 nprobe4 | 225x | +10.63% | 98% |
| IVF nlist1024 nprobe8 | 116x | +10.62% | 98% |
| IVF nlist1024 nprobe16 | 60x | +10.81% | 100% |
| IVF nlist1024 nprobe32 | 30x | +10.82% | 100% |
_(tiny->prid ds300K s22 done in 95s | 133 runs | 4.9h/9.5h elapsed)_

### seed 22 [prid->tiny ds300K s22]  Ndb=300,000  LM-only=2.9629  brute=+13.82% (λ=0.75)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +13.48% | 98% |
| dodeca-uniform P2 probe2 | 1x | +13.82% | 100% |
| dodeca-WHITENED P2 probe1 | 114x | +11.28% | 82% |
| dodeca-WHITENED P2 probe2 | 31x | +13.69% | 99% |
| dodeca-WHITENED P3 probe1 | 829x | +7.68% | 56% |
| dodeca-WHITENED P3 probe2 | 146x | +13.22% | 96% |
| IVF nlist256 nprobe1 | 202x | +13.86% | 100% |
| IVF nlist256 nprobe2 | 107x | +13.79% | 100% |
| IVF nlist256 nprobe4 | 55x | +13.81% | 100% |
| IVF nlist256 nprobe8 | 28x | +13.81% | 100% |
| IVF nlist256 nprobe16 | 14x | +13.82% | 100% |
| IVF nlist256 nprobe32 | 7x | +13.82% | 100% |
| IVF nlist1024 nprobe1 | 781x | +13.57% | 98% |
| IVF nlist1024 nprobe2 | 424x | +13.73% | 99% |
| IVF nlist1024 nprobe4 | 224x | +13.79% | 100% |
| IVF nlist1024 nprobe8 | 117x | +13.83% | 100% |
| IVF nlist1024 nprobe16 | 60x | +13.83% | 100% |
| IVF nlist1024 nprobe32 | 30x | +13.83% | 100% |
_(prid->tiny ds300K s22 done in 100s | 134 runs | 4.9h/9.5h elapsed)_

### seed 22 [tiny->omc_ ds300K s22]  Ndb=300,000  LM-only=5.6525  brute=+48.42% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +48.31% | 100% |
| dodeca-uniform P2 probe2 | 1x | +48.41% | 100% |
| dodeca-WHITENED P2 probe1 | 118x | +45.85% | 95% |
| dodeca-WHITENED P2 probe2 | 31x | +48.89% | 101% |
| dodeca-WHITENED P3 probe1 | 947x | +40.97% | 85% |
| dodeca-WHITENED P3 probe2 | 154x | +47.70% | 99% |
| IVF nlist256 nprobe1 | 193x | +48.56% | 100% |
| IVF nlist256 nprobe2 | 106x | +48.39% | 100% |
| IVF nlist256 nprobe4 | 56x | +48.41% | 100% |
| IVF nlist256 nprobe8 | 29x | +48.42% | 100% |
| IVF nlist256 nprobe16 | 15x | +48.42% | 100% |
| IVF nlist256 nprobe32 | 8x | +48.42% | 100% |
| IVF nlist1024 nprobe1 | 838x | +47.78% | 99% |
| IVF nlist1024 nprobe2 | 448x | +48.30% | 100% |
| IVF nlist1024 nprobe4 | 232x | +48.27% | 100% |
| IVF nlist1024 nprobe8 | 118x | +48.32% | 100% |
| IVF nlist1024 nprobe16 | 60x | +48.41% | 100% |
| IVF nlist1024 nprobe32 | 31x | +48.42% | 100% |
_(tiny->omc_ ds300K s22 done in 100s | 135 runs | 5.0h/9.5h elapsed)_

### seed 22 [tiny->prid ds600K s22]  Ndb=600,000  LM-only=2.7807  brute=+11.02% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 4x | +11.00% | 100% |
| dodeca-uniform P2 probe2 | 1x | +10.99% | 100% |
| dodeca-WHITENED P2 probe1 | 114x | +9.95% | 90% |
| dodeca-WHITENED P2 probe2 | 30x | +10.82% | 98% |
| dodeca-WHITENED P3 probe1 | 922x | +7.16% | 65% |
| dodeca-WHITENED P3 probe2 | 152x | +10.68% | 97% |
| IVF nlist256 nprobe1 | 202x | +10.87% | 99% |
| IVF nlist256 nprobe2 | 108x | +10.96% | 99% |
| IVF nlist256 nprobe4 | 56x | +10.99% | 100% |
| IVF nlist256 nprobe8 | 29x | +11.00% | 100% |
| IVF nlist256 nprobe16 | 15x | +11.00% | 100% |
| IVF nlist256 nprobe32 | 8x | +11.01% | 100% |
| IVF nlist1024 nprobe1 | 783x | +10.06% | 91% |
| IVF nlist1024 nprobe2 | 426x | +10.54% | 96% |
| IVF nlist1024 nprobe4 | 226x | +10.96% | 99% |
| IVF nlist1024 nprobe8 | 117x | +10.97% | 100% |
| IVF nlist1024 nprobe16 | 59x | +10.97% | 100% |
| IVF nlist1024 nprobe32 | 30x | +10.98% | 100% |
_(tiny->prid ds600K s22 done in 132s | 136 runs | 5.0h/9.5h elapsed)_

### seed 22 [prid->tiny ds600K s22]  Ndb=600,000  LM-only=2.9629  brute=+13.92% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +13.80% | 99% |
| dodeca-uniform P2 probe2 | 1x | +13.92% | 100% |
| dodeca-WHITENED P2 probe1 | 113x | +13.42% | 96% |
| dodeca-WHITENED P2 probe2 | 30x | +14.34% | 103% |
| dodeca-WHITENED P3 probe1 | 839x | +9.44% | 68% |
| dodeca-WHITENED P3 probe2 | 145x | +13.48% | 97% |
| IVF nlist256 nprobe1 | 190x | +13.99% | 101% |
| IVF nlist256 nprobe2 | 101x | +13.92% | 100% |
| IVF nlist256 nprobe4 | 53x | +13.92% | 100% |
| IVF nlist256 nprobe8 | 28x | +13.93% | 100% |
| IVF nlist256 nprobe16 | 14x | +13.93% | 100% |
| IVF nlist256 nprobe32 | 7x | +13.93% | 100% |
| IVF nlist1024 nprobe1 | 784x | +13.99% | 100% |
| IVF nlist1024 nprobe2 | 428x | +13.95% | 100% |
| IVF nlist1024 nprobe4 | 223x | +13.94% | 100% |
| IVF nlist1024 nprobe8 | 115x | +13.93% | 100% |
| IVF nlist1024 nprobe16 | 59x | +13.93% | 100% |
| IVF nlist1024 nprobe32 | 30x | +13.93% | 100% |
_(prid->tiny ds600K s22 done in 141s | 137 runs | 5.0h/9.5h elapsed)_

### seed 22 [tiny->omc_ ds600K s22]  Ndb=600,000  LM-only=5.6525  brute=+48.07% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +48.05% | 100% |
| dodeca-uniform P2 probe2 | 1x | +48.14% | 100% |
| dodeca-WHITENED P2 probe1 | 117x | +48.25% | 100% |
| dodeca-WHITENED P2 probe2 | 32x | +48.69% | 101% |
| dodeca-WHITENED P3 probe1 | 912x | +45.38% | 94% |
| dodeca-WHITENED P3 probe2 | 149x | +48.72% | 101% |
| IVF nlist256 nprobe1 | 199x | +48.14% | 100% |
| IVF nlist256 nprobe2 | 106x | +48.09% | 100% |
| IVF nlist256 nprobe4 | 55x | +48.07% | 100% |
| IVF nlist256 nprobe8 | 28x | +48.07% | 100% |
| IVF nlist256 nprobe16 | 14x | +48.08% | 100% |
| IVF nlist256 nprobe32 | 7x | +48.08% | 100% |
| IVF nlist1024 nprobe1 | 821x | +48.17% | 100% |
| IVF nlist1024 nprobe2 | 440x | +48.33% | 101% |
| IVF nlist1024 nprobe4 | 226x | +48.08% | 100% |
| IVF nlist1024 nprobe8 | 116x | +48.05% | 100% |
| IVF nlist1024 nprobe16 | 59x | +48.06% | 100% |
| IVF nlist1024 nprobe32 | 30x | +48.07% | 100% |
_(tiny->omc_ ds600K s22 done in 138s | 138 runs | 5.1h/9.5h elapsed)_

### seed 23 [tiny->prid ds300K s23]  Ndb=300,000  LM-only=2.7217  brute=+10.28% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +10.11% | 98% |
| dodeca-uniform P2 probe2 | 1x | +10.26% | 100% |
| dodeca-WHITENED P2 probe1 | 125x | +7.86% | 76% |
| dodeca-WHITENED P2 probe2 | 33x | +10.23% | 100% |
| dodeca-WHITENED P3 probe1 | 985x | +5.81% | 57% |
| dodeca-WHITENED P3 probe2 | 161x | +9.49% | 92% |
| IVF nlist256 nprobe1 | 185x | +9.95% | 97% |
| IVF nlist256 nprobe2 | 101x | +10.14% | 99% |
| IVF nlist256 nprobe4 | 54x | +10.23% | 100% |
| IVF nlist256 nprobe8 | 29x | +10.26% | 100% |
| IVF nlist256 nprobe16 | 15x | +10.29% | 100% |
| IVF nlist256 nprobe32 | 8x | +10.29% | 100% |
| IVF nlist1024 nprobe1 | 807x | +9.60% | 93% |
| IVF nlist1024 nprobe2 | 439x | +10.04% | 98% |
| IVF nlist1024 nprobe4 | 228x | +10.09% | 98% |
| IVF nlist1024 nprobe8 | 118x | +10.29% | 100% |
| IVF nlist1024 nprobe16 | 60x | +10.29% | 100% |
| IVF nlist1024 nprobe32 | 31x | +10.29% | 100% |
_(tiny->prid ds300K s23 done in 111s | 139 runs | 5.1h/9.5h elapsed)_

### seed 23 [prid->tiny ds300K s23]  Ndb=300,000  LM-only=3.0743  brute=+16.93% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +17.08% | 101% |
| dodeca-uniform P2 probe2 | 1x | +16.92% | 100% |
| dodeca-WHITENED P2 probe1 | 120x | +14.69% | 87% |
| dodeca-WHITENED P2 probe2 | 32x | +16.78% | 99% |
| dodeca-WHITENED P3 probe1 | 910x | +10.32% | 61% |
| dodeca-WHITENED P3 probe2 | 152x | +15.72% | 93% |
| IVF nlist256 nprobe1 | 204x | +16.73% | 99% |
| IVF nlist256 nprobe2 | 108x | +16.89% | 100% |
| IVF nlist256 nprobe4 | 55x | +16.92% | 100% |
| IVF nlist256 nprobe8 | 29x | +16.92% | 100% |
| IVF nlist256 nprobe16 | 15x | +16.92% | 100% |
| IVF nlist256 nprobe32 | 7x | +16.92% | 100% |
| IVF nlist1024 nprobe1 | 782x | +16.38% | 97% |
| IVF nlist1024 nprobe2 | 431x | +16.87% | 100% |
| IVF nlist1024 nprobe4 | 228x | +16.92% | 100% |
| IVF nlist1024 nprobe8 | 117x | +16.92% | 100% |
| IVF nlist1024 nprobe16 | 60x | +16.92% | 100% |
| IVF nlist1024 nprobe32 | 30x | +16.92% | 100% |
_(prid->tiny ds300K s23 done in 102s | 140 runs | 5.1h/9.5h elapsed)_

### seed 23 [tiny->omc_ ds300K s23]  Ndb=300,000  LM-only=5.6018  brute=+47.50% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +47.51% | 100% |
| dodeca-uniform P2 probe2 | 1x | +47.50% | 100% |
| dodeca-WHITENED P2 probe1 | 126x | +45.99% | 97% |
| dodeca-WHITENED P2 probe2 | 33x | +47.61% | 100% |
| dodeca-WHITENED P3 probe1 | 906x | +40.54% | 85% |
| dodeca-WHITENED P3 probe2 | 151x | +46.35% | 98% |
| IVF nlist256 nprobe1 | 197x | +47.19% | 99% |
| IVF nlist256 nprobe2 | 103x | +47.23% | 99% |
| IVF nlist256 nprobe4 | 54x | +47.37% | 100% |
| IVF nlist256 nprobe8 | 28x | +47.35% | 100% |
| IVF nlist256 nprobe16 | 15x | +47.50% | 100% |
| IVF nlist256 nprobe32 | 7x | +47.50% | 100% |
| IVF nlist1024 nprobe1 | 797x | +46.70% | 98% |
| IVF nlist1024 nprobe2 | 425x | +47.19% | 99% |
| IVF nlist1024 nprobe4 | 222x | +47.22% | 99% |
| IVF nlist1024 nprobe8 | 116x | +47.38% | 100% |
| IVF nlist1024 nprobe16 | 59x | +47.47% | 100% |
| IVF nlist1024 nprobe32 | 31x | +47.49% | 100% |
_(tiny->omc_ ds300K s23 done in 105s | 141 runs | 5.2h/9.5h elapsed)_

### seed 23 [tiny->prid ds600K s23]  Ndb=600,000  LM-only=2.7217  brute=+10.82% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +10.72% | 99% |
| dodeca-uniform P2 probe2 | 1x | +10.82% | 100% |
| dodeca-WHITENED P2 probe1 | 124x | +9.01% | 83% |
| dodeca-WHITENED P2 probe2 | 33x | +10.80% | 100% |
| dodeca-WHITENED P3 probe1 | 975x | +6.81% | 63% |
| dodeca-WHITENED P3 probe2 | 157x | +10.26% | 95% |
| IVF nlist256 nprobe1 | 203x | +10.47% | 97% |
| IVF nlist256 nprobe2 | 108x | +10.69% | 99% |
| IVF nlist256 nprobe4 | 57x | +10.72% | 99% |
| IVF nlist256 nprobe8 | 29x | +10.76% | 99% |
| IVF nlist256 nprobe16 | 15x | +10.80% | 100% |
| IVF nlist256 nprobe32 | 8x | +10.81% | 100% |
| IVF nlist1024 nprobe1 | 784x | +10.16% | 94% |
| IVF nlist1024 nprobe2 | 426x | +10.34% | 96% |
| IVF nlist1024 nprobe4 | 224x | +10.45% | 97% |
| IVF nlist1024 nprobe8 | 116x | +10.78% | 100% |
| IVF nlist1024 nprobe16 | 59x | +10.80% | 100% |
| IVF nlist1024 nprobe32 | 30x | +10.81% | 100% |
_(tiny->prid ds600K s23 done in 139s | 142 runs | 5.2h/9.5h elapsed)_

### seed 23 [prid->tiny ds600K s23]  Ndb=600,000  LM-only=3.0743  brute=+17.15% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +16.95% | 99% |
| dodeca-uniform P2 probe2 | 1x | +17.13% | 100% |
| dodeca-WHITENED P2 probe1 | 119x | +15.45% | 90% |
| dodeca-WHITENED P2 probe2 | 32x | +16.99% | 99% |
| dodeca-WHITENED P3 probe1 | 927x | +13.27% | 77% |
| dodeca-WHITENED P3 probe2 | 149x | +16.95% | 99% |
| IVF nlist256 nprobe1 | 191x | +17.12% | 100% |
| IVF nlist256 nprobe2 | 101x | +17.16% | 100% |
| IVF nlist256 nprobe4 | 52x | +17.15% | 100% |
| IVF nlist256 nprobe8 | 28x | +17.14% | 100% |
| IVF nlist256 nprobe16 | 14x | +17.14% | 100% |
| IVF nlist256 nprobe32 | 7x | +17.14% | 100% |
| IVF nlist1024 nprobe1 | 779x | +16.91% | 99% |
| IVF nlist1024 nprobe2 | 424x | +16.95% | 99% |
| IVF nlist1024 nprobe4 | 222x | +17.15% | 100% |
| IVF nlist1024 nprobe8 | 116x | +17.15% | 100% |
| IVF nlist1024 nprobe16 | 59x | +17.15% | 100% |
| IVF nlist1024 nprobe32 | 30x | +17.15% | 100% |
_(prid->tiny ds600K s23 done in 136s | 143 runs | 5.2h/9.5h elapsed)_

### seed 23 [tiny->omc_ ds600K s23]  Ndb=600,000  LM-only=5.6018  brute=+48.59% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +48.59% | 100% |
| dodeca-uniform P2 probe2 | 1x | +48.59% | 100% |
| dodeca-WHITENED P2 probe1 | 121x | +47.45% | 98% |
| dodeca-WHITENED P2 probe2 | 32x | +47.76% | 98% |
| dodeca-WHITENED P3 probe1 | 906x | +44.15% | 91% |
| dodeca-WHITENED P3 probe2 | 152x | +47.97% | 99% |
| IVF nlist256 nprobe1 | 202x | +48.32% | 99% |
| IVF nlist256 nprobe2 | 105x | +48.48% | 100% |
| IVF nlist256 nprobe4 | 55x | +48.52% | 100% |
| IVF nlist256 nprobe8 | 29x | +48.55% | 100% |
| IVF nlist256 nprobe16 | 15x | +48.58% | 100% |
| IVF nlist256 nprobe32 | 7x | +48.58% | 100% |
| IVF nlist1024 nprobe1 | 817x | +48.41% | 100% |
| IVF nlist1024 nprobe2 | 429x | +48.57% | 100% |
| IVF nlist1024 nprobe4 | 222x | +48.55% | 100% |
| IVF nlist1024 nprobe8 | 115x | +48.55% | 100% |
| IVF nlist1024 nprobe16 | 58x | +48.56% | 100% |
| IVF nlist1024 nprobe32 | 30x | +48.58% | 100% |
_(tiny->omc_ ds600K s23 done in 140s | 144 runs | 5.3h/9.5h elapsed)_

### seed 24 [tiny->prid ds300K s24]  Ndb=300,000  LM-only=2.7289  brute=+9.27% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 3x | +8.66% | 93% |
| dodeca-uniform P2 probe2 | 1x | +9.32% | 100% |
| dodeca-WHITENED P2 probe1 | 122x | +6.82% | 74% |
| dodeca-WHITENED P2 probe2 | 32x | +8.99% | 97% |
| dodeca-WHITENED P3 probe1 | 929x | +4.28% | 46% |
| dodeca-WHITENED P3 probe2 | 152x | +8.38% | 90% |
| IVF nlist256 nprobe1 | 205x | +8.92% | 96% |
| IVF nlist256 nprobe2 | 108x | +8.86% | 96% |
| IVF nlist256 nprobe4 | 57x | +8.91% | 96% |
| IVF nlist256 nprobe8 | 29x | +9.26% | 100% |
| IVF nlist256 nprobe16 | 15x | +9.28% | 100% |
| IVF nlist256 nprobe32 | 8x | +9.28% | 100% |
| IVF nlist1024 nprobe1 | 799x | +8.56% | 92% |
| IVF nlist1024 nprobe2 | 434x | +8.82% | 95% |
| IVF nlist1024 nprobe4 | 226x | +8.90% | 96% |
| IVF nlist1024 nprobe8 | 117x | +9.09% | 98% |
| IVF nlist1024 nprobe16 | 60x | +9.10% | 98% |
| IVF nlist1024 nprobe32 | 31x | +9.26% | 100% |
_(tiny->prid ds300K s24 done in 96s | 145 runs | 5.3h/9.5h elapsed)_

### seed 24 [prid->tiny ds300K s24]  Ndb=300,000  LM-only=3.1395  brute=+18.91% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +18.46% | 98% |
| dodeca-uniform P2 probe2 | 1x | +18.92% | 100% |
| dodeca-WHITENED P2 probe1 | 117x | +15.00% | 79% |
| dodeca-WHITENED P2 probe2 | 32x | +18.02% | 95% |
| dodeca-WHITENED P3 probe1 | 817x | +12.09% | 64% |
| dodeca-WHITENED P3 probe2 | 137x | +17.90% | 95% |
| IVF nlist256 nprobe1 | 204x | +18.62% | 98% |
| IVF nlist256 nprobe2 | 108x | +18.93% | 100% |
| IVF nlist256 nprobe4 | 57x | +18.92% | 100% |
| IVF nlist256 nprobe8 | 29x | +18.91% | 100% |
| IVF nlist256 nprobe16 | 15x | +18.91% | 100% |
| IVF nlist256 nprobe32 | 7x | +18.91% | 100% |
| IVF nlist1024 nprobe1 | 796x | +18.54% | 98% |
| IVF nlist1024 nprobe2 | 434x | +19.24% | 102% |
| IVF nlist1024 nprobe4 | 229x | +18.96% | 100% |
| IVF nlist1024 nprobe8 | 118x | +18.91% | 100% |
| IVF nlist1024 nprobe16 | 60x | +18.91% | 100% |
| IVF nlist1024 nprobe32 | 30x | +18.91% | 100% |
_(prid->tiny ds300K s24 done in 99s | 146 runs | 5.3h/9.5h elapsed)_

### seed 24 [tiny->omc_ ds300K s24]  Ndb=300,000  LM-only=5.5421  brute=+47.15% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +47.03% | 100% |
| dodeca-uniform P2 probe2 | 1x | +47.03% | 100% |
| dodeca-WHITENED P2 probe1 | 116x | +44.70% | 95% |
| dodeca-WHITENED P2 probe2 | 32x | +46.35% | 98% |
| dodeca-WHITENED P3 probe1 | 942x | +40.09% | 85% |
| dodeca-WHITENED P3 probe2 | 151x | +46.30% | 98% |
| IVF nlist256 nprobe1 | 190x | +46.74% | 99% |
| IVF nlist256 nprobe2 | 102x | +46.96% | 100% |
| IVF nlist256 nprobe4 | 53x | +47.12% | 100% |
| IVF nlist256 nprobe8 | 27x | +47.13% | 100% |
| IVF nlist256 nprobe16 | 14x | +47.14% | 100% |
| IVF nlist256 nprobe32 | 7x | +47.14% | 100% |
| IVF nlist1024 nprobe1 | 799x | +47.09% | 100% |
| IVF nlist1024 nprobe2 | 429x | +47.22% | 100% |
| IVF nlist1024 nprobe4 | 227x | +47.00% | 100% |
| IVF nlist1024 nprobe8 | 118x | +47.15% | 100% |
| IVF nlist1024 nprobe16 | 61x | +47.14% | 100% |
| IVF nlist1024 nprobe32 | 31x | +47.14% | 100% |
_(tiny->omc_ ds300K s24 done in 105s | 147 runs | 5.4h/9.5h elapsed)_

### seed 24 [tiny->prid ds600K s24]  Ndb=600,000  LM-only=2.7289  brute=+9.79% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 3x | +9.33% | 95% |
| dodeca-uniform P2 probe2 | 1x | +9.80% | 100% |
| dodeca-WHITENED P2 probe1 | 121x | +8.32% | 85% |
| dodeca-WHITENED P2 probe2 | 32x | +9.63% | 98% |
| dodeca-WHITENED P3 probe1 | 958x | +4.92% | 50% |
| dodeca-WHITENED P3 probe2 | 154x | +9.44% | 96% |
| IVF nlist256 nprobe1 | 201x | +9.44% | 96% |
| IVF nlist256 nprobe2 | 105x | +9.67% | 99% |
| IVF nlist256 nprobe4 | 56x | +9.73% | 99% |
| IVF nlist256 nprobe8 | 29x | +9.76% | 100% |
| IVF nlist256 nprobe16 | 15x | +9.78% | 100% |
| IVF nlist256 nprobe32 | 8x | +9.78% | 100% |
| IVF nlist1024 nprobe1 | 764x | +9.22% | 94% |
| IVF nlist1024 nprobe2 | 419x | +9.58% | 98% |
| IVF nlist1024 nprobe4 | 218x | +9.56% | 98% |
| IVF nlist1024 nprobe8 | 112x | +9.59% | 98% |
| IVF nlist1024 nprobe16 | 58x | +9.60% | 98% |
| IVF nlist1024 nprobe32 | 30x | +9.61% | 98% |
_(tiny->prid ds600K s24 done in 133s | 148 runs | 5.4h/9.5h elapsed)_

### seed 24 [prid->tiny ds600K s24]  Ndb=600,000  LM-only=3.1395  brute=+18.92% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +18.79% | 99% |
| dodeca-uniform P2 probe2 | 1x | +18.92% | 100% |
| dodeca-WHITENED P2 probe1 | 127x | +17.38% | 92% |
| dodeca-WHITENED P2 probe2 | 33x | +18.32% | 97% |
| dodeca-WHITENED P3 probe1 | 873x | +14.16% | 75% |
| dodeca-WHITENED P3 probe2 | 145x | +18.51% | 98% |
| IVF nlist256 nprobe1 | 193x | +18.71% | 99% |
| IVF nlist256 nprobe2 | 105x | +18.88% | 100% |
| IVF nlist256 nprobe4 | 54x | +18.91% | 100% |
| IVF nlist256 nprobe8 | 28x | +18.93% | 100% |
| IVF nlist256 nprobe16 | 14x | +18.92% | 100% |
| IVF nlist256 nprobe32 | 7x | +18.92% | 100% |
| IVF nlist1024 nprobe1 | 790x | +18.80% | 99% |
| IVF nlist1024 nprobe2 | 428x | +18.91% | 100% |
| IVF nlist1024 nprobe4 | 225x | +18.97% | 100% |
| IVF nlist1024 nprobe8 | 116x | +18.96% | 100% |
| IVF nlist1024 nprobe16 | 60x | +18.96% | 100% |
| IVF nlist1024 nprobe32 | 30x | +18.96% | 100% |
_(prid->tiny ds600K s24 done in 139s | 149 runs | 5.4h/9.5h elapsed)_

### seed 24 [tiny->omc_ ds600K s24]  Ndb=600,000  LM-only=5.5421  brute=+46.53% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +46.39% | 100% |
| dodeca-uniform P2 probe2 | 1x | +46.39% | 100% |
| dodeca-WHITENED P2 probe1 | 119x | +45.07% | 97% |
| dodeca-WHITENED P2 probe2 | 32x | +46.00% | 99% |
| dodeca-WHITENED P3 probe1 | 893x | +43.07% | 93% |
| dodeca-WHITENED P3 probe2 | 149x | +47.39% | 102% |
| IVF nlist256 nprobe1 | 179x | +46.23% | 99% |
| IVF nlist256 nprobe2 | 96x | +46.46% | 100% |
| IVF nlist256 nprobe4 | 49x | +46.49% | 100% |
| IVF nlist256 nprobe8 | 25x | +46.50% | 100% |
| IVF nlist256 nprobe16 | 13x | +46.50% | 100% |
| IVF nlist256 nprobe32 | 7x | +46.51% | 100% |
| IVF nlist1024 nprobe1 | 774x | +46.63% | 100% |
| IVF nlist1024 nprobe2 | 419x | +46.63% | 100% |
| IVF nlist1024 nprobe4 | 220x | +46.65% | 100% |
| IVF nlist1024 nprobe8 | 114x | +46.62% | 100% |
| IVF nlist1024 nprobe16 | 58x | +46.62% | 100% |
| IVF nlist1024 nprobe32 | 30x | +46.62% | 100% |
_(tiny->omc_ ds600K s24 done in 141s | 150 runs | 5.5h/9.5h elapsed)_

### seed 25 [tiny->prid ds300K s25]  Ndb=300,000  LM-only=2.7486  brute=+10.25% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +10.08% | 98% |
| dodeca-uniform P2 probe2 | 1x | +10.25% | 100% |
| dodeca-WHITENED P2 probe1 | 123x | +8.10% | 79% |
| dodeca-WHITENED P2 probe2 | 33x | +10.12% | 99% |
| dodeca-WHITENED P3 probe1 | 969x | +5.08% | 50% |
| dodeca-WHITENED P3 probe2 | 156x | +8.88% | 87% |
| IVF nlist256 nprobe1 | 196x | +10.19% | 99% |
| IVF nlist256 nprobe2 | 105x | +10.21% | 100% |
| IVF nlist256 nprobe4 | 55x | +10.23% | 100% |
| IVF nlist256 nprobe8 | 29x | +10.24% | 100% |
| IVF nlist256 nprobe16 | 16x | +10.25% | 100% |
| IVF nlist256 nprobe32 | 8x | +10.26% | 100% |
| IVF nlist1024 nprobe1 | 785x | +9.62% | 94% |
| IVF nlist1024 nprobe2 | 429x | +10.03% | 98% |
| IVF nlist1024 nprobe4 | 225x | +10.21% | 100% |
| IVF nlist1024 nprobe8 | 117x | +10.23% | 100% |
| IVF nlist1024 nprobe16 | 60x | +10.25% | 100% |
| IVF nlist1024 nprobe32 | 31x | +10.26% | 100% |
_(tiny->prid ds300K s25 done in 103s | 151 runs | 5.5h/9.5h elapsed)_

### seed 25 [prid->tiny ds300K s25]  Ndb=300,000  LM-only=3.0712  brute=+15.99% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +16.26% | 102% |
| dodeca-uniform P2 probe2 | 1x | +15.99% | 100% |
| dodeca-WHITENED P2 probe1 | 125x | +14.69% | 92% |
| dodeca-WHITENED P2 probe2 | 33x | +16.49% | 103% |
| dodeca-WHITENED P3 probe1 | 891x | +10.09% | 63% |
| dodeca-WHITENED P3 probe2 | 148x | +15.16% | 95% |
| IVF nlist256 nprobe1 | 201x | +15.92% | 100% |
| IVF nlist256 nprobe2 | 108x | +15.94% | 100% |
| IVF nlist256 nprobe4 | 56x | +15.96% | 100% |
| IVF nlist256 nprobe8 | 28x | +15.98% | 100% |
| IVF nlist256 nprobe16 | 15x | +15.99% | 100% |
| IVF nlist256 nprobe32 | 8x | +15.99% | 100% |
| IVF nlist1024 nprobe1 | 765x | +15.78% | 99% |
| IVF nlist1024 nprobe2 | 414x | +16.02% | 100% |
| IVF nlist1024 nprobe4 | 223x | +16.03% | 100% |
| IVF nlist1024 nprobe8 | 116x | +15.99% | 100% |
| IVF nlist1024 nprobe16 | 59x | +15.99% | 100% |
| IVF nlist1024 nprobe32 | 30x | +15.99% | 100% |
_(prid->tiny ds300K s25 done in 100s | 152 runs | 5.5h/9.5h elapsed)_

### seed 25 [tiny->omc_ ds300K s25]  Ndb=300,000  LM-only=5.5726  brute=+46.69% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +46.46% | 100% |
| dodeca-uniform P2 probe2 | 1x | +46.64% | 100% |
| dodeca-WHITENED P2 probe1 | 121x | +44.54% | 95% |
| dodeca-WHITENED P2 probe2 | 32x | +46.16% | 99% |
| dodeca-WHITENED P3 probe1 | 729x | +42.10% | 90% |
| dodeca-WHITENED P3 probe2 | 134x | +45.43% | 97% |
| IVF nlist256 nprobe1 | 196x | +46.39% | 99% |
| IVF nlist256 nprobe2 | 105x | +46.66% | 100% |
| IVF nlist256 nprobe4 | 56x | +46.58% | 100% |
| IVF nlist256 nprobe8 | 29x | +46.64% | 100% |
| IVF nlist256 nprobe16 | 15x | +46.68% | 100% |
| IVF nlist256 nprobe32 | 7x | +46.69% | 100% |
| IVF nlist1024 nprobe1 | 805x | +45.97% | 98% |
| IVF nlist1024 nprobe2 | 423x | +46.65% | 100% |
| IVF nlist1024 nprobe4 | 220x | +46.66% | 100% |
| IVF nlist1024 nprobe8 | 115x | +46.73% | 100% |
| IVF nlist1024 nprobe16 | 60x | +46.75% | 100% |
| IVF nlist1024 nprobe32 | 31x | +46.75% | 100% |
_(tiny->omc_ ds300K s25 done in 101s | 153 runs | 5.6h/9.5h elapsed)_

### seed 25 [tiny->prid ds600K s25]  Ndb=600,000  LM-only=2.7486  brute=+10.55% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +10.49% | 99% |
| dodeca-uniform P2 probe2 | 1x | +10.55% | 100% |
| dodeca-WHITENED P2 probe1 | 120x | +9.51% | 90% |
| dodeca-WHITENED P2 probe2 | 32x | +10.93% | 104% |
| dodeca-WHITENED P3 probe1 | 921x | +6.53% | 62% |
| dodeca-WHITENED P3 probe2 | 151x | +10.14% | 96% |
| IVF nlist256 nprobe1 | 204x | +10.47% | 99% |
| IVF nlist256 nprobe2 | 107x | +10.55% | 100% |
| IVF nlist256 nprobe4 | 56x | +10.56% | 100% |
| IVF nlist256 nprobe8 | 29x | +10.58% | 100% |
| IVF nlist256 nprobe16 | 15x | +10.58% | 100% |
| IVF nlist256 nprobe32 | 8x | +10.58% | 100% |
| IVF nlist1024 nprobe1 | 761x | +10.17% | 96% |
| IVF nlist1024 nprobe2 | 422x | +10.35% | 98% |
| IVF nlist1024 nprobe4 | 224x | +10.34% | 98% |
| IVF nlist1024 nprobe8 | 115x | +10.55% | 100% |
| IVF nlist1024 nprobe16 | 59x | +10.55% | 100% |
| IVF nlist1024 nprobe32 | 30x | +10.56% | 100% |
_(tiny->prid ds600K s25 done in 145s | 154 runs | 5.6h/9.5h elapsed)_

### seed 25 [prid->tiny ds600K s25]  Ndb=600,000  LM-only=3.0712  brute=+16.23% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +16.38% | 101% |
| dodeca-uniform P2 probe2 | 1x | +16.23% | 100% |
| dodeca-WHITENED P2 probe1 | 115x | +14.87% | 92% |
| dodeca-WHITENED P2 probe2 | 31x | +17.06% | 105% |
| dodeca-WHITENED P3 probe1 | 872x | +12.10% | 75% |
| dodeca-WHITENED P3 probe2 | 146x | +15.79% | 97% |
| IVF nlist256 nprobe1 | 192x | +16.22% | 100% |
| IVF nlist256 nprobe2 | 103x | +16.21% | 100% |
| IVF nlist256 nprobe4 | 55x | +16.25% | 100% |
| IVF nlist256 nprobe8 | 29x | +16.24% | 100% |
| IVF nlist256 nprobe16 | 15x | +16.24% | 100% |
| IVF nlist256 nprobe32 | 8x | +16.24% | 100% |
| IVF nlist1024 nprobe1 | 788x | +16.21% | 100% |
| IVF nlist1024 nprobe2 | 431x | +16.23% | 100% |
| IVF nlist1024 nprobe4 | 230x | +16.24% | 100% |
| IVF nlist1024 nprobe8 | 119x | +16.24% | 100% |
| IVF nlist1024 nprobe16 | 60x | +16.24% | 100% |
| IVF nlist1024 nprobe32 | 31x | +16.24% | 100% |
_(prid->tiny ds600K s25 done in 137s | 155 runs | 5.6h/9.5h elapsed)_

### seed 25 [tiny->omc_ ds600K s25]  Ndb=600,000  LM-only=5.5726  brute=+47.96% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +47.78% | 100% |
| dodeca-uniform P2 probe2 | 1x | +47.96% | 100% |
| dodeca-WHITENED P2 probe1 | 123x | +46.25% | 96% |
| dodeca-WHITENED P2 probe2 | 32x | +46.97% | 98% |
| dodeca-WHITENED P3 probe1 | 822x | +43.90% | 92% |
| dodeca-WHITENED P3 probe2 | 141x | +47.60% | 99% |
| IVF nlist256 nprobe1 | 196x | +47.55% | 99% |
| IVF nlist256 nprobe2 | 104x | +47.82% | 100% |
| IVF nlist256 nprobe4 | 55x | +47.92% | 100% |
| IVF nlist256 nprobe8 | 28x | +47.91% | 100% |
| IVF nlist256 nprobe16 | 15x | +47.94% | 100% |
| IVF nlist256 nprobe32 | 7x | +47.95% | 100% |
| IVF nlist1024 nprobe1 | 810x | +47.13% | 98% |
| IVF nlist1024 nprobe2 | 431x | +47.60% | 99% |
| IVF nlist1024 nprobe4 | 226x | +47.93% | 100% |
| IVF nlist1024 nprobe8 | 117x | +47.94% | 100% |
| IVF nlist1024 nprobe16 | 60x | +47.94% | 100% |
| IVF nlist1024 nprobe32 | 30x | +47.93% | 100% |
_(tiny->omc_ ds600K s25 done in 141s | 156 runs | 5.7h/9.5h elapsed)_

### seed 26 [tiny->prid ds300K s26]  Ndb=300,000  LM-only=2.7913  brute=+11.44% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 2x | +10.53% | 92% |
| dodeca-uniform P2 probe2 | 1x | +11.00% | 96% |
| dodeca-WHITENED P2 probe1 | 122x | +8.29% | 72% |
| dodeca-WHITENED P2 probe2 | 33x | +10.11% | 88% |
| dodeca-WHITENED P3 probe1 | 985x | +5.63% | 49% |
| dodeca-WHITENED P3 probe2 | 159x | +9.81% | 86% |
| IVF nlist256 nprobe1 | 207x | +10.59% | 93% |
| IVF nlist256 nprobe2 | 109x | +10.94% | 96% |
| IVF nlist256 nprobe4 | 57x | +10.94% | 96% |
| IVF nlist256 nprobe8 | 30x | +11.18% | 98% |
| IVF nlist256 nprobe16 | 15x | +11.43% | 100% |
| IVF nlist256 nprobe32 | 8x | +11.43% | 100% |
| IVF nlist1024 nprobe1 | 810x | +10.40% | 91% |
| IVF nlist1024 nprobe2 | 431x | +11.03% | 96% |
| IVF nlist1024 nprobe4 | 223x | +11.18% | 98% |
| IVF nlist1024 nprobe8 | 115x | +11.22% | 98% |
| IVF nlist1024 nprobe16 | 60x | +11.21% | 98% |
| IVF nlist1024 nprobe32 | 30x | +11.44% | 100% |
_(tiny->prid ds300K s26 done in 104s | 157 runs | 5.7h/9.5h elapsed)_

### seed 26 [prid->tiny ds300K s26]  Ndb=300,000  LM-only=3.0935  brute=+17.05% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +16.97% | 100% |
| dodeca-uniform P2 probe2 | 1x | +17.05% | 100% |
| dodeca-WHITENED P2 probe1 | 122x | +13.60% | 80% |
| dodeca-WHITENED P2 probe2 | 32x | +16.18% | 95% |
| dodeca-WHITENED P3 probe1 | 1040x | +9.24% | 54% |
| dodeca-WHITENED P3 probe2 | 164x | +15.35% | 90% |
| IVF nlist256 nprobe1 | 203x | +17.06% | 100% |
| IVF nlist256 nprobe2 | 107x | +17.03% | 100% |
| IVF nlist256 nprobe4 | 56x | +17.05% | 100% |
| IVF nlist256 nprobe8 | 29x | +17.05% | 100% |
| IVF nlist256 nprobe16 | 15x | +17.04% | 100% |
| IVF nlist256 nprobe32 | 7x | +17.04% | 100% |
| IVF nlist1024 nprobe1 | 786x | +16.67% | 98% |
| IVF nlist1024 nprobe2 | 426x | +16.60% | 97% |
| IVF nlist1024 nprobe4 | 224x | +17.03% | 100% |
| IVF nlist1024 nprobe8 | 117x | +17.04% | 100% |
| IVF nlist1024 nprobe16 | 60x | +17.04% | 100% |
| IVF nlist1024 nprobe32 | 31x | +17.04% | 100% |
_(prid->tiny ds300K s26 done in 109s | 158 runs | 5.7h/9.5h elapsed)_

### seed 26 [tiny->omc_ ds300K s26]  Ndb=300,000  LM-only=5.4572  brute=+46.81% (λ=0.85)
| method | speedup | gain | retained |
|---|---|---|---|
| dodeca-uniform P2 probe1 | 1x | +46.81% | 100% |
| dodeca-uniform P2 probe2 | 1x | +46.81% | 100% |
| dodeca-WHITENED P2 probe1 | 121x | +44.86% | 96% |
| dodeca-WHITENED P2 probe2 | 33x | +47.29% | 101% |
| dodeca-WHITENED P3 probe1 | 879x | +39.42% | 84% |
| dodeca-WHITENED P3 probe2 | 145x | +45.57% | 97% |
| IVF nlist256 nprobe1 | 200x | +46.53% | 99% |
| IVF nlist256 nprobe2 | 103x | +46.65% | 100% |
| IVF nlist256 nprobe4 | 55x | +46.66% | 100% |
| IVF nlist256 nprobe8 | 29x | +46.78% | 100% |
| IVF nlist256 nprobe16 | 15x | +46.79% | 100% |
| IVF nlist256 nprobe32 | 7x | +46.79% | 100% |
| IVF nlist1024 nprobe1 | 826x | +46.47% | 99% |
| IVF nlist1024 nprobe2 | 438x | +46.89% | 100% |
| IVF nlist1024 nprobe4 | 225x | +46.85% | 100% |
| IVF nlist1024 nprobe8 | 116x | +46.79% | 100% |
| IVF nlist1024 nprobe16 | 60x | +46.80% | 100% |
| IVF nlist1024 nprobe32 | 31x | +46.80% | 100% |
_(tiny->omc_ ds300K s26 done in 119s | 159 runs | 5.8h/9.5h elapsed)_

### seed 26 [tiny->prid ds600K s26]  Ndb=600,000  LM-only=2.7913  brute=+10.85% (λ=0.80)
| method | speedup | gain | retained |
|---|---|---|---|
