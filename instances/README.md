# Instances

| Folder | Instances | Source |
|---|---|---|
| `song/1005`, `song/2005`, `song/1510`, `song/2010` | 100 synthetic FJSP instances per size (10×5, 20×5, 15×10, 20×10) | Song et al. [1] |
| `brandimarte` | Mk01–Mk10 | Brandimarte [2] |
| `hurink/edata`, `hurink/rdata`, `hurink/vdata` | la01–la40 of each set | Hurink et al. [3] |
| `dauzere` | 01a–18a | Dauzère-Pérès and Paulli [4] |
| `barnes` | mt10c1–seti5xyz (21) | Chambers and Barnes [5] |
| `behnke/50x40`, `behnke/100x40`, `behnke/50x60`, `behnke/100x60` | Behnke31–35, 36–40, 51–55, 56–60 | Behnke and Geiger [6] |
| `taillard/15x15`, `taillard/20x15`, `taillard/20x20` | ta01–ta10, ta11–ta20, ta21–ta30 (JSP) | Taillard [7] |
| `demirkol/20x15`, `demirkol/20x20` | 10 instances each (JSP) | Demirkol et al. [8] |

The Brandimarte, Hurink, Dauzère-Pérès and Paulli, Barnes and Behnke instances are taken from the collection of SchedulingLab,
distributed under the MIT License (`LICENSE-SchedulingLab`). Data may contain errors; always refer to the original publications.

## Format

- FJSP (`.fjs`, `.txt`): the first line gives the numbers of jobs and machines (some files add the average number of machines per
  operation); each further line describes one job: the number of operations and, for each operation, the number of eligible
  machines followed by `machine processing-time` pairs.
- JSP (`.jsp`): the first line gives the numbers of jobs and machines; each further line lists the `machine processing-time` pairs
  of one job in processing order.

`utils/instance_parser.py` reads both formats and detects whether machines are numbered from 0 or 1.

## References

1. W. Song, X. Chen, Q. Li, Z. Cao. Flexible job-shop scheduling via graph neural network and deep reinforcement learning. IEEE Transactions on Industrial Informatics, 19(2):1600–1610, 2023.
2. P. Brandimarte. Routing and scheduling in a flexible job shop by tabu search. Annals of Operations Research, 41(3):157–183, 1993.
3. J. Hurink, B. Jurisch, M. Thole. Tabu search for the job-shop scheduling problem with multi-purpose machines. Operations-Research-Spektrum, 15(4):205–215, 1994.
4. S. Dauzère-Pérès, J. Paulli. Solving the general multiprocessor job-shop scheduling problem. Technical report, Rotterdam School of Management, Erasmus Universiteit Rotterdam, 1994.
5. J. B. Chambers, J. W. Barnes. Flexible job shop scheduling by tabu search. Technical Report ORP96-09, The University of Texas at Austin, 1996.
6. D. Behnke, M. J. Geiger. Test instances for the flexible job shop scheduling problem with work centers. Research Paper, Helmut-Schmidt-Universität, Hamburg, 2012.
7. E. Taillard. Benchmarks for basic scheduling problems. European Journal of Operational Research, 64(2):278–285, 1993.
8. E. Demirkol, S. Mehta, R. Uzsoy. Benchmarks for shop scheduling problems. European Journal of Operational Research, 109(1):137–141, 1998.
