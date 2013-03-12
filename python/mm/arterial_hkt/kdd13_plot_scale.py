'''
Created on Feb 22, 2013

@author: tjhunter
'''
import matplotlib
#matplotlib.use("Agg")
from matplotlib import rc
from mm.arterial_hkt.validation.validation import save_fig
#rc('font',**{'family':'serif','serif':['Times New Roman']})
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
#matplotlib.rcParams['text.usetex'] = True
rc('font', **{'family':'serif'})


import pylab as pl
import numpy as np
import re

exp_3="""
[1361594793.5102] big_3: Loaded network = 58 links
[1361594793.5105] big_3: TT graph building
[1361594793.5105] big_3: reading tt graph from /mnt//experiments/big_3//tt_graph.pkl
[1361594793.7788] big_3: Loaded TT graph = 1318 edges, 5424 variables
[1361594793.7788] big_3: simulating sstat building
[1361594793.7978] big_3: done simulating sstat building
[1361594793.7979] big_3: GMRF learning
[1361594793.8160] gmrf_learn_cov_cholmod: m=5424, active m=5424
[1361594793.8183] gmrf_learn_cov_cholmod: Computing symbolic cholesky factorization of the graph...
[1361594793.8226] gmrf_learn_cov_cholmod: Cholesky done
[1361594793.8226] covsel_cvx_cholmod: smallest ev
[1361594793.8860] covsel_cvx_cholmod: min_ei is -0.091073
[1361594793.8861] run_cvx_cholmod: Iter=0
[1361594793.8861] iter_cholmod: computing gradient
[1361594793.9466] iter_cholmod: Newton decrement squared: 4.30920e+02
[1361594793.9517] iter_cholmod: Current objective value: 1439.35222853
[1361594793.9541] iter_cholmod: lsiter=0 fn=inf
[1361594793.9566] iter_cholmod: lsiter=1 fn=inf
[1361594793.9615] iter_cholmod: lsiter=2 fn=1413.79207093
[1361594793.9616] iter_cholmod: Update lsiter=2
[1361594793.9616] run_cvx_cholmod: Iter=1
[1361594793.9616] iter_cholmod: computing gradient
[1361594794.0181] iter_cholmod: Newton decrement squared: 3.58173e+02
[1361594794.0232] iter_cholmod: Current objective value: 1413.79207093
[1361594794.0256] iter_cholmod: lsiter=0 fn=inf
[1361594794.0280] iter_cholmod: lsiter=1 fn=inf
[1361594794.0328] iter_cholmod: lsiter=2 fn=1415.28761598
[1361594794.0377] iter_cholmod: lsiter=3 fn=1407.12411578
[1361594794.0377] iter_cholmod: Update lsiter=3
[1361594794.0377] run_cvx_cholmod: Iter=2
[1361594794.0377] iter_cholmod: computing gradient
[1361594794.0948] iter_cholmod: Newton decrement squared: 3.23099e+02
[1361594794.0998] iter_cholmod: Current objective value: 1407.12411578
[1361594794.1022] iter_cholmod: lsiter=0 fn=inf
[1361594794.1047] iter_cholmod: lsiter=1 fn=inf
[1361594794.1094] iter_cholmod: lsiter=2 fn=1414.93067519
[1361594794.1141] iter_cholmod: lsiter=3 fn=1404.35905634
[1361594794.1141] iter_cholmod: Update lsiter=3
[1361594794.1141] run_cvx_cholmod: Iter=3
[1361594794.1142] iter_cholmod: computing gradient
[1361594794.1708] iter_cholmod: Newton decrement squared: 3.17821e+02
[1361594794.1759] iter_cholmod: Current objective value: 1404.35905634
[1361594794.1783] iter_cholmod: lsiter=0 fn=inf
[1361594794.1807] iter_cholmod: lsiter=1 fn=inf
[1361594794.1854] iter_cholmod: lsiter=2 fn=1416.47135433
[1361594794.1902] iter_cholmod: lsiter=3 fn=1403.65437565
[1361594794.1902] iter_cholmod: Update lsiter=3
[1361594794.1902] run_cvx_cholmod: Iter=4
[1361594794.1902] iter_cholmod: computing gradient
[1361594794.2474] iter_cholmod: Newton decrement squared: 3.26818e+02
[1361594794.2525] iter_cholmod: Current objective value: 1403.65437565
[1361594794.2549] iter_cholmod: lsiter=0 fn=inf
[1361594794.2573] iter_cholmod: lsiter=1 fn=inf
[1361594794.2621] iter_cholmod: lsiter=2 fn=1415.49269651
[1361594794.2669] iter_cholmod: lsiter=3 fn=1402.8031026
[1361594794.2669] iter_cholmod: Update lsiter=3
[1361594794.2669] run_cvx_cholmod: Iter=5
[1361594794.2669] iter_cholmod: computing gradient
[1361594794.3235] iter_cholmod: Newton decrement squared: 3.13241e+02
[1361594794.3286] iter_cholmod: Current objective value: 1402.8031026
[1361594794.3310] iter_cholmod: lsiter=0 fn=inf
[1361594794.3335] iter_cholmod: lsiter=1 fn=inf
[1361594794.3382] iter_cholmod: lsiter=2 fn=1414.36509228
[1361594794.3430] iter_cholmod: lsiter=3 fn=1402.29376423
[1361594794.3430] iter_cholmod: Update lsiter=3
[1361594794.3430] run_cvx_cholmod: Iter=6
[1361594794.3430] iter_cholmod: computing gradient
[1361594794.3990] iter_cholmod: Newton decrement squared: 3.13196e+02
[1361594794.4041] iter_cholmod: Current objective value: 1402.29376423
[1361594794.4065] iter_cholmod: lsiter=0 fn=inf
[1361594794.4090] iter_cholmod: lsiter=1 fn=inf
[1361594794.4138] iter_cholmod: lsiter=2 fn=1416.17738902
[1361594794.4187] iter_cholmod: lsiter=3 fn=1402.63056878
[1361594794.4234] iter_cholmod: lsiter=4 fn=1400.87627591
[1361594794.4234] iter_cholmod: Update lsiter=4
[1361594794.4235] run_cvx_cholmod: Iter=7
[1361594794.4235] iter_cholmod: computing gradient
[1361594794.4802] iter_cholmod: Newton decrement squared: 3.05245e+02
[1361594794.4854] iter_cholmod: Current objective value: 1400.87627591
[1361594794.4878] iter_cholmod: lsiter=0 fn=inf
[1361594794.4902] iter_cholmod: lsiter=1 fn=inf
[1361594794.4950] iter_cholmod: lsiter=2 fn=1415.13683469
[1361594794.4999] iter_cholmod: lsiter=3 fn=1401.74793041
[1361594794.5047] iter_cholmod: lsiter=4 fn=1399.79430428
[1361594794.5048] iter_cholmod: Update lsiter=4
[1361594794.5048] run_cvx_cholmod: Iter=8
[1361594794.5048] iter_cholmod: computing gradient
[1361594794.5617] iter_cholmod: Newton decrement squared: 2.94977e+02
[1361594794.5668] iter_cholmod: Current objective value: 1399.79430428
[1361594794.5692] iter_cholmod: lsiter=0 fn=inf
[1361594794.5717] iter_cholmod: lsiter=1 fn=inf
[1361594794.5764] iter_cholmod: lsiter=2 fn=1414.79935945
[1361594794.5812] iter_cholmod: lsiter=3 fn=1401.2410945
[1361594794.5860] iter_cholmod: lsiter=4 fn=1399.0552665
[1361594794.5860] iter_cholmod: Update lsiter=4
[1361594794.5860] run_cvx_cholmod: Iter=9
[1361594794.5860] iter_cholmod: computing gradient
[1361594794.6427] iter_cholmod: Newton decrement squared: 3.00717e+02
[1361594794.6479] iter_cholmod: Current objective value: 1399.0552665
[1361594794.6503] iter_cholmod: lsiter=0 fn=inf
[1361594794.6527] iter_cholmod: lsiter=1 fn=inf
[1361594794.6575] iter_cholmod: lsiter=2 fn=1414.42951766
[1361594794.6624] iter_cholmod: lsiter=3 fn=1400.65566766
[1361594794.6673] iter_cholmod: lsiter=4 fn=1398.37832344
[1361594794.6673] iter_cholmod: Update lsiter=4
[1361594794.6673] run_cvx_cholmod: Iter=10
[1361594794.6673] iter_cholmod: computing gradient
[1361594794.7236] iter_cholmod: Newton decrement squared: 2.85452e+02
[1361594794.7287] iter_cholmod: Current objective value: 1398.37832344
[1361594794.7311] iter_cholmod: lsiter=0 fn=inf
"""

exp0="""
[1361590987.4730] big_0: Loaded network = 6362 links
Creating dir /mnt//experiments/big_0/
[1361590987.4827] big_0: TT graph building
[1361590987.4828] big_0: creating empty tt graph
[1361590987.4829] big_0: creating empty hmm graph
[1361590987.4829] : Loading trajectory conversion...
[1361590987.5232] : Done loading trajectory conversion and mode counts
[1361590988.4546] big_0: done creating empty hmm graph  
[1361590988.4547] big_0: saving hmm graph pickle
[1361590989.6312] big_0: done saving hmm graph pickle   
Saving travel time graph structure in /mnt//experiments/big_0/
[1361590993.7604] big_0: Loaded TT graph = 12724 edges, 64368 variables
[1361590993.7605] big_0: simulating sstat building
[1361590994.0385] big_0: done simulating sstat building 
[1361590994.0386] big_0: saving tt graph pickle
[1361590999.5705] big_0: done saving tt graph pickle
[1361590999.5706] big_0: GMRF learning
[1361590999.8351] gmrf_learn_cov_cholmod: m=64368, active m=64368
[1361590999.8621] gmrf_learn_cov_cholmod: Computing symbolic cholesky factorization of the graph...
[1361590999.8976] gmrf_learn_cov_cholmod: Cholesky done 
[1361590999.8977] covsel_cvx_cholmod: smallest ev
[1361591000.6495] covsel_cvx_cholmod: min_ei is -0.393701
[1361591000.6498] run_cvx_cholmod: Iter=0
[1361591000.6498] iter_cholmod: computing gradient
[1361591001.5421] iter_cholmod: Newton decrement squared: 7.11576e+03
[1361591001.6185] iter_cholmod: Current objective value: 17746.1694964
/home/ubuntu/arterial-estimation-tt/python/mm/arterial_hkt/gmrf_learning/psd.py:25: CholmodWarning: ../Supernodal/t_cholmod_super_numeric.c:613: matrix not positive defi
nite (code 1)
  full_factor = cholesky(X) if factor is None else factor.cholesky(X)
[1361591001.6374] iter_cholmod: lsiter=0 fn=inf
[1361591001.6550] iter_cholmod: lsiter=1 fn=inf
[1361591001.7655] iter_cholmod: lsiter=2 fn=17124.6055027
[1361591001.7656] iter_cholmod: Update lsiter=2
[1361591001.7656] run_cvx_cholmod: Iter=1
[1361591001.7656] iter_cholmod: computing gradient
[1361591002.7495] iter_cholmod: Newton decrement squared: 5.89109e+03
[1361591002.8781] iter_cholmod: Current objective value: 17124.6055027
[1361591002.8954] iter_cholmod: lsiter=0 fn=inf
[1361591002.9121] iter_cholmod: lsiter=1 fn=inf
[1361591003.0212] iter_cholmod: lsiter=2 fn=17224.612965
[1361591003.1292] iter_cholmod: lsiter=3 fn=17007.0697276
[1361591003.1293] iter_cholmod: Update lsiter=3
[1361591003.1293] run_cvx_cholmod: Iter=2
[1361591003.1294] iter_cholmod: computing gradient
[1361591004.1090] iter_cholmod: Newton decrement squared: 5.62386e+03
[1361591004.2352] iter_cholmod: Current objective value: 17007.0697276
[1361591004.2529] iter_cholmod: lsiter=0 fn=inf
[1361591004.2694] iter_cholmod: lsiter=1 fn=inf
[1361591004.3092] iter_cholmod: lsiter=2 fn=inf
[1361591004.4179] iter_cholmod: lsiter=3 fn=16947.4620437
[1361591004.4180] iter_cholmod: Update lsiter=3
[1361591004.4180] run_cvx_cholmod: Iter=3
[1361591004.4180] iter_cholmod: computing gradient
[1361591005.3836] iter_cholmod: Newton decrement squared: 5.67186e+03
[1361591005.5071] iter_cholmod: Current objective value: 16947.4620437
[1361591005.5248] iter_cholmod: lsiter=0 fn=inf
[1361591005.5411] iter_cholmod: lsiter=1 fn=inf
[1361591005.6498] iter_cholmod: lsiter=2 fn=17237.6916787
[1361591005.7585] iter_cholmod: lsiter=3 fn=16915.0748719
[1361591005.7586] iter_cholmod: Update lsiter=3
[1361591005.7586] run_cvx_cholmod: Iter=4
[1361591005.7586] iter_cholmod: computing gradient
[1361591006.7015] iter_cholmod: Newton decrement squared: 5.76024e+03
[1361591006.8272] iter_cholmod: Current objective value: 16915.0748719
[1361591006.8453] iter_cholmod: lsiter=0 fn=inf
[1361591006.8623] iter_cholmod: lsiter=1 fn=inf
[1361591006.9021] iter_cholmod: lsiter=2 fn=inf
[1361591007.0121] iter_cholmod: lsiter=3 fn=16894.4669123
[1361591007.0122] iter_cholmod: Update lsiter=3
[1361591007.0122] run_cvx_cholmod: Iter=5

[1361591072.0332] iter_cholmod: Update lsiter=7
[1361591072.0332] run_cvx_cholmod: Iter=47
[1361591072.0332] iter_cholmod: computing gradient
[1361591072.9867] iter_cholmod: Newton decrement squared: 4.95646e+03
[1361591073.1112] iter_cholmod: Current objective value: 16674.9776188
[1361591073.1293] iter_cholmod: lsiter=0 fn=inf
[1361591073.1465] iter_cholmod: lsiter=1 fn=inf
[1361591073.1709] iter_cholmod: lsiter=2 fn=inf
[1361591073.2805] iter_cholmod: lsiter=3 fn=16813.8000781
[1361591073.3897] iter_cholmod: lsiter=4 fn=16704.4974141
[1361591073.4991] iter_cholmod: lsiter=5 fn=16680.1994927
[1361591073.6084] iter_cholmod: lsiter=6 fn=16675.2285055
[1361591073.7182] iter_cholmod: lsiter=7 fn=16674.5144447
[1361591073.7183] iter_cholmod: Update lsiter=7
[1361591073.7183] run_cvx_cholmod: Iter=48
[1361591073.7184] iter_cholmod: computing gradient
[1361591074.7078] iter_cholmod: Newton decrement squared: 4.90293e+03
[1361591074.8351] iter_cholmod: Current objective value: 16674.5144447
[1361591074.8534] iter_cholmod: lsiter=0 fn=inf
[1361591074.8711] iter_cholmod: lsiter=1 fn=inf
[1361591074.9170] iter_cholmod: lsiter=2 fn=inf
[1361591075.0271] iter_cholmod: lsiter=3 fn=16810.8441156
[1361591075.1380] iter_cholmod: lsiter=4 fn=16703.4406717
[1361591075.2482] iter_cholmod: lsiter=5 fn=16679.5747465
[1361591075.3590] iter_cholmod: lsiter=6 fn=16674.7162625
[1361591075.4685] iter_cholmod: lsiter=7 fn=16674.0344604
[1361591075.4686] iter_cholmod: Update lsiter=7
[1361591075.4687] run_cvx_cholmod: Iter=49
[1361591075.4687] iter_cholmod: computing gradient
[1361591076.4362] iter_cholmod: Newton decrement squared: 4.97506e+03
[1361591076.5628] iter_cholmod: Current objective value: 16674.0344604
[1361591076.5812] iter_cholmod: lsiter=0 fn=inf
[1361591076.5986] iter_cholmod: lsiter=1 fn=inf
[1361591076.6248] iter_cholmod: lsiter=2 fn=inf
[1361591076.7352] iter_cholmod: lsiter=3 fn=16813.1795878
[1361591076.8442] iter_cholmod: lsiter=4 fn=16703.5437594
[1361591076.9525] iter_cholmod: lsiter=5 fn=16679.2000624
[1361591077.0611] iter_cholmod: lsiter=6 fn=16674.2428289
[1361591077.1692] iter_cholmod: lsiter=7 fn=16673.5461884
[1361591077.1693] iter_cholmod: Update lsiter=7
[1361591077.3747] big_0: Done GMRF learning
[1361591077.3747] big_0: saving gmrf pickle
[1361591078.3557] big_0: done saving gmrf pickle
[1361591078.3558] big_0: GMRF estimation
Q shape (4000, 12724)
[1361591154.9920] big_0: End of learning
"""

exp_2 = """
[1361590311.6494] big_2: Loaded network = 53466 links
Creating dir /mnt//experiments/big_2/
[1361590312.0446] big_2: TT graph building
[1361590312.0497] big_2: creating empty tt graph
[1361590312.0533] big_2: creating empty hmm graph
[1361590312.0533] : Loading trajectory conversion...
[1361590312.5169] : Done loading trajectory conversion and mode counts
[1361590323.7904] big_2: done creating empty hmm graph
[1361590323.7905] big_2: saving hmm graph pickle
[1361590338.6291] big_2: done saving hmm graph pickle
Saving travel time graph structure in /mnt//experiments/big_2/
[1361590394.0367] big_2: Loaded TT graph = 106932 edges, 543128 variables
[1361590394.0368] big_2: simulating sstat building
[1361590396.8128] big_2: done simulating sstat building
[1361590396.8130] big_2: saving tt graph pickle
[1361590453.5559] big_2: done saving tt graph pickle
[1361590453.5560] big_2: GMRF learning
[1361590456.3626] gmrf_learn_cov_cholmod: m=543128, active m=543128
[1361590394.0368] big_2: simulating sstat building
[1361590396.8128] big_2: done simulating sstat building
[1361590396.8130] big_2: saving tt graph pickle
[1361590453.5559] big_2: done saving tt graph pickle
[1361590453.5560] big_2: GMRF learning
[1361590456.3626] gmrf_learn_cov_cholmod: m=543128, active m=543128
[1361590456.6859] gmrf_learn_cov_cholmod: Computing symbolic cholesky factorization of the graph...
[1361590457.1132] gmrf_learn_cov_cholmod: Cholesky done
[1361590457.1133] covsel_cvx_cholmod: smallest ev
[1361590489.0967] covsel_cvx_cholmod: min_ei is -0.413315
[1361590489.0980] run_cvx_cholmod: Iter=0
[1361590489.0981] iter_cholmod: computing gradient
[1361590498.3498] iter_cholmod: Newton decrement squared: 6.18179e+04
[1361590499.0844] iter_cholmod: Current objective value: 151232.732412
/home/ubuntu/arterial-estimation-tt/python/mm/arterial_hkt/gmrf_learning/psd.py:25: CholmodWarning: ../Supernodal/t_cholmod_super_numeric.c:613: matrix not positive definite (code 1)
  full_factor = cholesky(X) if factor is None else factor.cholesky(X)
[1361590499.3581] iter_cholmod: lsiter=0 fn=inf
[1361590499.6018] iter_cholmod: lsiter=1 fn=inf
[1361590500.9143] iter_cholmod: lsiter=2 fn=145685.034113
[1361590500.9144] iter_cholmod: Update lsiter=2
[1361590500.9145] run_cvx_cholmod: Iter=1
[1361590500.9145] iter_cholmod: computing gradient
[1361590510.3454] iter_cholmod: Newton decrement squared: 5.06726e+04
[1361590511.6573] iter_cholmod: Current objective value: 145685.034113
[1361590511.9112] iter_cholmod: lsiter=0 fn=inf
[1361590512.1529] iter_cholmod: lsiter=1 fn=inf
[1361590512.4233] iter_cholmod: lsiter=2 fn=inf
[1361590513.7323] iter_cholmod: lsiter=3 fn=144636.515984
[1361590513.7323] iter_cholmod: Update lsiter=3
[1361590513.7324] run_cvx_cholmod: Iter=2
[1361590513.7324] iter_cholmod: computing gradient
[1361590523.1053] iter_cholmod: Newton decrement squared: 4.84327e+04
[1361590524.4172] iter_cholmod: Current objective value: 144636.515984
[1361590524.6835] iter_cholmod: lsiter=0 fn=inf
[1361590524.9835] iter_cholmod: lsiter=1 fn=inf
[1361590525.3185] iter_cholmod: lsiter=2 fn=inf
[1361590526.6860] iter_cholmod: lsiter=3 fn=144090.141585
[1361590526.6860] iter_cholmod: Update lsiter=3
[1361590526.6861] run_cvx_cholmod: Iter=3
[1361590526.6861] iter_cholmod: computing gradient
[1361590536.0361] iter_cholmod: Newton decrement squared: 4.89614e+04
[1361590537.3508] iter_cholmod: Current objective value: 144090.141585
[1361590537.5984] iter_cholmod: lsiter=0 fn=inf
[1361590537.8381] iter_cholmod: lsiter=1 fn=inf
[1361590538.1616] iter_cholmod: lsiter=2 fn=inf
[1361590539.4683] iter_cholmod: lsiter=3 fn=143781.607892
[1361590539.4684] iter_cholmod: Update lsiter=3
[1361590539.4684] run_cvx_cholmod: Iter=4
[1361590539.4684] iter_cholmod: computing gradient
[1361590548.8569] iter_cholmod: Newton decrement squared: 4.98226e+04
[1361590550.2071] iter_cholmod: Current objective value: 143781.607892
[1361590550.4712] iter_cholmod: lsiter=0 fn=inf
[1361590550.7293] iter_cholmod: lsiter=1 fn=inf
[1361590551.1599] iter_cholmod: lsiter=2 fn=inf
[1361590552.5054] iter_cholmod: lsiter=3 fn=143600.487581
[1361590552.5055] iter_cholmod: Update lsiter=3
[1361590552.5055] run_cvx_cholmod: Iter=5
[1361590552.5056] iter_cholmod: computing gradient
[1361590562.3063] iter_cholmod: Newton decrement squared: 5.12667e+04
[1361590563.6304] iter_cholmod: Current objective value: 143600.487581
[1361590563.8704] iter_cholmod: lsiter=0 fn=inf
[1361590564.1153] iter_cholmod: lsiter=1 fn=inf
[1361590564.3775] iter_cholmod: lsiter=2 fn=inf
[1361590565.7060] iter_cholmod: lsiter=3 fn=143493.505871
[1361590565.7061] iter_cholmod: Update lsiter=3
[1361590565.7061] run_cvx_cholmod: Iter=6
[1361590565.7061] iter_cholmod: computing gradient
[1361590575.2480] iter_cholmod: Newton decrement squared: 5.22656e+04
[1361590576.5740] iter_cholmod: Current objective value: 143493.505871
[1361590576.8172] iter_cholmod: lsiter=0 fn=inf
[1361590577.0621] iter_cholmod: lsiter=1 fn=inf
[1361590577.3527] iter_cholmod: lsiter=2 fn=inf
[1361590578.6701] iter_cholmod: lsiter=3 fn=143428.314381
[1361590580.0059] iter_cholmod: lsiter=4 fn=143070.605166
[1361590580.0060] run_cvx_cholmod: Iter=7
[1361590580.0060] iter_cholmod: computing gradient
[1361590589.6222] iter_cholmod: Newton decrement squared: 4.86293e+04
[1361590590.9439] iter_cholmod: Current objective value: 143070.605166
[1361590591.2004] iter_cholmod: lsiter=0 fn=inf
[1361590591.4490] iter_cholmod: lsiter=1 fn=inf
[1361590591.7323] iter_cholmod: lsiter=2 fn=inf
[1361590593.0536] iter_cholmod: lsiter=3 fn=143282.381917
[1361590594.3672] iter_cholmod: lsiter=4 fn=142815.099531
[1361590594.3673] iter_cholmod: Update lsiter=4
[1361590594.3673] run_cvx_cholmod: Iter=8
[1361590594.3674] iter_cholmod: computing gradient
[1361590603.9171] iter_cholmod: Newton decrement squared: 4.70513e+04
[1361590605.2544] iter_cholmod: Current objective value: 142815.099531
[1361590605.5197] iter_cholmod: lsiter=0 fn=inf
[1361590605.7775] iter_cholmod: lsiter=1 fn=inf
[1361590606.0507] iter_cholmod: lsiter=2 fn=inf
[1361590607.3504] iter_cholmod: lsiter=3 fn=143198.390484
[1361590608.6577] iter_cholmod: lsiter=4 fn=142653.394143
[1361590608.6578] iter_cholmod: Update lsiter=4
[1361590608.6578] run_cvx_cholmod: Iter=9
[1361590608.6579] iter_cholmod: computing gradient
[1361590618.0683] iter_cholmod: Newton decrement squared: 4.67048e+04
[1361590619.4347] iter_cholmod: Current objective value: 142653.394143
[1361590619.6970] iter_cholmod: lsiter=0 fn=inf
[1361590619.9455] iter_cholmod: lsiter=1 fn=inf
[1361590620.2332] iter_cholmod: lsiter=2 fn=inf
[1361590621.6237] iter_cholmod: lsiter=3 fn=143147.611686
[1361590622.9760] iter_cholmod: lsiter=4 fn=142546.202776
[1361590622.9761] iter_cholmod: Update lsiter=4
[1361590622.9761] run_cvx_cholmod: Iter=10
[1361590622.9762] iter_cholmod: computing gradient
[1361590632.6046] iter_cholmod: Newton decrement squared: 4.64562e+04
[1361590633.9365] iter_cholmod: Current objective value: 142546.202776
[1361590634.1820] iter_cholmod: lsiter=0 fn=inf
[1361590634.4255] iter_cholmod: lsiter=1 fn=inf
[1361590634.6776] iter_cholmod: lsiter=2 fn=inf
[1361590635.9992] iter_cholmod: lsiter=3 fn=143111.895433
[1361590637.3125] iter_cholmod: lsiter=4 fn=142473.306206
[1361590637.3126] iter_cholmod: Update lsiter=4
[1361590637.3126] run_cvx_cholmod: Iter=11
[1361590637.3126] iter_cholmod: computing gradient
[1361590646.8282] iter_cholmod: Newton decrement squared: 4.64308e+04
[1361590648.1388] iter_cholmod: Current objective value: 142473.306206
[1361590648.3831] iter_cholmod: lsiter=0 fn=inf
[1361590648.6232] iter_cholmod: lsiter=1 fn=inf
[1361590648.8700] iter_cholmod: lsiter=2 fn=inf
[1361590650.1747] iter_cholmod: lsiter=3 fn=143091.454551
[1361590651.4888] iter_cholmod: lsiter=4 fn=142423.936326
[1361590651.4889] iter_cholmod: Update lsiter=4
[1361590651.4890] run_cvx_cholmod: Iter=12
[1361590651.4890] iter_cholmod: computing gradient
[1361590660.8817] iter_cholmod: Newton decrement squared: 4.64019e+04
[1361590662.2388] iter_cholmod: Current objective value: 142423.936326
[1361590662.4970] iter_cholmod: lsiter=0 fn=inf
[1361590662.7404] iter_cholmod: lsiter=1 fn=inf
[1361590663.0104] iter_cholmod: lsiter=2 fn=inf
[1361590664.3192] iter_cholmod: lsiter=3 fn=143082.079015
[1361590665.6345] iter_cholmod: lsiter=4 fn=142391.002537
[1361590665.6346] iter_cholmod: Update lsiter=4
[1361590665.6347] run_cvx_cholmod: Iter=13
[1361590665.6347] iter_cholmod: computing gradient
[1361590674.9839] iter_cholmod: Newton decrement squared: 4.67354e+04
[1361590676.3272] iter_cholmod: Current objective value: 142391.002537
[1361590676.5709] iter_cholmod: lsiter=0 fn=inf
[1361590676.8150] iter_cholmod: lsiter=1 fn=inf
[1361590677.0629] iter_cholmod: lsiter=2 fn=inf
"""

exp_108 = """
[1361589390.7970] big: Loaded network = 506585 links
[1361589390.7972] big: TT graph building
[1361589390.7972] big: reading tt graph from /mnt//experiments/big//tt_graph.pkl
[1361589741.1922] big: Loaded TT graph = 1013170 edges, 4794908 variables
[1361589741.1923] big: simulating sstat building
[1361589767.6658] big: done simulating sstat building
[1361589767.6659] big: saving tt graph pickle
[1361590265.8523] big: done saving tt graph pickle
[1361590265.8524] big: GMRF learning
[1361590299.4268] gmrf_learn_cov_cholmod: m=4794908, active m=4794908
[1361590303.0192] gmrf_learn_cov_cholmod: Computing symbolic cholesky factorization of the graph...
[1361590310.4560] gmrf_learn_cov_cholmod: Cholesky done
[1361590310.4561] covsel_cvx_cholmod: smallest ev
[1361590352.3454] covsel_cvx_cholmod: min_ei is -0.555896
[1361590352.3934] run_cvx_cholmod: Iter=0
[1361590352.3935] iter_cholmod: computing gradient
[1361590437.6903] iter_cholmod: Newton decrement squared: 6.97686e+05
[1361590447.8395] iter_cholmod: Current objective value: 1577400.31679
/home/ubuntu/arterial-estimation-tt/python/mm/arterial_hkt/gmrf_learning/psd.py:25: CholmodWarning: ../Supernodal/t_cholmod_super_numeric.c:613: matrix not positive definite (code 1)
  full_factor = cholesky(X) if factor is None else factor.cholesky(X)
[1361590451.9515] iter_cholmod: lsiter=0 fn=inf
[1361590456.1122] iter_cholmod: lsiter=1 fn=inf
[1361590468.3337] iter_cholmod: lsiter=2 fn=1498517.16289
[1361590468.3338] iter_cholmod: Update lsiter=2
[1361590468.3579] run_cvx_cholmod: Iter=1
[1361590468.3580] iter_cholmod: computing gradient
[1361590559.2682] iter_cholmod: Newton decrement squared: 5.41483e+05
[1361590571.8029] iter_cholmod: Current objective value: 1498517.16289
[1361590576.0083] iter_cholmod: lsiter=0 fn=inf
[1361590580.2290] iter_cholmod: lsiter=1 fn=inf
[1361590584.3681] iter_cholmod: lsiter=2 fn=inf
[1361590589.9191] iter_cholmod: lsiter=3 fn=inf
[1361590602.2078] iter_cholmod: lsiter=4 fn=1487477.04206
[1361590602.2079] iter_cholmod: Update lsiter=4
[1361590602.2713] run_cvx_cholmod: Iter=2
[1361590602.2714] iter_cholmod: computing gradient
[1361590693.9141] iter_cholmod: Newton decrement squared: 4.95919e+05
[1361590706.2833] iter_cholmod: Current objective value: 1487477.04206
[1361590710.5891] iter_cholmod: lsiter=0 fn=inf
[1361590714.8596] iter_cholmod: lsiter=1 fn=inf
[1361590718.9964] iter_cholmod: lsiter=2 fn=inf
[1361590731.5634] iter_cholmod: lsiter=3 fn=1477486.4984
[1361590731.5635] iter_cholmod: Update lsiter=3
[1361590731.5935] run_cvx_cholmod: Iter=3
[1361590731.5936] iter_cholmod: computing gradient
[1361590822.9442] iter_cholmod: Newton decrement squared: 5.16256e+05
[1361590835.5094] iter_cholmod: Current objective value: 1477486.4984
[1361590839.7334] iter_cholmod: lsiter=0 fn=inf
[1361590844.0610] iter_cholmod: lsiter=1 fn=inf
[1361590848.2421] iter_cholmod: lsiter=2 fn=inf
[1361590860.9842] iter_cholmod: lsiter=3 fn=1471651.44129
[1361590860.9843] iter_cholmod: Update lsiter=3
[1361590861.0401] run_cvx_cholmod: Iter=4
[1361590861.0402] iter_cholmod: computing gradient
[1361590952.8412] iter_cholmod: Newton decrement squared: 5.33267e+05
[1361590965.3062] iter_cholmod: Current objective value: 1471651.44129
[1361590969.4982] iter_cholmod: lsiter=0 fn=inf
[1361590973.7135] iter_cholmod: lsiter=1 fn=inf
[1361590977.9892] iter_cholmod: lsiter=2 fn=inf
[1361590990.5011] iter_cholmod: lsiter=3 fn=1468128.91542
[1361590990.5012] iter_cholmod: Update lsiter=3
[1361590990.5320] run_cvx_cholmod: Iter=5
[1361590990.5321] iter_cholmod: computing gradient
[1361591083.3653] iter_cholmod: Newton decrement squared: 5.54914e+05
[1361591095.7975] iter_cholmod: Current objective value: 1468128.91542
[1361591099.9938] iter_cholmod: lsiter=0 fn=inf
[1361591104.2688] iter_cholmod: lsiter=1 fn=inf
[1361591108.4765] iter_cholmod: lsiter=2 fn=inf
[1361591120.8114] iter_cholmod: lsiter=3 fn=1466143.91395
[1361591120.8115] iter_cholmod: Update lsiter=3
[1361591120.8637] run_cvx_cholmod: Iter=6
[1361591120.8638] iter_cholmod: computing gradient
[1361591209.0457] iter_cholmod: Newton decrement squared: 5.77154e+05
[1361591221.2161] iter_cholmod: Current objective value: 1466143.91395
[1361591225.3885] iter_cholmod: lsiter=0 fn=inf
"""

def scrape_iter_times(lines):
  times = []
  iters = []
  for line in lines:
    if "Iter=" in line:
      x = re.findall("\d+.\d+", line)[0]
      it = re.findall("=\d+", line)[0]
      times.append(float(x))
      iters.append(int(it[1:]))
  return times,iters

all_lines = [s.split('\n') for s in [exp_3,exp0,exp_2,exp_108]]
zs = [scrape_iter_times(lines) for lines in all_lines]

num_variables = [1318,12724,106932,1013170]

def iter_med_time(z):    
  times_, iters_ = z
  times = np.array(times_)
  iters = np.array(iters_)
  return np.median(times[1:]-times[:-1])

med_iter_times = np.array([iter_med_time(z) for z in zs])

def intWithCommas(x):
    if type(x) not in [type(0), type(0L)]:
        raise TypeError("Parameter must be an integer.")
    if x < 0:
        return '-' + intWithCommas(-x)
    result = ''
    while x >= 1000:
        x, r = divmod(x, 1000)
        result = ",%03d%s" % (r, result)
    return "%d%s" % (x, result)

fig = pl.figure(1,figsize=(5,1))
ax = fig.gca()
# 50 for each iteration, and 50 for increasing k
ax.loglog(num_variables,med_iter_times*50*50,'-o')
ax.xaxis.set_ticks(num_variables)
ax.xaxis.set_ticklabels([str(intWithCommas(n)) for n in num_variables])
ax.set_ylabel("Training time(seconds)")
ax.set_xlabel("Number of variables in the GMRF")
ax.set_xlim(1000,1.2e6)
save_fig('perf_gmrf.pdf')

