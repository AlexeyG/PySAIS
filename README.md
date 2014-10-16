A Python C module wrapper for the [SA-IS algorithm implementation by Yuta Mori](https://sites.google.com/site/yuta256/sais).
Additionally includes an implementation of [a linear LCP construction algirthm](http://www.cs.helsinki.fi/u/tpkarkka/opetus/11s/spa/lecture10.pdf).

The idea to create a C module wrapper was inspyred by the [CTypes wrapper](https://github.com/davehughes/sais) from David Hughes.

Installation:
---------
```
./setup.py build
./setup.py install
```

Example:
------------
```python
import pysais
import numpy as np

sequence = 'mississippi$'
sa = pysais.sais(sequence)
lcp, lcp_lm, lcp_mr = pysais.lcp(sequence, sa)

for off in sa :
    print '%3d : %s' % (off, sequence[off:])

array = np.array([2, 3, 1, 0], dtype = np.int32)
sa_int = pysais.sais_int(array, 4)
lcp_int, lcp_lm_int, lcp_mr_int = pysais.lcp_int(array, sa_int)
```
