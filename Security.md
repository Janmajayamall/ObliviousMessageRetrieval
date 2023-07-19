# Security

> **Note:** Source code of OMR has not been audited and cannot guarantee security against implementation bugs. The security below is only concerned with parameter security of the schemes used in OMR.

LWE parameters used by OMR has security of **122 bits**

LWE parameters used by PVW scheme has security of **120 bits**.

## OMR security

To verify the security of OMR you can use [lattice estimator](https://github.com/malb/lattice-estimator).

The parameters we use for OMR are as following:

$n = 2^{15}$ <br/>
$logQ = 760$ <br/>
$logP = 160$ <br/>
$logQP = 920$ <br/>

The secret key is sampled from sparse ternary distribution with hamming weight $n/2$.

To check security we will be using the cost model used by [HES](http://homomorphicencryption.org/wp-content/uploads/2018/11/HomomorphicEncryptionStandardv1.1.pdf), that is `RC.BDGL16`.

To verify the security, first install sage and clone lattice estimator. The run the following sage script:
Warning: The script can take a few hours to finish.

```
from estimator import *

omr920 = LWE.Parameters(
    n=2**15,
    q=2**920,
    Xs=ND.SparseTernary(2**15,2**13,2**13),
    Xe=ND.DiscreteGaussian(3.19),
    m=2**15,
    tag="OMR920",
)

res = LWE.estimate(omr920, red_cost_model = RC.BDGL16)
print(res)
```

The script should output the following output

```
usvp                 :: rop: ≈2^122.4, red: ≈2^122.4, δ: 1.004864, β: 298, d: 63052, tag: usvp
bdd                  :: rop: ≈2^122.2, red: ≈2^122.1, svp: ≈2^117.1, β: 297, η: 345, d: 65384, tag: bdd
bdd_hybrid           :: rop: ≈2^122.2, red: ≈2^122.1, svp: ≈2^117.1, β: 297, η: 345, ζ: 0, |S|: 1, d: 65537, prob: 1, ↻: 1, tag: hybrid
bdd_mitm_hybrid      :: rop: ≈2^181.6, red: ≈2^180.6, svp: ≈2^180.6, β: 297, η: 2, ζ: 135, |S|: ≈2^180.2, d: 65402, prob: ≈2^-56.3, ↻: ≈2^58.5, tag: hybrid
dual                 :: rop: ≈2^122.4, mem: ≈2^33.8, m: ≈2^15.0, β: 298, d: 65536, ↻: 1, tag: dual
dual_hybrid          :: rop: ≈2^122.4, mem: ≈2^106.2, m: ≈2^15.0, β: 298, d: 65475, ↻: 1, ζ: 61, tag: dual_hybrid
```

`rop` indicates estimated runtime of different LWE attacks on LWE parameters of OMR. As visible, for each attack `rop` is atleast 2^122.

## PVW Security

TODO
