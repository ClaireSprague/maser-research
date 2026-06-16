# Synthetic Data: A Student Guide

## Why we do this

We have very few masers (53 in the X-ray sample, 114 in the WISE sample). If we
try lots of models and feature sets on the *real* data and then write up
whatever happened to score best, we will almost certainly be fooling ourselves:
with this little signal, something always looks good by luck. That is p-hacking,
and it is how a result that does not replicate ends up in a paper.

So we develop and tune the whole pipeline on *simulated* data, freeze the plan,
then look at the real data once. People do exactly this. Sky surveys build entire
simulated datasets to develop and validate their analysis pipelines before the
real data arrive (for example the LSST DESC DC2 simulated sky; LSST DESC 2021),
and in statistics, generating data with a known truth is the standard way to
evaluate a method before trusting it (Morris et al. 2019). `src/synth_data.py` is
our small version of that, and the sandbox where exploration is free, because
nothing you learn here can bias the real result.

## The one rule

**You may look at the real *features* all you want. You may never tune your
choices on the real *labels*.**

Looking at the distribution of L₁₂μm, the WISE colors, how they correlate, how
far away the galaxies are: all fine. None of that tells you which galaxy is a
maser, so it cannot bias you. The numbers baked into `synth_data.py` are exactly
these feature summaries. What you must not do is decide "interaction-LR is our
model" because it won the real-data bake-off. That decision gets made here, on
synthetic data, *before* you unblind.

`synth_data.py` enforces the important part structurally: it samples from real
feature columns only and then invents synthetic labels from the chosen scenario.
The real labels never define the boundary, the model choice, or the winner.


## What it can and cannot tell you

It **can** tell you about statistical behavior:
- How much do cross-validation scores wobble at our sample size?
- If the truth really is a wedge of a given strength, can our test even detect
  it? (This is statistical power.)
- Does our evaluation code leak (score too well on data where we know the
  truth)?
- How are each model's predicted probabilities distorted at 3 to 8 percent
  prevalence?

It **cannot** tell you which model the real astrophysics wants. The "truth" in a
simulation is whatever you put in. If you build the data with a wedge, of course
the wedge model wins. So always explore across several plausible truths and pick
an approach that holds up across all of them, never one that wins on a single
synthetic scenario.

## Where things stand (read before you start)

This guide includes a baseline simulation run in "What the baseline run shows"
below. Treat those results as **provisional**: each one came from a single set of
modeling choices (one signal strength, one shape for each scenario, default
seeds, mostly the X-ray plane). Every one of those is an arbitrary knob, and a
conclusion that only holds for one setting is not a conclusion.

Your job is to **stress-test and extend the baseline**: change the knobs, sweep
the strengths, try the WISE plane, and see which conclusions survive. The "Open
questions to chase" section lists the specific ones that matter. The reason this
is real work and not busywork: we are about to freeze a pre-registration on these
conclusions, and freezing on one unexamined run is exactly the mistake the whole
blind-analysis exercise is meant to prevent. A second pair of hands on the
assumptions is the safeguard.

## Start here: the notebook

The fastest way in is **`notebooks/synthetic_data_walkthrough.ipynb`**. It plots
the synthetic datasets and runs the initial analysis with commentary. Open it,
run it top to bottom, then start changing things.

## The two modules

- **`src/synth_data.py`** makes the data. `make_dataset(...)` and
  `make_fused_dataset(...)` return a **pandas DataFrame** whose feature and
  target columns are named exactly as in the real data (`maser_data.py`), plus
  synthetic-only `z_true` and `detected`.
- **`src/synth_analysis.py`** is the harness that evaluates models. It selects
  columns by name, so the same calls run on synthetic *and* real frames.

Run the built-in demos from the repo root:

```
python src/synth_data.py       # dataset summaries
python src/synth_analysis.py   # the initial method-comparison study
```

In a notebook or script:

```python
import sys, os; sys.path.insert(0, os.path.abspath("src"))
import synth_data as sd
import synth_analysis as sa
from maser_data import FEATURES, TARGET

df = sd.make_dataset(plane="xray", scenario="wedge", seed=0)   # a DataFrame
X, y = df[FEATURES["xray"]], df[TARGET["xray"]]   # SAME call works on real data
df.head()
```

### The knobs (on `make_dataset`)

- `plane`: `"xray"` (2 features, ~8% positives, n=641) or `"wise"` (2 colors,
  ~3% positives, n≈4400). These match the real datasets.
- `scenario`: the **true** boundary you are simulating.
  - `"linear"`: positives on one side of a slanted line. A plain LR is enough.
  - `"wedge"`: positives need both risk coordinates high, graded (soft corner).
  - `"box"`: a hard jump where both risk coordinates are above their mean.
  - `"interaction"`: a genuine multiplicative interaction; the effect of one
    feature grows with the other. This is the case matched to interaction-LR.
  - `"blob"`: positives form a central island. Needs quadratic terms; an
    interaction term alone cannot enclose it.
- Risk coordinates are oriented a priori: for X-ray, high risk means high
  `L_12um_bestfit_1` and low `Lob`; for WISE, high risk means high `w1w2` and
  high `w2w3`. This keeps the synthetic positives in the scientifically
  expected quadrant without fitting the boundary to real labels.
- `strength`: how strong the signal is (bigger = easier to separate). Use it to
  ask "how big would the effect have to be before we could detect it?"
- `feature_model`: how feature positions are drawn.
  - `"bootstrap"` (default): resample real feature rows with replacement and add
    a little jitter. This preserves skew, clumps, bounds, holes in the 2-D plane,
    and feature-distance associations without using real labels. It is the most
    faithful small-sample mimic, but at huge `n` it can show repeated jittered
    copies of the same empirical rows.
  - `"gmm"`: fit a Gaussian mixture to the real feature rows plus log-distance,
    then sample from that smooth density. This is still label-blind, and is the
    better default for large synthetic population/effect-ceiling plots where a
    smooth density is useful.
  - `"gaussian"`: draw from a correlated-Gaussian summary. This is useful as a
    deliberately simple sensitivity check and for seeing whether conclusions
    depend on the feature-density model.
- `jitter`: the bootstrap smoothing scale as a fraction of each feature's real
  standard deviation. Default is `0.03`; set `0` for an exact row bootstrap.
- `distance_noise`: if `True`, distant true masers can go undetected. The column
  `z_true` is the real label, the target column (`is_megamaser_plus` etc.) is
  what you would actually observe, and `detected` flags which true masers were
  seen. Comparing them shows the bias.

For the **fusion** question (does adding WISE help X-ray?) use
`make_fused_dataset(scenario=..., wise_signal=...)`. It returns 4 feature columns
`[L_12um_bestfit_1, Lob, wise_w1w2, wise_w2w3]`. The `wise_signal` knob controls
how much of the truth lives in the part of the WISE colors that X-ray cannot
reproduce: `0.0` is "redundant" (WISE adds nothing), larger values "add signal."

### Specifying a comparison

A comparison is two **arms**, each a `(model_builder, feature_columns)` tuple:

- *model* comparison (same columns, different model):
  `(sa.interaction_lr, FEATURES["xray"])` vs `(sa.plain_lr, FEATURES["xray"])`.
- *feature* comparison (same model, different columns):
  `(sa.plain_lr, FEATURES["fused"])` vs `(sa.plain_lr, FEATURES["xray"])`.

A **maker** turns a seed into a DataFrame: `sa.maker("xray", scenario="wedge")`
or `sa.fused_maker(scenario="linear", wise_signal=1.5)`. Models available:
`plain_lr`, `interaction_lr`, `quadratic_lr`, `rf` (random forest), `gbt`
(boosted trees).

### The analysis helpers (in `synth_analysis`)

`effect_ceiling(make_big, arm_a, arm_b, target)` fits both arms on a huge sample,
so noise is gone, and reports how much one truly beats the other. Near zero means
there is nothing for any finite-sample test to find.

```python
XF, XT = FEATURES["xray"], TARGET["xray"]
big = sa.maker("xray", scenario="blob", n=40000, feature_model="gmm")
sa.effect_ceiling(big, (sa.quadratic_lr, XF), (sa.plain_lr, XF), XT)
# -> large positive dAUC   (the blob genuinely needs the quadratic)
```

`power_study(make, arm_a, arm_b, target)` repeats the full cross-validation
comparison over many simulated datasets and reports how often the decision rule
fires (adopt the more complex arm only if the corrected CI excludes zero *and*
the gain is at least 2 extra masers in the top-k list):

```python
sa.power_study(sa.maker("xray", scenario="interaction"),
               (sa.interaction_lr, XF), (sa.plain_lr, XF), XT, n_sims=50)
# -> {'rate': 0.0, ...}   power to detect the interaction at n=641
```

`optimism(build, make, features, target)` reports resubstitution AUC minus honest
CV AUC: how much a model flatters itself on the data it was trained on. It grows
with model flexibility. `k_sensitivity(build, make, features, target)` reports
the across-fold spread of precision@k, so you can see how noisy a "top-50"
operating point is at our n.

## What the baseline run shows (provisional)

The demo prints a small study. Here is what that baseline configuration shows
and why it matters. Treat each as a hypothesis to confirm or break, not a
settled fact: every number below rests on the arbitrary modeling choices flagged
above.

1. **The nonlinear X-ray shapes have real large-n effects.** In the X-ray risk
   coordinates (high L12, low observed X-ray luminosity), interaction-LR beats
   plain LR at n=40,000 for the wedge (ΔAUC ≈ +0.084 with the bootstrap sampler,
   ≈+0.088 with GMM), for the hard box (ΔAUC ≈ +0.026), and for the explicit
   interaction (ΔAUC ≈ +0.091). The hard box uses above-mean risk coordinates,
   covers roughly 10-12 percent of the X-ray cloud, and contains about 60
   percent of positives in a large synthetic draw. The blob is the strong
   positive-control nonlinearity (quadratic LR ΔAUC ≈ +0.40).

2. **Those effects are detectable at n=641 under the pre-registered rule.** With
   the decision rule (CI excludes zero and at least +2 masers in a full-sample
   **top-50 campaign**), interaction-LR vs plain-LR fires at n=641 with power
   roughly 0.7 for the wedge, 0.8 for the hard box, and 0.4 for the explicit
   interaction at strength 3, with mean effects of +3 to +5 masers in a top-50
   list; the blob positive control is ~1.0 and the linear null is ~0. So under
   realistic features these comparisons are worth keeping. The effect is measured
   at the full-sample top-50 campaign scale, not a per-fold top-10; those differ
   by the number of folds, so the campaign size has to be stated exactly (it is,
   in the pre-registration).

3. **The fusion question behaves much better.** Adding WISE colors to the X-ray
   features helps not at all when WISE is redundant (`wise_signal=0`: ceiling
   ΔAUC ≈ 0, power ≈ 0), and is detected reliably when WISE carries real
   independent signal (`wise_signal=1.5`: ceiling ΔAUC ≈ +0.21, power ≈ 1.0 at
   n=602). So "does WISE add information?" is a comparison worth running: it can
   actually come back positive.

4. **Flexible models flatter themselves, boosting most of all.** Optimism (resub
   AUC minus honest CV AUC) was ≈+0.001 for plain LR, ≈+0.005 for interaction
   LR, ≈+0.158 for the random forest, and ≈+0.167 for boosted trees in the
   baseline bootstrap run. RF is a bit less optimistic than boosting, but both
   ensembles look much more overfit than the logistic models under the
   interaction truth.

5. **The harness is not broken.** The positive control (a genuinely nonlinear
   blob, where the quadratic truly wins) fires with power ≈1.0, and the
   false-positive rate under a linear truth is ≈0. When there is a large effect
   the test finds it; when there is not, it stays quiet. That is what lets us
   trust a null result.

The takeaway is not "give up." It is: spend the confirmatory comparisons where
the simulation says we have a chance, set the effect-size threshold and the
top-k where they are meaningful, and frame the likely "can't tell" outcomes
honestly in the pre-registration. Use the open questions below to check whether
these readings hold up.

## Open questions to chase

These are the specific things the baseline run does *not* settle. Pick one and
see whether the provisional findings survive. Each should be checked across a
few parameterizations, not one.

1. **How robust are the nonlinear-shape results?** Try sharper/softer corners, a
   non-monotone boundary, and multiple feature models
   (`feature_model="bootstrap"`, `"gmm"`, and `"gaussian"`). If a result only
   appears for one shape or one feature sampler, it is not robust enough to
   drive the pre-registration.
2. **Strength sweeps, not single points.** At what interaction `strength` does
   the interaction-vs-plain test start to have power? At what `wise_signal` does
   the fusion test start firing? Those curves are what justify the +2-maser
   threshold and the choice of k, instead of asserting them.
3. **The WISE plane (114 positives) is barely tested.** Repeat the power and
   ceiling studies with `plane="wise"`. More positives may change which
   comparisons are worth keeping confirmatory.
4. **Distance noise on each comparison.** Turn on `distance_noise` and see how
   much it shifts power and the apparent effect, for both the model and the
   feature comparisons. Compare a "well-searched" subsample to the full set.
5. **Is top-50 the right operating point?** Use `k_sensitivity` across planes
   and scenarios to decide where precision@k stops being dominated by fold luck.
6. **Calibration.** The harness uses ranking metrics. If we will report
   probabilities, add a reliability/Brier check and see how RF vs LR compare at
   our prevalence (the base-rate compression story).

When a conclusion holds across several settings, it is safe to write into the
pre-registration. When it does not, that instability is itself the finding, and
it means the data cannot support that decision.

## The workflow

1. Hack here. Start from the baseline run, then change the knobs: models,
   features, scenarios, strengths, distance noise, planes. Work the open
   questions above.
2. Keep only the conclusions that are *robust across scenarios* and that the
   power studies say are *detectable*.
3. Write those decisions into `docs/PREREGISTRATION.md` and freeze it.
4. Only then run the frozen pipeline on the real data, once.

## Cautions

- **Explore several scenarios, not one.** A choice that only wins under one
  synthetic truth is not robust.
- **Do not fit the true boundary to the real labels.** Match the simulator to
  the real *features* (already done), invent the boundary yourself.
- **Feature realism is not truth realism.** The default bootstrap makes the
  synthetic cloud look like the real feature cloud, but the label boundary is
  still invented. Do not tune scenarios until your favorite model wins.
- **Check multiple feature samplers.** If a conclusion flips between bootstrap,
  GMM, and Gaussian features, that instability is itself important and should
  temper the preregistration claim.
- **Do not rabbit-hole.** "Plausible and spanning a few truths" is the bar, not
  "perfectly realistic." The simulator's job is to calibrate our expectations
  and our plan, then get out of the way.

## References

- LSST Dark Energy Science Collaboration, Abolfathi, B., et al. (2021). The LSST
  DESC DC2 Simulated Sky Survey. *The Astrophysical Journal Supplement Series*,
  253, 31. [arXiv:2010.05926](https://arxiv.org/abs/2010.05926). A simulated sky
  built to develop and validate survey analysis pipelines before real data.
- Morris, T. P., White, I. R., & Crowther, M. J. (2019). Using simulation studies
  to evaluate statistical methods. *Statistics in Medicine*, 38, 2074–2102.
  [doi:10.1002/sim.8086](https://doi.org/10.1002/sim.8086). The standard guide to
  designing a simulation study to choose and validate a method.
