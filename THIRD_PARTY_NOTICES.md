# Third-party notices

## CVDD (pytextad/models/cvdd.py)
Re-implemented from https://github.com/lukasruff/CVDD-PyTorch

MIT License

Copyright (c) 2019 lukasruff

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## RSRAE (pytextad/models/embedding/rsrae.py)
Ported to PyTorch from https://github.com/dmzou/RSRAE

MIT License

Copyright (c) 2019-present Chieh-Hsin Lai, Dongmian Zou and Gilad Lerman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## DATE (pytextad/models/date.py)
Re-implemented from https://github.com/bit-ml/date, whose model code lives in a
modified copy of simpletransformers distributed under the Apache License 2.0.
No source code is copied; the logic was re-written against modern transformers.

## FATE (pytextad/models/fate.py)
Re-implemented from https://github.com/arav1ndajay/fate, which has no licence file.
No source code is copied; the logic was re-written and checked numerically against it.

## ADERH (pytextad/models/embedding/_vendor/aderh.py)
Copied unchanged from https://github.com/Walid10010/ADERH (aderh/_aderh.py).

MIT License

Copyright (c) 2025-2026 Walid Durani

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## TCCM (pytextad/models/embedding/_vendor/tccm.py)
Copied from https://github.com/ZhongLIFR/TCCM-NIPS (FMAD/functions.py, FMAD/FlowMatchingAD.py),
with one line added to move training data to the model's device. The TCCM repository is
released under the Creative Commons Attribution-ShareAlike 4.0 licence
(https://creativecommons.org/licenses/by-sa/4.0/); this file is distributed under that
licence, not under PyTextAD's BSD 2-Clause licence. Authors: Zhong Li, Qi Huang, Yuxuan Zhu,
Lincen Yang, Mohammad Mohammadi Amiri, Niki van Stein, Matthijs van Leeuwen.

## Embedding baselines (pytextad/models/embedding/_vendor/)
Each file starts with its full source chain. Licences of the sources:

* dte.py, dte_nonparametric.py, ddpm.py: https://github.com/vicliv/DTE, MIT License,
  Copyright (c) 2023 Victor Livernoche (licence text as for ADERH above, with this notice).
* ddae.py: https://github.com/sattarov/AnoDDAE, MIT License, Copyright (c) 2025 Timur Sattarov.
* dagmm.py, ganomaly.py, drocc.py, goad.py, icl.py, mcm.py, slad.py, normalizing_flow.py: copies
  from https://github.com/ZhongLIFR/TCCM-NIPS (baselines/), released under CC BY-SA 4.0; these
  files are distributed under CC BY-SA 4.0. Their earlier sources: ADBench (BSD-2-Clause) for
  DAGMM and GANomaly; the DTE repository (MIT) for DAGMM, DROCC, GOAD, ICL, SLAD and the flow;
  official code of DROCC (microsoft/EdgeML, MIT), SLAD (xuhongzuo/scale-learning, MIT), ICL
  (OpenReview supplementary material, no licence) and MCM (OpenReview).
* goad.py additionally derives from https://github.com/lironber/GOAD, under the Yissum
  SOFTWARE RESEARCH LICENSE, which permits research use only and excludes commercial use.
  GOAD in PyTextAD is for research use only.
* drl.py: rewrite by Yang Cao of https://github.com/HangtingYe/DRL (no licence file; used
  with the author's agreement).
