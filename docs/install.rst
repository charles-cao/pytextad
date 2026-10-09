Installation
============

.. code-block:: bash

   pip install pytextad

From source:

.. code-block:: bash

   git clone https://github.com/charles-cao/pytextad.git
   cd pytextad
   pip install -e ".[test,docs]"

Install the PyTorch build that matches your CUDA version first (https://pytorch.org).

Using models from a local Hugging Face cache
--------------------------------------------

If your encoders were downloaded earlier with ``from_pretrained(..., cache_dir=...)``,
point Hugging Face to that folder and switch off network checks:

.. code-block:: bash

   export HF_HUB_CACHE=/path/to/cache   # PowerShell: $env:HF_HUB_CACHE = "D:\models"
   export HF_HUB_OFFLINE=1              # PowerShell: $env:HF_HUB_OFFLINE = "1"

In mainland China, ``HF_ENDPOINT=https://hf-mirror.com`` downloads through a mirror.
