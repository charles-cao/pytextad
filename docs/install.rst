Installation
============

.. code-block:: bash

   pip install pytextad

PyTextAD requires Python 3.9 or later, PyTorch 1.13 or later and transformers 4.30 or
later. Install the PyTorch build that matches your CUDA version first, following
https://pytorch.org.

To install from source:

.. code-block:: bash

   git clone https://github.com/charles-cao/pytextad.git
   cd pytextad
   pip install -e ".[test]"

Models in a local folder
------------------------

Every class that loads a Hugging Face model takes ``cache_dir``, the folder used with
``from_pretrained(..., cache_dir=...)``. To work offline:

.. code-block:: bash

   export HF_HUB_OFFLINE=1            # PowerShell: $env:HF_HUB_OFFLINE = "1"

Set ``HF_ENDPOINT`` to download models and datasets through a mirror.
