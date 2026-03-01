GPT From Scratch - Notebook Naming Reference
=========================================

Purpose
-------
This file records model-variable renames in `Notebooks/03_Architecture.ipynb`
so they can be documented in `README.md` later.

Model Variable Renames
----------------------
- model -> dummy_model         (DummyGPTModel demo)
- model -> gpt_model           (GPT architecture part 6 test)
- model -> gpt_block_model     ("all blocks" GPT section)
- model -> trained_gpt         (training loop + decoding section)
- model -> saved_model         (save weights demo)
- model -> loaded_model        (load weights demo)
- model -> ckpt_model          (checkpoint load demo)
- gpt1  -> gpt_pretrained      (pretrained GPT-2 usage)
- model2 -> gpt_classifier     (classification fine-tuning model)
- new_model -> classifier_reloaded (reloaded classifier model)

Additional Fix
--------------
- The previous load cell in the spam-classifier section was made explicit and safe:
  `review_model` is now instantiated with the correct classification head
  before loading `../Weights/review_classifier.pth`.

Notes for README
----------------
- Main reason for these renames: avoid accidental overwrites across notebook sections.
- Result: each section now has a distinct model variable with clearer intent.
