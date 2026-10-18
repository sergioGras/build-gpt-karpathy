# Remote compute plan for the GPT tutorial

This document adapts the attached RunPod playbook to this repository. It is a planning guide: no RunPod resources or credentials have been configured here, and no cloud run has been started.

## Learning path from Lesson 8

The first tutorial's next source-code pieces assemble the ideas already introduced:

1. Build a causal self-attention head with learned query, key, and value projections.
2. Combine heads, add the feed-forward network, residual connections, and the pre-normalized Transformer block.
3. Add token and position embeddings as the input representation.
4. Assemble the decoder model from stacked blocks, produce next-token logits, and generate text.
5. Train the small character-level model on Tiny Shakespeare, recording train/validation loss and generated samples.

The first full-model training lesson uses a reduced two-layer, 16-channel configuration and 100 optimizer updates. It runs locally on CPU; a GPU is reserved for a later scale comparison after the learner has inspected the training curve and measured a larger run.

These are small teaching examples. The reference notebook uses a short context, a 64-channel embedding, four heads, and four layers. We should first run the end-to-end model on the local CPU and measure it. A GPU is useful when it gives a clear comparison or shortens an experiment, but it is not a prerequisite for understanding the architecture.

Keep a later GPT-2 track distinct from this from-scratch exercise. The smallest standard GPT-2 checkpoint has 124 million parameters, uses byte-level BPE with a 50,257-token vocabulary, and was trained with context length 1,024. A useful progression is to load that pretrained checkpoint for inference, inspect its tokenizer and generation, then try a small fine-tuning experiment. Pretraining a full GPT-2 from scratch is a substantially different compute and data project. See the [GPT-2 model card](https://huggingface.co/openai-community/gpt2) for its specification.

## Which RunPod workflow fits

For lessons that need a remote interactive machine, start with a RunPod **Pod** using a pinned PyTorch image and connect from VS Code over SSH. RunPod currently documents SSH for long-running processes and direct VS Code/Cursor connections. Prefer Secure Cloud when public SSH access is required; Community Cloud does not guarantee a public IP. A local VS Code window can still edit and run the same notebook in the remote Linux environment.

Use a Pod for training, debugging, plotting, and inspecting checkpoints. A Serverless endpoint is a better later option for demonstrating a request/response inference API; it does not provide the same interactive development session. Check live GPU stock, hourly prices, image compatibility, and datacenter availability for each run instead of pinning a price in this document.

For an initial short run, avoid persistent storage if source, corpus, and results can be transferred in and out. Use a network volume only when repeated model downloads or checkpoints justify its ongoing storage cost. Network volumes are separate billable resources, are currently available to Secure Cloud Pods, and constrain Pod placement to their datacenter. Stopping a Pod does not stop network-volume charges.

## Reuse from the attached playbook

Reuse these workflow patterns:

- Give each experiment a question, unique run ID, source revision, configuration, expected outputs, and a time limit.
- Upload an explicit allowlist of source and data files; record hashes so the remote inputs can be checked.
- Check the live configuration and price before launch. Set a maximum hourly rate, total spend, runtime, instance count, retry count, and storage allowance.
- Permit one controller to manage an active run. If a create request times out, look up the unique Pod name and reconcile its state before attempting another create.
- Capture logs and machine-readable training metrics such as step, elapsed time, train loss, and validation loss. Retrieve plots and checkpoints before deleting ephemeral storage.
- Delete the Pod and verify it is gone. If a network volume was used, review and delete that separately when it is no longer needed.
- Keep API keys, SSH private keys, and Git credentials on the controlling computer. Never include them in uploads or notebook outputs.

Do not carry over the other project's Pod IDs, hashes, budget files, price quotes, approvals, experiment inputs, or controller implementation. Its playbook describes one tested pilot and notes that its controller still needs integrated remote validation. Rebuild and test this project's launch path against the provider's current API/CLI and the exact workload before relying on it.

## Spending and shutdown controls

Before any paid run, prepare a short owner-reviewed run card with the exact GPU/CPU shape, image, region flexibility, current price, storage plan, maximum runtime and total spend, outputs to upload, and cleanup method. A provider token grants access; it does not set a per-experiment budget or authorize an unspecified rental.

Use a verified provider-side expiry when the selected Pod workflow supports it. Otherwise, require a tested local watchdog that remains powered on and online, plus a separate cleanup path and deletion verification. Do not rely on disconnecting SSH, stopping the guest process, or a low-balance alert as proof that a Pod is deleted or as a hard experiment budget. RunPod bills compute and storage separately, and persistent network storage can continue billing while compute is stopped.

Before launch automation is implemented, its local tests should cover duplicate-create prevention, timeout reconciliation, wrong-resource detection, watchdog failure, deletion retries, and confirmation that no unexpected resources remain. Keep raw logs and local state out of Git; commit only reviewed source, experiment configuration, and shareable summaries.

## First practical cloud experiment

After the Transformer and full-model lessons run locally, make one short, bounded GPU comparison on the Tiny Shakespeare training loop. Upload only the needed lesson code and corpus, write periodic metrics to a small JSONL or CSV file, and collect a plot and checkpoint locally. That experiment will tell us whether this tutorial benefits from the GPU and validate the remote workflow before we consider a larger GPT-2 checkpoint.

Provider references: [RunPod Pod connections](https://docs.runpod.io/pods/connect-to-a-pod), [Pod CLI and lifecycle](https://docs.runpod.io/runpodctl/reference/runpodctl-pod), [billing](https://docs.runpod.io/accounts-billing/billing), and [network volumes](https://docs.runpod.io/storage/network-volumes).
