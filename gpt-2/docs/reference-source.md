# Reference checkout

The original code lives in a separate Git checkout at `gpt-2/build-nanogpt/`. It is ignored by the outer learning repository, so adding our lessons does not accidentally register an embedded repository without submodule metadata. The existing checkout has been left untouched.

- Upstream: [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt)
- Revision inspected when creating the scaffold: `6104ab1b53920f6e2159749676073ff7d815c1fa`
- Commit: `fix guard of torch.cuda based on device type`

On a fresh clone of this learning project, recreate the reference from the repository root:

```bash
git clone https://github.com/karpathy/build-nanogpt.git gpt-2/build-nanogpt
git -C gpt-2/build-nanogpt checkout 6104ab1b53920f6e2159749676073ff7d815c1fa
```

Those commands are for a missing/fresh reference checkout. To inspect an existing checkout without changing it, use `git -C gpt-2/build-nanogpt status` and `git -C gpt-2/build-nanogpt rev-parse HEAD`.

The original commit history is useful for studying the video progression. Our executable lessons live alongside the checkout. If a future lesson needs a newer upstream revision, record that change explicitly here.
