# Contributing

Bug reports, measurements on other Macs and feature requests are welcome.

Pull requests are not accepted for now. This package is developed in a private repository and published here with
each release, so a pull request here cannot be merged. If you have a fix in mind, open an issue and describe it; a
code snippet in the issue is welcome.

## Reporting a bug

Open an issue with:

- what you ran and what happened (the oMLX log lines around the error help most);
- Mac model and memory, macOS version;
- `python3 install.py --omlx /path/to/omlx --check` output;
- your model settings (`moe_expert_offload_resident_fraction`) and environment variables.

For security problems, follow `SECURITY.md` instead of opening an issue.

## Sharing measurements

Post them in [Discussions](https://github.com/martinsbrezauckis/omlx-inkling2-offload/discussions), category
Show and tell: Mac model, memory, settings, prompt and generation tok/s, and how you measured.

## Asking for a feature

Post it in Discussions, category Ideas.

## License

By contributing you agree that your contribution is licensed under the MIT License of this project (`LICENSE`).
