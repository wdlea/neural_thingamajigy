#!/usr/bin/env bash

# Format, then check with all possible combinations of features, finally execute unit tests
cargo fmt && cargo clippy && cargo clippy --all-features && cargo clippy --features train && cargo clippy --features serde && cargo t --all-features