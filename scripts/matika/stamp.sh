#!/bin/sh
# Matika: write install-stamp.json before `docker build`, so the image reports a real version
# (e.g. 0.23.1+matika.22 = release 0.23.1 plus 22 Matika commits) instead of "unknown".
# Run from the repo root on the build host: sh scripts/matika/stamp.sh && docker compose build
set -e
tag=$(git describe --tags --abbrev=0 --match 'v[0-9]*' --exclude '*canary*' HEAD)
base=${tag#v}
n=$(git rev-list --count "$tag"..HEAD)
python3 scripts/write_install_stamp.py --output install-stamp.json --distribution docker \
  --update-mechanism external --source matika --commit "$(git rev-parse HEAD)" \
  --branch "$(git rev-parse --abbrev-ref HEAD)" --base-version "$base" \
  --display-version "$base+matika.$n" --distance "$n" --commit-date "$(git log -1 --format=%ct)"
