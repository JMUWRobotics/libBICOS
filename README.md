# libBICOS

BInary COrrespondence Search for multi-shot stereo imaging, with optional CUDA acceleration.

## Build:

Dependencies:

- `gcc` or equivalent C compiler with C++20 support (build)
- [`meson`](https://github.com/mesonbuild/meson) >= 1.1.0 (build)
- `opencv 4.x`
- `cuda 12.x` including toolkit 
- [`cxxopts`](https://github.com/jarro2783/cxxopts) (optional, for cli)
- [`benchmark`](https://github.com/google/benchmark) (optional, for executing benchmarks)

```bash
$ git clone https://github.com/JMUWRobotics/libBICOS
$ cd libBICOS

# optional but recommended, pick a release depending on your needs
$ git checkout v1.1.0

$ meson setup builddir --buildtype release

# optional, if you do not have access to a CUDA-capable GPU
$ meson configure -Dimplementation=cpu builddir

$ meson install -C builddir
```

The versioning scheme of the library is [Semantic Versioning](https://semver.org/).

## Benchmarking:

```console
$ meson test --benchmark -C builddir --interactive
```

## License

This library is licensed under the GNU Lesser General Public License v3.0 (or later).
Please see [COPYING](/COPYING) and [COPYING.LESSER](/COPYING.LESSER) for details.

![LGPL-3.0-Logo](https://www.gnu.org/graphics/lgplv3-147x51.png)
