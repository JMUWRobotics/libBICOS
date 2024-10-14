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

## Usage

### Linking
After installing, you can include `/usr/local/include/BICOS/*` and link against `/usr/local/lib/libBICOS.so`.

### Commandline-interface
Alternatively, this project builds `bicos-cli`. To give you a feel for the parameters of BICOS, there is an example dataset provided in [data](/data);
Calling:
```console
$ bicos-cli data/{left,right} -q data/Q.cvstore --threshold 0.98 --stacksize 12 --variance 2.5 --step 0.1 -o /tmp/result.png
```
will get you:

#### Disparity
![Example disparity](/example-disp.png)

#### Pointcloud:
![Example pointcloud](/example-pcl.png)

While only requiring ~70ms (RTX4090) and ~1.5GB VRAM for matching on two stacks of 12 images each.

The most significant parameters can be summarized:

- `--threshold`: increasing this reduces match-outliers. High values rely on a reliable pattern.
- `--variance`: high values reduce coarse outliers, e.g. pixels where no pattern was projected. May reduce correct matches on low pattern contrast.
- `--step`: optional value for subpixel-interpolation around matches.

Other settings are available; Try `bicos-cli -h` for details.

## Benchmarking:

```console
$ meson test --benchmark -C builddir --interactive
```

## License

This library is licensed under the GNU Lesser General Public License v3.0 (or later).
Please see [COPYING](/COPYING) and [COPYING.LESSER](/COPYING.LESSER) for details.

![LGPL-3.0-Logo](https://www.gnu.org/graphics/lgplv3-147x51.png)
