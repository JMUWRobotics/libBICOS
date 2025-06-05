##  libBICOS: binary correspondence search on multishot stereo imagery
##  Copyright (C) 2024-2025  Robotics Group @ Julius-Maximilian University
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU Lesser General Public License as
##  published by the Free Software Foundation, either version 3 of the
##  License, or (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU Lesser General Public License for more details.
##
##  You should have received a copy of the GNU Lesser General Public License
##  along with this program.  If not, see <https://www.gnu.org/licenses/>.
##

project('libBICOS', 'cpp', version: '2.2.0', license: 'LGPL-3.0-or-later', license_files: ['COPYING', 'COPYING.LESSER'], meson_version: '>=1.1.0')

pkg = import('pkgconfig')
fs  = import('fs')
cc  = meson.get_compiler('cpp')

cppver = 'c++17'
override_options = [f'cpp_std=@cppver@']

impl = get_option('implementation')
srcs = files('src/lib.cpp', 'src/exception.cpp', 'src/formatable.cpp')
incls = include_directories('include')
deps = [
    dependency('opencv4'),
    dependency('fmt', version: '>=11.0.0')
]

is_debug = get_option('buildtype') in ['debug', 'debugoptimized']
if is_debug
    add_project_arguments('-DBICOS_DEBUG', language: ['cpp', 'cuda'])
endif

# complaints about opencv inlines
add_project_arguments('-Wno-deprecated-enum-enum-conversion', language: 'cpp')
add_project_arguments(
    '-Xcompiler=-Wno-deprecated-enum-enum-conversion',
    '--expt-relaxed-constexpr',
language: 'cuda')

cudasrcs = files()

if impl == 'cpu'
    add_project_arguments('-DBICOS_CPU', language: 'cpp')
    srcs += files('src/impl/cpu.cpp')
elif impl == 'cuda'
    add_languages('cuda', native: false)
    add_project_arguments('-DBICOS_CUDA', language: ['cpp', 'cuda'])
    override_options += f'cuda_std=@cppver@'
    cog = find_program('cog', required: true)
    cudasrcs += custom_target('cuda.cu',
        output: 'cuda.cu',
        input: 'src/impl/cuda.cu.in',
        command: [cog, '-o', '@OUTPUT@', '@INPUT@']
    )
    cudasrcs += files('src/impl/cuda/cutil.cu')

    srcs += cudasrcs

    has_builtin_uint128 = meson.get_compiler('cuda').compiles('''
        __device__ __uint128_t longlonglong() { return __uint128_t(0xBAADF00D); }
    ''', name: 'cuda __uint128_t', args: f'-std=@cppver@')

    if has_builtin_uint128
        add_project_arguments('-DBICOS_CUDA_HAS_UINT128', language: ['cpp', 'cuda'])
    else
        warning('CUDA runtime does not support 128-bit integers. Compiling without 128-bit support.')
    endif
else
    error('unsupported implementation')
endif

project_version = meson.project_version()

libs = both_libraries(
    'BICOS',
    sources: srcs,
    include_directories: incls,
    dependencies: deps,
    install: true,
    override_options: override_options,
    version: project_version.split('-')[0],
)

# enable static linkage as subproject
libbicos_dep = declare_dependency(
    include_directories: incls,
    dependencies: deps,
    objects: libs.get_static_lib().extract_all_objects(recursive: true)
)
meson.override_dependency('BICOS', libbicos_dep)

py = import('python').find_installation(pure: false)
pybind11 = dependency('pybind11', version: '>=2.0.0')
if py.found() and pybind11.found()
    py.extension_module('pybicos',
        'src/pybicos.cpp',
        install: true,
        dependencies: [pybind11, deps],
        link_with: libs.get_shared_lib(),
        include_directories: incls,
    )
endif

cxxopts = dependency('cxxopts', version: '>=3.0.0')
cli_tgt = executable(
    'bicos-cli',
    sources: ['src/cli.cpp', 'src/fileutils.cpp', 'src/exception.cpp'],
    include_directories: incls,
    dependencies: deps + cxxopts,
    link_with: libs,
    install: true,
    override_options: override_options,
)

if impl == 'cuda'
    datdir = join_paths(meson.project_source_root(), 'data')
    srcs = cudasrcs + files('src/impl/cpu.cpp', 'src/fileutils.cpp', 'src/exception.cpp', 'src/formatable.cpp')
    incls = [incls, include_directories('test/include')]

    depths = [32, 64]
    if has_builtin_uint128
        depths += 128
    endif

    foreach input_depth : [8, 16]
        foreach descriptor_depth : depths
            testname = f'descriptor_transform_@input_depth@_@descriptor_depth@'
            foreach limited : [0, 1]
                if limited == 0
                    testname = 'full_' + testname
                else
                    testname = 'limited_' + testname
                endif
                test(testname,
                    executable('test_' + testname,
                        sources: srcs + 'test/descriptor_transform.cu',
                        dependencies: deps,
                        include_directories: incls,
                        cuda_args: ['-DBICOS_CUDA', f'-DINPUT_TYPE=uint@input_depth@_t', f'-DDESCRIPTOR_TYPE=uint@descriptor_depth@_t', f'-DTRANSFORM_LIMITED=@limited@'],
                        cpp_args: '-DBICOS_CPU',
                        override_options: override_options,
                    )
                )
            endforeach
        endforeach
        foreach agree_cfg : [0, 1]
            if agree_cfg == 1
                testname = f'agree_subpixel_@input_depth@'
            else
                testname = f'agree_@input_depth@'
            endif
            test(testname,
                executable('test_' + testname,
                    sources: srcs + f'test/agree.cu',
                    dependencies: deps,
                    include_directories: incls,
                    cuda_args: ['-DBICOS_CUDA', f'-DINPUT_TYPE=uint@input_depth@_t', f'-DTEST_SUBPIXEL=@agree_cfg@'],
                    cpp_args: '-DBICOS_CPU',
                    override_options: override_options
                )
            )            
        endforeach
    endforeach

    foreach descriptor_depth : depths
        foreach variant : ['NODUPES', 'CONSISTENCY']
            testname = f'bicos_@descriptor_depth@_' + variant.to_lower()
            test(testname,
                executable('test_' + testname,
                    sources: srcs + 'test/bicos.cu',
                    dependencies: deps,
                    include_directories: incls,
                    cuda_args: ['-DBICOS_CUDA', f'-DDESCRIPTOR_TYPE=uint@descriptor_depth@_t', f'-DBICOS_VARIANT=BICOSFLAGS_@variant@'],
                    cpp_args: '-DBICOS_CPU',
                    override_options: override_options,
                )
            )
        endforeach
    endforeach

    if is_debug
        # TODO: investigate why (and how) release optimization messes with fp-computations
        test('integration',
            executable('test_integration',
                sources: srcs + 'test/integration.cu',
                dependencies: deps,
                include_directories: incls,
                cuda_args: '-DBICOS_CUDA',
                cpp_args: '-DBICOS_CPU',
                override_options: override_options,
            ),
            args: [datdir / 'left', datdir / 'right'],
            timeout: -1
        )
    endif

    test('cutil',
        executable('test_cutil',
            sources: srcs + 'test/cutil.cu',
            dependencies: deps,
            include_directories: incls,
            cuda_args: '-DBICOS_CUDA',
            cpp_args: '-DBICOS_CPU',
            override_options: override_options
        )
    )

    has_data = fs.is_dir(datdir / 'left') and fs.is_dir(datdir / 'right')
    if has_data
        foreach bicosflags: [0, 1, 2]
            testname = f'integration_raw_@bicosflags@'
            test(testname,
                executable('test_' + testname,
                    sources: srcs + 'test/integration_raw.cu',
                    dependencies: deps,
                    include_directories: incls,
                    cuda_args: ['-DBICOS_CUDA', f'-DBICOSFLAGS=@bicosflags@'],
                    cpp_args: '-DBICOS_CPU',
                    override_options: override_options,
                ),
                args: [datdir / 'left', datdir / 'right'],
                timeout: 150
            )
        endforeach

        if impl == 'cuda'
            test('regress_cuda',
                find_program('test/regress.bash'),
                depends: cli_tgt,
                workdir: meson.current_source_dir(),
                should_fail: not has_builtin_uint128,
                verbose: true
            )
        endif
    else
        message('Test data not available, run \'data/prepare.sh\' and reconfigure for regress tests.')
    endif
endif

googlebench = dependency('benchmark', required: false, not_found_message: 'library google/benchmark not found, skipping benchmark build.')
if googlebench.found()
    sourceroot = meson.project_source_root()
    deps += googlebench
    if impl == 'cuda'

        nvidia_smi = find_program('nvidia-smi', required: false)
        
        if nvidia_smi.found()
            available_gpus = run_command(nvidia_smi, '--query-gpu=gpu_name', '--format=csv,noheader', check: true).stdout().strip()
        else
            available_gpus = 'no-gpu-detected'
        endif

        benchmark('cuda',
            executable('bench_cuda',
                sources: srcs + 'bench/cuda.cu',
                dependencies: deps,
                include_directories: incls,
                cuda_args: f'-DSOURCE_ROOT="@sourceroot@"',
                override_options: override_options
            ),
            args: [f'--benchmark_context=available_gpus=@available_gpus@']
        )

        benchmark('cpu',
            executable('bench_cpu',
                sources: srcs + 'bench/cpu.cpp',
                dependencies: deps,
                include_directories: incls,
                override_options: override_options
            ),
        )
    endif
endif

configure_file(
    output: 'config.hpp',
    configuration: {
        'BICOS_VERSION': f'"@project_version@"',
        'BICOS_CPU': impl == 'cpu',
        'BICOS_CUDA': impl == 'cuda',
    },
    install: true,
    install_dir: 'include/BICOS',
)
install_headers('include/common.hpp', 'include/match.hpp', subdir: 'BICOS')
if pkg.found()
    pkg.generate(libs,
        name: 'libBICOS',
        version: project_version,
        description: 'GPU-accelerated library for binary correspondence search on multishot stereo imagery',
        requires: 'opencv4',
        url: 'https://github.com/JMUWRobotics/libBICOS',
        subdirs: 'BICOS'
    )
else
    message('pkg-config not found, skipping .pc generation.')
endif
