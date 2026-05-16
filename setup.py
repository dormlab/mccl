"""
MCCL build script — Distributed Metal GPU Runtime.

Builds the C++/Obj-C++ extension with the DMEM layer, coherence protocol,
and Distributed Metal Runtime on macOS Apple Silicon.
"""
import os
import platform
import shutil
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _shaders_metal_path() -> str:
    for parts in (
        ("csrc", "metal", "shaders.metal"),
        ("mccl", "shaders.metal"),
    ):
        p = os.path.join(_REPO_ROOT, *parts)
        if os.path.isfile(p):
            return p
    return os.path.join(_REPO_ROOT, "csrc", "metal", "shaders.metal")


def _torch_include_dirs():
    import sysconfig
    import torch
    from torch.utils.cpp_extension import include_paths
    torch_root = os.path.join(os.path.dirname(torch.__file__), "include")
    python_inc = sysconfig.get_path("include")
    return include_paths() + [python_inc]


def _torch_library_dirs():
    from torch.utils.cpp_extension import library_paths
    return library_paths()


class MCCLBuildExt(build_ext):
    """Custom build_ext that handles mixed .cpp/.mm compilation on macOS."""

    def build_extensions(self):
        if platform.system() != "Darwin":
            raise RuntimeError(
                "MCCL can only be built on macOS. "
                "This machine reports platform: " + platform.system()
            )

        arch = platform.machine()
        if arch not in ("arm64", "aarch64"):
            raise RuntimeError(
                f"MCCL requires Apple Silicon (arm64). Detected: {arch}"
            )

        sdk_path = subprocess.check_output(
            ["xcrun", "--show-sdk-path"], text=True
        ).strip()

        for ext in self.extensions:
            ext.extra_compile_args = ext.extra_compile_args or []
            ext.extra_link_args = ext.extra_link_args or []

            cpp_flags = [
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Wno-unused-parameter",
                "-fvisibility=hidden",
                "-DDISTRO_BUILD",
                "-march=armv8.5-a+crc",
                "-isysroot", sdk_path,
            ]
            objcpp_flags = cpp_flags + ["-fobjc-arc"]

            ext._cpp_flags = cpp_flags
            ext._objcpp_flags = objcpp_flags

            ext.extra_link_args += [
                "-framework", "Metal",
                "-framework", "Foundation",
                "-framework", "MetalPerformanceShaders",
                "-framework", "Accelerate",
                "-isysroot", sdk_path,
            ]

        super().build_extensions()

    def build_extension(self, ext):
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")

        sources_cpp = [s for s in ext.sources if s.endswith(".cpp")]
        sources_mm = [s for s in ext.sources if s.endswith(".mm")]

        objects = []

        for src in sources_cpp:
            flags = ext._cpp_flags
            if src.endswith("bindings.cpp"):
                flags = [f for f in flags if f != "-fvisibility=hidden"]
            obj = self._compile_single(src, flags, ext)
            objects.append(obj)

        for src in sources_mm:
            obj = self._compile_single(src, ext._objcpp_flags, ext)
            objects.append(obj)

        ext.extra_link_args += [
            f"-L{torch_lib}",
            "-ltorch",
            "-ltorch_cpu",
            "-ltorch_python",
            "-lc10",
            f"-Wl,-rpath,{torch_lib}",
            "-undefined", "dynamic_lookup",
        ]

        self._link_shared_object(objects, ext)

    def _compile_single(self, src, flags, ext):
        obj = src + ".o"
        obj_path = os.path.join(self.build_temp, obj)
        os.makedirs(os.path.dirname(obj_path), exist_ok=True)

        include_flags = []
        for d in _torch_include_dirs() + (ext.include_dirs or []):
            include_flags += ["-I", d]

        cmd = ["clang++"] + flags + include_flags + ["-c", src, "-o", obj_path]

        self.announce(f"Compiling {src}", level=2)
        subprocess.check_call(cmd)
        return obj_path

    def _link_shared_object(self, objects, ext):
        ext_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)

        cmd = (
            ["clang++", "-shared", "-o", ext_path]
            + objects
            + (ext.extra_link_args or [])
        )

        self.announce(f"Linking {ext_path}", level=2)
        subprocess.check_call(cmd)

        self._fixup_rpath(ext_path)
        out_dir = os.path.dirname(ext_path)
        self._install_shaders_metal_for_ext(ext, out_dir)
        self._compile_metallib(ext, out_dir)

    @staticmethod
    def _fixup_rpath(ext_path):
        import torch
        runtime_torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        result = subprocess.run(
            ["otool", "-l", ext_path], capture_output=True, text=True
        )
        for i, line in enumerate(result.stdout.splitlines()):
            if "path" in line and "torch" in line and "pip-build-env" in line:
                stale = line.strip().split()[1]
                subprocess.check_call([
                    "install_name_tool", "-delete_rpath", stale, ext_path
                ])
        subprocess.run([
            "install_name_tool", "-add_rpath", runtime_torch_lib, ext_path
        ], capture_output=True)

    @staticmethod
    def _detect_metal_std():
        mac_ver = platform.mac_ver()[0]
        if mac_ver:
            major = int(mac_ver.split(".")[0])
            if major >= 15:
                return "metal3.1"
        return "metal3.0"

    def _shader_bundle_dest_dirs(self, ext, primary_out_dir):
        dirs = [os.path.normpath(primary_out_dir)]
        parts = ext.name.split(".")
        if len(parts) >= 2:
            pkg_path = os.path.normpath(os.path.join(_REPO_ROOT, *parts[:-1]))
            if os.path.isdir(pkg_path):
                dirs.append(pkg_path)
        seen_real = set()
        unique = []
        for d in dirs:
            try:
                key = os.path.realpath(d)
            except OSError:
                key = d
            if key not in seen_real:
                seen_real.add(key)
                unique.append(d)
        return unique

    def _install_shaders_metal_for_ext(self, ext, primary_out_dir):
        shader_src = _shaders_metal_path()
        if not os.path.isfile(shader_src):
            raise RuntimeError(
                "MCCL build requires shaders.metal at "
                f"{os.path.join(_REPO_ROOT, 'csrc', 'metal', 'shaders.metal')}"
            )
        for d in self._shader_bundle_dest_dirs(ext, primary_out_dir):
            os.makedirs(d, exist_ok=True)
            dst = os.path.join(d, "shaders.metal")
            shutil.copy2(shader_src, dst)
            self.announce(f"Installed shaders.metal next to extension: {dst}", level=2)

    def _compile_metallib(self, ext, output_dir):
        shader_src = _shaders_metal_path()
        if not os.path.isfile(shader_src):
            raise RuntimeError(f"MCCL build requires {shader_src} in the source tree.")

        require_metallib = os.environ.get("MCCL_REQUIRE_METALLIB", "").strip().lower() in (
            "1", "true", "yes",
        )

        try:
            subprocess.check_output(
                ["xcrun", "--find", "metal"], text=True, stderr=subprocess.STDOUT
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            msg = (
                "Metal shader compiler not found. Skipping precompiled "
                "distro_shaders.metallib; shaders.metal installed for runtime JIT."
            )
            if require_metallib:
                raise RuntimeError(
                    "MCCL_REQUIRE_METALLIB=1 but `xcrun metal` is not available."
                ) from e
            self.warn(msg)
            return

        air_path = os.path.join(self.build_temp, "distro_shaders.air")
        lib_path = os.path.join(output_dir, "distro_shaders.metallib")
        os.makedirs(os.path.dirname(air_path), exist_ok=True)

        sdk_path = subprocess.check_output(
            ["xcrun", "--show-sdk-path"], text=True
        ).strip()

        metal_std = self._detect_metal_std()
        self.announce(f"Compiling {shader_src} -> .air (std={metal_std})", level=2)
        subprocess.check_call([
            "xcrun", "metal",
            "-c", shader_src,
            "-o", air_path,
            f"-std={metal_std}",
            "-isysroot", sdk_path,
        ])

        self.announce(f"Linking .air -> {lib_path}", level=2)
        subprocess.check_call([
            "xcrun", "metallib",
            air_path,
            "-o", lib_path,
        ])

        if not os.path.isfile(lib_path):
            raise RuntimeError(f"metallib build did not produce {lib_path}")

        self.announce(f"Precompiled metallib: {lib_path}", level=2)

        for d in self._shader_bundle_dest_dirs(ext, output_dir):
            if os.path.normpath(d) == os.path.normpath(output_dir):
                continue
            dup = os.path.join(d, "distro_shaders.metallib")
            shutil.copy2(lib_path, dup)
            self.announce(f"Synced metallib to {dup}", level=2)


CPP_SOURCES = [
    "csrc/runtime/Rendezvous.cpp",
    "csrc/backend/WorkMCCL.cpp",
    "csrc/backend/PeerMesh.cpp",
    "csrc/backend/ProcessGroupMCCL.cpp",
    "csrc/backend/Registration.cpp",
    "csrc/python/bindings.cpp",
]

MM_SOURCES = []

ext = Extension(
    name="distro._C",
    sources=CPP_SOURCES + MM_SOURCES,
    include_dirs=["csrc"],
    language="c++",
)

setup(
    name="distro",
    version="0.4.0",
    description="Distributed Metal GPU Runtime for Apple Silicon clusters",
    packages=["distro", "distro.distributed"],
    ext_modules=[ext],
    cmdclass={"build_ext": MCCLBuildExt},
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "distro=distro.cli:main",
        ],
    },
)
