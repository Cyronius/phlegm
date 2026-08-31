// Build the extern "C" xrt-shim (npu-engine/xrt-shim/xrt_shim.cpp) into a static
// lib and link it + XRT's import lib. Only runs when the `npu` feature is on so
// the CPU engine builds without MSVC/XRT present.
//
// Uses cl.exe via vcvars64 (same toolchain as m0/build_driver.cmd) with no extra
// crates. Override the vcvars path with the VCVARS64 env var if needed.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only build the shim when the npu feature is enabled.
    if std::env::var("CARGO_FEATURE_NPU").is_err() {
        return;
    }

    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let shim_src = manifest.join("xrt-shim").join("xrt_shim.cpp");
    let shim_hdr = manifest.join("xrt-shim").join("xrt_shim.h");
    let xrt_inc = manifest
        .join("deps")
        .join("XRT")
        .join("src")
        .join("runtime_src")
        .join("core")
        .join("include");
    // XRT import lib (gendef'd over the system xrt_coreutil.dll by build_m0.cmd).
    let xrt_lib_dir = manifest.join("m0").join("out");

    println!("cargo:rerun-if-changed={}", shim_src.display());
    println!("cargo:rerun-if-changed={}", shim_hdr.display());
    println!("cargo:rerun-if-env-changed=VCVARS64");

    if !xrt_lib_dir.join("xrt_coreutil.lib").exists() {
        panic!(
            "missing XRT import lib {}\\xrt_coreutil.lib — build it first with \
             npu-engine/m0/build_m0.cmd",
            xrt_lib_dir.display()
        );
    }

    let vcvars = std::env::var("VCVARS64").unwrap_or_else(|_| {
        "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat".to_string()
    });

    let obj = out_dir.join("xrt_shim.obj");
    let lib = out_dir.join("xrt_shim.lib");

    // A .cmd wrapper so we can `call vcvars64` then cl/lib in one MSVC env.
    let cmd_path = out_dir.join("build_shim.cmd");
    let script = format!(
        "@echo off\r\n\
         call \"{vcvars}\" >nul\r\n\
         if errorlevel 1 exit /b 1\r\n\
         cl /nologo /c /EHsc /std:c++17 /Zc:__cplusplus /O2 /I \"{inc}\" \"{src}\" /Fo\"{obj}\"\r\n\
         if errorlevel 1 exit /b 1\r\n\
         lib /nologo /out:\"{lib}\" \"{obj}\"\r\n\
         if errorlevel 1 exit /b 1\r\n",
        vcvars = vcvars,
        inc = xrt_inc.display(),
        src = shim_src.display(),
        obj = obj.display(),
        lib = lib.display(),
    );
    std::fs::write(&cmd_path, script).expect("write build_shim.cmd");

    let status = Command::new("cmd.exe")
        .arg("/c")
        .arg(&cmd_path)
        .status()
        .expect("failed to launch cmd.exe for shim build");
    if !status.success() {
        panic!("xrt-shim compile failed (see cl.exe output above)");
    }

    // Link the shim static lib + XRT import lib.
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=xrt_shim");
    println!("cargo:rustc-link-search=native={}", xrt_lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=xrt_coreutil");
}
