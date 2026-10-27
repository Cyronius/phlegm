// Build the XRT C++ shim only under the `npu` feature. Windows uses MSVC;
// Linux uses a local XRT install and the configured C++ toolchain.

use std::path::PathBuf;
use std::process::Command;

mod build_release;

fn main() {
    build_release::emit();
    // Only build the shim when the npu feature is enabled.
    if std::env::var("CARGO_FEATURE_NPU").is_err() {
        return;
    }

    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let shim_src = manifest.join("xrt-shim").join("xrt_shim.cpp");
    let shim_hdr = manifest.join("xrt-shim").join("xrt_shim.h");
    println!("cargo:rerun-if-changed={}", shim_src.display());
    println!("cargo:rerun-if-changed={}", shim_hdr.display());
    match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("windows") => build_windows(&manifest, &out_dir, &shim_src),
        Ok("linux") => build_linux(&out_dir, &shim_src),
        _ => panic!("the npu feature supports Windows and Linux only"),
    }
}

fn build_linux(out_dir: &std::path::Path, shim_src: &std::path::Path) {
    for name in ["CXX", "AR", "XILINX_XRT", "XRT_INCLUDE_DIR", "XRT_LIB_DIR"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    let xrt_root = PathBuf::from(std::env::var("XILINX_XRT").unwrap_or_else(|_| "/opt/xilinx/xrt".into()));
    let include = std::env::var_os("XRT_INCLUDE_DIR").map(PathBuf::from)
        .unwrap_or_else(|| xrt_root.join("include"));
    let lib_dir = std::env::var_os("XRT_LIB_DIR").map(PathBuf::from)
        .unwrap_or_else(|| xrt_root.join("lib"));
    if !include.join("xrt/xrt_device.h").is_file() || !lib_dir.join("libxrt_coreutil.so").is_file() {
        panic!("Linux XRT headers or libxrt_coreutil.so are missing under {} and {}",
               include.display(), lib_dir.display());
    }
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "g++".into());
    let ar = std::env::var("AR").unwrap_or_else(|_| "ar".into());
    let obj = out_dir.join("xrt_shim.o");
    let lib = out_dir.join("libxrt_shim.a");
    let status = Command::new(&cxx)
        .args(["-std=c++17", "-O2", "-fPIC", "-c", "-I"])
        .arg(&include).arg(shim_src).arg("-o").arg(&obj)
        .status().expect("failed to launch Linux C++ compiler for XRT shim");
    if !status.success() { panic!("Linux XRT shim compilation failed"); }
    let status = Command::new(&ar).arg("crs").arg(&lib).arg(&obj)
        .status().expect("failed to launch ar for XRT shim");
    if !status.success() { panic!("Linux XRT shim archive failed"); }
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=xrt_shim");
    println!("cargo:rustc-link-lib=dylib=xrt_coreutil");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn build_windows(manifest: &std::path::Path, out_dir: &std::path::Path, shim_src: &std::path::Path) {
    let xrt_inc = manifest
        .join("deps")
        .join("XRT")
        .join("src")
        .join("runtime_src")
        .join("core")
        .join("include");
    // build_import_lib.cmd makes it from the system xrt_coreutil.dll; XRT ships no import lib.
    let xrt_lib_dir = manifest.join("xrt-shim").join("out");

    println!("cargo:rerun-if-env-changed=VCVARS64");

    if !xrt_lib_dir.join("xrt_coreutil.lib").exists() {
        panic!(
            "missing XRT import lib {}\\xrt_coreutil.lib — build it first with \
             phlegm-core/xrt-shim/build_import_lib.cmd",
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
