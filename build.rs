fn main() {
    // ── Cyclo ZKP static libraries ───────────────────────────────────────────
    // Enabled by `--features cyclo`.  Requires `zig build` to have been run
    // in ../VELA/libVELA/cyclo/ first (produces libcyclo.a + libntt_shim.a).
    #[cfg(feature = "cyclo")]
    {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let cyclo_dir = std::path::Path::new(&manifest_dir)
            .parent()
            .unwrap()
            .join("VELA/libVELA/cyclo");

        println!(
            "cargo:rustc-link-search=native={}",
            cyclo_dir.join("zig-out/lib").display()
        );
        println!(
            "cargo:rustc-link-search=native={}",
            cyclo_dir.join("ntt_shim/target/release").display()
        );
        // The -l flags are declared via #[link(...)] in benches/cyclo_bench.rs
        // so that they are scoped to the bench target rather than the whole crate.

        // libntt_shim.a is a Rust staticlib (panic=abort), so it bundles the
        // full `core` runtime.  Linking it into another Rust binary causes
        // duplicate-symbol errors for every core symbol.  Both copies are
        // compiled from the same source with the same settings, so it is safe
        // to let the linker pick the first definition and drop the second.
        println!("cargo:rustc-link-arg=-Wl,--allow-multiple-definition");

        println!(
            "cargo:rerun-if-changed={}",
            cyclo_dir.join("zig-out/lib/libcyclo.a").display()
        );
    }

    // When the "incomplete-rexl" feature is enabled (default), everything is
    // pure Rust — no external linking required.
    //
    // When "incomplete-rexl" is disabled, we fall back to the C++ Intel HEXL
    // shared library built via `make hexl && make wrapper`.
    #[cfg(not(feature = "incomplete-rexl"))]
    {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

        println!("cargo:rustc-link-lib=dylib=hexl_wrapper");
        println!("cargo:rustc-link-search=native=.");
        println!("cargo:rustc-link-search=native=./hexl-bindings/hexl/build/hexl/lib");
        println!("cargo:rustc-link-search=native=./hexl-bindings/hexl/build/hexl/lib64");

        // Embed rpath so the binary can find shared libraries at runtime
        // without needing to set LD_LIBRARY_PATH
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", manifest_dir);
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}/hexl-bindings/hexl/build/hexl/lib",
            manifest_dir
        );
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}/hexl-bindings/hexl/build/hexl/lib64",
            manifest_dir
        );
    }
}
