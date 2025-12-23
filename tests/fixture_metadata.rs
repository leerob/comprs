mod support;

use std::fs;
use std::path::Path;

use support::fixture_info::read_png_header;

const FIXTURES: &str = "tests/fixtures";

#[test]
fn provided_fixtures_have_expected_metadata_and_sizes() -> Result<(), Box<dyn std::error::Error>> {
    let uncompressed = Path::new(FIXTURES).join("uncompressed.png");
    let comprs_result = Path::new(FIXTURES).join("comprs-result.png");
    let benchmark = Path::new(FIXTURES).join("benchmark-result.png");

    let uncompressed_hdr = read_png_header(&uncompressed)?;
    let comprs_hdr = read_png_header(&comprs_result)?;
    let benchmark_hdr = read_png_header(&benchmark)?;

    // All fixtures should be the same dimensions.
    assert_eq!(
        (uncompressed_hdr.width, uncompressed_hdr.height),
        (911, 1662)
    );
    assert_eq!(
        (comprs_hdr.width, comprs_hdr.height),
        (uncompressed_hdr.width, uncompressed_hdr.height)
    );
    assert_eq!(
        (benchmark_hdr.width, benchmark_hdr.height),
        (uncompressed_hdr.width, uncompressed_hdr.height)
    );

    // Color types: source RGBA (6), our output RGB (2), benchmark palette (3).
    assert_eq!(uncompressed_hdr.color_type, 6, "expected RGBA source");
    assert_eq!(comprs_hdr.color_type, 2, "expected RGB comprs output");
    assert_eq!(benchmark_hdr.color_type, 3, "expected indexed benchmark");

    // Bit depth must be 8 for all.
    assert_eq!(uncompressed_hdr.bit_depth, 8);
    assert_eq!(comprs_hdr.bit_depth, 8);
    assert_eq!(benchmark_hdr.bit_depth, 8);

    let uncompressed_size = fs::metadata(&uncompressed)?.len();
    let comprs_size = fs::metadata(&comprs_result)?.len();
    let benchmark_size = fs::metadata(&benchmark)?.len();

    // Sanity: benchmark is much smaller than ours; ours is much smaller than source.
    assert!(
        benchmark_size < comprs_size && benchmark_size < uncompressed_size,
        "benchmark should be the smallest (bench: {benchmark_size}, comprs: {comprs_size}, src: {uncompressed_size})"
    );
    assert!(
        comprs_size < uncompressed_size,
        "comprs result should be smaller than source"
    );

    // Guardrails to catch large regressions and to encode the target scale.
    assert!(
        benchmark_size <= 110_000,
        "benchmark PNG unexpectedly large: {benchmark_size}"
    );
    assert!(
        comprs_size >= 250_000,
        "existing comprs result unexpectedly small: {comprs_size}"
    );

    Ok(())
}
