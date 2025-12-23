use comprs::png::{self, FilterStrategy, PngOptions, QuantizationMode, QuantizationOptions};
use comprs::ColorType;
use image::GenericImageView;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let fixture = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("tests/fixtures/uncompressed.png");

    let img = image::open(&fixture)?;
    let (w, h) = img.dimensions();
    let rgba = img.to_rgba8();
    let pixels = rgba.as_raw();

    // Baseline: no quantization
    let baseline_opts = PngOptions {
        compression_level: 6,
        filter_strategy: FilterStrategy::AdaptiveSampled { interval: 2 },
        ..Default::default()
    };
    let baseline = png::encode_with_options(pixels, w, h, ColorType::Rgba, &baseline_opts)?;

    // Quantized: force palette, filter none
    let quant_opts = PngOptions {
        compression_level: 6,
        filter_strategy: FilterStrategy::None,
        quantization: QuantizationOptions {
            mode: QuantizationMode::Force,
            max_colors: 256,
            dithering: false,
        },
    };
    let quantized = png::encode_with_options(pixels, w, h, ColorType::Rgba, &quant_opts)?;

    println!("Fixture: {fixture}");
    println!("Dimensions: {}x{}", w, h);
    println!(
        "Baseline (no quant, lvl6 adaptive-sampled): {} bytes",
        baseline.len()
    );
    println!(
        "Quantized (force, lvl6 filter-none): {} bytes (indexed={}, reduction: {:.1}%)",
        quantized.len(),
        quantized.get(25).map(|b| *b == 3).unwrap_or(false),
        (1.0 - quantized.len() as f64 / baseline.len() as f64) * 100.0
    );

    Ok(())
}
