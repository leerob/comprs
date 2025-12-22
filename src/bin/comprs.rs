//! comprs CLI - Image compression tool
//!
//! A command-line interface for the comprs image compression library.
//! Supports PNG, JPEG, and PPM/PGM input formats.

use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};

use comprs::jpeg::{JpegOptions, Subsampling};
use comprs::png::{FilterStrategy, PngOptions};
use comprs::ColorType;

/// A minimal-dependency, high-performance image compression tool.
///
/// Supports PNG, JPEG, and PPM/PGM input formats.
#[derive(Parser, Debug)]
#[command(name = "comprs")]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input image file (PNG, JPEG, PPM, or PGM)
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Output file path (format detected from extension)
    #[arg(short, long, value_name = "OUTPUT")]
    output: Option<PathBuf>,

    /// Output format (overrides extension detection)
    #[arg(short, long, value_enum)]
    format: Option<OutputFormat>,

    /// JPEG quality (1-100, higher = better quality)
    #[arg(short, long, default_value = "85", value_parser = clap::value_parser!(u8).range(1..=100))]
    quality: u8,

    /// PNG compression level (1-9, higher = smaller file)
    #[arg(short = 'c', long, default_value = "2", value_parser = clap::value_parser!(u8).range(1..=9))]
    compression: u8,

    /// JPEG chroma subsampling
    #[arg(long, value_enum, default_value = "s444")]
    subsampling: SubsamplingArg,

    /// PNG filter strategy
    #[arg(long, value_enum, default_value = "adaptivefast")]
    filter: FilterArg,

    /// PNG preset (overrides compression/filter when set)
    #[arg(long, value_enum)]
    png_preset: Option<PngPresetArg>,

    /// Interval for adaptive-sampled filter (rows between full evaluations)
    #[arg(
        long,
        default_value = "4",
        value_parser = clap::value_parser!(u32).range(1..=1_000_000)
    )]
    adaptive_sample_interval: u32,

    /// Convert to grayscale
    #[arg(long)]
    grayscale: bool,

    /// Show verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputFormat {
    /// PNG format (lossless)
    Png,
    /// JPEG format (lossy)
    Jpeg,
    /// JPEG format (alias for jpeg)
    Jpg,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SubsamplingArg {
    /// 4:4:4 - No subsampling (best quality)
    S444,
    /// 4:2:0 - 2x2 chroma downsample (smaller file)
    S420,
}

impl From<SubsamplingArg> for Subsampling {
    fn from(arg: SubsamplingArg) -> Self {
        match arg {
            SubsamplingArg::S444 => Subsampling::S444,
            SubsamplingArg::S420 => Subsampling::S420,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum FilterArg {
    /// No filter (fastest)
    None,
    /// Sub filter
    Sub,
    /// Up filter
    Up,
    /// Average filter
    Average,
    /// Paeth filter
    Paeth,
    /// Adaptive filter selection (best compression)
    Adaptive,
    /// Adaptive with reduced trials and early cutoffs (faster)
    AdaptiveFast,
    /// Adaptive on sampled rows, reuse chosen filter between samples
    AdaptiveSampled,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PngPresetArg {
    /// Fastest settings (level 2, AdaptiveFast)
    Fast,
    /// Balanced settings (level 6, Adaptive)
    Balanced,
    /// Maximum compression (level 9, AdaptiveSampled interval=2)
    Max,
}

impl FilterArg {
    fn to_strategy(self, sampled_interval: u32) -> FilterStrategy {
        match self {
            FilterArg::None => FilterStrategy::None,
            FilterArg::Sub => FilterStrategy::Sub,
            FilterArg::Up => FilterStrategy::Up,
            FilterArg::Average => FilterStrategy::Average,
            FilterArg::Paeth => FilterStrategy::Paeth,
            FilterArg::Adaptive => FilterStrategy::Adaptive,
            FilterArg::AdaptiveFast => FilterStrategy::AdaptiveFast,
            FilterArg::AdaptiveSampled => FilterStrategy::AdaptiveSampled {
                interval: sampled_interval.max(1),
            },
        }
    }
}

/// Decoded image data.
struct DecodedImage {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    color_type: ColorType,
    input_format: &'static str,
}

/// Detect input format from file header bytes.
fn detect_format(path: &PathBuf) -> Result<&'static str, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut header = [0u8; 8];
    file.read_exact(&mut header)?;

    // PNG: 89 50 4E 47 0D 0A 1A 0A
    if header.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
        return Ok("png");
    }

    // JPEG: FF D8 FF
    if header.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Ok("jpeg");
    }

    // PPM: P6
    if header.starts_with(b"P6") {
        return Ok("ppm");
    }

    // PGM: P5
    if header.starts_with(b"P5") {
        return Ok("pgm");
    }

    Err("Unknown image format. Supported: PNG, JPEG, PPM (P6), PGM (P5)".into())
}

/// Decode a PNG file.
fn decode_png(path: &PathBuf) -> Result<DecodedImage, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info()?;

    let mut pixels = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut pixels)?;
    pixels.truncate(info.buffer_size());

    let color_type = match info.color_type {
        png::ColorType::Grayscale => ColorType::Gray,
        png::ColorType::GrayscaleAlpha => ColorType::GrayAlpha,
        png::ColorType::Rgb => ColorType::Rgb,
        png::ColorType::Rgba => ColorType::Rgba,
        png::ColorType::Indexed => {
            return Err("Indexed PNG not supported. Convert to RGB first.".into())
        }
    };

    Ok(DecodedImage {
        width: info.width,
        height: info.height,
        pixels,
        color_type,
        input_format: "PNG",
    })
}

/// Decode a JPEG file.
fn decode_jpeg(path: &PathBuf) -> Result<DecodedImage, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut decoder = jpeg_decoder::Decoder::new(BufReader::new(file));
    let pixels = decoder.decode()?;
    let info = decoder.info().ok_or("Failed to get JPEG info")?;

    let color_type = match info.pixel_format {
        jpeg_decoder::PixelFormat::L8 => ColorType::Gray,
        jpeg_decoder::PixelFormat::L16 => return Err("16-bit grayscale JPEG not supported.".into()),
        jpeg_decoder::PixelFormat::RGB24 => ColorType::Rgb,
        jpeg_decoder::PixelFormat::CMYK32 => {
            return Err("CMYK JPEG not supported. Convert to RGB first.".into())
        }
    };

    Ok(DecodedImage {
        width: info.width as u32,
        height: info.height as u32,
        pixels,
        color_type,
        input_format: "JPEG",
    })
}

/// Decode a PPM (P6) or PGM (P5) file.
fn decode_pnm(path: &PathBuf) -> Result<DecodedImage, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read magic number
    let mut magic = String::new();
    read_token(&mut reader, &mut magic)?;

    let (color_type, input_format) = match magic.as_str() {
        "P5" => (ColorType::Gray, "PGM"),
        "P6" => (ColorType::Rgb, "PPM"),
        _ => {
            return Err(
                format!("Unsupported format '{magic}'. Expected P5 (PGM) or P6 (PPM)",).into(),
            )
        }
    };

    // Read dimensions
    let mut token = String::new();

    read_token(&mut reader, &mut token)?;
    let width: u32 = token.parse()?;

    token.clear();
    read_token(&mut reader, &mut token)?;
    let height: u32 = token.parse()?;

    token.clear();
    read_token(&mut reader, &mut token)?;
    let max_val: u32 = token.parse()?;

    if max_val != 255 {
        return Err(format!("Unsupported max value {max_val}. Only 8-bit (255) supported",).into());
    }

    // Read pixel data
    let bytes_per_pixel = color_type.bytes_per_pixel();
    let expected_size = width as usize * height as usize * bytes_per_pixel;

    let mut pixels = vec![0u8; expected_size];
    reader.read_exact(&mut pixels)?;

    Ok(DecodedImage {
        width,
        height,
        pixels,
        color_type,
        input_format,
    })
}

/// Read next whitespace-delimited token, skipping comments.
fn read_token<R: BufRead>(reader: &mut R, token: &mut String) -> std::io::Result<()> {
    token.clear();
    let mut in_comment = false;

    loop {
        let mut byte = [0u8; 1];
        if reader.read(&mut byte)? == 0 {
            break;
        }

        let ch = byte[0] as char;

        if in_comment {
            if ch == '\n' {
                in_comment = false;
            }
            continue;
        }

        if ch == '#' {
            in_comment = true;
            continue;
        }

        if ch.is_ascii_whitespace() {
            if !token.is_empty() {
                break;
            }
            continue;
        }

        token.push(ch);
    }

    Ok(())
}

/// Load and decode an image file.
fn load_image(path: &PathBuf) -> Result<DecodedImage, Box<dyn std::error::Error>> {
    let format = detect_format(path)?;

    match format {
        "png" => decode_png(path),
        "jpeg" => decode_jpeg(path),
        "ppm" | "pgm" => decode_pnm(path),
        _ => Err(format!("Unsupported format: {format}").into()),
    }
}

/// Convert image to grayscale.
fn to_grayscale(pixels: &[u8], color_type: ColorType) -> Vec<u8> {
    match color_type {
        ColorType::Gray => pixels.to_vec(),
        ColorType::GrayAlpha => pixels.iter().step_by(2).copied().collect(),
        ColorType::Rgb => pixels
            .chunks_exact(3)
            .map(|rgb| {
                // ITU-R BT.601 luma coefficients
                let r = rgb[0] as u32;
                let g = rgb[1] as u32;
                let b = rgb[2] as u32;
                ((77 * r + 150 * g + 29 * b + 128) >> 8) as u8
            })
            .collect(),
        ColorType::Rgba => pixels
            .chunks_exact(4)
            .map(|rgba| {
                let r = rgba[0] as u32;
                let g = rgba[1] as u32;
                let b = rgba[2] as u32;
                ((77 * r + 150 * g + 29 * b + 128) >> 8) as u8
            })
            .collect(),
    }
}

/// Convert RGBA to RGB by dropping alpha channel.
fn rgba_to_rgb(pixels: &[u8]) -> Vec<u8> {
    pixels
        .chunks_exact(4)
        .flat_map(|rgba| [rgba[0], rgba[1], rgba[2]])
        .collect()
}

/// Convert GrayAlpha to Gray by dropping alpha channel.
fn gray_alpha_to_gray(pixels: &[u8]) -> Vec<u8> {
    pixels.iter().step_by(2).copied().collect()
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Load input image
    let start = Instant::now();
    let img = load_image(&args.input)?;
    let load_time = start.elapsed();

    let width = img.width;
    let height = img.height;
    let input_format = img.input_format;

    if args.verbose {
        let input = &args.input;
        let ct = img.color_type;
        eprintln!("Loaded: {input:?}");
        eprintln!("  Input format: {input_format}");
        eprintln!("  Dimensions: {width}x{height}");
        eprintln!("  Color type: {ct:?}");
        eprintln!("  Load time: {load_time:.2?}");
    }

    // Determine output format
    let output_path = args.output.clone().unwrap_or_else(|| {
        let mut path = args.input.clone();
        let ext = match determine_format(&args) {
            OutputFormat::Png => "png",
            OutputFormat::Jpeg | OutputFormat::Jpg => "jpg",
        };
        path.set_extension(format!("compressed.{ext}"));
        path
    });

    let format = args.format.unwrap_or_else(|| {
        output_path
            .extension()
            .and_then(|e| e.to_str())
            .and_then(|e| match e.to_lowercase().as_str() {
                "png" => Some(OutputFormat::Png),
                "jpg" | "jpeg" => Some(OutputFormat::Jpeg),
                _ => None,
            })
            .unwrap_or(OutputFormat::Jpeg)
    });

    // Convert to appropriate color format
    let (pixels, color_type) = if args.grayscale {
        (to_grayscale(&img.pixels, img.color_type), ColorType::Gray)
    } else {
        match format {
            OutputFormat::Png => {
                // PNG supports all color types
                (img.pixels, img.color_type)
            }
            OutputFormat::Jpeg | OutputFormat::Jpg => {
                // JPEG only supports Gray and RGB
                match img.color_type {
                    ColorType::Gray => (img.pixels, ColorType::Gray),
                    ColorType::GrayAlpha => (gray_alpha_to_gray(&img.pixels), ColorType::Gray),
                    ColorType::Rgb => (img.pixels, ColorType::Rgb),
                    ColorType::Rgba => (rgba_to_rgb(&img.pixels), ColorType::Rgb),
                }
            }
        }
    };

    // Encode
    let encode_start = Instant::now();
    let mut output_data = Vec::new();
    match format {
        OutputFormat::Png => {
            let mut options = match args.png_preset {
                Some(PngPresetArg::Fast) => PngOptions::fast(),
                Some(PngPresetArg::Balanced) => PngOptions::balanced(),
                Some(PngPresetArg::Max) => PngOptions::max_compression(),
                None => PngOptions {
                    compression_level: args.compression,
                    filter_strategy: args.filter.to_strategy(args.adaptive_sample_interval),
                },
            };
            // Allow explicit overrides if preset is provided but user also set flags.
            options.compression_level = args.compression;
            options.filter_strategy = args.filter.to_strategy(args.adaptive_sample_interval);

            comprs::png::encode_into(
                &mut output_data,
                &pixels,
                width,
                height,
                color_type,
                &options,
            )?
        }
        OutputFormat::Jpeg | OutputFormat::Jpg => {
            let options = JpegOptions {
                quality: args.quality,
                subsampling: args.subsampling.into(),
                restart_interval: None,
            };
            comprs::jpeg::encode_with_options_into(
                &mut output_data,
                &pixels,
                width,
                height,
                args.quality,
                color_type,
                &options,
            )?
        }
    };
    let encode_time = encode_start.elapsed();

    // Write output
    fs::write(&output_path, &output_data)?;

    // Report results
    let input_size = fs::metadata(&args.input)?.len();
    let output_size = output_data.len() as u64;
    let ratio = if input_size > 0 {
        (output_size as f64 / input_size as f64) * 100.0
    } else {
        0.0
    };

    if args.verbose {
        eprintln!("Output: {output_path:?}");
        eprintln!("  Format: {format:?}");
        eprintln!("  Color type: {color_type:?}");
        match format {
            OutputFormat::Png => {
                let compression = args.compression;
                let filter = &args.filter;
                eprintln!("  Compression level: {compression}");
                eprintln!("  Filter: {filter:?}");
            }
            OutputFormat::Jpeg | OutputFormat::Jpg => {
                let quality = args.quality;
                let subsampling = &args.subsampling;
                eprintln!("  Quality: {quality}");
                eprintln!("  Subsampling: {subsampling:?}");
            }
        }
        eprintln!("  Encode time: {encode_time:.2?}");
        eprintln!(
            "  Size: {} -> {} ({:.1}%)",
            format_size(input_size),
            format_size(output_size),
            ratio
        );
    } else {
        println!(
            "{} -> {} ({:.1}%)",
            format_size(input_size),
            format_size(output_size),
            ratio
        );
    }

    Ok(())
}

fn determine_format(args: &Args) -> OutputFormat {
    args.format.unwrap_or_else(|| {
        args.output
            .as_ref()
            .and_then(|p| p.extension())
            .and_then(|e| e.to_str())
            .and_then(|e| match e.to_lowercase().as_str() {
                "png" => Some(OutputFormat::Png),
                "jpg" | "jpeg" => Some(OutputFormat::Jpeg),
                _ => None,
            })
            .unwrap_or(OutputFormat::Jpeg)
    })
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;

    if bytes >= MB {
        let mb = bytes as f64 / MB as f64;
        format!("{mb:.2} MB")
    } else if bytes >= KB {
        let kb = bytes as f64 / KB as f64;
        format!("{kb:.2} KB")
    } else {
        format!("{bytes} B")
    }
}
