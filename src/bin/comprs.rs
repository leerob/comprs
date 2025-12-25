//! comprs CLI - Image compression tool
//!
//! A command-line interface for the comprs image compression library.
//! Supports PNG, JPEG, and PPM/PGM input formats.

use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Cursor, Read, Write};
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
#[command(after_help = "\
EXAMPLES:
    comprs photo.png -o photo.jpg              Convert PNG to JPEG
    comprs photo.png -o photo.jpg -q 90        JPEG with higher quality
    comprs input.jpg -o output.png -c 9        Maximum PNG compression
    comprs image.png --png-preset max          Use PNG optimization preset
    comprs photo.png -o gray.jpg --grayscale   Convert to grayscale
    comprs photo.png -v                        Verbose output with timing

More info: https://github.com/leerob/comprs/blob/main/docs/cli.md")]
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
    /// Optimize JPEG Huffman tables (smaller files, slower)
    #[arg(long, default_value_t = false)]
    jpeg_optimize_huffman: bool,
    /// JPEG restart interval in MCUs (0 to disable, 1-65535 to enable). Use to improve error resilience.
    #[arg(
        long,
        value_parser = clap::value_parser!(u16).range(0..=65535),
        default_value = "0"
    )]
    jpeg_restart_interval: u16,

    /// PNG compression level (1-9, higher = smaller file)
    #[arg(short = 'c', long, default_value = "2", value_parser = clap::value_parser!(u8).range(1..=9))]
    compression: u8,

    /// JPEG chroma subsampling
    #[arg(long, value_enum, default_value = "s444")]
    subsampling: SubsamplingArg,

    /// PNG filter strategy
    #[arg(long, value_enum, default_value = "adaptive-fast")]
    filter: FilterArg,

    /// PNG preset (overrides compression/filter when set)
    #[arg(long, value_enum)]
    png_preset: Option<PngPresetArg>,

    /// Optimize fully transparent pixels by zeroing color channels (PNG)
    #[arg(long, default_value_t = false)]
    png_optimize_alpha: bool,

    /// Reduce color type when lossless-safe (e.g., RGBA→RGB/GrayAlpha, RGB→Gray)
    #[arg(long, default_value_t = false)]
    png_reduce_color: bool,

    /// Strip non-critical metadata (tEXt/zTXt/iTXt/tIME) from PNG output
    #[arg(long, default_value_t = false)]
    png_strip_metadata: bool,

    /// Convert to grayscale
    #[arg(long)]
    grayscale: bool,

    /// Show verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Suppress all output except errors
    #[arg(long)]
    quiet: bool,

    /// Output results as JSON (for scripting)
    #[arg(long)]
    json: bool,

    /// Preview the operation without writing any files
    #[arg(long, short = 'n')]
    dry_run: bool,
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
    /// Min-sum filter selection (oxipng-style)
    Minsum,
    /// Adaptive filter selection (best compression)
    Adaptive,
    /// Adaptive with reduced trials and early cutoffs (faster)
    AdaptiveFast,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PngPresetArg {
    /// Fastest settings (level 2, AdaptiveFast, no optimizations)
    Fast,
    /// Balanced settings (level 6, Adaptive, all optimizations)
    Balanced,
    /// Maximum compression (level 9, MinSum, all optimizations)
    Max,
}

impl FilterArg {
    fn to_strategy(self) -> FilterStrategy {
        match self {
            FilterArg::None => FilterStrategy::None,
            FilterArg::Sub => FilterStrategy::Sub,
            FilterArg::Up => FilterStrategy::Up,
            FilterArg::Average => FilterStrategy::Average,
            FilterArg::Paeth => FilterStrategy::Paeth,
            FilterArg::Minsum => FilterStrategy::MinSum,
            FilterArg::Adaptive => FilterStrategy::Adaptive,
            FilterArg::AdaptiveFast => FilterStrategy::AdaptiveFast,
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

/// Read all bytes from stdin
fn read_stdin() -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut buffer = Vec::new();
    io::stdin().read_to_end(&mut buffer)?;
    Ok(buffer)
}

/// Detect format from raw bytes
fn detect_format_from_bytes(data: &[u8]) -> Result<&'static str, Box<dyn std::error::Error>> {
    if data.len() < 8 {
        return Err("Input too small to detect format".into());
    }

    // PNG: 89 50 4E 47 0D 0A 1A 0A
    if data.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
        return Ok("png");
    }

    // JPEG: FF D8 FF
    if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Ok("jpeg");
    }

    // PPM: P6
    if data.starts_with(b"P6") {
        return Ok("ppm");
    }

    // PGM: P5
    if data.starts_with(b"P5") {
        return Ok("pgm");
    }

    Err("Unknown image format. Supported: PNG, JPEG, PPM (P6), PGM (P5)".into())
}

fn load_image(path: &PathBuf) -> Result<DecodedImage, Box<dyn std::error::Error>> {
    let format = detect_format(path)?;

    match format {
        "png" => decode_png(path),
        "jpeg" => decode_jpeg(path),
        "ppm" | "pgm" => decode_pnm(path),
        _ => Err(format!("Unsupported format: {format}").into()),
    }
}

fn load_image_from_bytes(data: Vec<u8>) -> Result<DecodedImage, Box<dyn std::error::Error>> {
    let format = detect_format_from_bytes(&data)?;

    match format {
        "png" => decode_png_from_bytes(data),
        "jpeg" => decode_jpeg_from_bytes(data),
        "ppm" | "pgm" => decode_pnm_from_bytes(data, format),
        _ => Err(format!("Unsupported format: {format}").into()),
    }
}

fn decode_png_from_bytes(data: Vec<u8>) -> Result<DecodedImage, Box<dyn std::error::Error>> {
    let decoder = png::Decoder::new(Cursor::new(data));
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

fn decode_jpeg_from_bytes(data: Vec<u8>) -> Result<DecodedImage, Box<dyn std::error::Error>> {
    let mut decoder = jpeg_decoder::Decoder::new(Cursor::new(data));
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

fn decode_pnm_from_bytes(
    data: Vec<u8>,
    format_hint: &str,
) -> Result<DecodedImage, Box<dyn std::error::Error>> {
    let mut reader = BufReader::new(Cursor::new(data));

    // Read magic number
    let mut magic = String::new();
    read_token(&mut reader, &mut magic)?;

    let (color_type, input_format) = match magic.as_str() {
        "P5" => (ColorType::Gray, "PGM"),
        "P6" => (ColorType::Rgb, "PPM"),
        _ => {
            return Err(
                format!("Unsupported format '{magic}'. Expected P5 (PGM) or P6 (PPM)").into(),
            )
        }
    };

    // Validate format hint matches
    if (format_hint == "ppm" && magic != "P6") || (format_hint == "pgm" && magic != "P5") {
        return Err(format!("Format mismatch: expected {format_hint}, got {magic}").into());
    }

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
        return Err(format!("Unsupported max value {max_val}. Only 8-bit (255) supported").into());
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

fn rgba_to_rgb(pixels: &[u8]) -> Vec<u8> {
    pixels
        .chunks_exact(4)
        .flat_map(|rgba| [rgba[0], rgba[1], rgba[2]])
        .collect()
}

fn gray_alpha_to_gray(pixels: &[u8]) -> Vec<u8> {
    pixels.iter().step_by(2).copied().collect()
}

fn main() {
    // Show concise help if no arguments provided
    if std::env::args().len() == 1 {
        print_concise_help();
        std::process::exit(0);
    }

    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn print_concise_help() {
    eprintln!("comprs - A minimal-dependency, high-performance image compression tool");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("    comprs <INPUT> [OPTIONS]");
    eprintln!();
    eprintln!("EXAMPLES:");
    eprintln!("    comprs photo.png -o photo.jpg         Convert PNG to JPEG");
    eprintln!("    comprs photo.png -o photo.jpg -q 90   JPEG with higher quality");
    eprintln!("    comprs input.jpg -o output.png -c 9   Maximum PNG compression");
    eprintln!();
    eprintln!("For more options, run: comprs --help");
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let is_stdin = args.input.as_os_str() == "-";
    let is_stdout = args
        .output
        .as_ref()
        .map(|p| p.as_os_str() == "-")
        .unwrap_or(false);

    // Load input image
    let start = Instant::now();
    let img = if is_stdin {
        let data = read_stdin().map_err(|e| format!("Can't read from stdin: {e}"))?;
        load_image_from_bytes(data)?
    } else {
        load_image(&args.input).map_err(|e| {
            if args.input.exists() {
                format!("Can't read '{}': {e}", args.input.display())
            } else {
                format!(
                    "File not found: '{}'. Check that the path is correct.",
                    args.input.display()
                )
            }
        })?
    };
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
    // When reading from stdin, output must be specified
    let output_path = if is_stdin {
        args.output.clone().ok_or(
            "When reading from stdin (-), you must specify an output file with -o/--output",
        )?
    } else {
        args.output.clone().unwrap_or_else(|| {
            let mut path = args.input.clone();
            let ext = match determine_format(&args) {
                OutputFormat::Png => "png",
                OutputFormat::Jpeg | OutputFormat::Jpg => "jpg",
            };
            path.set_extension(format!("compressed.{ext}"));
            path
        })
    };

    let format = args.format.unwrap_or_else(|| {
        // When writing to stdout, require explicit format
        if is_stdout {
            return OutputFormat::Jpeg; // Default to JPEG for stdout
        }
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
            let mut builder = PngOptions::builder()
                .compression_level(args.compression)
                .filter_strategy(args.filter.to_strategy())
                .optimize_alpha(args.png_optimize_alpha)
                .reduce_color_type(args.png_reduce_color)
                .strip_metadata(args.png_strip_metadata)
                .reduce_palette(args.png_reduce_color)
                .verbose_filter_log(args.verbose);

            if let Some(preset) = args.png_preset {
                let preset_id = match preset {
                    PngPresetArg::Fast => 0,
                    PngPresetArg::Balanced => 1,
                    PngPresetArg::Max => 2,
                };
                builder = builder.preset(preset_id);
                // Explicit flags still override preset
                builder = builder
                    .compression_level(args.compression)
                    .filter_strategy(args.filter.to_strategy())
                    .optimize_alpha(args.png_optimize_alpha)
                    .reduce_color_type(args.png_reduce_color)
                    .strip_metadata(args.png_strip_metadata)
                    .reduce_palette(args.png_reduce_color)
                    .verbose_filter_log(args.verbose);
            }

            let options = builder.build();

            if args.verbose {
                eprintln!(
                    "PNG options: preset={:?}, level={}, filter={:?}, optimize_alpha={}, reduce_color_type={}, reduce_palette={}, strip_metadata={}",
                    args.png_preset.unwrap_or(PngPresetArg::Fast),
                    options.compression_level,
                    options.filter_strategy,
                    options.optimize_alpha,
                    options.reduce_color_type,
                    options.reduce_palette,
                    options.strip_metadata
                );
            }

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
            let options = JpegOptions::builder()
                .quality(args.quality)
                .subsampling(args.subsampling.into())
                .restart_interval(if args.jpeg_restart_interval == 0 {
                    None
                } else {
                    Some(args.jpeg_restart_interval)
                })
                .optimize_huffman(args.jpeg_optimize_huffman)
                .progressive(false)
                .trellis_quant(false)
                .build();
            if args.verbose {
                eprintln!(
                    "JPEG options: quality={}, subsampling={:?}, restart_interval={:?}, optimize_huffman={}",
                    options.quality,
                    options.subsampling,
                    options.restart_interval,
                    options.optimize_huffman
                );
            }
            comprs::jpeg::encode_with_options_into(
                &mut output_data,
                &pixels,
                width,
                height,
                color_type,
                &options,
            )?
        }
    };
    let encode_time = encode_start.elapsed();

    // Report results
    let input_size = if is_stdin {
        // For stdin, use the raw pixel data size as a rough estimate
        // (actual compressed input size is unknown)
        (width * height * color_type.bytes_per_pixel() as u32) as u64
    } else {
        fs::metadata(&args.input)?.len()
    };
    let output_size = output_data.len() as u64;
    let ratio = if input_size > 0 {
        (output_size as f64 / input_size as f64) * 100.0
    } else {
        0.0
    };

    let input_display = if is_stdin {
        "<stdin>".to_string()
    } else {
        args.input.display().to_string()
    };
    let output_display = if is_stdout {
        "<stdout>".to_string()
    } else {
        output_path.display().to_string()
    };

    // Write output (unless dry-run)
    if args.dry_run {
        if !args.quiet {
            if args.json {
                println!(
                    r#"{{"dry_run":true,"input":"{input_display}","output":"{output_display}","input_size":{input_size},"output_size":{output_size},"ratio":{ratio:.1}}}"#
                );
            } else {
                eprintln!("Dry run: would write to {output_display}");
                println!(
                    "{} -> {} ({:.1}%)",
                    format_size(input_size),
                    format_size(output_size),
                    ratio
                );
            }
        }
        return Ok(());
    }

    // Write output
    if is_stdout {
        io::stdout()
            .write_all(&output_data)
            .map_err(|e| format!("Can't write to stdout: {e}"))?;
    } else {
        fs::write(&output_path, &output_data).map_err(|e| {
            format!(
                "Can't write to '{}': {}. Check that the directory exists and is writable.",
                output_path.display(),
                e
            )
        })?;
    }

    // Output results (to stderr if writing to stdout, to avoid mixing with output data)
    let print_results = |msg: &str| {
        if is_stdout {
            eprintln!("{msg}");
        } else {
            println!("{msg}");
        }
    };

    if args.json {
        let json_output = format!(
            r#"{{"input":"{input_display}","output":"{output_display}","input_size":{input_size},"output_size":{output_size},"ratio":{ratio:.1}}}"#
        );
        print_results(&json_output);
    } else if args.verbose {
        eprintln!("Output: {output_display}");
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
    } else if !args.quiet {
        print_results(&format!(
            "{} -> {} ({:.1}%)",
            format_size(input_size),
            format_size(output_size),
            ratio
        ));
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
