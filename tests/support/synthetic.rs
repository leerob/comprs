//! Synthetic test image generation.
//!
//! Generates deterministic test patterns for controlled testing of
//! compression algorithms. All functions produce reproducible output
//! suitable for regression testing.

#![allow(dead_code)]

/// Generate a solid color image.
pub fn solid_color(width: u32, height: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
    let pixel_count = (width * height) as usize;
    let mut pixels = Vec::with_capacity(pixel_count * 3);
    for _ in 0..pixel_count {
        pixels.extend_from_slice(&[r, g, b]);
    }
    pixels
}

/// Generate a solid color RGBA image.
pub fn solid_color_rgba(width: u32, height: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
    let pixel_count = (width * height) as usize;
    let mut pixels = Vec::with_capacity(pixel_count * 4);
    for _ in 0..pixel_count {
        pixels.extend_from_slice(&[r, g, b, a]);
    }
    pixels
}

/// Generate a grayscale image.
pub fn solid_gray(width: u32, height: u32, gray: u8) -> Vec<u8> {
    vec![gray; (width * height) as usize]
}

/// Generate a horizontal gradient (left to right).
pub fn gradient_horizontal(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for _y in 0..height {
        for x in 0..width {
            let v = ((x * 255) / width.max(1)) as u8;
            pixels.extend_from_slice(&[v, v, v]);
        }
    }
    pixels
}

/// Generate a vertical gradient (top to bottom).
pub fn gradient_vertical(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        let v = ((y * 255) / height.max(1)) as u8;
        for _x in 0..width {
            pixels.extend_from_slice(&[v, v, v]);
        }
    }
    pixels
}

/// Generate a diagonal gradient (top-left to bottom-right).
pub fn gradient_diagonal(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    let max_dist = (width + height).max(1);
    for y in 0..height {
        for x in 0..width {
            let v = (((x + y) * 255) / max_dist) as u8;
            pixels.extend_from_slice(&[v, v, v]);
        }
    }
    pixels
}

/// Generate an RGB gradient (red horizontal, green vertical, blue diagonal).
pub fn gradient_rgb(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 255) / width.max(1)) as u8;
            let g = ((y * 255) / height.max(1)) as u8;
            let b = (((x + y) * 127) / (width + height).max(1)) as u8;
            pixels.extend_from_slice(&[r, g, b]);
        }
    }
    pixels
}

/// Generate a checkerboard pattern.
pub fn checkerboard(width: u32, height: u32, cell_size: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    let cell_size = cell_size.max(1);
    for y in 0..height {
        for x in 0..width {
            let cell_x = x / cell_size;
            let cell_y = y / cell_size;
            let is_white = (cell_x + cell_y) % 2 == 0;
            let v = if is_white { 255 } else { 0 };
            pixels.extend_from_slice(&[v, v, v]);
        }
    }
    pixels
}

/// Generate a colored checkerboard pattern.
pub fn checkerboard_color(
    width: u32,
    height: u32,
    cell_size: u32,
    color1: [u8; 3],
    color2: [u8; 3],
) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    let cell_size = cell_size.max(1);
    for y in 0..height {
        for x in 0..width {
            let cell_x = x / cell_size;
            let cell_y = y / cell_size;
            let color = if (cell_x + cell_y) % 2 == 0 {
                color1
            } else {
                color2
            };
            pixels.extend_from_slice(&color);
        }
    }
    pixels
}

/// Generate a high-frequency pattern (1px alternating black/white).
/// This is challenging for JPEG compression.
pub fn high_frequency(width: u32, height: u32) -> Vec<u8> {
    checkerboard(width, height, 1)
}

/// Generate horizontal stripes.
pub fn stripes_horizontal(width: u32, height: u32, stripe_height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    let stripe_height = stripe_height.max(1);
    for y in 0..height {
        let is_white = (y / stripe_height) % 2 == 0;
        let v = if is_white { 255 } else { 0 };
        for _x in 0..width {
            pixels.extend_from_slice(&[v, v, v]);
        }
    }
    pixels
}

/// Generate vertical stripes.
pub fn stripes_vertical(width: u32, height: u32, stripe_width: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    let stripe_width = stripe_width.max(1);
    for _y in 0..height {
        for x in 0..width {
            let is_white = (x / stripe_width) % 2 == 0;
            let v = if is_white { 255 } else { 0 };
            pixels.extend_from_slice(&[v, v, v]);
        }
    }
    pixels
}

/// Generate a radial gradient (circular, from center).
pub fn gradient_radial(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    let cx = width as f32 / 2.0;
    let cy = height as f32 / 2.0;
    let max_dist = (cx * cx + cy * cy).sqrt();

    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let v = ((dist / max_dist) * 255.0).min(255.0) as u8;
            pixels.extend_from_slice(&[v, v, v]);
        }
    }
    pixels
}

/// Generate pseudo-random noise using a simple LCG.
/// The pattern is deterministic based on the seed.
pub fn noise(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    let mut state = seed;

    for _ in 0..(width * height) {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let r = (state >> 16) as u8;
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let g = (state >> 16) as u8;
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let b = (state >> 16) as u8;
        pixels.extend_from_slice(&[r, g, b]);
    }
    pixels
}

/// Generate grayscale noise.
pub fn noise_gray(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height) as usize);
    let mut state = seed;

    for _ in 0..(width * height) {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        pixels.push((state >> 16) as u8);
    }
    pixels
}

/// Generate an image with sharp edges (good for testing filter strategies).
pub fn sharp_edges(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);

    for y in 0..height {
        for x in 0..width {
            // Create a pattern with sharp transitions
            let region = match (x * 4 / width.max(1), y * 4 / height.max(1)) {
                (0, 0) | (2, 2) => [255, 0, 0],     // Red
                (1, 0) | (3, 2) => [0, 255, 0],     // Green
                (0, 1) | (2, 3) => [0, 0, 255],     // Blue
                (1, 1) | (3, 3) => [255, 255, 0],   // Yellow
                (2, 0) | (0, 2) => [255, 0, 255],   // Magenta
                (3, 0) | (1, 2) => [0, 255, 255],   // Cyan
                (2, 1) | (0, 3) => [255, 255, 255], // White
                _ => [0, 0, 0],                     // Black
            };
            pixels.extend_from_slice(&region);
        }
    }
    pixels
}

/// Generate a test pattern similar to a TV test card.
/// Contains bars, gradients, and sharp edges.
pub fn test_pattern(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);

    for y in 0..height {
        for x in 0..width {
            let color = if y < height / 3 {
                // Top third: color bars
                let bar = (x * 8 / width.max(1)) as usize;
                match bar {
                    0 => [255, 255, 255], // White
                    1 => [255, 255, 0],   // Yellow
                    2 => [0, 255, 255],   // Cyan
                    3 => [0, 255, 0],     // Green
                    4 => [255, 0, 255],   // Magenta
                    5 => [255, 0, 0],     // Red
                    6 => [0, 0, 255],     // Blue
                    _ => [0, 0, 0],       // Black
                }
            } else if y < 2 * height / 3 {
                // Middle third: grayscale gradient
                let v = ((x * 255) / width.max(1)) as u8;
                [v, v, v]
            } else {
                // Bottom third: checkerboard
                let cell = 8u32;
                let cx = x / cell;
                let cy = (y - 2 * height / 3) / cell;
                if (cx + cy) % 2 == 0 {
                    [255, 255, 255]
                } else {
                    [0, 0, 0]
                }
            };
            pixels.extend_from_slice(&color);
        }
    }
    pixels
}

/// Predefined test image dimensions for edge case testing.
pub const EDGE_CASE_DIMENSIONS: &[(u32, u32, &str)] = &[
    (1, 1, "minimum"),
    (2, 2, "tiny"),
    (7, 7, "not_power_of_2"),
    (8, 8, "single_mcu"),
    (9, 9, "just_over_mcu"),
    (16, 16, "two_mcus"),
    (15, 17, "odd_dimensions"),
    (1, 100, "tall_narrow"),
    (100, 1, "wide_short"),
    (256, 256, "standard_small"),
    (512, 512, "standard_medium"),
    (1000, 1000, "large_non_pow2"),
    (1024, 1024, "large_pow2"),
];

/// Generate a set of standard test images for comprehensive testing.
/// Returns (name, width, height, RGB pixels).
#[allow(clippy::vec_init_then_push)]
pub fn generate_test_suite() -> Vec<(String, u32, u32, Vec<u8>)> {
    let mut suite = Vec::new();

    // Solid colors
    suite.push((
        "solid_black".to_string(),
        64,
        64,
        solid_color(64, 64, 0, 0, 0),
    ));
    suite.push((
        "solid_white".to_string(),
        64,
        64,
        solid_color(64, 64, 255, 255, 255),
    ));
    suite.push((
        "solid_red".to_string(),
        64,
        64,
        solid_color(64, 64, 255, 0, 0),
    ));
    suite.push((
        "solid_green".to_string(),
        64,
        64,
        solid_color(64, 64, 0, 255, 0),
    ));
    suite.push((
        "solid_blue".to_string(),
        64,
        64,
        solid_color(64, 64, 0, 0, 255),
    ));

    // Gradients
    suite.push((
        "gradient_h".to_string(),
        256,
        64,
        gradient_horizontal(256, 64),
    ));
    suite.push((
        "gradient_v".to_string(),
        64,
        256,
        gradient_vertical(64, 256),
    ));
    suite.push((
        "gradient_d".to_string(),
        256,
        256,
        gradient_diagonal(256, 256),
    ));
    suite.push(("gradient_rgb".to_string(), 256, 256, gradient_rgb(256, 256)));
    suite.push((
        "gradient_radial".to_string(),
        256,
        256,
        gradient_radial(256, 256),
    ));

    // Patterns
    suite.push(("checker_8".to_string(), 256, 256, checkerboard(256, 256, 8)));
    suite.push(("checker_1".to_string(), 64, 64, high_frequency(64, 64)));
    suite.push((
        "stripes_h".to_string(),
        256,
        64,
        stripes_horizontal(256, 64, 4),
    ));
    suite.push((
        "stripes_v".to_string(),
        64,
        256,
        stripes_vertical(64, 256, 4),
    ));

    // Complex patterns
    suite.push(("sharp_edges".to_string(), 256, 256, sharp_edges(256, 256)));
    suite.push(("test_pattern".to_string(), 512, 512, test_pattern(512, 512)));

    // Noise
    suite.push(("noise_42".to_string(), 64, 64, noise(64, 64, 42)));

    // Edge case dimensions
    for &(w, h, name) in EDGE_CASE_DIMENSIONS.iter().take(8) {
        suite.push((format!("dim_{name}"), w, h, gradient_rgb(w, h)));
    }

    suite
}

/// Generate a minimal test suite for quick smoke tests.
pub fn generate_minimal_test_suite() -> Vec<(String, u32, u32, Vec<u8>)> {
    vec![
        (
            "solid".to_string(),
            32,
            32,
            solid_color(32, 32, 128, 128, 128),
        ),
        ("gradient".to_string(), 64, 64, gradient_rgb(64, 64)),
        ("checker".to_string(), 32, 32, checkerboard(32, 32, 8)),
        ("1x1".to_string(), 1, 1, solid_color(1, 1, 255, 0, 0)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solid_color_size() {
        let pixels = solid_color(10, 20, 255, 128, 0);
        assert_eq!(pixels.len(), 10 * 20 * 3);
    }

    #[test]
    fn test_gradient_values() {
        let pixels = gradient_horizontal(256, 1);
        // First pixel should be dark
        assert!(pixels[0] < 10);
        // Last pixel should be bright
        assert!(pixels[pixels.len() - 3] > 245);
    }

    #[test]
    fn test_checkerboard_pattern() {
        let pixels = checkerboard(4, 4, 2);
        // Top-left 2x2 should be white
        assert_eq!(pixels[0], 255);
        // Next 2x2 should be black
        assert_eq!(pixels[6], 0);
    }

    #[test]
    fn test_noise_deterministic() {
        let a = noise(32, 32, 12345);
        let b = noise(32, 32, 12345);
        assert_eq!(a, b);
    }

    #[test]
    fn test_generate_test_suite() {
        let suite = generate_test_suite();
        assert!(!suite.is_empty());
        for (name, w, h, pixels) in suite {
            assert_eq!(
                pixels.len(),
                (w * h * 3) as usize,
                "Size mismatch for {name}"
            );
        }
    }
}
