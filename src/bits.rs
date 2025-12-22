//! Bit-level I/O utilities for binary encoding.

/// A bit writer that packs bits into bytes, LSB first (for DEFLATE).
#[derive(Debug)]
pub struct BitWriter {
    buffer: Vec<u8>,
    current_byte: u8,
    bit_position: u8,
}

impl BitWriter {
    /// Create a new bit writer with default capacity.
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    /// Create a new bit writer with specified byte capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            current_byte: 0,
            bit_position: 0,
        }
    }

    /// Write bits to the stream, LSB first.
    ///
    /// # Arguments
    /// * `value` - The value to write (only lower `num_bits` are used)
    /// * `num_bits` - Number of bits to write (1-32)
    #[inline]
    pub fn write_bits(&mut self, value: u32, num_bits: u8) {
        debug_assert!(num_bits <= 32);

        let mut value = value;
        let mut remaining = num_bits;

        while remaining > 0 {
            let available = 8 - self.bit_position;
            let to_write = remaining.min(available);

            // Extract the bits we want to write
            let mask = (1u32 << to_write) - 1;
            let bits = (value & mask) as u8;

            // Add to current byte at the correct position
            self.current_byte |= bits << self.bit_position;

            self.bit_position += to_write;
            value >>= to_write;
            remaining -= to_write;

            // If byte is full, flush it
            if self.bit_position == 8 {
                self.buffer.push(self.current_byte);
                self.current_byte = 0;
                self.bit_position = 0;
            }
        }
    }

    /// Write a single bit.
    #[inline]
    pub fn write_bit(&mut self, bit: bool) {
        self.write_bits(bit as u32, 1);
    }

    /// Write a byte-aligned value (flushes partial byte first with zeros).
    pub fn write_byte(&mut self, byte: u8) {
        if self.bit_position == 0 {
            self.buffer.push(byte);
        } else {
            self.write_bits(byte as u32, 8);
        }
    }

    /// Write multiple bytes.
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.write_byte(byte);
        }
    }

    /// Flush any remaining bits, padding with zeros.
    pub fn flush(&mut self) {
        if self.bit_position > 0 {
            self.buffer.push(self.current_byte);
            self.current_byte = 0;
            self.bit_position = 0;
        }
    }

    /// Consume the writer and return the byte buffer.
    pub fn finish(mut self) -> Vec<u8> {
        self.flush();
        self.buffer
    }

    /// Get the current length in bytes (not counting partial byte).
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the writer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty() && self.bit_position == 0
    }

    /// Get current bit position within the current byte.
    pub fn bit_position(&self) -> u8 {
        self.bit_position
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// A fast bit writer using a 64-bit accumulator (LSB-first) for DEFLATE.
#[derive(Debug)]
pub struct BitWriter64 {
    buffer: Vec<u8>,
    acc: u64,
    bits_in_acc: u8,
}

impl BitWriter64 {
    /// Create a new bit writer with default capacity.
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    /// Create a new bit writer with specified byte capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            acc: 0,
            bits_in_acc: 0,
        }
    }

    /// Write bits to the stream, LSB first.
    #[inline]
    pub fn write_bits(&mut self, value: u32, num_bits: u8) {
        debug_assert!(num_bits <= 32);
        debug_assert!(self.bits_in_acc <= 63);
        let val64 = (value & ((1u32 << num_bits) - 1)) as u64;
        self.acc |= val64 << self.bits_in_acc;
        self.bits_in_acc += num_bits;

        // Flush full bytes
        while self.bits_in_acc >= 8 {
            self.buffer.push(self.acc as u8);
            self.acc >>= 8;
            self.bits_in_acc -= 8;
        }
    }

    /// Write a single bit.
    #[inline]
    pub fn write_bit(&mut self, bit: bool) {
        self.write_bits(bit as u32, 1);
    }

    /// Flush any remaining bits, padding with zeros.
    pub fn flush(&mut self) {
        if self.bits_in_acc > 0 {
            self.buffer.push(self.acc as u8);
            self.acc = 0;
            self.bits_in_acc = 0;
        }
    }

    /// Consume the writer and return the byte buffer.
    pub fn finish(mut self) -> Vec<u8> {
        self.flush();
        self.buffer
    }

    /// Get the current length in bytes (not counting partial byte).
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the writer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty() && self.bits_in_acc == 0
    }
}

/// A bit writer that packs bits MSB first (for JPEG).
#[derive(Debug)]
pub struct BitWriterMsb {
    buffer: Vec<u8>,
    current_byte: u8,
    bit_position: u8, // Counts from 8 down to 0
}

impl BitWriterMsb {
    /// Create a new MSB bit writer.
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    /// Create a new MSB bit writer with specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            current_byte: 0,
            bit_position: 8,
        }
    }

    /// Write bits to the stream, MSB first.
    ///
    /// Optimized to process multiple bits at once instead of bit-by-bit.
    #[inline]
    pub fn write_bits(&mut self, value: u32, num_bits: u8) {
        debug_assert!(num_bits <= 32);

        let mut remaining = num_bits;
        let val = value;

        while remaining > 0 {
            let space = self.bit_position;
            let to_write = remaining.min(space);

            // Extract the top `to_write` bits from the remaining value
            // and place them in the current byte at the correct position
            let shift = remaining - to_write;
            let mask = (1u32 << to_write) - 1;
            let bits = ((val >> shift) & mask) as u8;

            self.bit_position -= to_write;
            self.current_byte |= bits << self.bit_position;
            remaining -= to_write;

            // If byte is full, flush it with JPEG byte stuffing
            if self.bit_position == 0 {
                self.flush_byte_with_stuffing();
            }
        }
    }

    /// Flush the current byte and apply JPEG byte stuffing if needed.
    #[inline]
    fn flush_byte_with_stuffing(&mut self) {
        self.buffer.push(self.current_byte);
        // JPEG byte stuffing: if we wrote 0xFF, add 0x00
        if self.current_byte == 0xFF {
            self.buffer.push(0x00);
        }
        self.current_byte = 0;
        self.bit_position = 8;
    }

    /// Write a single bit.
    #[inline]
    pub fn write_bit(&mut self, bit: bool) {
        self.write_bits(bit as u32, 1);
    }

    /// Flush remaining bits, padding with 1s (as per JPEG spec).
    pub fn flush(&mut self) {
        if self.bit_position < 8 {
            // Pad with 1s
            self.current_byte |= (1 << self.bit_position) - 1;
            self.buffer.push(self.current_byte);
            if self.current_byte == 0xFF {
                self.buffer.push(0x00);
            }
            self.current_byte = 0;
            self.bit_position = 8;
        }
    }

    /// Consume the writer and return the byte buffer.
    pub fn finish(mut self) -> Vec<u8> {
        self.flush();
        self.buffer
    }

    /// Write raw bytes (must be byte-aligned).
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        debug_assert_eq!(self.bit_position, 8, "Must be byte-aligned");
        self.buffer.extend_from_slice(bytes);
    }

    /// Get the current buffer length.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty() && self.bit_position == 8
    }
}

impl Default for BitWriterMsb {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_writer_single_bits() {
        let mut writer = BitWriter::new();
        // Write 8 bits: 10110100 LSB first
        writer.write_bit(false); // bit 0
        writer.write_bit(false); // bit 1
        writer.write_bit(true); // bit 2
        writer.write_bit(false); // bit 3
        writer.write_bit(true); // bit 4
        writer.write_bit(true); // bit 5
        writer.write_bit(false); // bit 6
        writer.write_bit(true); // bit 7

        let result = writer.finish();
        assert_eq!(result, vec![0b10110100]);
    }

    #[test]
    fn test_bit_writer_multi_bits() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b101, 3);
        writer.write_bits(0b11, 2);
        writer.write_bits(0b001, 3);

        let result = writer.finish();
        // LSB first: 101 + 11 + 001 = 00111101
        assert_eq!(result, vec![0b00111101]);
    }

    #[test]
    fn test_bit_writer_cross_byte() {
        let mut writer = BitWriter::new();
        writer.write_bits(0xFF, 8);
        writer.write_bits(0x0F, 4);

        let result = writer.finish();
        assert_eq!(result, vec![0xFF, 0x0F]);
    }

    #[test]
    fn test_bit_writer_msb() {
        let mut writer = BitWriterMsb::new();
        writer.write_bits(0b10110100, 8);

        let result = writer.finish();
        assert_eq!(result, vec![0b10110100]);
    }

    #[test]
    fn test_bit_writer_msb_partial() {
        let mut writer = BitWriterMsb::new();
        writer.write_bits(0b101, 3);

        let result = writer.finish();
        // MSB first, padded with 1s: 101_11111 = 0xBF
        assert_eq!(result, vec![0b10111111]);
    }
}
