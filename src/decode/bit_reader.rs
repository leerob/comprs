//! Bit-level reader for DEFLATE and JPEG decoding.
//!
//! Provides efficient bit reading with peek/consume semantics for Huffman decoding.

use crate::error::{Error, Result};

/// Bit reader for LSB-first bit streams (DEFLATE).
///
/// Maintains a bit buffer filled from the input byte stream.
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_buf: u64,
    bits_in_buf: u8,
}

impl<'a> BitReader<'a> {
    /// Create a new bit reader from a byte slice.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bit_buf: 0,
            bits_in_buf: 0,
        }
    }

    /// Ensure at least `n` bits are available in the buffer.
    #[inline]
    fn ensure(&mut self, n: u8) -> Result<()> {
        while self.bits_in_buf < n {
            if self.pos >= self.data.len() {
                return Err(Error::InvalidDecode("unexpected end of stream".into()));
            }
            self.bit_buf |= (self.data[self.pos] as u64) << self.bits_in_buf;
            self.pos += 1;
            self.bits_in_buf += 8;
        }
        Ok(())
    }

    /// Peek at the next `n` bits without consuming them (LSB-first).
    #[inline]
    pub fn peek_bits(&mut self, n: u8) -> Result<u32> {
        debug_assert!(n <= 32);
        self.ensure(n)?;
        Ok((self.bit_buf & ((1u64 << n) - 1)) as u32)
    }

    /// Try to peek at `n` bits, returning available bits and count if stream ends.
    /// Returns Ok((bits, actual_count)) where actual_count <= n.
    /// This is useful for Huffman decoding near end of stream.
    #[inline]
    pub fn try_peek_bits(&mut self, n: u8) -> Result<(u32, u8)> {
        debug_assert!(n <= 32);
        // Try to load as many bits as possible
        while self.bits_in_buf < n && self.pos < self.data.len() {
            self.bit_buf |= (self.data[self.pos] as u64) << self.bits_in_buf;
            self.pos += 1;
            self.bits_in_buf += 8;
        }
        let actual = self.bits_in_buf.min(n);
        Ok(((self.bit_buf & ((1u64 << actual) - 1)) as u32, actual))
    }

    /// Consume `n` bits from the buffer.
    #[inline]
    pub fn consume(&mut self, n: u8) {
        debug_assert!(n <= self.bits_in_buf);
        self.bit_buf >>= n;
        self.bits_in_buf -= n;
    }

    /// Read `n` bits LSB-first (for DEFLATE).
    #[inline]
    pub fn read_bits(&mut self, n: u8) -> Result<u32> {
        let val = self.peek_bits(n)?;
        self.consume(n);
        Ok(val)
    }

    /// Align to byte boundary (discard remaining bits in current byte).
    pub fn align_to_byte(&mut self) {
        let discard = self.bits_in_buf % 8;
        if discard > 0 {
            self.bit_buf >>= discard;
            self.bits_in_buf -= discard;
        }
    }

    /// Read a byte directly (after aligning).
    #[allow(dead_code)]
    pub fn read_byte(&mut self) -> Result<u8> {
        self.read_bits(8).map(|v| v as u8)
    }

    /// Read bytes directly into a buffer (assumes byte-aligned).
    pub fn read_bytes(&mut self, buf: &mut [u8]) -> Result<()> {
        // First drain any buffered bits
        let buffered_bytes = (self.bits_in_buf / 8) as usize;
        let from_buf = buffered_bytes.min(buf.len());
        for byte in buf.iter_mut().take(from_buf) {
            *byte = (self.bit_buf & 0xFF) as u8;
            self.bit_buf >>= 8;
            self.bits_in_buf -= 8;
        }

        // Read remaining directly from input
        let remaining = &mut buf[from_buf..];
        if self.pos + remaining.len() > self.data.len() {
            return Err(Error::InvalidDecode("unexpected end of stream".into()));
        }
        remaining.copy_from_slice(&self.data[self.pos..self.pos + remaining.len()]);
        self.pos += remaining.len();
        Ok(())
    }

    /// Check if we've reached the end of the stream.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.pos >= self.data.len() && self.bits_in_buf == 0
    }

    /// Get current byte position (for debugging).
    #[allow(dead_code)]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Get remaining bytes in stream.
    #[allow(dead_code)]
    pub fn remaining_bytes(&self) -> usize {
        self.data.len() - self.pos + (self.bits_in_buf as usize / 8)
    }
}

/// Bit reader for MSB-first bit streams (JPEG).
///
/// JPEG uses MSB-first bit ordering and has special byte-stuffing rules
/// where 0xFF bytes are followed by 0x00 which must be skipped.
pub struct MsbBitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_buf: u32,
    bits_in_buf: u8,
}

impl<'a> MsbBitReader<'a> {
    /// Create a new MSB-first bit reader.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bit_buf: 0,
            bits_in_buf: 0,
        }
    }

    /// Read the next byte, handling JPEG byte stuffing.
    fn next_byte(&mut self) -> Result<u8> {
        if self.pos >= self.data.len() {
            return Err(Error::InvalidDecode("unexpected end of JPEG data".into()));
        }
        let byte = self.data[self.pos];
        self.pos += 1;

        // Handle JPEG byte stuffing: 0xFF 0x00 -> 0xFF
        if byte == 0xFF {
            if self.pos >= self.data.len() {
                return Err(Error::InvalidDecode("unexpected end after 0xFF".into()));
            }
            let next = self.data[self.pos];
            if next == 0x00 {
                // Stuffed byte, consume the 0x00
                self.pos += 1;
            } else if (0xD0..=0xD7).contains(&next) {
                // Restart marker (RST0-RST7) - skip it entirely and continue.
                // These markers reset the DC predictor but contain no data.
                // IMPORTANT: Clear the bit buffer because the bitstream is padded
                // with 1-bits to byte boundary before restart markers, and decoding
                // resumes byte-aligned after the marker.
                self.pos += 1;
                self.bit_buf = 0;
                self.bits_in_buf = 0;
                return self.next_byte();
            } else {
                // Other marker - indicates end of entropy-coded data
                // Back up so caller can see the marker
                self.pos -= 1;
                return Err(Error::InvalidDecode("marker in entropy data".into()));
            }
        }
        Ok(byte)
    }

    /// Ensure at least `n` bits are available in the buffer.
    #[inline]
    fn ensure(&mut self, n: u8) -> Result<()> {
        while self.bits_in_buf < n {
            let byte = self.next_byte()?;
            // MSB-first: new byte goes to the left
            self.bit_buf = (self.bit_buf << 8) | (byte as u32);
            self.bits_in_buf += 8;
        }
        Ok(())
    }

    /// Peek at the next `n` bits without consuming them (MSB-first).
    #[inline]
    pub fn peek_bits(&mut self, n: u8) -> Result<u32> {
        debug_assert!(n <= 25); // Limited by buffer size minus overhead
        self.ensure(n)?;
        // Extract top n bits
        Ok((self.bit_buf >> (self.bits_in_buf - n)) & ((1 << n) - 1))
    }

    /// Consume `n` bits from the buffer.
    #[inline]
    pub fn consume(&mut self, n: u8) {
        debug_assert!(n <= self.bits_in_buf);
        self.bits_in_buf -= n;
        // Clear consumed bits (use checked_shl to avoid overflow when bits_in_buf is 32)
        self.bit_buf &= 1u32
            .checked_shl(self.bits_in_buf as u32)
            .unwrap_or(0)
            .wrapping_sub(1);
    }

    /// Read `n` bits MSB-first.
    #[inline]
    pub fn read_bits(&mut self, n: u8) -> Result<u32> {
        let val = self.peek_bits(n)?;
        self.consume(n);
        Ok(val)
    }

    /// Get current position in data.
    #[allow(dead_code)]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Check if we can read more bits.
    #[allow(dead_code)]
    pub fn has_more(&self) -> bool {
        self.pos < self.data.len() || self.bits_in_buf > 0
    }

    /// Skip to a specific position (for restart markers).
    #[allow(dead_code)]
    pub fn seek(&mut self, pos: usize) {
        self.pos = pos;
        self.bit_buf = 0;
        self.bits_in_buf = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_reader_basic() {
        let data = [0b10110100, 0b11001010];
        let mut reader = BitReader::new(&data);

        // Read 4 bits: should get 0b0100 (LSB first from 0b10110100)
        assert_eq!(reader.read_bits(4).unwrap(), 0b0100);
        // Read 4 more: should get 0b1011
        assert_eq!(reader.read_bits(4).unwrap(), 0b1011);
        // Read 8 more: should get 0b11001010
        assert_eq!(reader.read_bits(8).unwrap(), 0b11001010);
    }

    #[test]
    fn test_bit_reader_peek_consume() {
        let data = [0b10110100];
        let mut reader = BitReader::new(&data);

        // Peek without consuming
        assert_eq!(reader.peek_bits(4).unwrap(), 0b0100);
        assert_eq!(reader.peek_bits(4).unwrap(), 0b0100); // Same value

        // Now consume
        reader.consume(4);
        assert_eq!(reader.peek_bits(4).unwrap(), 0b1011);
    }

    #[test]
    fn test_bit_reader_align() {
        let data = [0xFF, 0xAB];
        let mut reader = BitReader::new(&data);

        // Read 3 bits
        reader.read_bits(3).unwrap();
        // Align to byte
        reader.align_to_byte();
        // Should now read second byte
        assert_eq!(reader.read_byte().unwrap(), 0xAB);
    }

    #[test]
    fn test_bit_reader_align_already_aligned() {
        let data = [0xAB, 0xCD];
        let mut reader = BitReader::new(&data);

        // Read 8 bits (aligned)
        reader.read_bits(8).unwrap();
        // Align should do nothing
        reader.align_to_byte();
        // Should read next byte
        assert_eq!(reader.read_byte().unwrap(), 0xCD);
    }

    #[test]
    fn test_bit_reader_read_bytes() {
        let data = [0x01, 0x02, 0x03, 0x04];
        let mut reader = BitReader::new(&data);

        let mut buf = [0u8; 4];
        reader.read_bytes(&mut buf).unwrap();
        assert_eq!(buf, [0x01, 0x02, 0x03, 0x04]);
    }

    #[test]
    fn test_bit_reader_read_bytes_partial() {
        let data = [0x01, 0x02, 0x03, 0x04, 0x05];
        let mut reader = BitReader::new(&data);

        let mut buf = [0u8; 3];
        reader.read_bytes(&mut buf).unwrap();
        assert_eq!(buf, [0x01, 0x02, 0x03]);

        let mut buf2 = [0u8; 2];
        reader.read_bytes(&mut buf2).unwrap();
        assert_eq!(buf2, [0x04, 0x05]);
    }

    #[test]
    fn test_bit_reader_eof() {
        let data = [0xFF];
        let mut reader = BitReader::new(&data);

        reader.read_bits(8).unwrap();
        assert!(reader.read_bits(1).is_err());
    }

    #[test]
    fn test_bit_reader_empty() {
        let data: &[u8] = &[];
        let mut reader = BitReader::new(data);

        assert!(reader.read_bits(1).is_err());
        assert!(reader.is_empty());
    }

    #[test]
    fn test_bit_reader_position() {
        let data = [0x01, 0x02, 0x03];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.position(), 0);
        reader.read_bits(8).unwrap();
        assert_eq!(reader.position(), 1);
    }

    #[test]
    fn test_bit_reader_remaining_bytes() {
        let data = [0x01, 0x02, 0x03, 0x04];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.remaining_bytes(), 4);
        reader.read_bits(16).unwrap();
        // After reading 16 bits, 2 bytes consumed
        assert_eq!(reader.remaining_bytes(), 2);
    }

    #[test]
    fn test_msb_reader_basic() {
        let data = [0b10110100];
        let mut reader = MsbBitReader::new(&data);

        // Read 4 bits MSB-first: should get 0b1011
        assert_eq!(reader.read_bits(4).unwrap(), 0b1011);
        // Read 4 more: should get 0b0100
        assert_eq!(reader.read_bits(4).unwrap(), 0b0100);
    }

    #[test]
    fn test_msb_reader_byte_stuffing() {
        // 0xFF followed by 0x00 should yield just 0xFF
        let data = [0xFF, 0x00, 0xAB];
        let mut reader = MsbBitReader::new(&data);

        assert_eq!(reader.read_bits(8).unwrap(), 0xFF);
        assert_eq!(reader.read_bits(8).unwrap(), 0xAB);
    }

    #[test]
    fn test_msb_reader_peek_consume() {
        let data = [0b11001100];
        let mut reader = MsbBitReader::new(&data);

        assert_eq!(reader.peek_bits(4).unwrap(), 0b1100);
        reader.consume(2);
        assert_eq!(reader.peek_bits(4).unwrap(), 0b0011);
    }

    #[test]
    fn test_msb_reader_position() {
        let data = [0x01, 0x02, 0x03];
        let mut reader = MsbBitReader::new(&data);

        assert_eq!(reader.position(), 0);
        reader.read_bits(8).unwrap();
        assert_eq!(reader.position(), 1);
    }

    #[test]
    fn test_msb_reader_has_more() {
        // Use a non-0xFF byte to avoid byte stuffing complexity
        let data = [0xAB];
        let mut reader = MsbBitReader::new(&data);

        assert!(reader.has_more());
        reader.read_bits(8).unwrap();
        assert!(!reader.has_more());
    }

    #[test]
    fn test_msb_reader_seek() {
        let data = [0x01, 0x02, 0x03];
        let mut reader = MsbBitReader::new(&data);

        reader.read_bits(8).unwrap();
        assert_eq!(reader.position(), 1);

        reader.seek(0);
        assert_eq!(reader.position(), 0);
        assert_eq!(reader.read_bits(8).unwrap(), 0x01);
    }

    #[test]
    fn test_msb_reader_empty() {
        let data: &[u8] = &[];
        let mut reader = MsbBitReader::new(data);

        assert!(!reader.has_more());
        assert!(reader.read_bits(1).is_err());
    }

    #[test]
    fn test_msb_reader_cross_byte() {
        let data = [0b11110000, 0b00001111];
        let mut reader = MsbBitReader::new(&data);

        // Read 6 bits from first byte
        assert_eq!(reader.read_bits(6).unwrap(), 0b111100);
        // Read 6 bits crossing the byte boundary
        assert_eq!(reader.read_bits(6).unwrap(), 0b000000);
    }

    #[test]
    fn test_msb_reader_consume_zero_with_full_buffer() {
        // Verify that consume(0) doesn't overflow when buffer has 32 bits.
        // peek_bits(25) requires 4 bytes to fill buffer (32 bits total).
        let data = [0x12, 0x34, 0x56, 0x78, 0x9A];
        let mut reader = MsbBitReader::new(&data);
        reader.peek_bits(25).unwrap(); // Forces buffer to have 32 bits
        reader.consume(0); // Previously caused 1 << 32 overflow
                           // Buffer should still have all 32 bits intact
        assert_eq!(reader.peek_bits(8).unwrap(), 0x12);
    }

    #[test]
    fn test_msb_reader_restart_marker_clears_bit_buffer() {
        // Simulate JPEG restart marker scenario:
        // - Read some bits (leaving residual bits in buffer)
        // - Encounter restart marker (0xFF 0xD0)
        // - Verify bit buffer is cleared and next byte is read fresh
        //
        // Data: 0xAB (partial read), 0xFF 0xD0 (RST0 marker), 0xCD (fresh data)
        // Before fix: residual bits from 0xAB would corrupt reading of 0xCD
        let data = [0xAB, 0xFF, 0xD0, 0xCD];
        let mut reader = MsbBitReader::new(&data);

        // Read 4 bits from first byte (0xAB = 0b10101011)
        // This leaves 4 bits (0b1011) in the buffer
        assert_eq!(reader.read_bits(4).unwrap(), 0b1010);

        // Now read 8 bits - this will:
        // 1. Try to get more bits via ensure()
        // 2. Call next_byte() which sees 0xFF 0xD0 (restart marker)
        // 3. With the fix: clears bit buffer, returns 0xCD
        // 4. Without fix: would have stale bits corrupting the result
        let next_byte = reader.read_bits(8).unwrap();

        // After restart marker, bit buffer should be cleared and we get 0xCD fresh
        assert_eq!(next_byte, 0xCD);
    }

    #[test]
    fn test_msb_reader_restart_marker_mid_byte() {
        // Test that restart marker properly resets even with full byte in buffer
        // Data: 0x12, 0x34, 0xFF, 0xD1 (RST1), 0xAB
        let data = [0x12, 0x34, 0xFF, 0xD1, 0xAB];
        let mut reader = MsbBitReader::new(&data);

        // Read 16 bits (consumes 0x12 and 0x34)
        assert_eq!(reader.read_bits(16).unwrap(), 0x1234);

        // Next read will hit the restart marker
        // Should get 0xAB with no residual bits
        assert_eq!(reader.read_bits(8).unwrap(), 0xAB);
    }
}
