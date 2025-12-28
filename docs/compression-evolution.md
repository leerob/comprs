# The Evolution of Compression: How Small Gains Compound

This document explores the history and philosophy behind advanced compression techniques. Rather than implementation details, we focus on the _ideas_ that drive compression improvements and the patterns they share.

## The Fundamental Challenge

All compression faces the same core tension:

```text
┌─────────────────────────────────────────────────────────────────┐
│                    The Compression Tradeoff                      │
│                                                                  │
│    Information Theory Limit                                      │
│    ════════════════════════                                      │
│    Claude Shannon proved (1948) that every data source has       │
│    a minimum number of bits required to represent it.            │
│    You cannot do better than this limit.                         │
│                                                                  │
│    The Game                                                      │
│    ════════                                                      │
│    Get as close to the theoretical limit as possible,            │
│    while balancing:                                              │
│    • Encoding speed                                              │
│    • Decoding speed                                              │
│    • Memory usage                                                │
│    • Implementation complexity                                   │
└─────────────────────────────────────────────────────────────────┘
```

Modern codecs like mozjpeg don't use one clever trick. They use _dozens_ of small optimizations that each shave off 1-5%. These compound:

```text
0.95 × 0.97 × 0.95 × 0.98 × 0.96 = 0.82
```

Five optimizations of 2-5% each combine to reduce file size by 18%.

---

## The Three Pillars of Compression Improvement

Every major compression advancement falls into one of three categories:

### 1. Better Representation

**Core idea**: Transform data into a form that's easier to compress.

The classic example is the **Discrete Cosine Transform (DCT)** used in JPEG. Natural images have a property: nearby pixels are usually similar. The DCT exploits this by converting spatial data (pixel values) into frequency data (how fast values change).

```text
Spatial Domain          Frequency Domain
┌───────────────┐      ┌───────────────┐
│ 52 55 61 66   │      │ 420  -8  -2   │
│ 70 61 64 73   │ DCT  │  -3   5  -1   │
│ 63 59 55 90   │ ───► │   2  -4   1   │
│ 67 61 68 104  │      │   1   0  -1   │
└───────────────┘      └───────────────┘

Most energy concentrates in the top-left corner.
The rest becomes small numbers, often zero.
```

**Why it works**: Images don't change randomly. Smooth gradients become a single number (the DC coefficient). Only edges and textures need the high-frequency coefficients.

**Historical evolution**:

- 1974: DCT proposed by Ahmed, Natarajan, and Rao
- 1988: AAN fast algorithm reduces multiplications from 64 to 5
- 1990s: Integer DCT produces reproducible results across platforms

### 2. Smarter Encoding

**Core idea**: Use fewer bits for common patterns, more bits for rare patterns.

**Huffman coding** (1952) is the foundation. It assigns shorter codes to frequent symbols:

```text
Frequency-Based Codes
═════════════════════
Letter   Frequency   Naive Code   Huffman Code
  E        12.7%      00101         10
  T         9.1%      10100         110
  A         8.2%      00001         1110
  Z         0.07%     11010         1111111110

Common letters get 2-3 bits instead of 5.
Rare letters get more bits, but they're rare.
```

But Huffman has limits. It must assign whole bits, so a symbol with 90% probability still gets at least 1 bit when theory says 0.15 bits would suffice.

**Arithmetic coding** (1970s) solves this by treating the entire message as a single number between 0 and 1. In practice, it achieves 5-10% better compression than Huffman for many data types.

### 3. Rate-Distortion Optimization

**Core idea**: Accept some quality loss to save bits, but choose _which_ losses give the best tradeoff.

This is the most sophisticated approach. Instead of asking "how do I encode this data?", we ask "what data should I encode?"

```text
The R-D Framework
═════════════════
Every encoding decision has two costs:

  Rate (R):       How many bits does this choice cost?
  Distortion (D): How much quality does this choice lose?

The goal: Minimize R + λD

Where λ is a parameter that controls the tradeoff.
Higher λ → smaller files, more quality loss
Lower λ → larger files, less quality loss
```

This framework unifies many optimizations. Quantization, progressive encoding, and trellis optimization are all R-D problems in disguise.

---

## Case Study: The Journey from JPEG to mozjpeg

The original JPEG standard (1992) made practical choices for 1990s hardware. Over 30 years, each limitation became an optimization opportunity.

### Stage 1: Baseline JPEG (1992)

The original JPEG encoder:

- **Floating-point DCT**: Fast on FPUs, but results vary by platform
- **Simple quantization**: Round to nearest integer
- **Fixed Huffman tables**: Pre-defined, not optimized for the image
- **Sequential encoding**: Top-to-bottom, one pass

This was revolutionary for 1992. But each choice left performance on the table.

### Stage 2: Optimized Huffman Tables (1990s)

**The insight**: Pre-defined Huffman tables are generic. Tables tuned to the actual image content compress better.

**The technique**: Two-pass encoding

1. First pass: Count symbol frequencies
2. Build optimal Huffman tables from frequencies
3. Second pass: Encode with optimal tables

**Typical gain**: 2-5% smaller files

**Why it works**: Different images have different coefficient distributions. A sunset photo (smooth gradients) has very different statistics than a text document (sharp edges).

### Stage 3: Integer DCT (1990s-2000s)

**The problem**: Floating-point arithmetic varies by platform. The same image encoded on different machines produces slightly different files.

**The insight**: Fixed-point integer math is:

- Reproducible across platforms
- Often faster (no FPU needed)
- Can be tuned for optimal coefficient distribution

**The technique**: Scale all DCT constants by 2^13, use integer arithmetic, round carefully at each stage.

```text
Floating-point:  cos(π/8) = 0.9238795...
Fixed-point:     (0.9238795 × 8192) ≈ 7568

Multiplication becomes:
  result = (a × 7568) >> 13  // Shift replaces division
```

**Typical gain**: 1-3% from better coefficient distribution

**Why it works**: Careful rounding in integer math can actually produce _better_ coefficient distributions than floating-point, because the small rounding errors happen in predictable, controllable ways.

### Stage 4: Progressive Encoding (1990s-2000s)

**The insight**: Not all parts of an image are equally important. The overall brightness matters more than fine texture details.

**The technique**: Instead of encoding each 8×8 block completely before moving to the next, split the encoding into multiple "scans":

```text
Progressive Scan Structure
══════════════════════════

Scan 1: DC coefficients only (overall brightness)
        ┌─────────────────────────────────────┐
        │ ██                                  │  Just the average
        │                                     │  brightness of each
        │                                     │  8×8 block
        └─────────────────────────────────────┘

Scan 2: Low-frequency AC (basic shapes)
        ┌─────────────────────────────────────┐
        │ ██ ░░                               │  Basic gradients
        │ ░░                                  │  and large shapes
        │                                     │
        └─────────────────────────────────────┘

Scan 3-N: Higher frequencies (details, texture)
        ┌─────────────────────────────────────┐
        │ ██ ░░ ░░ ▒▒ ▒▒ ▒▒ ░░ ░░            │  Fine details
        │ ░░ ▒▒ ▒▒ ▒▒ ░░ ░░ ░░ ░░            │  and textures
        │ ░░ ▒▒ ░░ ░░ ░░ ░░ ░░               │
        └─────────────────────────────────────┘
```

**Typical gain**: 5-15% smaller files

**Why it works**:

1. **Better Huffman statistics**: Each scan contains similar data, improving compression
2. **Spectral correlation**: DC coefficients across blocks are similar, as are low-frequency AC coefficients
3. **User experience**: Image appears quickly (blurry), then sharpens progressively

### Stage 5: Trellis Quantization (2000s-2010s)

**The insight**: Simple rounding isn't optimal. Sometimes rounding _away_ from the nearest integer produces a smaller file with negligible quality loss.

**The problem in detail**:

```text
Quantization Decision
═════════════════════
DCT coefficient: 47.3
Quantization divisor: 10
Simple round: 47.3/10 = 4.73 → 5

But wait—what does "5" cost to encode?
- If the previous coefficient was 4, encoding "5" might cost 3 bits
- If the previous coefficient was 0, encoding "5" might cost 6 bits
- And encoding "4" instead only adds 0.09 distortion...

The optimal choice depends on context!
```

**The technique**: Model this as a graph problem

```text
Trellis Graph
═════════════
Each coefficient position has multiple candidates.
Edges connect candidates with their encoding costs.
Find the path that minimizes total (Rate + λ × Distortion).

Position 1        Position 2        Position 3
    [4]───────────────[2]───────────────[0]
     │╲               │╲               │
     │ ╲              │ ╲              │
    [5]──╲───────────[3]──╲───────────[1]
          ╲               ╲              ╲
          [6]─────────────[4]─────────────[2]

Use Viterbi algorithm to find optimal path.
```

**Typical gain**: 3-8% smaller files

**Why it works**: Huffman coding costs depend on context (run-length of zeros, magnitude of values). By considering these costs during quantization, we make globally better decisions.

### Stage 6: mozjpeg (2014-present)

mozjpeg combines all the above, plus:

- **Scan optimization**: Automatically select the best progressive scan script
- **DC coefficient optimization**: Special handling for DC differences
- **Quantization table tuning**: Tables optimized for human perception, not just PSNR

**Combined result**: 10-30% smaller than baseline JPEG at equivalent quality.

---

## Common Patterns Across Techniques

These compression improvements share several recurring themes:

### Pattern 1: Two-Pass Is Often Better Than One

Many optimizations require knowing something about the data before encoding it:

| Technique            | First Pass                   | Second Pass                    |
| -------------------- | ---------------------------- | ------------------------------ |
| Optimized Huffman    | Count frequencies            | Encode with optimal tables     |
| Trellis Quantization | Compute all DCT coefficients | Find optimal quantization path |
| Progressive          | Store all coefficients       | Encode in optimal scan order   |

**The tradeoff**: 2× encoding time for 5-15% size reduction. Usually worth it for storage/transmission.

### Pattern 2: Context Improves Compression

Information theory tells us: the more you know about what comes next, the fewer bits you need to encode it.

| Technique              | Context Used                            |
| ---------------------- | --------------------------------------- |
| Differential DC coding | Previous block's DC value               |
| Run-length encoding    | How many zeros came before              |
| Progressive scans      | Similar-frequency coefficients together |
| Trellis quantization   | Previous coefficient values             |

**The insight**: Don't just encode values—encode _differences_ from predictions.

### Pattern 3: Human Perception Is Non-Linear

Our eyes and brains don't perceive all information equally:

```text
What Humans Notice vs. What Compression Sees
════════════════════════════════════════════

High sensitivity:           Low sensitivity:
• Edges and contours        • Uniform areas
• Faces                     • High-frequency noise
• Text                      • Blue channel details
• Low-frequency patterns    • Areas with texture masking
```

Smart compression allocates bits where humans will notice, not uniformly.

### Pattern 4: Exact vs. Good Enough

Many optimizations distinguish between:

- **Mathematically optimal**: The provably best answer (often expensive to compute)
- **Good enough**: 99% as good, 10× faster to compute

| Technique         | Optimal Approach  | Practical Approach |
| ----------------- | ----------------- | ------------------ |
| Huffman tables    | Arithmetic coding | Canonical Huffman  |
| Trellis           | Full Viterbi      | Pruned trellis     |
| Progressive scans | Exhaustive search | Heuristic scripts  |

**The wisdom**: Diminishing returns are real. The last 1% improvement often costs 10× the computation.

### Pattern 5: Iterative Refinement

Some optimizations face a chicken-and-egg problem: the optimal choice at step A depends on what happens at step B, but step B depends on step A.

The classic example is **LZ77 + Huffman** (used in Deflate/PNG):

```text
The Chicken-and-Egg Problem
════════════════════════════
LZ77 parsing decides which matches to use.
Huffman coding assigns bit lengths based on symbol frequencies.

But: The best LZ77 parse depends on Huffman bit costs.
And: Huffman costs depend on which symbols LZ77 produces.

Which do we optimize first?
```

**The solution**: Iterate until convergence.

```text
Iterative Refinement (Zopfli-style)
═══════════════════════════════════

1. Initial pass:    Greedy LZ77 → baseline symbol frequencies
2. Calculate costs: Convert frequencies to bit lengths via entropy
3. Re-parse:        Optimal LZ77 using entropy-based costs
4. Iterate:         Repeat steps 2-3 for N iterations
5. Track best:      Keep the result with smallest actual size

Each iteration refines the cost model, enabling better parsing decisions.
Typically converges in 5-15 iterations.
```

| Iteration | What Happens                                              |
| --------- | --------------------------------------------------------- |
| 1         | Greedy parse, rough frequency estimates                   |
| 2-5       | Costs stabilize, parse improves significantly             |
| 5-15      | Diminishing returns, occasional escapes from local minima |

**Why it works**: The cost function and the parsing algorithm co-evolve toward a better solution. Neither alone finds the optimum, but together they converge.

**Real-world examples**:

- Zopfli uses 15 iterations by default
- Video codecs iterate between motion estimation and rate control
- Machine learning uses gradient descent (same principle: iterate to refine)

---

## The Compounding Effect

Here's why mature codecs like mozjpeg are hard to beat:

```text
Starting with baseline JPEG (100%)
═══════════════════════════════════

  Optimized Huffman tables:     ×0.97  →  97%
  Integer DCT precision:        ×0.98  →  95%
  Progressive encoding:         ×0.92  →  87%
  Trellis quantization:         ×0.95  →  83%
  Chroma subsampling:           ×0.85  →  71%
  DC coefficient optimization:  ×0.98  →  69%

  Final: ~70% of baseline size at same quality
```

Each technique contributes a small factor. Together, they're transformative.

---

## Applying These Lessons

When implementing compression:

1. **Start with the standard algorithm**. Understand why each step exists before optimizing.

2. **Measure before optimizing**. Profile to find where bits are actually being spent.

3. **Look for context**. Whatever you're encoding, ask: what did we just encode that predicts this?

4. **Consider R-D tradeoffs**. Not all bits are equal—some matter more than others.

5. **Accept diminishing returns**. The first 80% of improvement takes 20% of the effort. Know when to stop.

6. **Learn from existing work**. mozjpeg, zopfli, and other mature codecs encode decades of optimization wisdom.

---

## Further Reading

### Foundational Papers

- Shannon, C. E. (1948). "A Mathematical Theory of Communication"
- Huffman, D. (1952). "A Method for the Construction of Minimum-Redundancy Codes"
- Ahmed, Natarajan, Rao (1974). "Discrete Cosine Transform"
- Arai, Agui, Nakajima (1988). "A Fast DCT-SQ Scheme for Images"

### JPEG Specific

- ITU-T T.81: The original JPEG specification
- Lakhani, G. (2003). "Optimal Huffman Coding of DCT Blocks"
- mozjpeg documentation: <https://github.com/mozilla/mozjpeg>

### Rate-Distortion Theory

- Berger, T. (1971). "Rate Distortion Theory"
- Sullivan & Wiegand (1998). "Rate-Distortion Optimization for Video Compression"

### Deflate/PNG Specific

- RFC 1951: DEFLATE Compressed Data Format Specification
- Zopfli documentation: <https://github.com/google/zopfli>
- Pinho et al. (2004). "A note on Zeng's technique for color reindexing" (palette optimization)

### Implementation References

- libjpeg source code (especially jfdctint.c, jchuff.c)
- Independent JPEG Group documentation
- zopfli source code (especially squeeze.c for iterative refinement)
